import tensorflow as tf
import argparse
import fs
from fs.zipfs import ZipFS

# To avoid getting the 'No module named _tkinter, please install the python-tk package' error
# when running on GCP
import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt

import numpy as np
import os
from skimage.transform import resize

import trainer.model as model

REAL_DATASET_PATHS_FILE = "real-files.txt"
MASKED_DATASET_PATHS_FILE = "masked-files.txt"
REFERENCE_DATASET_PATHS_FILE = "reference-files.txt"

PATH_FILE_BUFFER_SIZE = 1000000

IMAGE_SIZE = 128
PATCH_SIZE = 32

DATASET_BUFFER = 10000
SHUFFLE_BUFFER_SIZE = 1000
PARALLEL_MAP_THREADS = 16

MAX_STEPS = 1e6
EPOCHS = 50
BATCHES_PER_PRINT = 20
BATCHES_PER_CHECKPOINT = 100

GEN_LEARNING_RATE = 1e-4
DISC_LEARNING_RATE = 1e-4

LAMBDA_REC = 1.0
LAMBDA_ADV_GEN = 0.0
LAMBDA_ADV_DISC = 1.0


def get_reference_image_path_fn(train_reference_paths_dict):
  def get_reference_image_path(image_path):
    # Need to do this because when calling this function using tf.py_func,
    # the image_path is passed as bytes instead of string.
    image_path = image_path.decode('UTF-8')

    identity = image_path.split('/')[-2]
    reference_paths = train_reference_paths_dict[identity]
    idx = np.random.randint(len(reference_paths))
    image_file_name = reference_paths[idx]

    return os.path.join(identity, image_file_name)

  return get_reference_image_path


def fix_image_encoding(image):
  if (image.ndim == 2):
    # Add new dimension for channels
    image = image[:, :, np.newaxis]
  if (image.shape[-1] == 1):
    # Convert greyscale to RGB
    image = np.concatenate((image,) * 3, axis=-1)
  return image


def create_reference_paths_dict(reference_dict_file_path):
  tf.logging.info('Creating reference paths dictionary')
  reference_dict = {}
  reference_images_file = tf.gfile.GFile(reference_dict_file_path, mode='r')
  for line in reference_images_file:
    split_line = line.rstrip('\n').split(':')
    identity = split_line[0]
    image_paths = split_line[1].split(',')
    reference_dict[identity] = image_paths
    assert len(image_paths) > 0
  tf.logging.info('Finished creating reference paths dictionary')
  return reference_dict


def get_mask_fn(img_size, patch_size, use_batch=False):
  patch_start = (img_size - patch_size) // 2
  img_size_after_patch = img_size - (patch_start + patch_size)

  def mask_fn(image):
    """
    Applies a mask of zeroes of size (patch_size x patch_size) at the center of the image.
    Returns a tuple of the masked image and the original image.
    """
    upper_edge = tf.ones([patch_start, img_size, 3], tf.float32)
    lower_edge = tf.ones([img_size_after_patch, img_size, 3], tf.float32)

    middle_left = tf.ones([patch_size, patch_start, 3], tf.float32)
    middle_right = tf.ones([patch_size, img_size_after_patch, 3],
                           tf.float32)

    zeros = tf.zeros([patch_size, patch_size, 3], tf.float32)

    middle = tf.concat([middle_left, zeros, middle_right], axis=1)
    mask = tf.concat([upper_edge, middle, lower_edge], axis=0)

    if use_batch:
      mask = tf.expand_dims(mask, axis=0)

    return image * mask

  return mask_fn


def patch_image(patch, image):
  """
  Apply the given patch to the image.
  The patch is applied at the center of the image, assuming a 7x7 patch and a 28x28 image.
  """

  patch_start = (IMAGE_SIZE - PATCH_SIZE) // 2
  patch_end = patch_start + PATCH_SIZE

  # TODO: See if this could be done more efficiently.

  upper_edge = image[:, :patch_start, :, :]
  lower_edge = image[:, patch_end:, :, :]

  middle_left = image[:, patch_start:patch_end, :patch_start, :]
  middle_right = image[:, patch_start:patch_end, patch_end:, :]

  middle = tf.concat([middle_left, patch, middle_right], axis=2)
  return tf.concat([upper_edge, middle, lower_edge], axis=1)


def generate_images(generator, images, reference_images):
  # make sure the training parameter is set to False because we
  # don't want to train the batchnorm layer when doing inference.
  mask_fn = get_mask_fn(IMAGE_SIZE, PATCH_SIZE, use_batch=True)
  mask_images = mask_fn(images)
  reference_images = tf.constant(reference_images, tf.float32)
  patches = generator([mask_images, reference_images], training=False)
  generated_images = patch_image(patches, mask_images)

  return generated_images


def train_step(sess, gen_optimizer, disc_optimizer, gen_loss, disc_loss,
               global_step):
  _, gen_loss_value = sess.run([gen_optimizer, gen_loss])
  _, disc_loss_value, step_value = sess.run(
    [disc_optimizer, disc_loss, global_step])

  # Divide by two because we increment the global_step twice per each train_step.
  if (step_value // 2) % (BATCHES_PER_PRINT // 2) == 0:
    tf.logging.info(
      "Step: {} - Gen_loss: {} - Disc_loss: {}".format(step_value,
                                                       gen_loss_value,
                                                       disc_loss_value))


def train(dataset, generator, discriminator, validation_images,
          validation_references, experiment_dir):
  global_step = tf.train.get_or_create_global_step()

  dataset = dataset.repeat()
  iterator = dataset.make_one_shot_iterator()

  train_batch = iterator.get_next()
  full_images = train_batch[0]
  (masked_images, unmasked_images, masked_reference_images) = train_batch[1]

  generated_patches = generator([masked_images, masked_reference_images],
                                training=True)
  generated_images = patch_image(generated_patches, masked_images)

  real_output = discriminator(full_images, training=True)
  generated_output = discriminator(
    [generated_images, masked_reference_images], training=True)

  gen_loss = model.generator_loss(unmasked_images, generated_images,
                                  generated_output, LAMBDA_REC,
                                  LAMBDA_ADV_GEN)
  disc_loss = model.discriminator_loss(real_output, generated_output,
                                       LAMBDA_ADV_DISC)

  gen_optimizer = tf.train.AdamOptimizer(GEN_LEARNING_RATE).minimize(
    gen_loss, var_list=generator.variables,
    global_step=global_step)
  disc_optimizer = tf.train.AdamOptimizer(DISC_LEARNING_RATE).minimize(
    disc_loss, var_list=discriminator.variables,
    global_step=global_step)

  tf.summary.scalar('gen_loss', gen_loss)
  tf.summary.scalar('disc_loss', disc_loss)
  tf.summary.image('generated_train_images', generated_images, max_outputs=9)

  # TODO see why validation images are not being generated correctly.
  # generated_validation_images = generate_images(generator,
  #                                               validation_images,
  #                                               validation_references)
  # tf.summary.image('generated_validation_images', generated_validation_images,
  #                  max_outputs=9)

  hooks = [tf.train.StopAtStepHook(num_steps=MAX_STEPS)]
  with tf.train.MonitoredTrainingSession(
          checkpoint_dir=os.path.join(experiment_dir, "train"),
          hooks=hooks) as sess:
    while not sess.should_stop():
      train_step(sess, gen_optimizer, disc_optimizer, gen_loss, disc_loss,
                 global_step)


def get_read_images_from_fs_fn(dataset_fs, base_path):
  def read_images_from_fs(image_path):
    image_path = image_path.decode('UTF-8')
    image_full_path = os.path.join(base_path, image_path)
    with dataset_fs.open(image_full_path, 'rb') as img_file:
      img_content = img_file.read()
    return img_content

  return read_images_from_fs


def get_load_and_preprocess_image_fn(dataset_fs, base_path, reference_base_path,
                                     reference_dict, masked=False):
  def load_and_preprocess_image(img_filename):
    image_content = tf.py_func(
      get_read_images_from_fs_fn(dataset_fs, base_path), [img_filename],
      tf.string)
    image = tf.image.decode_image(image_content, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_SIZE,
                                                   IMAGE_SIZE)
    image = tf.image.convert_image_dtype(image, tf.float32)

    if masked:
      reference_image_filename = tf.py_func(
        get_reference_image_path_fn(reference_dict),
        [img_filename], tf.string)
      reference_content = tf.py_func(
        get_read_images_from_fs_fn(dataset_fs, reference_base_path),
        [reference_image_filename],
        tf.string)

      reference = tf.image.decode_image(reference_content, channels=3)
      reference = tf.image.resize_image_with_crop_or_pad(reference, IMAGE_SIZE,
                                                         IMAGE_SIZE)
      reference = tf.image.convert_image_dtype(reference, tf.float32)

      mask_image = get_mask_fn(IMAGE_SIZE, PATCH_SIZE)(image)
      return mask_image, image, reference
    else:
      return image

  return load_and_preprocess_image


def copy_dataset_to_mem_fs(mem_fs, dataset_zip_file_path):
  tf.logging.info('Copying dataset to in-memory filesystem.')
  dataset_path, dataset_zip_filename = os.path.split(dataset_zip_file_path)
  with fs.open_fs(dataset_path) as host_fs:  # Could be local or GCS
    with host_fs.open(dataset_zip_filename, 'rb') as zip_file:
      with ZipFS(zip_file) as zip_fs:
        fs.copy.copy_dir(zip_fs, '.', mem_fs, '.')

def main(args):
  BATCH_SIZE = args.batch_size

  DATASET_PATH = args.dataset_path
  DATASET_TRAIN_PATH = "train"
  TRAIN_REAL_PATH = os.path.join(DATASET_TRAIN_PATH, "real")
  TRAIN_MASKED_PATH = os.path.join(DATASET_TRAIN_PATH, "masked")
  TRAIN_REFERENCE_PATH = os.path.join(DATASET_TRAIN_PATH, "reference")

  if (args.dataset_path.endswith('.zip')):
    tf.logging.info('Creating memory filesystem')
    base_dataset_path, _ = os.path.split(args.dataset_path)
    dataset_fs_path = 'mem://'
  else:
    tf.logging.info('Using base path as filesystem')
    base_dataset_path = args.dataset_path
    dataset_fs_path = args.dataset_path
  tf.logging.info("Filesystem: {}".format(dataset_fs_path))

  REAL_IMAGES_PATHS_FILE = os.path.join(base_dataset_path,
                                        REAL_DATASET_PATHS_FILE)
  MASKED_IMAGES_PATHS_FILE = os.path.join(base_dataset_path,
                                          MASKED_DATASET_PATHS_FILE)
  REFERENCE_DICT_FILE = os.path.join(base_dataset_path,
                                     REFERENCE_DATASET_PATHS_FILE)

  with fs.open_fs(dataset_fs_path) as dataset_fs:
    if (dataset_fs_path.startswith('mem://')):
      copy_dataset_to_mem_fs(dataset_fs, args.dataset_path)

    train_reference_paths_dict = create_reference_paths_dict(
      REFERENCE_DICT_FILE)

    # Real dataset
    real_dataset = tf.data.TextLineDataset(REAL_IMAGES_PATHS_FILE,
                                           buffer_size=PATH_FILE_BUFFER_SIZE)

    real_dataset = real_dataset.shuffle(SHUFFLE_BUFFER_SIZE)

    real_dataset = real_dataset.map(
      get_load_and_preprocess_image_fn(dataset_fs, TRAIN_REAL_PATH,
                                       TRAIN_REFERENCE_PATH,
                                       train_reference_paths_dict),
      num_parallel_calls=PARALLEL_MAP_THREADS)
    real_dataset = real_dataset.prefetch(BATCH_SIZE * 2)
    real_dataset = real_dataset.batch(BATCH_SIZE, drop_remainder=True)

    # Masked dataset
    masked_dataset = tf.data.TextLineDataset(MASKED_IMAGES_PATHS_FILE,
                                             buffer_size=PATH_FILE_BUFFER_SIZE)
    masked_dataset = masked_dataset.shuffle(SHUFFLE_BUFFER_SIZE)

    masked_dataset = masked_dataset.map(
      get_load_and_preprocess_image_fn(dataset_fs, TRAIN_MASKED_PATH,
                                       TRAIN_REFERENCE_PATH,
                                       train_reference_paths_dict, masked=True),
      num_parallel_calls=PARALLEL_MAP_THREADS)
    masked_dataset = masked_dataset.prefetch(BATCH_SIZE * 2)
    masked_dataset = masked_dataset.batch(BATCH_SIZE, drop_remainder=True)

    train_dataset = tf.data.Dataset.zip((real_dataset, masked_dataset))
    train_dataset = train_dataset.prefetch(1)

    # TODO Change to tf and fs
    # Validation images
    # VALIDATION_IDENTITIES = [
    #   "0005366",
    #   "0005367",
    #   "0005370",
    #   "0005371",
    #   "0005373",
    #   "0005376",
    #   "0005378",
    #   "0005379",
    #   "0005381"
    # ]
    #
    # validation_images = []
    # validation_references = []
    # for identity in VALIDATION_IDENTITIES:
    #   full_identity_dir = os.path.join(DATASET_PATH, "validation", identity)
    #
    #   mask_image_file = tf.gfile.GFile(
    #     os.path.join(full_identity_dir, "001.jpg"),
    #     mode='rb')
    #   mask_image = plt.imread(mask_image_file)
    #
    #   reference_image_file = tf.gfile.GFile(
    #     os.path.join(full_identity_dir, "002.jpg"),
    #     mode='rb')
    #   reference_image = plt.imread(reference_image_file)
    #
    #   mask_image = fix_image_encoding(mask_image)
    #   reference_image = fix_image_encoding(reference_image)
    #
    #   mask_image = resize(mask_image, (IMAGE_SIZE, IMAGE_SIZE))
    #   reference_image = resize(reference_image, (IMAGE_SIZE, IMAGE_SIZE))
    #
    #   validation_images.append(mask_image)
    #   validation_references.append(reference_image)
    #
    # validation_images = np.array(validation_images).astype('float32')
    # validation_references = np.array(validation_references).astype('float32')
    validation_images = None
    validation_references = None


    generator, discriminator = model.make_models()

    train(train_dataset, generator, discriminator,
          validation_images, validation_references, args.experiment_dir)


if __name__ == "__main__":
  tf.logging.info("Parsing flags")
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--dataset_path',
    help='GCS or local path to the dataset. If ends with zip extension, will load dataset in memory-filesystem.',
    default='gs://first-ml-project-222122-mlengine/data')
  parser.add_argument(
    '--experiment_dir',
    help='GCS or local path where checkpoints will be stored.')
  parser.add_argument(
    '--batch_size',
    type=int,
    help='Batch size for training.',
    default=16)
  parser.add_argument(
    '--verbosity',
    choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
    default='INFO')

  args, _ = parser.parse_known_args()

  # Set python level verbosity
  tf.logging.set_verbosity(args.verbosity)
  # Set C++ Graph Execution level verbosity
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
    tf.logging.__dict__[args.verbosity] / 10)

  tf.logging.info('Dataset: {} - checkpoints: {}'.format(args.dataset_path,
                                                         args.experiment_dir))

  tf.logging.info("Starting training")
  main(args)
