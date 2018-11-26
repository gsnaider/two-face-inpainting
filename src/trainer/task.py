import time

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

IMAGE_SIZE = 128
PATCH_SIZE = 32

PATH_FILE_BUFFER_SIZE = 1000000
SHUFFLE_BUFFER_SIZE = 1000
PARALLEL_MAP_THREADS = 16

MAX_STEPS = 1e6

STEPS_PER_PRINT = 20
EVAL_SAVE_SECS=120

GEN_LEARNING_RATE = 1e-4
DISC_LEARNING_RATE = 1e-4

LAMBDA_REC = 1.0
LAMBDA_ADV_LOCAL = 0.0  # 0.01
LAMBDA_ADV_GLOBAL = 0.0  # 0.01
LAMBDA_ID = 0.0  # 0.1

LAMBDA_LOCAL_DISC = 0.0  # 0.1
LAMBDA_GLOBAL_DISC = 0.0  # 0.1


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


def extract_patch(image):
  patch_start = (IMAGE_SIZE - PATCH_SIZE) // 2
  patch_end = patch_start + PATCH_SIZE
  return image[:, patch_start:patch_end, patch_start:patch_end, :]


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


def train_step(sess, optimizers, gen_loss, local_disc_loss,
               global_disc_loss,
               global_step):
  (_, gen_loss_value, local_disc_loss_value, global_disc_loss_value,
   step_value) = sess.run(
    [optimizers, gen_loss, local_disc_loss, global_disc_loss, global_step])

  # Divide by 3 because we increment the global_step 3 times per each train_step.
  if (step_value // 3) % (STEPS_PER_PRINT // 3) == 0:
    tf.logging.info(
      "Step: {} - Gen_loss: {} - Local_disc_loss: {} - Global_disc_loss: {}".format(
        step_value,
        gen_loss_value,
        local_disc_loss_value,
        global_disc_loss_value))


def expand_patches(patches):
  paddings = tf.constant([[0, 0], [8, 8], [8, 8], [0, 0]])
  return tf.pad(patches, paddings, "CONSTANT")


def train(dataset, generator, local_discriminator, global_discriminator,
          facenet, experiment_dir):
  global_step = tf.train.get_or_create_global_step()

  dataset = dataset.repeat()
  iterator = dataset.make_one_shot_iterator()

  train_batch = iterator.get_next()
  full_images = train_batch[0]
  (masked_images, unmasked_images, reference_images) = train_batch[1]

  generated_patches = generator([masked_images, reference_images],
                                training=True)

  generated_images = patch_image(generated_patches, masked_images)

  # Local discriminator

  # Expand patches to because in tf <= 1.10 VGG min input size is 48x48
  expanded_real_patches = expand_patches(extract_patch(full_images))
  expanded_gen_patches = expand_patches(generated_patches)

  local_real_output = local_discriminator(expanded_real_patches,
                                          training=True)
  local_generated_output = local_discriminator(expanded_gen_patches,
                                               training=True)
  local_disc_loss = model.discriminator_loss(local_real_output,
                                             local_generated_output,
                                             LAMBDA_LOCAL_DISC)

  # Global discriminator
  global_real_output = global_discriminator(full_images, training=True)
  global_generated_output = global_discriminator(generated_images,
                                                 training=True)
  global_disc_loss = model.discriminator_loss(global_real_output,
                                              global_generated_output,
                                              LAMBDA_GLOBAL_DISC)

  # Generator
  gen_loss = model.generator_loss(unmasked_images, generated_images,
                                  reference_images, local_generated_output,
                                  global_generated_output, LAMBDA_REC,
                                  LAMBDA_ADV_LOCAL, LAMBDA_ADV_GLOBAL,
                                  LAMBDA_ID, facenet)

  # TODO check that this is the correct way to use optimizer with keras.
  gen_optimizer = tf.train.AdamOptimizer(GEN_LEARNING_RATE).minimize(
    gen_loss, var_list=generator.variables,
    global_step=global_step)

  # TODO check that this works
  # tf.logging.info('Generator variables {}'.format([v.name for v in generator.variables]))

  # TODO Seems that the disc optimizer is propagating changes to the generator.
  local_disc_optimizer = tf.train.AdamOptimizer(DISC_LEARNING_RATE).minimize(
    local_disc_loss, var_list=local_discriminator.variables,
    global_step=global_step)

  # tf.logging.info('Local discriminator variables {}'.format([v.name for v in local_discriminator.variables]))

  global_disc_optimizer = tf.train.AdamOptimizer(DISC_LEARNING_RATE).minimize(
    global_disc_loss, var_list=global_discriminator.variables,
    global_step=global_step)

  # This is required for the batch_normalization layers.
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    optimizers = tf.group([gen_optimizer, local_disc_optimizer, global_disc_optimizer])

  # tf.logging.info('Global discriminator variables {}'.format([v.name for v in global_discriminator.variables]))

  tf.summary.scalar('gen_loss', gen_loss)
  tf.summary.scalar('local_disc_loss', local_disc_loss)
  tf.summary.scalar('global_disc_loss', global_disc_loss)
  tf.summary.image('generated_train_images', generated_images, max_outputs=8)
  tf.summary.image('reference_images', reference_images, max_outputs=8)

  hooks = [tf.train.StopAtStepHook(num_steps=MAX_STEPS)]
  with tf.train.MonitoredTrainingSession(
          checkpoint_dir=os.path.join(experiment_dir, "train"),
          hooks=hooks) as sess:
    while not sess.should_stop():
      train_step(sess, optimizers, gen_loss, local_disc_loss,
                 global_disc_loss,
                 global_step)


def evaluate(dataset, generator, local_discriminator, global_discriminator,
             facenet, experiment_dir):

  generator.trainable = False
  local_discriminator.trainable = False
  global_discriminator.trainable = False
  facenet.trainable = False

  global_step = tf.train.get_or_create_global_step()

  dataset = dataset.repeat()
  iterator = dataset.make_one_shot_iterator()

  eval_batch = iterator.get_next()
  full_images = eval_batch[0]
  (masked_images, unmasked_images, reference_images) = eval_batch[1]

  generated_patches = generator([masked_images, reference_images],
                                training=False)

  generated_images = patch_image(generated_patches, masked_images)

  # Local discriminator

  # Expand patches to because in tf <= 1.10 VGG min input size is 48x48
  expanded_real_patches = expand_patches(extract_patch(full_images))
  expanded_gen_patches = expand_patches(generated_patches)

  local_real_output = local_discriminator(expanded_real_patches,
                                          training=False)
  local_generated_output = local_discriminator(expanded_gen_patches,
                                               training=False)
  local_disc_loss = model.discriminator_loss(local_real_output,
                                             local_generated_output,
                                             LAMBDA_LOCAL_DISC)

  # Global discriminator
  global_real_output = global_discriminator(full_images, training=False)
  global_generated_output = global_discriminator(generated_images,
                                                 training=False)
  global_disc_loss = model.discriminator_loss(global_real_output,
                                              global_generated_output,
                                              LAMBDA_GLOBAL_DISC)

  # Generator
  gen_loss = model.generator_loss(unmasked_images, generated_images,
                                  reference_images, local_generated_output,
                                  global_generated_output, LAMBDA_REC,
                                  LAMBDA_ADV_LOCAL, LAMBDA_ADV_GLOBAL,
                                  LAMBDA_ID, facenet)

  tf.summary.scalar('gen_loss', gen_loss)
  tf.summary.scalar('local_disc_loss', local_disc_loss)
  tf.summary.scalar('global_disc_loss', global_disc_loss)
  tf.summary.image('original_eval_images', unmasked_images, max_outputs=8)
  tf.summary.image('generated_eval_images', generated_images, max_outputs=8)
  tf.summary.image('reference_eval_images', reference_images, max_outputs=8)

  # hooks = [tf.train.SummarySaverHook(
  #   save_secs=EVAL_SAVE_SECS,
  #   output_dir=os.path.join(experiment_dir, "eval"),
  #   summary_op=tf.summary.merge_all())]

  writer = tf.summary.FileWriter(os.path.join(experiment_dir, "eval"))
  summary_op = tf.summary.merge_all()

  # Have to do this because for some reason the checkpoints are not being updated.
  # TODO see if this could be done better.
  while True:
    with tf.train.SingularMonitoredSession(
            checkpoint_dir=os.path.join(experiment_dir, "train")) as sess:
      tf.logging.info("Starting evaluation.")
      gen_loss_value, local_disc_loss_value, global_disc_loss_value, global_step_value = sess.run(
        [gen_loss, local_disc_loss, global_disc_loss, global_step])
      tf.logging.info(
        'Gen_loss: {} - Local_disc_loss: {} - Global_disc_loss: {}'.format(
          gen_loss_value, local_disc_loss_value, global_disc_loss_value))
      writer.add_summary(sess.run(summary_op), global_step_value)
    time.sleep(EVAL_SAVE_SECS)

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
  if args.train:
    tf.logging.info("Starting training")
  else:
    tf.logging.info("Starting evaluation")

  BATCH_SIZE = args.batch_size

  # TODO this shouldn't be required now, change the train and eval directories to be equal inside
  DATASET_PATH = "train" if args.train else "validation"
  REAL_IMGS_PATH = os.path.join(DATASET_PATH, "real")
  MASKED_IMGS_PATH = os.path.join(DATASET_PATH, "masked")
  REFERENCE_IMGS_PATH = os.path.join(DATASET_PATH, "reference")

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
      get_load_and_preprocess_image_fn(dataset_fs, REAL_IMGS_PATH,
                                       REFERENCE_IMGS_PATH,
                                       train_reference_paths_dict),
      num_parallel_calls=PARALLEL_MAP_THREADS)
    real_dataset = real_dataset.prefetch(BATCH_SIZE * 2)
    real_dataset = real_dataset.batch(BATCH_SIZE, drop_remainder=True)

    # Masked dataset
    masked_dataset = tf.data.TextLineDataset(MASKED_IMAGES_PATHS_FILE,
                                             buffer_size=PATH_FILE_BUFFER_SIZE)
    masked_dataset = masked_dataset.shuffle(SHUFFLE_BUFFER_SIZE)

    masked_dataset = masked_dataset.map(
      get_load_and_preprocess_image_fn(dataset_fs, MASKED_IMGS_PATH,
                                       REFERENCE_IMGS_PATH,
                                       train_reference_paths_dict, masked=True),
      num_parallel_calls=PARALLEL_MAP_THREADS)
    masked_dataset = masked_dataset.prefetch(BATCH_SIZE * 2)
    masked_dataset = masked_dataset.batch(BATCH_SIZE, drop_remainder=True)

    full_dataset = tf.data.Dataset.zip((real_dataset, masked_dataset))
    full_dataset = full_dataset.prefetch(1)

    generator, local_discriminator, global_discriminator, facenet = model.make_models(
      args.facenet_dir)

    if args.train:
      train(full_dataset, generator, local_discriminator, global_discriminator,
            facenet, args.experiment_dir)
    else:
      evaluate(full_dataset, generator, local_discriminator, global_discriminator,
           facenet, args.experiment_dir)

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
    '--facenet_dir',
    help='Directory where the weights and model of Facenet are stored.')
  parser.add_argument(
    '--batch_size',
    type=int,
    help='Batch size for training.',
    default=16)
  parser.add_argument(
    '--train',
    dest='train',
    help="True if it's a training run, False if validation run.",
    action='store_true')
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

  main(args)
