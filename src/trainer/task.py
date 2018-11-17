import tensorflow as tf
import argparse

# To avoid getting the 'No module named _tkinter, please install the python-tk package' error
# when running on GCP
import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt

import numpy as np
import os
import time
from skimage.transform import resize

import trainer.model as model

REAL_DATASET_PATHS_FILE = "real-files.txt"
MASKED_DATASET_PATHS_FILE = "masked-files.txt"
REFERENCE_DATASET_PATHS_FILE = "reference-files.txt"

IMAGE_SIZE = 128
PATCH_SIZE = 32

BATCH_SIZE = 16

DATASET_BUFFER = 10000
SHUFFLE_BUFFER_SIZE = 1000
PARALLEL_MAP_THREADS = 8

EPOCHS = 50
BATCHES_PER_PRINT = 2
BATCHES_PER_CHECKPOINT = 100

# Use tf eager execution for the whole app.
tf.enable_eager_execution()


def get_os_join_fn(base_path):
  def os_join(x):
    x = x.decode('UTF-8')
    return os.path.join(base_path, x)

  return os_join


def get_reference_image(image, image_path):
  # Need to do this because when calling this function using tf.py_func,
  # the image_path is passed as bytes instead of string.
  image_path = image_path.decode('UTF-8')

  identity = image_path.split('/')[-2]
  references = train_reference_dict[identity]
  idx = np.random.randint(len(references))
  return (image, references[idx])


def get_reference_image_path_fn(train_reference_path,
                                train_reference_paths_dict):
  def get_reference_image_path(image_path):
    # Need to do this because when calling this function using tf.py_func,
    # the image_path is passed as bytes instead of string.
    image_path = image_path.decode('UTF-8')

    identity = image_path.split('/')[-2]
    reference_paths = train_reference_paths_dict[identity]
    idx = np.random.randint(len(reference_paths))
    image_file_name = reference_paths[idx]

    return os.path.join(train_reference_path, identity, image_file_name)

  return get_reference_image_path


def fix_image_encoding(image):
  if (image.ndim == 2):
    # Add new dimension for channels
    image = image[:, :, np.newaxis]
  if (image.shape[-1] == 1):
    # Convert greyscale to RGB
    image = np.concatenate((image,) * 3, axis=-1)
  return image


def create_reference_paths_dict(base_path):
  tf.logging.info('Creating reference paths dictionary')
  reference_dict = {}
  reference_images_file = tf.gfile.GFile(
    os.path.join(base_path, REFERENCE_DATASET_PATHS_FILE),
    mode='r')
  for line in reference_images_file:
    split_line = line.rstrip('\n').split(':')
    identity = split_line[0]
    image_paths = split_line[1].split(',')
    reference_dict[identity] = image_paths
    assert len(image_paths) > 0
  tf.logging.info('Finished creating reference paths dictionary')
  return reference_dict


def get_mask_fn(img_size, patch_size):
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


def generate_images(generator, masked_images, reference_images):
  # make sure the training parameter is set to False because we
  # don't want to train the batchnorm layer when doing inference.
  patches = generator([masked_images, reference_images], training=False)
  generated_images = patch_image(patches, masked_images)

  return generated_images


def train_step(full_images,
               full_reference_images,
               masked_images,
               masked_reference_images,
               generator,
               discriminator,
               generator_optimizer,
               discriminator_optimizer):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_patches = generator([masked_images, masked_reference_images],
                                  training=True)
    generated_images = patch_image(generated_patches, masked_images)

    real_output = discriminator([full_images, full_reference_images],
                                training=True)
    generated_output = discriminator(
      [generated_images, masked_reference_images], training=True)

    gen_loss = model.generator_loss(generated_output)
    disc_loss = model.discriminator_loss(real_output, generated_output)

  gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                  discriminator.variables)

  generator_optimizer.apply_gradients(
    zip(gradients_of_generator, generator.variables))
  discriminator_optimizer.apply_gradients(
    zip(gradients_of_discriminator, discriminator.variables))

  return gen_loss, disc_loss


def train(dataset, epochs, generator, discriminator, validation_masked_images,
          validation_references, checkpoints_dir):
  # train_step = tf.contrib.eager.defun(train_step)

  generator_optimizer = tf.train.AdamOptimizer(1e-4)
  discriminator_optimizer = tf.train.AdamOptimizer(1e-4)

  checkpoint_prefix = os.path.join(checkpoints_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                   discriminator_optimizer=discriminator_optimizer,
                                   generator=generator,
                                   discriminator=discriminator)

  gen_losses = []
  disc_losses = []

  global_step = tf.train.get_or_create_global_step()

  logdir = checkpoints_dir
  writer = tf.contrib.summary.create_file_writer(logdir)
  writer.set_as_default()

  for epoch in range(epochs):
    epoch_start = time.time()
    batch_start = time.time()
    for images in dataset:
      global_step.assign_add(1)

      # See if we can get rid of this (we are already checking below)
      with tf.contrib.summary.record_summaries_every_n_global_steps(
              BATCHES_PER_PRINT):
        (full_images, full_reference_images) = images[0]
        (masked_images, unmasked_images, masked_reference_images) = \
          images[1]
        gen_loss, disc_loss = train_step(full_images,
                                         full_reference_images,
                                         masked_images,
                                         masked_reference_images,
                                         generator,
                                         discriminator,
                                         generator_optimizer,
                                         discriminator_optimizer)

        tf.contrib.summary.scalar('gen_loss', gen_loss)
        tf.contrib.summary.scalar('disc_loss', disc_loss)
        if (global_step.numpy() % BATCHES_PER_PRINT == 0):
          generated_images = generate_images(generator,
                                             validation_masked_images,
                                             validation_references)
          tf.contrib.summary.image('generated_images',
                                   generated_images, max_images=9)

          batch_end = time.time()
          batch_time = (batch_end - batch_start) / BATCHES_PER_PRINT
          batch_start = time.time()  # Restart the timer.
          global_steps_per_second = 1 / batch_time if batch_time > 0 else 0
          tf.contrib.summary.scalar('global_step',
                                    global_steps_per_second)

          tf.logging.info(
            'Gen loss: {} - Disc loss: {} - Steps per second: {} - Current step {}'.format(
              gen_loss,
              disc_loss,
              global_steps_per_second,
              global_step.numpy()))

      if (global_step.numpy() % BATCHES_PER_CHECKPOINT == 0):
        checkpoint.save(file_prefix=checkpoint_prefix)

    tf.logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                               time.time() - epoch_start))


def get_load_image_fn(base_path):
  def load_image(img_filename):
    image_full_path = tf.py_func(get_os_join_fn(base_path), [img_filename],
                                 tf.string)
    return (tf.image.decode_image(
      tf.read_file(image_full_path), channels=3), image_full_path)

  return load_image


def get_load_and_preprocess_image_fn(base_path, reference_base_path,
                                     reference_dict, masked=False):
  def load_and_preprocess_image(img_filename):
    image_full_path = tf.py_func(get_os_join_fn(base_path), [img_filename],
                                 tf.string)
    image = tf.image.decode_image(tf.read_file(image_full_path), channels=3)

    reference_image_path = tf.py_func(
      get_reference_image_path_fn(reference_base_path, reference_dict),
      [image_full_path], tf.string)

    reference = tf.image.decode_image(tf.read_file(reference_image_path),
                                      channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_SIZE,
                                                   IMAGE_SIZE)
    reference = tf.image.resize_image_with_crop_or_pad(reference, IMAGE_SIZE,
                                                       IMAGE_SIZE)

    image = tf.image.convert_image_dtype(image, tf.float32)
    reference = tf.image.convert_image_dtype(reference, tf.float32)

    if masked:
      mask_image = get_mask_fn(IMAGE_SIZE, PATCH_SIZE)(image)
      return mask_image, image, reference
    else:
      return image, reference

  return load_and_preprocess_image

def main(args):
  DATASET_PATH = args.dataset_path
  DATASET_TRAIN_PATH = os.path.join(DATASET_PATH, "train")
  TRAIN_REAL_PATH = os.path.join(DATASET_TRAIN_PATH, "real")
  TRAIN_MASKED_PATH = os.path.join(DATASET_TRAIN_PATH, "masked")
  TRAIN_REFERENCE_PATH = os.path.join(DATASET_TRAIN_PATH, "reference")

  REAL_IMAGES_PATHS_FILE = os.path.join(TRAIN_REAL_PATH,
                                        REAL_DATASET_PATHS_FILE)
  MASKED_IMAGES_PATHS_FILE = os.path.join(TRAIN_MASKED_PATH,
                                          MASKED_DATASET_PATHS_FILE)

  train_reference_paths_dict = create_reference_paths_dict(
    TRAIN_REFERENCE_PATH)

  real_dataset = tf.data.TextLineDataset(REAL_IMAGES_PATHS_FILE)

  # TODO tal vez los maps pueden combinarse
  real_dataset = real_dataset.map(
    get_load_and_preprocess_image_fn(TRAIN_REAL_PATH, TRAIN_REFERENCE_PATH,
                                     train_reference_paths_dict),
    num_parallel_calls=PARALLEL_MAP_THREADS)
  real_dataset = real_dataset.prefetch(BATCH_SIZE * 2)
  real_dataset = real_dataset.batch(BATCH_SIZE, drop_remainder=True)

  masked_dataset = tf.data.TextLineDataset(MASKED_IMAGES_PATHS_FILE)
  masked_dataset = masked_dataset.map(
    get_load_and_preprocess_image_fn(TRAIN_MASKED_PATH, TRAIN_REFERENCE_PATH,
                                     train_reference_paths_dict, masked=True),
    num_parallel_calls=PARALLEL_MAP_THREADS)
  masked_dataset = masked_dataset.prefetch(BATCH_SIZE * 2)
  masked_dataset = masked_dataset.batch(BATCH_SIZE, drop_remainder=True)

  train_dataset = tf.data.Dataset.zip((real_dataset, masked_dataset))
  train_dataset = train_dataset.prefetch(1)

  VALIDATION_IDENTITIES = [
    "0005366",
    "0005367",
    "0005370",
    "0005371",
    "0005373",
    "0005376",
    "0005378",
    "0005379",
    "0005381"
  ]

  validation_images = []
  validation_references = []
  for identity in VALIDATION_IDENTITIES:
    full_identity_dir = os.path.join(DATASET_PATH, "validation", identity)

    mask_image_file = tf.gfile.GFile(
      os.path.join(full_identity_dir, "001.jpg"),
      mode='rb')
    mask_image = plt.imread(mask_image_file)

    reference_image_file = tf.gfile.GFile(
      os.path.join(full_identity_dir, "002.jpg"),
      mode='rb')
    reference_image = plt.imread(reference_image_file)

    mask_image = fix_image_encoding(mask_image)
    reference_image = fix_image_encoding(reference_image)

    mask_image = resize(mask_image, (IMAGE_SIZE, IMAGE_SIZE))
    reference_image = resize(reference_image, (IMAGE_SIZE, IMAGE_SIZE))

    validation_images.append(mask_image)
    validation_references.append(reference_image)

  validation_masked_images = []
  mask_fn = get_mask_fn(IMAGE_SIZE, PATCH_SIZE)
  for mask_image, reference_image in zip(validation_images,
                                         validation_references):
    mask_image = mask_fn(mask_image)
    validation_masked_images.append(mask_image.numpy())

  validation_images = np.array(validation_images).astype('float32')
  validation_references = np.array(validation_references).astype('float32')
  validation_masked_images = np.array(validation_masked_images).astype(
    'float32')

  generator, discriminator = model.make_models()

  train(train_dataset, EPOCHS, generator, discriminator,
        validation_masked_images, validation_references, args.checkpoints_dir)


if __name__ == "__main__":
  tf.logging.info("Parsing flags")
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--dataset_path',
    help='GCS or local path to the dataset.',
    default='gs://first-ml-project-222122-mlengine/data')
  parser.add_argument(
    '--checkpoints_dir',
    help='GCS or local path where checkpoints will be stored.')
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
                                                         args.checkpoints_dir))

  tf.logging.info("Starting training")
  main(args)