import json
import time

import tensorflow as tf
import argparse
import fs
from fs.zipfs import ZipFS

# To avoid getting the 'No module named _tkinter, please install the python-tk package' error
# when running on GCP
import matplotlib

matplotlib.use('agg')

import numpy as np
import os

import trainer.model as model
from trainer.generator import Generator
from trainer.local_discriminator import LocalDiscriminator
from trainer.global_discriminator import GlobalDiscriminator

TRAIN_RUN_MODE = 'TRAIN'
EVAL_RUN_MODE = 'EVAL'
SAVE_MODEL_RUN_MODE = 'SAVE_MODEL'

REAL_DATASET_PATHS_FILE = "real-files.txt"
MASKED_DATASET_PATHS_FILE = "masked-files.txt"
REFERENCE_DATASET_PATHS_FILE = "reference-files.txt"

STEPS_PER_PRINT = 20

EVAL_SAVE_SECS = 120

IMAGE_SIZE = 128
PATCH_SIZE = 32

PATH_FILE_BUFFER_SIZE = 1000000
SHUFFLE_BUFFER_SIZE = 1000
PARALLEL_MAP_THREADS = 16


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


def get_mask_fn(img_size, patch_size):
  def mask_fn(image):
    """
    Applies a mask of zeroes of size (patch_size x patch_size) at a random place in the image.
    Returns a tuple of the masked image and the binary mask matrix (0=mask, 1=visible).
    """

    # Upper left point of patch
    patch_x = tf.random.uniform([1], minval=0, maxval=img_size - patch_size + 1, dtype=tf.int32)
    patch_y = tf.random.uniform([1], minval=0, maxval=img_size - patch_size + 1, dtype=tf.int32)

    # TODO testing (remove). Trying border cases
    # patch_x = 0
    # patch_y = 0
    # patch_x = img_size - patch_size + 1
    # patch_y = img_size - patch_size + 1

    # TODO need to convert the patch_size and img_size to tensors for this to work
    column_width_after_patch = img_size - (patch_x + patch_size)
    row_height_after_patch = img_size - (patch_y + patch_size)

    upper_edge = tf.ones([patch_y, img_size, 3], tf.float32)
    lower_edge = tf.ones([row_height_after_patch, img_size, 3], tf.float32)

    middle_left = tf.ones([patch_size, patch_x, 3], tf.float32)
    middle_right = tf.ones([patch_size, column_width_after_patch, 3],
                           tf.float32)

    zeros = tf.zeros([patch_size, patch_size, 3], tf.float32)

    middle = tf.concat([middle_left, zeros, middle_right], axis=1)
    mask = tf.concat([upper_edge, middle, lower_edge], axis=0)

    # Mask is created with 3 channels in order to multiply with image, but since
    # all the channels are the same, we can just return one of them.
    return image * mask, tf.expand_dims(mask[:, :, 0], axis=2)

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


def train(dataset, generator, local_discriminator, global_discriminator,
          facenet, args):
  global_step = tf.train.get_or_create_global_step()

  dataset = dataset.repeat()
  iterator = dataset.make_one_shot_iterator()

  train_batch = iterator.get_next()
  real_images = train_batch[0]
  (masked_images, masks, original_images, reference_images) = train_batch[1]

  # TODO pass the masks to the generator as well.
  generated_patches = generator([masked_images, reference_images],
                                training=True)

  generated_images = patch_image(generated_patches, masked_images)

  # Local discriminator
  local_real_output = local_discriminator(extract_patch(real_images),
                                          training=True)
  local_generated_output = local_discriminator(generated_patches,
                                               training=True)
  local_disc_loss = model.discriminator_loss(local_real_output,
                                             local_generated_output,
                                             args.lambda_local_disc)

  # Global discriminator
  global_real_output = global_discriminator(real_images, training=True)
  global_generated_output = global_discriminator(generated_images,
                                                 training=True)
  global_disc_loss = model.discriminator_loss(global_real_output,
                                              global_generated_output,
                                              args.lambda_global_disc)

  # Generator
  gen_loss = model.generator_loss(original_images, generated_images,
                                  reference_images, local_generated_output,
                                  global_generated_output, args.lambda_rec,
                                  args.lambda_adv_local, args.lambda_adv_global,
                                  args.lambda_id, facenet)

  # This is required for the batch_normalization layers.
  gen_update_ops = generator.updates
  local_disc_update_ops = local_discriminator.updates
  global_disc_update_ops = global_discriminator.updates

  tf.logging.debug("num_gen_update_ops: {}".format(len(gen_update_ops)))
  tf.logging.debug("gen_update_ops: {}".format(gen_update_ops))

  tf.logging.debug(
    "num_local_disc_update_ops: {}".format(len(local_disc_update_ops)))
  tf.logging.debug(
    "local_disc_update_ops: {}".format(local_disc_update_ops))

  tf.logging.debug(
    "num_global_disc_update_ops: {}".format(len(global_disc_update_ops)))
  tf.logging.debug(
    "global_disc_update_ops: {}".format(global_disc_update_ops))

  update_ops = gen_update_ops + local_disc_update_ops + global_disc_update_ops

  tf.logging.debug(
    "num_update_ops: {}".format(len(update_ops)))
  tf.logging.debug(
    "update_ops: {}".format(update_ops))

  with tf.control_dependencies(update_ops):
    # TODO check that this is the correct way to use optimizer with keras.
    gen_optimizer = tf.train.AdamOptimizer(args.gen_learning_rate).minimize(
      gen_loss, var_list=generator.variables,
      global_step=global_step)
    tf.logging.debug(
      'Generator variables {}'.format([v.name for v in generator.variables]))

    # TODO Seems that the disc optimizer is propagating changes to the generator.
    local_disc_optimizer = tf.train.AdamOptimizer(
      args.disc_learning_rate).minimize(
      local_disc_loss, var_list=local_discriminator.variables,
      global_step=global_step)
    tf.logging.debug('Local discriminator variables {}'.format(
      [v.name for v in local_discriminator.variables]))

    global_disc_optimizer = tf.train.AdamOptimizer(
      args.disc_learning_rate).minimize(
      global_disc_loss, var_list=global_discriminator.variables,
      global_step=global_step)
    tf.logging.debug('Global discriminator variables {}'.format(
      [v.name for v in global_discriminator.variables]))

  optimizers = tf.group(
    [gen_optimizer, local_disc_optimizer, global_disc_optimizer])

  if (args.batch_normalization):
    gen_bn_layers = [generator.layers[8], generator.layers[11]]
    local_disc_bn_layers = [local_discriminator.layers[2],
                            local_discriminator.layers[5]]
    global_disc_bn_layers = [global_discriminator.layers[2],
                             global_discriminator.layers[5],
                             global_discriminator.layers[9]]

    tf.logging.debug("GENERATOR BN LAYERS {}".format(gen_bn_layers))
    tf.logging.debug("LOCAL_DISC BN LAYERS {}".format(local_disc_bn_layers))
    tf.logging.debug("GLOBAL_DISC BN LAYERS {}".format(global_disc_bn_layers))

    tf.logging.debug("PRINTING BN WEIGHTS")
    for bn_layer in (
            gen_bn_layers + local_disc_bn_layers + global_disc_bn_layers):
      tf.logging.debug("Weights: {}".format(bn_layer.weights))

  tf.logging.info("GENERATOR")
  generator.summary()
  tf.logging.info("LOCAL DISCRIMINATOR")
  local_discriminator.summary()
  tf.logging.info("GLOBAL DISCRIMINATOR")
  global_discriminator.summary()

  tf.summary.scalar('gen_loss', gen_loss)
  tf.summary.scalar('local_disc_loss', local_disc_loss)
  tf.summary.scalar('global_disc_loss', global_disc_loss)

  tf.summary.image('generated_train_images', generated_images, max_outputs=8)
  tf.summary.image('masked_train_image', masked_images, max_outputs=8)
  tf.summary.image('original_train_image', original_images, max_outputs=8)
  tf.summary.image('reference_train_images', reference_images, max_outputs=8)

  hooks = [tf.train.StopAtStepHook(num_steps=args.max_steps)]
  with tf.train.MonitoredTrainingSession(
          checkpoint_dir=os.path.join(args.experiment_dir, "train"),
          hooks=hooks) as sess:
    while not sess.should_stop():
      if (args.batch_normalization):
        # TODO remove these (and all other related debug logs) after verifying BN works correctly.
        tf.logging.debug(
          "GEN_BN_1: Gamma {} - Beta {} - Moving_mean {} - Moving_variance {}".format(
            *sess.run(
              gen_bn_layers[0].weights)))
        tf.logging.debug(
          "GEN_BN_2: Gamma {} - Beta {} - Moving_mean {} - Moving_variance {}".format(
            *sess.run(
              gen_bn_layers[1].weights)))

        tf.logging.debug(
          "LOCAL_DISC_BN_1: Gamma {} - Beta {} - Moving_mean {} - Moving_variance {}".format(
            *sess.run(
              local_disc_bn_layers[0].weights)))
        tf.logging.debug(
          "LOCAL_DISC_BN_2: Gamma {} - Beta {} - Moving_mean {} - Moving_variance {}".format(
            *sess.run(
              local_disc_bn_layers[1].weights)))

        tf.logging.debug(
          "GLOBAL_DISC_BN_1: Gamma {} - Beta {} - Moving_mean {} - Moving_variance {}".format(
            *sess.run(
              global_disc_bn_layers[0].weights)))
        tf.logging.debug(
          "GLOBAL_DISC_BN_2: Gamma {} - Beta {} - Moving_mean {} - Moving_variance {}".format(
            *sess.run(
              global_disc_bn_layers[1].weights)))
        tf.logging.debug(
          "GLOBAL_DISC_BN_3: Gamma {} - Beta {} - Moving_mean {} - Moving_variance {}".format(
            *sess.run(
              global_disc_bn_layers[2].weights)))

      train_step(sess, optimizers, gen_loss, local_disc_loss,
                 global_disc_loss,
                 global_step)


def evaluate(dataset, generator, local_discriminator, global_discriminator,
             facenet, args):
  # TODO verify that global step is updated every 10 minutes (when running train and eval in parallel).
  # This is because the summaries from the train run are saved every 10 minutes.
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
  local_real_output = local_discriminator(extract_patch(full_images),
                                          training=False)
  local_generated_output = local_discriminator(generated_patches,
                                               training=False)
  local_disc_loss = model.discriminator_loss(local_real_output,
                                             local_generated_output,
                                             args.lambda_local_disc)

  # Global discriminator
  global_real_output = global_discriminator(full_images, training=False)
  global_generated_output = global_discriminator(generated_images,
                                                 training=False)
  global_disc_loss = model.discriminator_loss(global_real_output,
                                              global_generated_output,
                                              args.lambda_global_disc)

  # Generator
  gen_loss = model.generator_loss(unmasked_images, generated_images,
                                  reference_images, local_generated_output,
                                  global_generated_output, args.lambda_rec,
                                  args.lambda_adv_local, args.lambda_adv_global,
                                  args.lambda_id, facenet)

  tf.logging.debug("num_gen_update_ops: {}".format(
    len(generator.get_updates_for(generator.inputs))))
  tf.logging.debug(
    "num_local_disc_update_ops: {}".format(
      len(local_discriminator.get_updates_for(
        local_discriminator.inputs))))
  tf.logging.debug(
    "num_global_disc_update_ops: {}".format(
      len(global_discriminator.get_updates_for(
        global_discriminator.inputs))))

  if (args.batch_normalization):
    gen_bn_layers = [generator.layers[8], generator.layers[11]]
    local_disc_bn_layers = [local_discriminator.layers[2],
                            local_discriminator.layers[5]]
    global_disc_bn_layers = [global_discriminator.layers[2],
                             global_discriminator.layers[5],
                             global_discriminator.layers[9]]

    tf.logging.debug("GENERATOR BN LAYERS {}".format(gen_bn_layers))
    tf.logging.debug("LOCAL_DISC BN LAYERS {}".format(local_disc_bn_layers))
    tf.logging.debug("GLOBAL_DISC BN LAYERS {}".format(global_disc_bn_layers))

    tf.logging.debug("PRINTING BN WEIGHTS")
    for bn_layer in (
            gen_bn_layers + local_disc_bn_layers + global_disc_bn_layers):
      tf.logging.debug("Weights: {}".format(bn_layer.weights))

  tf.logging.info("GENERATOR")
  generator.summary()
  tf.logging.info("LOCAL DISCRIMINATOR")
  local_discriminator.summary()
  tf.logging.info("GLOBAL DISCRIMINATOR")
  global_discriminator.summary()

  tf.summary.scalar('gen_loss', gen_loss)
  tf.summary.scalar('local_disc_loss', local_disc_loss)
  tf.summary.scalar('global_disc_loss', global_disc_loss)

  tf.summary.image('generated_eval_images', generated_images, max_outputs=8)
  tf.summary.image('original_eval_images', unmasked_images, max_outputs=8)
  tf.summary.image('masked_eval_image', masked_images, max_outputs=8)
  tf.summary.image('reference_eval_images', reference_images, max_outputs=8)

  # hooks = [tf.train.SummarySaverHook(
  #   save_secs=EVAL_SAVE_SECS,
  #   output_dir=os.path.join(experiment_dir, "eval"),
  #   summary_op=tf.summary.merge_all())]

  summary_op = tf.summary.merge_all()

  # Have to do this because for some reason the checkpoints are not being updated.
  # TODO see if this could be done better.
  with tf.train.SingularMonitoredSession(
          checkpoint_dir=os.path.join(args.experiment_dir, "train")) as sess:
    tf.logging.info("Starting evaluation.")

    writer = tf.summary.FileWriter(os.path.join(args.experiment_dir, "eval"),
                                   sess.graph)

    while True:
      if (args.batch_normalization):
        tf.logging.debug(
          "GEN_BN_1: Gamma {} - Beta {} - Moving_mean {} - Moving_variance {}".format(
            *sess.run(
              gen_bn_layers[0].weights)))
        tf.logging.debug(
          "GEN_BN_2: Gamma {} - Beta {} - Moving_mean {} - Moving_variance {}".format(
            *sess.run(
              gen_bn_layers[1].weights)))

        tf.logging.debug(
          "LOCAL_DISC_BN_1: Gamma {} - Beta {} - Moving_mean {} - Moving_variance {}".format(
            *sess.run(
              local_disc_bn_layers[0].weights)))
        tf.logging.debug(
          "LOCAL_DISC_BN_2: Gamma {} - Beta {} - Moving_mean {} - Moving_variance {}".format(
            *sess.run(
              local_disc_bn_layers[1].weights)))

        tf.logging.debug(
          "GLOBAL_DISC_BN_1: Gamma {} - Beta {} - Moving_mean {} - Moving_variance {}".format(
            *sess.run(
              global_disc_bn_layers[0].weights)))
        tf.logging.debug(
          "GLOBAL_DISC_BN_2: Gamma {} - Beta {} - Moving_mean {} - Moving_variance {}".format(
            *sess.run(
              global_disc_bn_layers[1].weights)))
        tf.logging.debug(
          "GLOBAL_DISC_BN_3: Gamma {} - Beta {} - Moving_mean {} - Moving_variance {}".format(
            *sess.run(
              global_disc_bn_layers[2].weights)))

      gen_loss_value, local_disc_loss_value, global_disc_loss_value, global_step_value = sess.run(
        [gen_loss, local_disc_loss, global_disc_loss, global_step])
      tf.logging.info(
        'Gen_loss: {} - Local_disc_loss: {} - Global_disc_loss: {}'.format(
          gen_loss_value, local_disc_loss_value, global_disc_loss_value))
      writer.add_summary(sess.run(summary_op), global_step_value)

      time.sleep(EVAL_SAVE_SECS)


def save_model(generator, experiment_dir, model_number):
  # TODO if we add namespace to the models, we shouldn't need to create the
  # whole model here, only the generator

  tf.keras.backend.set_learning_phase(0)

  global_step = tf.train.get_or_create_global_step()

  masked_images = tf.placeholder(tf.float32,
                                 shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3],
                                 name='masked_images_ph')
  reference_images = tf.placeholder(tf.float32,
                                    shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3],
                                    name='reference_images_ph')

  generated_patches = generator([masked_images, reference_images],
                                training=False)

  generated_images = patch_image(generated_patches, masked_images)

  saver = tf.train.Saver()
  with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint(
      os.path.join(experiment_dir, 'train')))
    tf.logging.info('Checkpoints restored.')

    tf.saved_model.simple_save(sess,
                               os.path.join(experiment_dir, "saved_model",
                                            str(model_number)),
                               inputs={'masked_image': masked_images,
                                       'reference_image': reference_images},
                               outputs={'output_image': generated_images})


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
    """
    Loads the image from the dataset filesystem, converts it to float32 and
    resizes it to IMAGE_SIZE x IMAGE_SIZE.
    If the masked flag is set to True, it also searches for a reference image in
    the reference_dict, applying the same process as for the original image, and
    applies a binary mask to the original image.

    Returns the loaded image if masked is set to False. Otherwise returns a
    tuple of the masked image, the binary mask matrix (0=mask, 1=visible), the
    original image, and the reference image,
    """
    image_content = tf.py_func(
      get_read_images_from_fs_fn(dataset_fs, base_path), [img_filename],
      tf.string)

    image = tf.image.decode_jpeg(image_content, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, [IMAGE_SIZE,
                                                   IMAGE_SIZE])

    if masked:
      reference_image_filename = tf.py_func(
        get_reference_image_path_fn(reference_dict),
        [img_filename], tf.string)
      reference_content = tf.py_func(
        get_read_images_from_fs_fn(dataset_fs, reference_base_path),
        [reference_image_filename],
        tf.string)

      reference = tf.image.decode_jpeg(reference_content, channels=3)
      reference = tf.image.convert_image_dtype(reference, tf.float32)
      reference = tf.image.resize_images(reference, [IMAGE_SIZE,
                                                         IMAGE_SIZE])

      mask_image, mask = get_mask_fn(IMAGE_SIZE, PATCH_SIZE)(image)
      return mask_image, mask, image, reference
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
  if args.run_mode == TRAIN_RUN_MODE:
    # all new operations will be in train mode from now on
    tf.keras.backend.set_learning_phase(1)
    tf.logging.info("Starting training")
  elif args.run_mode == EVAL_RUN_MODE:
    # all new operations will be in test mode from now on
    tf.keras.backend.set_learning_phase(0)
    tf.logging.info("Starting evaluation")
  else:
    tf.keras.backend.set_learning_phase(0)
    assert args.run_mode == SAVE_MODEL_RUN_MODE
    tf.logging.info("Creating model for inference")

  generator = Generator()
  local_discriminator = LocalDiscriminator()
  global_discriminator = GlobalDiscriminator()
  facenet = model.make_identity_model(args.facenet_dir)

  if args.run_mode == SAVE_MODEL_RUN_MODE:
    save_model(generator, args.experiment_dir, args.model_number)
    return

  BATCH_SIZE = args.batch_size

  # TODO this shouldn't be required now, change the train and eval directories to be equal inside
  DATASET_PATH = "train" if (args.run_mode == TRAIN_RUN_MODE) else "validation"
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

    if args.run_mode == TRAIN_RUN_MODE:
      train(full_dataset, generator, local_discriminator, global_discriminator,
            facenet, args)
    elif args.run_mode == EVAL_RUN_MODE:
      evaluate(full_dataset, generator, local_discriminator,
               global_discriminator,
               facenet, args)


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
    '--run_mode',
    choices=[TRAIN_RUN_MODE, EVAL_RUN_MODE, SAVE_MODEL_RUN_MODE],
    help="TRAIN for training the model, EVAL for evaluating on validation data, SAVE_MODEL for storing the model for inference.",
    default=SAVE_MODEL_RUN_MODE)
  parser.add_argument(
    '--model_number',
    type=int,
    help='Model number for saving model. Only when using run_mode=SAVE_MODEL.')
  parser.add_argument(
    '--verbosity',
    choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
    default='INFO')

  parser.add_argument('--batch_normalization', dest='batch_normalization',
                      action='store_true')
  parser.add_argument('--no_batch_normalization', dest='batch_normalization',
                      action='store_false')
  parser.set_defaults(batch_normalization=False)

  parser.add_argument(
    '--config_train_file',
    type=argparse.FileType(mode='r'))
  parser.add_argument('--batch_size', type=int, default=16)

  # float so we can express with scientific notation.
  parser.add_argument('--max_steps', type=float, default=1e3)

  parser.add_argument('--gen_learning_rate', type=float, default=0.0)
  parser.add_argument('--disc_learning_rate', type=float, default=0.0)
  parser.add_argument('--lambda_rec', type=float, default=1.0)
  parser.add_argument('--lambda_adv_local', type=float, default=0.01)
  parser.add_argument('--lambda_adv_global', type=float, default=0.001)
  parser.add_argument('--lambda_id', type=float, default=0.001)
  parser.add_argument('--lambda_local_disc', type=float, default=0.1)
  parser.add_argument('--lambda_global_disc', type=float, default=0.1)

  args, _ = parser.parse_known_args()

  if args.config_train_file:
    data = json.load(args.config_train_file)
    delattr(args, 'config_train_file')
    arg_dict = args.__dict__
    for key, value in data.items():
      if isinstance(value, list):
        arg_dict[key].extend(value)
      else:
        arg_dict[key] = value

  # Set python level verbosity
  tf.logging.set_verbosity(args.verbosity)
  # Set C++ Graph Execution level verbosity
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
    tf.logging.__dict__[args.verbosity] / 10)

  tf.logging.info('Dataset: {} - checkpoints: {}'.format(args.dataset_path,
                                                         args.experiment_dir))

  tf.logging.debug("Args: {}".format(args))

  main(args)
