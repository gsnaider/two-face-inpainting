import os

import tensorflow as tf

IMAGE_SIZE = 128

def get_read_images_from_fs_fn(dataset_fs, base_path):
  def read_images_from_fs(image_path):
    image_path = image_path.decode('UTF-8')
    image_full_path = os.path.join(base_path, image_path)
    with dataset_fs.open(image_full_path, 'rb') as img_file:
      img_content = img_file.read()
    return img_content

  return read_images_from_fs

def get_load_and_preprocess_image_fn(dataset_fs, base_path):
  def load_and_preprocess_image(img_filename):
    image_content = tf.py_func(
      get_read_images_from_fs_fn(dataset_fs, base_path), [img_filename],
      tf.string)
    image = tf.image.decode_jpeg(image_content, channels=3)

    image = tf.image.resize_images(image, [IMAGE_SIZE,
                                                   IMAGE_SIZE])
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image

  return load_and_preprocess_image

# Real dataset
dataset = tf.data.TextLineDataset(REAL_IMAGES_PATHS_FILE,
                                  buffer_size=PATH_FILE_BUFFER_SIZE)

dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)

dataset = dataset.map(
  get_load_and_preprocess_image_fn(dataset_fs, REAL_IMGS_PATH,
                                   REFERENCE_IMGS_PATH,
                                   train_reference_paths_dict),
  num_parallel_calls=PARALLEL_MAP_THREADS)