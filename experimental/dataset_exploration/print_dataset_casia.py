import os

import tensorflow as tf

import matplotlib.pyplot as plt

DATASET_PATH = "/home/gaston/workspace/datasets/CASIA-WebFace/CASIA-WebFace/data/test"


def _parse_function(filename):
  image_string = tf.read_file(filename)

  # image = tf.image.decode_image(image_string)
  # image = tf.image.resize_image_with_crop_or_pad(image, 28, 28)
  # image = tf.image.convert_image_dtype(image, tf.float32)

  image = tf.image.decode_jpeg(image_string, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  # image = tf.image.resize_images(image, [28, 28])

  return image


def show_imgs(images):
  plt.figure(figsize=[10,10])

  for i in range(images.shape[0]):
    plt.subplot(5, 5, i + 1)
    plt.imshow(images[i, :, :, :])
    plt.axis('off')

  plt.subplots_adjust(wspace=0.01, hspace=0.01)
  plt.show()

dataset = tf.data.Dataset.list_files(os.path.join(DATASET_PATH, "*/*.jpg")).map(
  _parse_function).batch(25)

iterator = dataset.make_one_shot_iterator()

with tf.Session() as sess:
  batch = sess.run(iterator.get_next())
  show_imgs(batch)
