import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
    plt.imshow(images[i, :, :], cmap='gray')
    plt.axis('off')

  plt.subplots_adjust(wspace=0.01, hspace=0.01)
  plt.show()

tf.enable_eager_execution()
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

print(len(train_images))
print(len(test_images))

test_images = test_images.reshape(test_images.shape[0], 28, 28).astype('float32')
test_images = (test_images - 127.5) / 127.5
np.random.shuffle(test_images)

print(test_images[:25].shape)
show_imgs(test_images[:25])
