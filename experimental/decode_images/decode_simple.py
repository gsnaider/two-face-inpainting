import tensorflow as tf


# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename):
  image_string = tf.read_file(filename)

  # image = tf.image.decode_image(image_string)
  # image = tf.image.resize_image_with_crop_or_pad(image, 28, 28)
  # image = tf.image.convert_image_dtype(image, tf.float32)

  image = tf.image.decode_jpeg(image_string)
  image = tf.image.convert_image_dtype(image, tf.float32)
  # image = tf.image.resize_images(image, [28, 28])

  return image


# A vector of filenames.
filenames = tf.constant([
  "/home/gaston/workspace/datasets/CASIA-WebFace/CASIA-WebFace/data/test/0000652/064.jpg",
  "/home/gaston/workspace/datasets/CASIA-WebFace/CASIA-WebFace/data/test/0000652/065.jpg"])

dataset = tf.data.Dataset.from_tensor_slices(filenames)
dataset = dataset.map(_parse_function)

iterator = dataset.make_one_shot_iterator()

with tf.Session() as sess:
  print(sess.run(iterator.get_next()))
