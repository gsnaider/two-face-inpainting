import tensorflow as tf

PATCH_SIZE = 32

class LocalDiscriminator(tf.keras.Model):

  def __init__(self):
    super(LocalDiscriminator, self).__init__(name='local_discriminator')

    with tf.name_scope("vgg"):
      vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False,
                                                weights='imagenet',
                                                input_tensor=None,
                                                input_shape=(
                                                  PATCH_SIZE,
                                                  PATCH_SIZE, 3))
      vgg16.trainable = False
      self.encoder = tf.keras.Model(inputs=vgg16.inputs,
                               outputs=vgg16.layers[10].output)
      self.encoder.trainable = False

    with tf.name_scope("local_disc_encoder"):
      self.conv_1 = tf.keras.layers.Conv2D(128, (3, 3),
                                        strides=(1, 1),
                                        padding='same',
                                        input_shape=(6, 6, 256))
      self.batch_norm_1 = tf.keras.layers.BatchNormalization()
      self.relu_1 = tf.keras.layers.LeakyReLU()

      self.conv_2 = tf.keras.layers.Conv2D(64, (3, 3),
                                        strides=(1, 1),
                                        padding='same',
                                        input_shape=(4, 4, 128))
      self.batch_norm_2 = tf.keras.layers.BatchNormalization()
      self.relu_2 = tf.keras.layers.LeakyReLU()

      self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
      self.flatten = tf.keras.layers.Flatten()

    with tf.name_scope("local_disc_classifier"):
      self.dense_1 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
      self.dense_2 = tf.keras.layers.Dense(1)


  def call(self, inputs, training=False):
    x = inputs

    x = self.encoder(x)

    x = self.conv_1(x)
    x = self.batch_norm_1(x, training=training)
    x = self.relu_1(x)

    x = self.conv_2(x)
    x = self. batch_norm_2(x, training=training)
    x = self.relu_2(x)

    x = self.max_pool(x)
    x = self.flatten(x)

    x = self.dense_1(x)
    x = self.dense_2(x)

    return x

  # TODO remove if not required
  # def compute_output_shape(self, input_shape):
  #   # You need to override this function if you want to use the subclassed model
  #   # as part of a functional-style model.
  #   # Otherwise, this method is optional.
  #   shape = tf.TensorShape(input_shape).as_list()
  #   shape[-1] = self.num_classes
  #   return tf.TensorShape(shape)