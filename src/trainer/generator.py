import tensorflow as tf

IMAGE_SIZE = 128

class ChannelWiseFCLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(ChannelWiseFCLayer, self).__init__()

  def build(self, input_shape):
    _, self.width, self.height, self.n_feat_map = input_shape.as_list()
    self.W = self.add_variable("W",
                               shape=[self.n_feat_map, self.width * self.height,
                                      self.width * self.height])

  def call(self, input):
    input_reshape = tf.reshape(input,
                               [-1, self.width * self.height, self.n_feat_map])
    input_transpose = tf.transpose(input_reshape, [2, 0, 1])
    output = tf.matmul(input_transpose, self.W)

    output_transpose = tf.transpose(output, [1, 2, 0])
    output_reshape = tf.reshape(output_transpose,
                                [-1, self.height, self.width, self.n_feat_map])
    return output_reshape


class GeneratorEncoder(tf.keras.Model):
  def __init__(self):
    super(GeneratorEncoder, self).__init__(name='generator_encoder')

    with tf.name_scope("vgg"):
      vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False,
                                                weights='imagenet',
                                                input_tensor=None,
                                                input_shape=(
                                                IMAGE_SIZE, IMAGE_SIZE, 3))

      vgg16.trainable = False

      self.encoder = tf.keras.Model(inputs=vgg16.inputs,
                                     outputs=vgg16.layers[10].output)

      self.encoder.trainable = False

    with tf.name_scope("gen_encoder"):
      self.conv_1 = tf.keras.layers.Conv2D(512, (3, 3),
                                           strides=(1, 1),
                                           padding='same')

      self.batch_norm_1 = tf.keras.layers.BatchNormalization()
      self.relu_1 = tf.keras.layers.LeakyReLU()

      self.conv_2 = tf.keras.layers.Conv2D(512, (3, 3),
                                           strides=(1, 1),
                                           padding='same')

      self.batch_norm_2 = tf.keras.layers.BatchNormalization()
      self.relu_2 = tf.keras.layers.LeakyReLU()

      self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

  def call(self, inputs, training=False):
    x = inputs

    x = self.encoder(x)

    x = self.conv_1(x)
    x = self.batch_norm_1(x, training=training)
    x = self.relu_1(x)

    x = self.conv_2(x)
    x = self.batch_norm_2(x, training=training)
    x = self.relu_2(x)

    x = self.max_pool(x)

    return x


class Generator(tf.keras.Model):
  def __init__(self):
    super(Generator, self).__init__(name='generator')

    with tf.name_scope('gen_masked_enconder'):
      self.masked_encoder = GeneratorEncoder()

    with tf.name_scope('gen_reference_enconder'):
      self.reference_encoder = GeneratorEncoder()


    self.concat = tf.keras.layers.Concatenate(axis=-1)

    with tf.name_scope('gen_channel_wise_fc'):
      self.channel_wise_fc = ChannelWiseFCLayer()

    with tf.name_scope('gen_decoder'):
      self.conv_transp_1 = tf.keras.layers.Conv2DTranspose(512, (3, 3),
                                                 strides=(1, 1),
                                                 padding='same',
                                                 use_bias=False,
                                                 input_shape=(8, 8, 1024))

      self.conv_transp_2 = tf.keras.layers.Conv2DTranspose(256, (3, 3),
                                                 strides=(1, 1),
                                                 padding='same',
                                                 use_bias=False,
                                                 input_shape=(8, 8, 512))

      self.up_sampling_1 = tf.keras.layers.UpSampling2D(size=(2, 2))

      self.conv_transp_3 = tf.keras.layers.Conv2DTranspose(128, (3, 3),
                                                 strides=(1, 1),
                                                 padding='same',
                                                 use_bias=False,
                                                 input_shape=(16, 16, 256))
      self.batch_norm_1 = tf.keras.layers.BatchNormalization()
      self.relu_1 = tf.keras.layers.LeakyReLU()

      self.conv_transp_4 = tf.keras.layers.Conv2DTranspose(64, (3, 3),
                                                 strides=(1, 1),
                                                 padding='same',
                                                 use_bias=False,
                                                 input_shape=(16, 16, 128))
      self.batch_norm_2 = tf.keras.layers.BatchNormalization()
      self.relu_2 = tf.keras.layers.LeakyReLU()

      self.up_sampling_2 = tf.keras.layers.UpSampling2D(size=(2, 2))

      self.conv_transp_5 = tf.keras.layers.Conv2DTranspose(3, (3, 3),
                                                        strides=(1, 1),
                                                        padding='same',
                                                        use_bias=False,
                                                        activation='sigmoid')

  def call(self, inputs, training=False):
    masked_image = inputs[0]
    reference_image = inputs[1]

    masked_encoding = self.masked_encoder(masked_image)
    reference_encoding = self.reference_encoder(reference_image)

    x = self.concat(masked_encoding, reference_encoding)

    x = self.channel_wise_fc(x)

    x = self.conv_transp_1(x)
    x = self.conv_transp_2(x)

    x = self.up_sampling_1(x)

    x = self.conv_transp_3(x)
    x = self.batch_norm_1(x, training=training)
    x = self.relu_1(x)

    x = self.conv_transp_4(x)
    x = self.batch_norm_2(x, training=training)
    x = self.relu_2(x)

    x = self.up_sampling_2(x)
    x = self.conv_transp_5(x)

    return x

