import tensorflow as tf

def make_encoders():
  """Returns the gen and disc encoders."""

  vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False, 
                                            weights='imagenet', 
                                            input_tensor=None,
                                            input_shape=(128, 128, 3))
  vgg16.trainable = False

  # Remove last 5 conv layers.
  encoder = tf.keras.Model(inputs = vgg16.inputs, outputs=vgg16.layers[-6].output)
  encoder.trainable = False

  return encoder, vgg16


def make_generator_model(gen_encoder):
  
  masked_image = tf.keras.Input(shape=(128, 128, 3,), name='masked_image')
  masked_encoding = gen_encoder(masked_image)
  # 16x16x512
  
  reference_image = tf.keras.Input(shape=(128, 128, 3,), name='reference_image')
  reference_encoding = gen_encoder(reference_image)
  # 16x16x512

  encoding = tf.keras.layers.concatenate([masked_encoding, reference_encoding], axis=-1)
  # 16x16x1024
  
  # Decoder
  encoding = tf.keras.layers.Conv2DTranspose(256, (2, 2), 
                                             strides=(1, 1), 
                                             padding='same', 
                                             use_bias=False, 
                                             input_shape=(16,16,1024))(encoding)
  encoding = tf.keras.layers.BatchNormalization()(encoding)
  encoding = tf.keras.layers.LeakyReLU()(encoding)
  # 16x16x256

  encoding = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', use_bias=False)(encoding)
  encoding = tf.keras.layers.BatchNormalization()(encoding)
  encoding = tf.keras.layers.LeakyReLU()(encoding)
  # 32x32x32

  generated_patch = tf.keras.layers.Conv2DTranspose(3, (5, 5), 
                                                    strides=(1, 1), 
                                                    padding='same', 
                                                    use_bias=False, 
                                                    activation='sigmoid')(encoding)
  # 32x32x3
  
  return tf.keras.Model(inputs=[masked_image, reference_image], outputs=generated_patch)


def make_discriminator_model(disc_encoder):
  
  image = tf.keras.Input(shape=(128, 128, 3,), name='image')
  image_encoding = disc_encoder(image)
  # 4x4x512
  
  image_encoding = tf.keras.layers.Conv2D(64, (1, 1),
                                                strides=(1, 1), 
                                                padding='same', 
                                                input_shape=(4, 4, 512))(image_encoding)
  image_encoding = tf.keras.layers.BatchNormalization()(image_encoding)
  image_encoding = tf.keras.layers.LeakyReLU()(image_encoding)
  # 4x4x64
  
  image_encoding = tf.keras.layers.Flatten()(image_encoding)
  # 1024
  
  reference_image = tf.keras.Input(shape=(128, 128, 3,), name='reference_image')
  reference_encoding = disc_encoder(reference_image)
  # 4x4x512
  
  reference_encoding = tf.keras.layers.Conv2D(64, (1, 1),
                                                strides=(1, 1), 
                                                padding='same', 
                                                input_shape=(4, 4, 512))(reference_encoding)
  reference_encoding = tf.keras.layers.BatchNormalization()(reference_encoding)
  reference_encoding = tf.keras.layers.LeakyReLU()(reference_encoding)
  # 4x4x64
  
  reference_encoding = tf.keras.layers.Flatten()(reference_encoding)
  # 1024
  
  encoding = tf.keras.layers.concatenate([image_encoding, reference_encoding], axis=-1)
  # 2048
  
  # Classifier
  encoding = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)(encoding)
  logits = tf.keras.layers.Dense(1)(encoding)
  
  return tf.keras.Model(inputs=[image, reference_image], outputs=logits)


def generator_loss(generated_output):
  return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)

def discriminator_loss(real_output, generated_output):
  real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)
  generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)
  total_loss = real_loss + generated_loss
  return total_loss


def make_models():

  gen_encoder, disc_encoder = make_encoders()

  generator = make_generator_model(gen_encoder)
  discriminator = make_discriminator_model(disc_encoder)

  return generator, discriminator