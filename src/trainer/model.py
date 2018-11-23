import tensorflow as tf
import os

FACENET_DIR = '/home/gaston/workspace/two-face/facenet'

# facenet model structure: https://github.com/serengil/tensorflow-101/blob/master/model/facenet_model.json
FACENET_MODEL_PATH = os.path.join(FACENET_DIR,
                                  'tensorflow-101/model/facenet_model_128.json')

#pre-trained weights https://drive.google.com/file/d/1971Xk5RwedbudGgTIrGAL4F7Aifu7id1/view?usp=sharing
FACENET_WEIGHTS_PATH = os.path.join(FACENET_DIR, 'facenet_weights.h5')

IMAGE_SIZE = 128
PATCH_SIZE = 32

# In tf <= 1.10, min input size for vgg is 48x48
EXPANDED_PATCH_SIZE = 48

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

def make_generator_encoder():
  vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False,
                                            weights='imagenet',
                                            input_tensor=None,
                                            input_shape=(128, 128, 3))

  vgg16.trainable = False

  input_encoder = tf.keras.Model(inputs=vgg16.inputs,
                               outputs=vgg16.layers[10].output)

  for layer in input_encoder.layers[:10]:
    layer.trainable = False

  # Generator encoder
  # Keep only first 10 layers of vgg for the generator
  gen_encoder = tf.keras.layers.Conv2D(512, (3, 3),
                                       strides=(1, 1),
                                       padding='same')(
    input_encoder.layers[-1].output)

  # 16x16x512
  gen_encoder = tf.keras.layers.BatchNormalization()(gen_encoder)
  gen_encoder = tf.keras.layers.LeakyReLU()(gen_encoder)

  gen_encoder = tf.keras.layers.Conv2D(512, (3, 3),
                                       strides=(1, 1),
                                       padding='same')(gen_encoder)
  gen_encoder = tf.keras.layers.BatchNormalization()(gen_encoder)
  gen_encoder = tf.keras.layers.LeakyReLU()(gen_encoder)
  # 16x16x512

  gen_encoder = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(
    gen_encoder)
  # 8x8x512

  return tf.keras.Model(inputs=input_encoder.inputs, outputs=gen_encoder)

def make_generator_model():

  mask_encoder = make_generator_encoder()
  reference_encoder = make_generator_encoder()

  masked_image = tf.keras.Input(shape=(128, 128, 3,), name='masked_image')
  masked_encoding = mask_encoder(masked_image)
  # 8x8x512

  reference_image = tf.keras.Input(shape=(128, 128, 3,), name='reference_image')
  reference_encoding = reference_encoder(reference_image)
  # 8x8x512

  encoding = tf.keras.layers.concatenate([masked_encoding, reference_encoding],
                                         axis=-1)
  # 8x8x1024

  encoding = ChannelWiseFCLayer()(encoding)
  # 8x8x1024

  # Decoder
  encoding = tf.keras.layers.Conv2DTranspose(512, (3, 3),
                                             strides=(1, 1),
                                             padding='same',
                                             use_bias=False,
                                             input_shape=(8, 8, 1024))(
    encoding)
  # 8x8x512

  encoding = tf.keras.layers.Conv2DTranspose(256, (3, 3),
                                             strides=(1, 1),
                                             padding='same',
                                             use_bias=False,
                                             input_shape=(8, 8, 512))(
    encoding)
  # 8x8x256

  encoding = tf.keras.layers.UpSampling2D(size=(2, 2))(encoding)
  # 16x16x256

  encoding = tf.keras.layers.Conv2DTranspose(128, (3, 3),
                                             strides=(1, 1),
                                             padding='same',
                                             use_bias=False,
                                             input_shape=(16, 16, 256))(
    encoding)
  encoding = tf.keras.layers.BatchNormalization()(encoding)
  encoding = tf.keras.layers.LeakyReLU()(encoding)
  # 16x16x128

  encoding = tf.keras.layers.Conv2DTranspose(64, (3, 3),
                                             strides=(1, 1),
                                             padding='same',
                                             use_bias=False,
                                             input_shape=(16, 16, 128))(
    encoding)
  encoding = tf.keras.layers.BatchNormalization()(encoding)
  encoding = tf.keras.layers.LeakyReLU()(encoding)
  # 16x16x64

  encoding = tf.keras.layers.UpSampling2D(size=(2, 2))(encoding)
  # 32x32x64

  generated_patch = tf.keras.layers.Conv2DTranspose(3, (3, 3),
                                                    strides=(1, 1),
                                                    padding='same',
                                                    use_bias=False,
                                                    activation='sigmoid')(
    encoding)
  # 32x32x3

  return tf.keras.Model(inputs=[masked_image, reference_image],
                        outputs=generated_patch)


def make_local_discriminator_model():
  patch = tf.keras.Input(shape=(EXPANDED_PATCH_SIZE, EXPANDED_PATCH_SIZE, 3,),
                         name='patch')

  vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False,
                                            weights='imagenet',
                                            input_tensor=None,
                                            input_shape=(
                                              EXPANDED_PATCH_SIZE,
                                              EXPANDED_PATCH_SIZE, 3))
  encoder = tf.keras.Model(inputs=vgg16.inputs,
                           outputs=vgg16.layers[10].output)
  encoder.trainable = False

  encoding = encoder(patch)
  #6x6x256

  encoding = tf.keras.layers.Conv2D(128, (3, 3),
                                          strides=(1, 1),
                                          padding='same',
                                          input_shape=(6, 6, 256))(encoding)
  encoding = tf.keras.layers.BatchNormalization()(encoding)
  encoding = tf.keras.layers.LeakyReLU()(encoding)
  # 6x6x128

  encoding = tf.keras.layers.Conv2D(64, (3, 3),
                                          strides=(1, 1),
                                          padding='same',
                                          input_shape=(4, 4, 128))(encoding)
  encoding = tf.keras.layers.BatchNormalization()(encoding)
  encoding = tf.keras.layers.LeakyReLU()(encoding)
  # 6x6x64

  encoding = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(
    encoding)
  # 3x3x64

  encoding = tf.keras.layers.Flatten()(encoding)
  # 1024

  # Classifier
  encoding = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)(encoding)
  logits = tf.keras.layers.Dense(1)(encoding)

  return tf.keras.Model(inputs=patch, outputs=logits)



def make_global_discriminator_model():


  vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False,
                                            weights='imagenet',
                                            input_tensor=None,
                                            input_shape=(128, 128, 3))
  vgg16.trainable = False

  disc_encoder = tf.keras.Model(inputs=vgg16.inputs,
                               outputs=vgg16.layers[10].output)

  for layer in disc_encoder.layers[:10]:
    layer.trainable = False

  image = tf.keras.Input(shape=(128, 128, 3,), name='image')

  disc_encoder = disc_encoder(image)
  # 4x4x64


  # Take the first 10 layers of vgg
  disc_encoder = tf.keras.layers.Conv2D(256, (3, 3),
                                        strides=(1, 1),
                                        padding='same')(disc_encoder)
  # 16x16x256
  disc_encoder = tf.keras.layers.BatchNormalization()(disc_encoder)
  disc_encoder = tf.keras.layers.LeakyReLU()(disc_encoder)

  disc_encoder = tf.keras.layers.Conv2D(128, (3, 3),
                                        strides=(1, 1),
                                        padding='same')(disc_encoder)
  disc_encoder = tf.keras.layers.BatchNormalization()(disc_encoder)
  disc_encoder = tf.keras.layers.LeakyReLU()(disc_encoder)
  # 16x16x128

  disc_encoder = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(
    disc_encoder)
  # 8x8x128

  disc_encoder = tf.keras.layers.Conv2D(64, (3, 3),
                                        strides=(1, 1),
                                        padding='same')(disc_encoder)
  disc_encoder = tf.keras.layers.BatchNormalization()(disc_encoder)
  disc_encoder = tf.keras.layers.LeakyReLU()(disc_encoder)
  # 8x8x64

  disc_encoder = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(
    disc_encoder)
  # 4x4x64


  image_encoding = tf.keras.layers.Flatten()(disc_encoder)
  # 2048

  # Classifier
  image_encoding = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)(
    image_encoding)
  logits = tf.keras.layers.Dense(1)(image_encoding)

  return tf.keras.Model(inputs=image, outputs=logits)


def make_identity_model():
  facenet = tf.keras.models.model_from_json(
    open(FACENET_MODEL_PATH, "r").read())
  facenet.load_weights(FACENET_WEIGHTS_PATH)
  facenet.trainable = False
  return facenet


def reconstruction_loss(original_image, patched_image):
  return tf.nn.l2_loss(original_image - patched_image) * 2.0 / (
            128.0 * 128.0 * 3.0)


def generator_adversarial_loss(generated_output):
  return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output),
                                         generated_output)


def identity_loss(original_image, patched_image, facenet):
  original_identity = facenet(original_image)
  patched_identity = facenet(patched_image)

  return tf.norm(original_identity - patched_identity)

def generator_loss(original_image, patched_image, local_generated_output,
                   global_generated_output, facenet, lambda_rec,
                   lambda_adv_local, lambda_adv_global, lambda_id):
  rec_loss = lambda_rec * reconstruction_loss(original_image, patched_image)
  local_adv_loss = lambda_adv_local * generator_adversarial_loss(
    local_generated_output)
  global_adv_loss = lambda_adv_global * generator_adversarial_loss(
    global_generated_output)
  id_loss = lambda_id * identity_loss(original_image, patched_image, facenet)

  tf.summary.scalar('rec_loss', rec_loss)
  tf.summary.scalar('local_adv_loss', local_adv_loss)
  tf.summary.scalar('global_adv_loss', global_adv_loss)
  tf.summary.scalar('identity_loss', id_loss)

  return rec_loss + local_adv_loss + global_adv_loss + id_loss


def discriminator_loss(real_output, generated_output, lambda_adv):
  real_loss = tf.losses.sigmoid_cross_entropy(
    multi_class_labels=tf.ones_like(real_output), logits=real_output)
  generated_loss = tf.losses.sigmoid_cross_entropy(
    multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)
  total_loss = real_loss + generated_loss
  return lambda_adv * total_loss


def make_models():

  generator = make_generator_model()
  local_discriminator = make_local_discriminator_model()
  global_discriminator = make_global_discriminator_model()
  facenet = make_identity_model()

  tf.logging.info('Generator')
  generator.summary()

  tf.logging.info('Local Discriminator')
  local_discriminator.summary()

  tf.logging.info('Global Discriminator')
  global_discriminator.summary()

  tf.logging.info('FaceNet')
  facenet.summary()

  return generator, local_discriminator, global_discriminator, facenet
