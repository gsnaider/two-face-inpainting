import tensorflow as tf

import os
import tempfile

def make_identity_model(facenet_dir):
  with tf.name_scope("facenet"):
    facenet_model_path = os.path.join(facenet_dir, 'facenet_model_128.json')
    facenet_weights_path = os.path.join(facenet_dir, 'facenet_weights.h5')

    facenet = tf.keras.models.model_from_json(
      tf.gfile.GFile(facenet_model_path, "r").read())

    # This has to be done in order to be able to read the model from GCS
    model_file = tf.gfile.GFile(facenet_weights_path, mode='rb')
    temp_model_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=True)
    temp_model_file.write(model_file.read())
    model_file.close()

    tf.logging.debug(
      'Reading Facenet weights from temp file: {}'.format(temp_model_file.name))
    facenet.load_weights(temp_model_file.name)
    facenet.trainable = False
    temp_model_file.close()

  return facenet


def reconstruction_loss(original_image, patched_image):
  return tf.nn.l2_loss(original_image - patched_image) * 2.0 / (
          128.0 * 128.0 * 3.0)


def generator_adversarial_loss(generated_output):
  return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output),
                                         generated_output)


def identity_loss(reference_image, patched_image, facenet):
  # Facenet takes input images in [-1,1] range
  reference_image = reference_image * 2.0 - 1.0
  patched_image = patched_image * 2.0 - 1.0

  reference_identity = facenet(reference_image)
  patched_identity = facenet(patched_image)

  # TODO Could also try normalizing and apply dot product
  return tf.norm(reference_identity - patched_identity)


def generator_loss(original_image, patched_image, reference_image,
                   local_generated_output,
                   global_generated_output, lambda_rec,
                   lambda_adv_local, lambda_adv_global, lambda_id, facenet):
  rec_loss = lambda_rec * reconstruction_loss(original_image, patched_image)
  local_adv_loss = lambda_adv_local * generator_adversarial_loss(
    local_generated_output)
  global_adv_loss = lambda_adv_global * generator_adversarial_loss(
    global_generated_output)
  id_loss = lambda_id * identity_loss(reference_image, patched_image, facenet)

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