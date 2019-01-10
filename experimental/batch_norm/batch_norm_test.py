import numpy as np
import tensorflow as tf


def make_model():
  x = tf.keras.layers.Input(shape=(10,), dtype=tf.float64, name='model_input')
  bn = tf.keras.layers.BatchNormalization()
  h1 = bn(x)  # This creates 2 updates.
  h2 = tf.keras.layers.ReLU()(h1)
  y = tf.keras.layers.Dense(1)(h2)

  model = tf.keras.models.Model(inputs=x, outputs=y)

  return model


model = make_model()

# Returns empty list
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# Returns same as model.updates
# update_ops = model.get_updates_for(model.inputs)

global_step = tf.train.get_or_create_global_step()

train_x = tf.convert_to_tensor(np.random.rand(100, 10))
train_y = tf.convert_to_tensor(np.random.rand(100, 1))

print(train_x)

predictions = model(train_x)

loss_op = tf.nn.l2_loss(predictions - train_y)

update_ops = model.updates
print(update_ops)
# with tf.control_dependencies(update_ops):
train_op = tf.train.AdamOptimizer().minimize(loss_op, global_step=global_step)

hooks = [tf.train.StopAtStepHook(num_steps=100000)]
with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
  while not sess.should_stop():
    (_, loss, step_value) = sess.run([train_op, loss_op, global_step])
    if (step_value % 100 == 0):
      print("Step {} - Loss: {} ".format(step_value, loss))
