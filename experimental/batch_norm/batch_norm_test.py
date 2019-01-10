import numpy as np
import tensorflow as tf


def make_model(input_tensor):

  # This works as long as the tensor is of the required shape,
  # it doesn't matter what specific value we use.
  x = tf.keras.layers.Input(shape=(1,), dtype=tf.float64, name='model_input',
                            tensor=tf.ones([131,1], dtype=tf.float64))

  # bn = tf.keras.layers.BatchNormalization()
  # h1 = bn(x)  # This creates 2 updates.
  # h2 = tf.keras.layers.ReLU()(h1)
  # y = tf.keras.layers.Dense(1)(h2)

  y = tf.keras.layers.Dense(1)(x)

  model = tf.keras.models.Model(inputs=x, outputs=y)

  return model


global_step = tf.train.get_or_create_global_step()

train_x = tf.convert_to_tensor(np.random.rand(100, 1))
train_x2 = tf.convert_to_tensor(np.random.rand(200, 1))
train_y = train_x * 5 + 2
train_y2 = train_x2 * 5 + 2

# model = make_model(None)
model = make_model(train_x)

# Returns empty list
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# Returns same as model.updates
# update_ops = model.get_updates_for(model.inputs)

predictions = model(train_x)
predictions2 = model(train_x2)

loss_op1 = tf.nn.l2_loss(predictions - train_y)
loss_op2 = tf.nn.l2_loss(predictions2 - train_y2)

loss_op = loss_op1 + loss_op2
# loss_op = loss_op2

update_ops = model.updates

print(update_ops)
with tf.control_dependencies(update_ops):
  train_op = tf.train.AdamOptimizer().minimize(loss_op, global_step=global_step)

hooks = [tf.train.StopAtStepHook(num_steps=100000)]
with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
  while not sess.should_stop():
    (_, loss, step_value, model_vars) = sess.run(
      [train_op, loss_op, global_step, model.weights])
    if (step_value % 100 == 0):
      print(
        "Step {} - Loss: {} - Model {}".format(step_value, loss, model_vars))
