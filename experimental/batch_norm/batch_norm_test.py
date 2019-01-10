import numpy as np
import tensorflow as tf


def make_model(input_tensor):
  with tf.name_scope('my_model'):
    x = tf.keras.layers.Input(shape=(1,), dtype=tf.float64, name='model_input',
                              tensor=input_tensor)

    bn = tf.keras.layers.BatchNormalization()
    h1 = bn(x)
    h2 = tf.keras.layers.ReLU()(h1)
    y = tf.keras.layers.Dense(1)(h2)

    # y = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inputs=x, outputs=y)

  return model


# tf.keras.backend.clear_session()

# TODO ver si esto afecta o no.
tf.keras.backend.set_learning_phase(1)

global_step = tf.train.get_or_create_global_step()

train_x = tf.convert_to_tensor(np.random.rand(100, 1), name="dataset_1")
train_y = train_x * 5 + 2

train_x2 = tf.convert_to_tensor(np.random.rand(200, 1), name="dataset_2")
train_y2 = train_x2 * 5 + 2

dataset1 = tf.data.Dataset.from_tensor_slices((train_x, train_y))
dataset2 = tf.data.Dataset.from_tensor_slices((train_x2, train_y2))

# dataset = tf.data.Dataset.zip((dataset1, dataset2))

dataset1 = dataset1.shuffle(50).batch(10).repeat()
dataset2 = dataset2.shuffle(50).batch(10).repeat()


# model = make_model(train_x)
# This works as long as the tensor is of the required shape,
# it doesn't matter what specific value we use.
model = make_model(tf.zeros([1, 1], dtype=tf.float64, name="my_constant"))

# Returns empty list
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# Returns same as model.updates
# update_ops = model.get_updates_for(model.inputs)

iterator_1 = dataset1.make_one_shot_iterator()
next_1 = iterator_1.get_next()

iterator_2 = dataset2.make_one_shot_iterator()
next_2 = iterator_2.get_next()

predictions = model(next_1[0])
predictions2 = model(next_2[0])

loss_op1 = tf.nn.l2_loss(predictions - next_1[1])
loss_op2 = tf.nn.l2_loss(predictions2 - next_2[1])

loss_op = loss_op1 + loss_op2
# loss_op = loss_op1
# loss_op = loss_op2

update_ops = model.updates
print("Update ops: {}".format(update_ops))

with tf.control_dependencies(update_ops):
  train_op = tf.train.AdamOptimizer().minimize(loss_op, global_step=global_step)

hooks = [tf.train.StopAtStepHook(num_steps=1000000)]
with tf.train.MonitoredTrainingSession(
        checkpoint_dir="/home/gaston/workspace/two-face/two-face-inpainting-experiments/local-runs/batch_norm/train",
        hooks=hooks) as sess:
  while not sess.should_stop():
    (_, loss, step_value, model_vars) = sess.run(
      [train_op, loss_op, global_step, model.weights])
    if (step_value % 100 == 0):
      print(
        "Step {} - Loss: {} - Model {}".format(step_value, loss, model_vars))
      print("Mean_1 {} - Variance_1 {} - Mean_2 {} - Variance_2 {}".format(
        *sess.run(
          ["model/batch_normalization/moments/mean:0",
           "model/batch_normalization/moments/variance:0",
           "model_1/batch_normalization/moments/mean:0",
           "model_1/batch_normalization/moments/variance:0"
           ])))
