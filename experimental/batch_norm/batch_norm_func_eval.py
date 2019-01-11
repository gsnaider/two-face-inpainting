import time

import numpy as np
import tensorflow as tf


def make_model(input_tensor):
  with tf.name_scope('my_model'):
    input = tf.keras.layers.Input(shape=(1,), dtype=tf.float64, name='model_input',
                              tensor=input_tensor)

    x = tf.keras.layers.Dense(5)(input)
    bn = tf.keras.layers.BatchNormalization()
    x = bn(x)
    x = tf.keras.layers.ReLU()(x)
    y = tf.keras.layers.Dense(1)(x)

    # y = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inputs=input, outputs=y)

  return model


# tf.keras.backend.clear_session()

# TODO ver si esto afecta o no.
tf.keras.backend.set_learning_phase(0)

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

# TODO ver si llamar dos veces al modelo (haciendo que se creen dos grafos con variables distintas)
# afecta en algo al entrenamiento..
predictions = model(next_1[0])
predictions2 = model(next_2[0])

loss_op1 = tf.nn.l2_loss(predictions - next_1[1])
loss_op2 = tf.nn.l2_loss(predictions2 - next_2[1])

loss_op = loss_op1 + loss_op2
# loss_op = loss_op1
# loss_op = loss_op2

bn_weights = model.layers[2].weights
print("BN Weights {}".format(bn_weights))

hooks = [tf.train.StopAtStepHook(num_steps=1000000)]
with tf.train.SingularMonitoredSession(
        checkpoint_dir="/home/gaston/workspace/two-face/two-face-inpainting-experiments/local-runs/batch_norm_func/train") as sess:

  writer = tf.summary.FileWriter(
    "/home/gaston/workspace/two-face/two-face-inpainting-experiments/local-runs/batch_norm_func/eval",
    sess.graph)

  print("Gamma {} - Beta {} - Moving mean {} - Moving variance {}".format(
    *sess.run(
      bn_weights)))

  while True:
    (loss, step_value, model_vars) = sess.run(
      [loss_op, global_step, model.weights])

    print("Step {} - Loss: {}".format(step_value, loss))
    print("Gamma {} - Beta {} - Moving mean {} - Moving variance {}".format(
      *sess.run(
        bn_weights)))

    time.sleep(10)
