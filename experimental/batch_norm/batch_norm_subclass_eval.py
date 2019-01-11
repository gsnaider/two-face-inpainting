import time

import numpy as np
import tensorflow as tf


class MyModel(tf.keras.Model):

  def __init__(self, num_classes=1):
    super(MyModel, self).__init__(name='my_model')
    self.num_classes = num_classes
    # Define your layers here.

    # self.dense_1 = tf.keras.layers.Dense(5)
    self.batch_norm_1 = tf.keras.layers.BatchNormalization()
    self.relu_1 = tf.keras.layers.ReLU()
    self.dense_2 = tf.keras.layers.Dense(1)

  def call(self, inputs, train):
    # Define your forward pass here,
    # using layers you previously defined (in `__init__`).

    x = inputs

    # x = self.dense_1(x)
    x = self.batch_norm_1(x, training=train)
    x = self.relu_1(x)
    return self.dense_2(x)

    # return self.dense_2(inputs)

  def compute_output_shape(self, input_shape):
    # You need to override this function if you want to use the subclassed model
    # as part of a functional-style model.
    # Otherwise, this method is optional.
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.num_classes
    return tf.TensorShape(shape)


# tf.keras.backend.clear_session()

# TODO ver si esto afecta o no.
tf.keras.backend.set_learning_phase(0)

global_step = tf.train.get_or_create_global_step()

train_x = tf.convert_to_tensor(np.random.rand(100, 1), name="dataset_1")
train_y = train_x * 5 + 2

train_x2 = tf.convert_to_tensor(np.random.rand(200, 1), name="dataset_2")
train_y2 = train_x2 * 5 + 2

dataset1 = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(
  50).batch(10).repeat()
dataset2 = tf.data.Dataset.from_tensor_slices((train_x2, train_y2)).shuffle(
  50).batch(10).repeat()

iterator_1 = dataset1.make_one_shot_iterator()
next_1 = iterator_1.get_next()

iterator_2 = dataset2.make_one_shot_iterator()
next_2 = iterator_2.get_next()

model = MyModel()

# TODO ver si llamar dos veces al modelo (haciendo que se creen dos grafos con variables distintas)
# afecta en algo al entrenamiento..
predictions = model(next_1[0], train=False)
predictions2 = model(next_2[0], train=False)

loss_op1 = tf.nn.l2_loss(predictions - next_1[1])
loss_op2 = tf.nn.l2_loss(predictions2 - next_2[1])

loss_op = loss_op1 + loss_op2
# loss_op = loss_op1
# loss_op = loss_op2

# tf.logging.info("Printing operations.")
# for op in tf.get_default_graph().get_operations():
#   print(op.name)

bn_weights = model.layers[0].weights
print("BN Weights {}".format(bn_weights))
print("BN layer: {}".format(model.layers[0]))

with tf.train.SingularMonitoredSession(
        checkpoint_dir="/home/gaston/workspace/two-face/two-face-inpainting-experiments/local-runs/batch_norm_2/train") as sess:
  tf.logging.info("Starting evaluation.")

  writer = tf.summary.FileWriter(
    "/home/gaston/workspace/two-face/two-face-inpainting-experiments/local-runs/batch_norm_2/eval", sess.graph)

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
