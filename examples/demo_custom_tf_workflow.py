# flake8: noqa
import os

# Set backend env to tensorflow
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf

from keras import Model
from keras import backend
from keras import initializers
from keras import layers
from keras import ops
from keras import optimizers


class MyDense(layers.Layer):
    def __init__(self, units, name=None):
        super().__init__(name=name)
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        w_shape = (input_dim, self.units)
        w_value = initializers.GlorotUniform()(w_shape)
        self.w = backend.Variable(w_value, name="kernel")

        b_shape = (self.units,)
        b_value = initializers.Zeros()(b_shape)
        self.b = backend.Variable(b_value, name="bias")

    def call(self, inputs):
        return ops.matmul(inputs, self.w) + self.b


class MyModel(Model):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.dense1 = MyDense(hidden_dim)
        self.dense2 = MyDense(hidden_dim)
        self.dense3 = MyDense(output_dim)

    def call(self, x):
        x = tf.nn.relu(self.dense1(x))
        x = tf.nn.relu(self.dense2(x))
        return self.dense3(x)


def Dataset():
    for _ in range(20):
        yield (
            np.random.random((32, 128)).astype("float32"),
            np.random.random((32, 4)).astype("float32"),
        )


def loss_fn(y_true, y_pred):
    return ops.sum((y_true - y_pred) ** 2)


model = MyModel(hidden_dim=256, output_dim=4)

optimizer = optimizers.SGD(learning_rate=0.001)
dataset = Dataset()


######### Custom TF workflow ###############


@tf.function(jit_compile=True)
def train_step(data):
    x, y = data
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = loss_fn(y, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


for data in dataset:
    loss = train_step(data)
    print("Loss:", float(loss))
