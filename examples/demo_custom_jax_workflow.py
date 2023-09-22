# flake8: noqa
import os

# Set backend env to JAX
os.environ["KERAS_BACKEND"] = "jax"

import jax
import numpy as np

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
        x = jax.nn.relu(self.dense1(x))
        x = jax.nn.relu(self.dense2(x))
        return self.dense3(x)


def Dataset():
    for _ in range(20):
        yield (np.random.random((32, 128)), np.random.random((32, 4)))


def loss_fn(y_true, y_pred):
    return ops.sum((y_true - y_pred) ** 2)


model = MyModel(hidden_dim=256, output_dim=4)

optimizer = optimizers.SGD(learning_rate=0.001)
dataset = Dataset()

# Build model
x = np.random.random((1, 128))
model(x)
# Build optimizer
optimizer.build(model.trainable_variables)


######### Custom JAX workflow ###############


def compute_loss_and_updates(
    trainable_variables, non_trainable_variables, x, y
):
    y_pred, non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, x
    )
    loss = loss_fn(y, y_pred)
    return loss, non_trainable_variables


grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)


@jax.jit
def train_step(state, data):
    trainable_variables, non_trainable_variables, optimizer_variables = state
    x, y = data
    (loss, non_trainable_variables), grads = grad_fn(
        trainable_variables, non_trainable_variables, x, y
    )
    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        optimizer_variables, grads, trainable_variables
    )
    # Return updated state
    return loss, (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
    )


trainable_variables = model.trainable_variables
non_trainable_variables = model.non_trainable_variables
optimizer_variables = optimizer.variables
state = trainable_variables, non_trainable_variables, optimizer_variables
# Training loop
for data in dataset:
    loss, state = train_step(state, data)
    print("Loss:", loss)

# Post-processing model state update
trainable_variables, non_trainable_variables, optimizer_variables = state
for variable, value in zip(model.trainable_variables, trainable_variables):
    variable.assign(value)
for variable, value in zip(
    model.non_trainable_variables, non_trainable_variables
):
    variable.assign(value)
