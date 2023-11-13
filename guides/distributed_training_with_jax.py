"""
Title: Multi-GPU distributed training with JAX
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2023/07/11
Last modified: 2023/07/11
Description: Guide to multi-GPU/TPU training for Keras models with JAX.
Accelerator: GPU
"""
"""
## Introduction

There are generally two ways to distribute computation across multiple devices:

**Data parallelism**, where a single model gets replicated on multiple devices or
multiple machines. Each of them processes different batches of data, then they merge
their results. There exist many variants of this setup, that differ in how the different
model replicas merge results, in whether they stay in sync at every batch or whether they
are more loosely coupled, etc.

**Model parallelism**, where different parts of a single model run on different devices,
processing a single batch of data together. This works best with models that have a
naturally-parallel architecture, such as models that feature multiple branches.

This guide focuses on data parallelism, in particular **synchronous data parallelism**,
where the different replicas of the model stay in sync after each batch they process.
Synchronicity keeps the model convergence behavior identical to what you would see for
single-device training.

Specifically, this guide teaches you how to use `jax.sharding` APIs to train Keras
models, with minimal changes to your code, on multiple GPUs or TPUS (typically 2 to 16)
installed on a single machine (single host, multi-device training). This is the
most common setup for researchers and small-scale industry workflows.
"""

"""
## Setup

Let's start by defining the function that creates the model that we will train,
and the function that creates the dataset we will train on (MNIST in this case).
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import jax
import numpy as np
import tensorflow as tf
import keras

from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def get_model():
    # Make a simple convnet with batch normalization and dropout.
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.Conv2D(
        filters=12, kernel_size=3, padding="same", use_bias=False
    )(x)
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(
        filters=24,
        kernel_size=6,
        use_bias=False,
        strides=2,
    )(x)
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size=6,
        padding="same",
        strides=2,
        name="large_k",
    )(x)
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    return model


def get_datasets():
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # Create TF Datasets
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    eval_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return train_data, eval_data


"""
## Single-host, multi-device synchronous training

In this setup, you have one machine with several GPUs or TPUs on it (typically 2 to 16).
Each device will run a copy of your model (called a **replica**). For simplicity, in
what follows, we'll assume we're dealing with 8 GPUs, at no loss of generality.

**How it works**

At each step of training:

- The current batch of data (called **global batch**) is split into 8 different
  sub-batches (called **local batches**). For instance, if the global batch has 512
  samples, each of the 8 local batches will have 64 samples.
- Each of the 8 replicas independently processes a local batch: they run a forward pass,
  then a backward pass, outputting the gradient of the weights with respect to the loss of
  the model on the local batch.
- The weight updates originating from local gradients are efficiently merged across the 8
  replicas. Because this is done at the end of every step, the replicas always stay in
  sync.

In practice, the process of synchronously updating the weights of the model replicas is
handled at the level of each individual weight variable. This is done through a using
a `jax.sharding.NamedSharding` that is configured to replicate the variables.

**How to use it**

To do single-host, multi-device synchronous training with a Keras model, you
would use the `jax.sharding` features. Here's how it works:

- We first create a device mesh using `mesh_utils.create_device_mesh`.
- We use `jax.sharding.Mesh`, `jax.sharding.NamedSharding` and
  `jax.sharding.PartitionSpec` to define how to partition JAX arrays.
    - We specify that we want to replicate the model and optimizer variables
      across all devices by using a spec with no axis.
    - We specify that we want to shard the data across devices by using a spec
      that splits along the batch dimension.
- We use `jax.device_put` to replicate the model and optimizer variables across
  devices. This happens once at the beginning.
- In the training loop, for each batch that we process, we use `jax.device_put`
  to split the batch across devices before invoking the train step.

Here's the flow, where each step is split into its own utility function:
"""

# Config
num_epochs = 2
batch_size = 64

train_data, eval_data = get_datasets()
train_data = train_data.batch(batch_size, drop_remainder=True)

model = get_model()
optimizer = keras.optimizers.Adam(1e-3)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Initialize all state with .build()
(one_batch, one_batch_labels) = next(iter(train_data))
model.build(one_batch)
optimizer.build(model.trainable_variables)


# This is the loss function that will be differentiated.
# Keras provides a pure functional forward pass: model.stateless_call
def compute_loss(trainable_variables, non_trainable_variables, x, y):
    y_pred, updated_non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, x
    )
    loss_value = loss(y, y_pred)
    return loss_value, updated_non_trainable_variables


# Function to compute gradients
compute_gradients = jax.value_and_grad(compute_loss, has_aux=True)


# Training step, Keras provides a pure functional optimizer.stateless_apply
@jax.jit
def train_step(train_state, x, y):
    (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
    ) = train_state
    (loss_value, non_trainable_variables), grads = compute_gradients(
        trainable_variables, non_trainable_variables, x, y
    )

    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        optimizer_variables, grads, trainable_variables
    )

    return loss_value, (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
    )


# Replicate the model and optimizer variable on all devices
def get_replicated_train_state(devices):
    # All variables will be replicated on all devices
    var_mesh = Mesh(devices, axis_names=("_"))
    # In NamedSharding, axes not mentioned are replicated (all axes here)
    var_replication = NamedSharding(var_mesh, P())

    # Apply the distribution settings to the model variables
    trainable_variables = jax.device_put(
        model.trainable_variables, var_replication
    )
    non_trainable_variables = jax.device_put(
        model.non_trainable_variables, var_replication
    )
    optimizer_variables = jax.device_put(optimizer.variables, var_replication)

    # Combine all state in a tuple
    return (trainable_variables, non_trainable_variables, optimizer_variables)


num_devices = len(jax.local_devices())
print(f"Running on {num_devices} devices: {jax.local_devices()}")
devices = mesh_utils.create_device_mesh((num_devices,))

# Data will be split along the batch axis
data_mesh = Mesh(devices, axis_names=("batch",))  # naming axes of the mesh
data_sharding = NamedSharding(
    data_mesh,
    P(
        "batch",
    ),
)  # naming axes of the sharded partition

# Display data sharding
x, y = next(iter(train_data))
sharded_x = jax.device_put(x.numpy(), data_sharding)
print("Data sharding")
jax.debug.visualize_array_sharding(jax.numpy.reshape(sharded_x, [-1, 28 * 28]))

train_state = get_replicated_train_state(devices)

# Custom training loop
for epoch in range(num_epochs):
    data_iter = iter(train_data)
    for data in data_iter:
        x, y = data
        sharded_x = jax.device_put(x.numpy(), data_sharding)
        loss_value, train_state = train_step(train_state, sharded_x, y.numpy())
    print("Epoch", epoch, "loss:", loss_value)

# Post-processing model state update to write them back into the model
trainable_variables, non_trainable_variables, optimizer_variables = train_state
for variable, value in zip(model.trainable_variables, trainable_variables):
    variable.assign(value)
for variable, value in zip(
    model.non_trainable_variables, non_trainable_variables
):
    variable.assign(value)

"""
That's it!
"""
