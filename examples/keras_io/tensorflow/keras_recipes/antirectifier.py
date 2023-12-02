"""
Title: Simple custom layer example: Antirectifier
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2016/01/06
Last modified: 2020/04/20
Description: Demonstration of custom layer creation.
Accelerator: GPU
"""
"""
## Introduction

This example shows how to create custom layers, using the Antirectifier layer
 (originally proposed as a Keras example script in January 2016), an alternative
to ReLU. Instead of zeroing-out the negative part of the input, it splits the negative
 and positive parts and returns the concatenation of the absolute value
of both. This avoids loss of information, at the cost of an increase in dimensionality.
 To fix the dimensionality increase, we linearly combine the
features back to a space of the original size.
"""

"""
## Setup
"""

import tensorflow as tf
import keras
from keras import layers

"""
## The Antirectifier layer
"""


class Antirectifier(layers.Layer):
    def __init__(self, initializer="he_normal", **kwargs):
        super().__init__(**kwargs)
        self.initializer = keras.initializers.get(initializer)

    def build(self, input_shape):
        output_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(output_dim * 2, output_dim),
            initializer=self.initializer,
            name="kernel",
            trainable=True,
        )

    def call(self, inputs):
        inputs -= tf.reduce_mean(inputs, axis=-1, keepdims=True)
        pos = tf.nn.relu(inputs)
        neg = tf.nn.relu(-inputs)
        concatenated = tf.concat([pos, neg], axis=-1)
        mixed = tf.matmul(concatenated, self.kernel)
        return mixed

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super().get_config()
        config = {"initializer": keras.initializers.serialize(self.initializer)}
        return dict(list(base_config.items()) + list(config.items()))


"""
## Let's test-drive it on MNIST
"""

# Training parameters
batch_size = 128
num_classes = 10
epochs = 20

# The data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Build the model
model = keras.Sequential(
    [
        keras.Input(shape=(784,)),
        layers.Dense(256),
        Antirectifier(),
        layers.Dense(256),
        Antirectifier(),
        layers.Dropout(0.5),
        layers.Dense(10),
    ]
)

# Compile the model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train the model
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.15,
)

# Test the model
model.evaluate(x_test, y_test)
