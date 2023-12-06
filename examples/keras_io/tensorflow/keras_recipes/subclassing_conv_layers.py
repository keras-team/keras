"""
Title: Customizing the convolution operation of a Conv2D layer
Author: [lukewood](https://lukewood.xyz)
Date created: 11/03/2021
Last modified: 11/03/2021
Description: This example shows how to implement custom convolution layers using the `Conv.convolution_op()` API.
Accelerator: GPU
"""
"""
## Introduction

You may sometimes need to implement custom versions of convolution layers like `Conv1D` and `Conv2D`.
Keras enables you do this without implementing the entire layer from scratch: you can reuse
most of the base convolution layer and just customize the convolution op itself via the
`convolution_op()` method.

This method was introduced in Keras 2.7. So before using the
`convolution_op()` API, ensure that you are running Keras version 2.7.0 or greater.
"""

"""
## A Simple `StandardizedConv2D` implementation

There are two ways to use the `Conv.convolution_op()` API. The first way
is to override the `convolution_op()` method on a convolution layer subclass.
Using this approach, we can quickly implement a
[StandardizedConv2D](https://arxiv.org/abs/1903.10520) as shown below.
"""
import tensorflow as tf
import keras
from keras import layers
import numpy as np


class StandardizedConv2DWithOverride(layers.Conv2D):
    def convolution_op(self, inputs, kernel):
        mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
        return tf.nn.conv2d(
            inputs,
            (kernel - mean) / tf.sqrt(var + 1e-10),
            padding="valid",
            strides=list(self.strides),
            name=self.__class__.__name__,
        )


"""
The other way to use the `Conv.convolution_op()` API is to directly call the
`convolution_op()` method from the `call()` method of a convolution layer subclass.
A comparable class implemented using this approach is shown below.
"""


class StandardizedConv2DWithCall(layers.Conv2D):
    def call(self, inputs):
        mean, var = tf.nn.moments(self.kernel, axes=[0, 1, 2], keepdims=True)
        result = self.convolution_op(
            inputs, (self.kernel - mean) / tf.sqrt(var + 1e-10)
        )
        if self.use_bias:
            result = result + self.bias
        return result


"""
## Example Usage

Both of these layers work as drop-in replacements for `Conv2D`. The following
demonstration performs classification on the MNIST dataset.
"""

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        StandardizedConv2DWithCall(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        StandardizedConv2DWithOverride(
            64, kernel_size=(3, 3), activation="relu"
        ),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()
"""

"""
batch_size = 128
epochs = 5

model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

model.fit(
    x_train, y_train, batch_size=batch_size, epochs=5, validation_split=0.1
)

"""
## Conclusion

The `Conv.convolution_op()` API provides an easy and readable way to implement custom
convolution layers. A `StandardizedConvolution` implementation using the API is quite
terse, consisting of only four lines of code.
"""
