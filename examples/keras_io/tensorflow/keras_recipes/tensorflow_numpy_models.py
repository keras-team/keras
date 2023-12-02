"""
Title: Writing Keras Models With TensorFlow NumPy
Author: [lukewood](https://lukewood.xyz)
Date created: 2021/08/28
Last modified: 2021/08/28
Description: Overview of how to use the TensorFlow NumPy API to write Keras models.
Accelerator: GPU
"""
"""
## Introduction

[NumPy](https://numpy.org/) is a hugely successful Python linear algebra library.

TensorFlow recently launched [tf_numpy](https://www.tensorflow.org/guide/tf_numpy), a
TensorFlow implementation of a large subset of the NumPy API.
Thanks to `tf_numpy`, you can write Keras layers or models in the NumPy style!

The TensorFlow NumPy API has full integration with the TensorFlow ecosystem.
Features such as automatic differentiation, TensorBoard, Keras model callbacks,
TPU distribution and model exporting are all supported.

Let's run through a few examples.
"""

"""
## Setup
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import keras
from keras import layers

"""
To test our models we will use the Boston housing prices regression dataset.
"""

(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data(
    path="boston_housing.npz", test_split=0.2, seed=113
)
input_dim = x_train.shape[1]


def evaluate_model(model: keras.Model):
    loss, percent_error = model.evaluate(x_test, y_test, verbose=0)
    print("Mean absolute percent error before training: ", percent_error)
    model.fit(x_train, y_train, epochs=200, verbose=0)
    loss, percent_error = model.evaluate(x_test, y_test, verbose=0)
    print("Mean absolute percent error after training:", percent_error)


"""
## Subclassing keras.Model with TNP

The most flexible way to make use of the Keras API is to subclass the
[`keras.Model`](https://keras.io/api/models/model/) class.  Subclassing the Model class
gives you the ability to fully customize what occurs in the training loop.  This makes
subclassing Model a popular option for researchers.

In this example, we will implement a `Model` subclass that performs regression over the
boston housing dataset using the TNP API.  Note that differentiation and gradient
descent is handled automatically when using the TNP API alongside keras.

First let's define a simple `TNPForwardFeedRegressionNetwork` class.
"""


class TNPForwardFeedRegressionNetwork(keras.Model):
    def __init__(self, blocks=None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(blocks, list):
            raise ValueError(f"blocks must be a list, got blocks={blocks}")
        self.blocks = blocks
        self.block_weights = None
        self.biases = None

    def build(self, input_shape):
        current_shape = input_shape[1]
        self.block_weights = []
        self.biases = []
        for i, block in enumerate(self.blocks):
            self.block_weights.append(
                self.add_weight(
                    shape=(current_shape, block),
                    trainable=True,
                    name=f"block-{i}",
                    initializer="glorot_normal",
                )
            )
            self.biases.append(
                self.add_weight(
                    shape=(block,),
                    trainable=True,
                    name=f"bias-{i}",
                    initializer="zeros",
                )
            )
            current_shape = block

        self.linear_layer = self.add_weight(
            shape=(current_shape, 1),
            name="linear_projector",
            trainable=True,
            initializer="glorot_normal",
        )

    def call(self, inputs):
        activations = inputs
        for w, b in zip(self.block_weights, self.biases):
            activations = tnp.matmul(activations, w) + b
            # ReLu activation function
            activations = tnp.maximum(activations, 0.0)

        return tnp.matmul(activations, self.linear_layer)


"""
Just like with any other Keras model we can utilize any supported optimizer, loss,
metrics or callbacks that we want.

Let's see how the model performs!
"""

model = TNPForwardFeedRegressionNetwork(blocks=[3, 3])
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=[keras.metrics.MeanAbsolutePercentageError()],
)
evaluate_model(model)

"""
Great! Our model seems to be effectively learning to solve the problem at hand.

We can also write our own custom loss function using TNP.
"""


def tnp_mse(y_true, y_pred):
    return tnp.mean(tnp.square(y_true - y_pred), axis=0)


keras.backend.clear_session()
model = TNPForwardFeedRegressionNetwork(blocks=[3, 3])
model.compile(
    optimizer="adam",
    loss=tnp_mse,
    metrics=[keras.metrics.MeanAbsolutePercentageError()],
)
evaluate_model(model)

"""
## Implementing a Keras Layer Based Model with TNP

If desired, TNP can also be used in layer oriented Keras code structure.  Let's
implement the same model, but using a layered approach!
"""


def tnp_relu(x):
    return tnp.maximum(x, 0)


class TNPDense(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.w = self.add_weight(
            name="weights",
            shape=(input_shape[1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        outputs = tnp.matmul(inputs, self.w) + self.bias
        if self.activation:
            return self.activation(outputs)
        return outputs


def create_layered_tnp_model():
    return keras.Sequential(
        [
            TNPDense(3, activation=tnp_relu),
            TNPDense(3, activation=tnp_relu),
            TNPDense(1),
        ]
    )


model = create_layered_tnp_model()
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=[keras.metrics.MeanAbsolutePercentageError()],
)
model.build((None, input_dim))
model.summary()

evaluate_model(model)

"""
You can also seamlessly switch between TNP layers and native Keras layers!
"""


def create_mixed_model():
    return keras.Sequential(
        [
            TNPDense(3, activation=tnp_relu),
            # The model will have no issue using a normal Dense layer
            layers.Dense(3, activation="relu"),
            # ... or switching back to tnp layers!
            TNPDense(1),
        ]
    )


model = create_mixed_model()
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=[keras.metrics.MeanAbsolutePercentageError()],
)
model.build((None, input_dim))
model.summary()

evaluate_model(model)

"""
The Keras API offers a wide variety of layers.  The ability to use them alongside NumPy
code can be a huge time saver in projects.
"""

"""
## Distribution Strategy

TensorFlow NumPy and Keras integrate with
[TensorFlow Distribution Strategies](https://www.tensorflow.org/guide/distributed_training).
This makes it simple to perform distributed training across multiple GPUs,
or even an entire TPU Pod.
"""

gpus = tf.config.list_logical_devices("GPU")
if gpus:
    strategy = tf.distribute.MirroredStrategy(gpus)
else:
    # We can fallback to a no-op CPU strategy.
    strategy = tf.distribute.get_strategy()
print("Running with strategy:", str(strategy.__class__.__name__))

with strategy.scope():
    model = create_layered_tnp_model()
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=[keras.metrics.MeanAbsolutePercentageError()],
    )
    model.build((None, input_dim))
    model.summary()
    evaluate_model(model)

"""
## TensorBoard Integration

One of the many benefits of using the Keras API is the ability to monitor training
through TensorBoard.  Using the TensorFlow NumPy API alongside Keras allows you to easily
leverage TensorBoard.
"""

keras.backend.clear_session()

"""
To load the TensorBoard from a Jupyter notebook, you can run the following magic:
```
%load_ext tensorboard
```

"""

models = [
    (
        TNPForwardFeedRegressionNetwork(blocks=[3, 3]),
        "TNPForwardFeedRegressionNetwork",
    ),
    (create_layered_tnp_model(), "layered_tnp_model"),
    (create_mixed_model(), "mixed_model"),
]
for model, model_name in models:
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=[keras.metrics.MeanAbsolutePercentageError()],
    )
    model.fit(
        x_train,
        y_train,
        epochs=200,
        verbose=0,
        callbacks=[keras.callbacks.TensorBoard(log_dir=f"logs/{model_name}")],
    )

"""
To load the TensorBoard from a Jupyter notebook you can use the `%tensorboard` magic:

```
%tensorboard --logdir logs
```

The TensorBoard monitor metrics and examine the training curve.

![Tensorboard training graph](https://i.imgur.com/wsOuFnz.png)

The TensorBoard also allows you to explore the computation graph used in your models.

![Tensorboard graph exploration](https://i.imgur.com/tOrezDL.png)

The ability to introspect into your models can be valuable during debugging.
"""

"""
## Conclusion

Porting existing NumPy code to Keras models using the `tensorflow_numpy` API is easy!
By integrating with Keras you gain the ability to use existing Keras callbacks, metrics
and optimizers, easily distribute your training and use Tensorboard.

Migrating a more complex model, such as a ResNet, to the TensorFlow NumPy API would be a
great follow up learning exercise.

Several open source NumPy ResNet implementations are available online.
"""
