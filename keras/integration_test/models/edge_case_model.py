"""Model that incorporates a set of edge case development patterns.
"""

import tensorflow as tf
from tensorflow import keras

from keras.integration_test.models.input_spec import InputSpec

INPUT_DIM = 32
NUM_CLASSES = 5


def get_data_spec(batch_size):
    return (
        InputSpec((batch_size, INPUT_DIM)),
        InputSpec((batch_size, NUM_CLASSES)),
    )


def get_input_preprocessor():
    return None


class LinearA(keras.layers.Layer):
    """Standard custom layer with 2 call() inputs."""

    def __init__(self, units=32, input_dim=32):
        super().__init__()
        self.w = self.add_weight(
            shape=(input_dim, units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(units,), initializer="zeros", trainable=True
        )

    def call(self, inputs_1, inputs_2):
        return (
            tf.matmul(inputs_1, self.w) + tf.matmul(inputs_2, self.w) + self.b
        )


class LinearB(keras.layers.Layer):
    """Layer that tracks weights in a dict attribute that gets updated later."""

    def __init__(self, units=32, input_dim=32, **kwargs):
        super().__init__(**kwargs)
        w_init = tf.random_normal_initializer()
        b_init = tf.zeros_initializer()
        self.state = {
            "kernel": tf.Variable(
                initial_value=w_init(shape=(input_dim, units), dtype="float32"),
                trainable=True,
                name="kernel",
            )
        }
        self.state["bias"] = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"),
            trainable=True,
            name="bias",
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.state["kernel"]) + self.state["bias"]


class LinearC(keras.layers.Layer):
    """Layer that creates weights in call()."""

    def __init__(self, units=32, input_dim=32, **kwargs):
        super().__init__(**kwargs)
        self._custom_built = False
        self.units = units
        self.input_dim = input_dim

    def call(self, inputs):
        if not self._custom_built:
            self.w = self.add_weight(
                shape=(self.input_dim, self.units),
                initializer="random_normal",
                trainable=True,
            )
            self.b = self.add_weight(
                shape=(self.units,), initializer="zeros", trainable=True
            )
            self._custom_built = True
        return tf.matmul(inputs, self.w) + self.b


class BatchNorm(keras.layers.Layer):
    """Layer with different training/test behavior and non-trainable updates."""

    def __init__(
        self, scale=True, center=True, epsilon=1e-6, momentum=0.9, **kwargs
    ):
        super().__init__(**kwargs)
        self.scale = scale
        self.center = center
        self.epsilon = epsilon
        self.momentum = momentum

    def build(self, input_shape):
        self.var = self.add_weight(
            shape=[input_shape[1]], initializer="ones", trainable=False
        )
        self.mean = self.add_weight(
            shape=[input_shape[1]], initializer="zeros", trainable=False
        )
        self.gamma = self.add_weight(shape=[input_shape[1]], initializer="ones")
        self.beta = self.add_weight(shape=[input_shape[1]], initializer="zeros")

    def call(self, inputs, training=False):
        if training:
            mean, var = tf.nn.moments(inputs, axes=[0])
            outputs = (inputs - mean) / (var + self.epsilon)
            self.var.assign(self.var * self.momentum + var * 0.1)
            self.mean.assign(self.mean * self.momentum + mean * 0.1)
        else:
            outputs = (inputs - self.mean) / (self.var + self.epsilon)
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs


class FunctionalSubclassModel(keras.Model):
    def __init__(self, **kwargs):
        inputs = keras.Input((INPUT_DIM,))
        x = inputs
        x = LinearA(32, INPUT_DIM)(x, x)
        x = LinearB(32, 32)(x)
        x = LinearC(32, 32)(x)
        x = BatchNorm()(x)
        outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
        super().__init__(inputs, outputs, **kwargs)


def get_model(
    build=False, compile=False, jit_compile=False, include_preprocessing=True
):
    model = FunctionalSubclassModel()
    if compile:
        model.compile("rmsprop", "mse", jit_compile=jit_compile)
    return model


def get_custom_objects():
    return {
        "LinearA": LinearA,
        "LinearB": LinearB,
        "LinearC": LinearC,
        "BatchNorm": BatchNorm,
    }
