"""Tests for optimizer."""

import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.optimizers.legacy import adadelta
from keras.optimizers.legacy import adagrad
from keras.optimizers.legacy import adam
from keras.optimizers.legacy import adamax
from keras.optimizers.legacy import ftrl
from keras.optimizers.legacy import nadam
from keras.optimizers.legacy import rmsprop
from keras.optimizers.legacy import sgd

adadelta_fn = tf.__internal__.test.combinations.NamedObject(
    "adadelta", lambda: adadelta.Adadelta(0.002)
)
adagrad_fn = tf.__internal__.test.combinations.NamedObject(
    "adagrad", lambda: adagrad.Adagrad(0.002)
)
adam_fn = tf.__internal__.test.combinations.NamedObject(
    "adam", lambda: adam.Adam(0.002)
)
adamax_fn = tf.__internal__.test.combinations.NamedObject(
    "adamax", lambda: adamax.Adamax(0.002)
)
ftrl_fn = tf.__internal__.test.combinations.NamedObject(
    "ftrl", lambda: ftrl.Ftrl(0.002)
)
gradient_descent_fn = tf.__internal__.test.combinations.NamedObject(
    "sgd", lambda: sgd.SGD(0.002)
)
nadam_fn = tf.__internal__.test.combinations.NamedObject(
    "nadam", lambda: nadam.Nadam(0.002)
)
rmsprop_fn = tf.__internal__.test.combinations.NamedObject(
    "rmsprop", lambda: rmsprop.RMSprop(0.002)
)

OPTIMIZER_FN = [
    adadelta_fn,
    adagrad_fn,
    adam_fn,
    adamax_fn,
    ftrl_fn,
    gradient_descent_fn,
    nadam_fn,
    rmsprop_fn,
]


class OptimizerFuntionalityTest(tf.test.TestCase, parameterized.TestCase):
    """Test the functionality of optimizer."""

    @parameterized.product(optimizer_fn=OPTIMIZER_FN)
    def testModelFit(self, optimizer_fn):
        model = keras.Sequential(
            [keras.layers.Input(shape=(1,)), keras.layers.Dense(1)]
        )
        optimizer = optimizer_fn()
        x = tf.expand_dims(tf.convert_to_tensor([1, 1, 1, 0, 0, 0]), axis=1)
        y = tf.expand_dims(tf.convert_to_tensor([1, 1, 1, 0, 0, 0]), axis=1)
        model.compile(loss="mse", optimizer=optimizer)
        model.fit(x, y, epochs=1, steps_per_epoch=5)


if __name__ == "__main__":
    tf.__internal__.distribute.multi_process_runner.test_main()
