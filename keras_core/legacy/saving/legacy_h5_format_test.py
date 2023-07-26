import os

import numpy as np
import pytest
import tensorflow as tf

import keras_core
from keras_core import layers
from keras_core import models
from keras_core import testing
from keras_core.legacy.saving import legacy_h5_format

# TODO: more thorough testing. Correctness depends
# on exact weight ordering for each layer, so we need
# to test across all types of layers.


def get_sequential_model(keras):
    return keras.Sequential(
        [
            keras.layers.Input((3,), batch_size=2),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.BatchNormalization(
                moving_mean_initializer="uniform", gamma_initializer="uniform"
            ),
            keras.layers.Dense(5, activation="softmax"),
        ]
    )


def get_functional_model(keras):
    inputs = keras.Input((3,), batch_size=2)
    x = keras.layers.Dense(4, activation="relu")(inputs)
    residual = x
    x = keras.layers.BatchNormalization(
        moving_mean_initializer="uniform", gamma_initializer="uniform"
    )(x)
    x = keras.layers.Dense(4, activation="relu")(x)
    x = keras.layers.add([x, residual])
    outputs = keras.layers.Dense(5, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def get_subclassed_model(keras):
    class MyModel(keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.dense_1 = keras.layers.Dense(3, activation="relu")
            self.dense_2 = keras.layers.Dense(1, activation="sigmoid")

        def call(self, x):
            return self.dense_2(self.dense_1(x))

    model = MyModel()
    model(np.random.random((2, 3)))
    return model


@pytest.mark.requires_trainable_backend
class LegacyH5WeightsTest(testing.TestCase):
    def _check_reloading_weights(self, ref_input, model, tf_keras_model):
        ref_output = tf_keras_model(ref_input)
        initial_weights = model.get_weights()
        # Check weights only file
        temp_filepath = os.path.join(self.get_temp_dir(), "weights.h5")
        tf_keras_model.save_weights(temp_filepath)
        model.load_weights(temp_filepath)
        output = model(ref_input)
        self.assertAllClose(ref_output, output, atol=1e-5)
        model.set_weights(initial_weights)
        model.load_weights(temp_filepath)
        output = model(ref_input)
        self.assertAllClose(ref_output, output, atol=1e-5)

    def test_sequential_model_weights(self):
        model = get_sequential_model(keras_core)
        tf_keras_model = get_sequential_model(tf.keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_weights(ref_input, model, tf_keras_model)

    def test_functional_model_weights(self):
        model = get_functional_model(keras_core)
        tf_keras_model = get_functional_model(tf.keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_weights(ref_input, model, tf_keras_model)

    def test_subclassed_model_weights(self):
        model = get_subclassed_model(keras_core)
        tf_keras_model = get_subclassed_model(tf.keras)
        ref_input = np.random.random((2, 3))
        self._check_reloading_weights(ref_input, model, tf_keras_model)


@pytest.mark.requires_trainable_backend
class LegacyH5WholeModelTest(testing.TestCase):
    def _check_reloading_model(self, ref_input, model):
        # Whole model file
        ref_output = model(ref_input)
        temp_filepath = os.path.join(self.get_temp_dir(), "model.h5")
        legacy_h5_format.save_model_to_hdf5(model, temp_filepath)
        loaded = legacy_h5_format.load_model_from_hdf5(temp_filepath)
        output = loaded(ref_input)
        self.assertAllClose(ref_output, output, atol=1e-5)

    def test_sequential_model(self):
        model = get_sequential_model(keras_core)
        ref_input = np.random.random((2, 3))
        self._check_reloading_model(ref_input, model)

    def test_functional_model_weights(self):
        model = get_functional_model(keras_core)
        ref_input = np.random.random((2, 3))
        self._check_reloading_model(ref_input, model)

    def test_compiled_model_with_various_layers(self):
        model = models.Sequential()
        model.add(layers.Dense(2, input_shape=(3,)))
        model.add(layers.RepeatVector(3))
        model.add(layers.TimeDistributed(layers.Dense(3)))

        model.compile(optimizer="rmsprop", loss="mse")
        ref_input = np.random.random((1, 3))
        self._check_reloading_model(ref_input, model)
