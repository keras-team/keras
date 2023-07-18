"""Tests for SavedModel functionality under tf implementation."""

import os

import numpy as np
import pytest
import tensorflow as tf

from keras_core import backend
from keras_core import layers
from keras_core import metrics
from keras_core import models
from keras_core import testing
from keras_core.saving import object_registration


@object_registration.register_keras_serializable(package="my_package")
class CustomModelX(models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = layers.Dense(1)
        self.dense2 = layers.Dense(1)

    def call(self, inputs):
        out = self.dense1(inputs)
        return self.dense2(out)

    def one(self):
        return 1


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="The SavedModel test can only run with TF backend.",
)
class SavedModelTest(testing.TestCase):
    def test_sequential(self):
        model = models.Sequential([layers.Dense(1)])
        model.compile(loss="mse", optimizer="adam")
        X_train = np.random.rand(100, 3)
        y_train = np.random.rand(100, 1)
        model.fit(X_train, y_train)
        path = os.path.join(self.get_temp_dir(), "my_keras_core_model")
        tf.saved_model.save(model, path)
        restored_model = tf.saved_model.load(path)
        self.assertAllClose(
            model(X_train),
            restored_model.signatures["serving_default"](
                tf.convert_to_tensor(X_train, dtype=tf.float32)
            )["output_0"],
            rtol=1e-4,
            atol=1e-4,
        )

    def test_functional(self):
        inputs = layers.Input(shape=(3,))
        x = layers.Dense(1, name="first_dense")(inputs)
        outputs = layers.Dense(1, name="second_dense")(x)
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer="adam",
            loss="mse",
        )
        X_train = np.random.rand(100, 3)
        y_train = np.random.rand(100, 1)
        model.fit(X_train, y_train)
        path = os.path.join(self.get_temp_dir(), "my_keras_core_model")
        tf.saved_model.save(model, path)
        restored_model = tf.saved_model.load(path)
        self.assertAllClose(
            model(X_train),
            restored_model.signatures["serving_default"](
                tf.convert_to_tensor(X_train, dtype=tf.float32)
            )["output_0"],
            rtol=1e-4,
            atol=1e-4,
        )

    def test_subclassed(self):
        model = CustomModelX()
        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=[metrics.Hinge(), "mse"],
        )
        X_train = np.random.rand(100, 3)
        y_train = np.random.rand(100, 1)
        model.fit(X_train, y_train)
        path = os.path.join(self.get_temp_dir(), "my_keras_core_model")
        tf.saved_model.save(model, path)
        restored_model = tf.saved_model.load(path)
        self.assertAllClose(
            model(X_train),
            restored_model.signatures["serving_default"](
                tf.convert_to_tensor(X_train, dtype=tf.float32)
            )["output_0"],
            rtol=1e-4,
            atol=1e-4,
        )
