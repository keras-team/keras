"""Tests for SavedModel functionality under tf implementation."""

import os

import numpy as np
import pytest
import tensorflow as tf

from keras import backend
from keras import layers
from keras import metrics
from keras import models
from keras import testing
from keras.saving import object_registration


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
        path = os.path.join(self.get_temp_dir(), "my_keras_model")
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
        path = os.path.join(self.get_temp_dir(), "my_keras_model")
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
        path = os.path.join(self.get_temp_dir(), "my_keras_model")
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

    def test_custom_model_and_layer(self):
        @object_registration.register_keras_serializable(package="my_package")
        class CustomLayer(layers.Layer):
            def __call__(self, inputs):
                return inputs

        @object_registration.register_keras_serializable(package="my_package")
        class Model(models.Model):
            def __init__(self):
                super().__init__()
                self.layer = CustomLayer()

            @tf.function(input_signature=[tf.TensorSpec([None, 1])])
            def call(self, inputs):
                return self.layer(inputs)

        model = Model()
        inp = np.array([[1.0]])
        result = model(inp)
        path = os.path.join(self.get_temp_dir(), "my_keras_model")
        tf.saved_model.save(model, path)
        restored_model = tf.saved_model.load(path)
        self.assertAllClose(
            result,
            restored_model.call(inp),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_multi_input_model(self):
        input_1 = layers.Input(shape=(3,))
        input_2 = layers.Input(shape=(5,))
        model = models.Model([input_1, input_2], [input_1, input_2])
        path = os.path.join(self.get_temp_dir(), "my_keras_model")

        tf.saved_model.save(model, path)
        restored_model = tf.saved_model.load(path)
        input_arr_1 = np.random.random((1, 3)).astype("float32")
        input_arr_2 = np.random.random((1, 5)).astype("float32")

        outputs = restored_model.signatures["serving_default"](
            inputs=tf.convert_to_tensor(input_arr_1, dtype=tf.float32),
            inputs_1=tf.convert_to_tensor(input_arr_2, dtype=tf.float32),
        )

        self.assertAllClose(
            input_arr_1, outputs["output_0"], rtol=1e-4, atol=1e-4
        )
        self.assertAllClose(
            input_arr_2, outputs["output_1"], rtol=1e-4, atol=1e-4
        )

    def test_multi_input_custom_model_and_layer(self):
        @object_registration.register_keras_serializable(package="my_package")
        class CustomLayer(layers.Layer):
            def __call__(self, *input_list):
                self.add_loss(input_list[-2] * 2)
                return sum(input_list)

        @object_registration.register_keras_serializable(package="my_package")
        class CustomModel(models.Model):
            def build(self, input_shape):
                super().build(input_shape)
                self.layer = CustomLayer()

            @tf.function
            def call(self, *inputs):
                inputs = list(inputs)
                return self.layer(*inputs)

        model = CustomModel()
        inp = [
            tf.constant(i, shape=[1, 1], dtype=tf.float32) for i in range(1, 4)
        ]
        expected = model(*inp)
        path = os.path.join(self.get_temp_dir(), "my_keras_model")
        tf.saved_model.save(model, path)
        restored_model = tf.saved_model.load(path)
        output = restored_model.call(*inp)
        self.assertAllClose(expected, output, rtol=1e-4, atol=1e-4)
