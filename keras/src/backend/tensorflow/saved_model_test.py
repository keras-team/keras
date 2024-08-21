"""Tests for SavedModel functionality under tf implementation."""

import os

import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import metrics
from keras.src import models
from keras.src import ops
from keras.src import optimizers
from keras.src import testing
from keras.src.saving import object_registration
from keras.src.testing.test_utils import named_product


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


@object_registration.register_keras_serializable(package="my_package")
class CustomSignatureModel(models.Model):
    def __init__(self):
        super(CustomSignatureModel, self).__init__()
        self.v = tf.Variable(1.0)

    @tf.function
    def __call__(self, x):
        return x * self.v

    @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
    def mutate(self, new_v):
        self.v.assign(new_v)


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="The SavedModel test can only run with TF backend.",
)
class SavedModelTest(testing.TestCase, parameterized.TestCase):
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

    @parameterized.named_parameters(
        named_product(struct_type=["tuple", "array", "dict"])
    )
    def test_model_with_input_structure(self, struct_type):

        class TupleModel(models.Model):

            def call(self, inputs):
                x, y = inputs
                return x + ops.mean(y, axis=1)

        class ArrayModel(models.Model):

            def call(self, inputs):
                x = inputs[0]
                y = inputs[1]
                return x + ops.mean(y, axis=1)

        class DictModel(models.Model):

            def call(self, inputs):
                x = inputs["x"]
                y = inputs["y"]
                return x + ops.mean(y, axis=1)

        input_x = tf.constant([1.0])
        input_y = tf.constant([[1.0, 0.0, 2.0]])
        if struct_type == "tuple":
            model = TupleModel()
            inputs = (input_x, input_y)
        elif struct_type == "array":
            model = ArrayModel()
            inputs = [input_x, input_y]
        elif struct_type == "dict":
            model = DictModel()
            inputs = {"x": input_x, "y": input_y}

        result = model(inputs)
        path = os.path.join(self.get_temp_dir(), "my_keras_model")
        tf.saved_model.save(model, path)
        restored_model = tf.saved_model.load(path)
        outputs = restored_model.signatures["serving_default"](
            inputs=input_x, inputs_1=input_y
        )
        self.assertAllClose(result, outputs["output_0"], rtol=1e-4, atol=1e-4)

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
            def build(self, *input_shape):
                self.built = True

            def call(self, *input_list):
                self.add_loss(input_list[-2] * 2)
                return sum(input_list)

        @object_registration.register_keras_serializable(package="my_package")
        class CustomModel(models.Model):
            def build(self, *input_shape):
                self.layer = CustomLayer()
                self.layer.build(*input_shape)
                self.built = True

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

    def test_list_trackable_children_tracking(self):
        @object_registration.register_keras_serializable(package="my_package")
        class CustomLayerList(layers.Layer):
            def __init__(self):
                super().__init__()
                self.sublayers = [
                    layers.Dense(2),
                    layers.Dense(2),
                ]

            def call(self, inputs):
                x = inputs
                for sublayer in self.sublayers:
                    x = sublayer(x)
                return x

        inputs = layers.Input(shape=(1,))
        outputs = CustomLayerList()(inputs)
        model = models.Model(inputs, outputs)

        inp = np.array([[1.0]])
        expected = model(inp)
        path = os.path.join(self.get_temp_dir(), "my_keras_model")
        tf.saved_model.save(model, path)
        restored_model = tf.saved_model.load(path)
        self.assertAllClose(
            expected,
            restored_model.signatures["serving_default"](
                tf.convert_to_tensor(inp, dtype=tf.float32)
            )["output_0"],
            rtol=1e-4,
            atol=1e-4,
        )

    def test_dict_trackable_children_tracking(self):
        @object_registration.register_keras_serializable(package="my_package")
        class CustomLayerDict(layers.Layer):
            def __init__(self):
                super().__init__()
                self.sublayers = {
                    "first_layer": layers.Dense(2),
                    "second_layer": layers.Dense(2),
                }

            def call(self, inputs):
                x = inputs
                for key, sublayer in self.sublayers.items():
                    x = sublayer(x)
                return x

        inputs = layers.Input(shape=(1,))
        outputs = CustomLayerDict()(inputs)
        model = models.Model(inputs, outputs)

        inp = np.array([[1.0]])
        expected = model(inp)
        path = os.path.join(self.get_temp_dir(), "my_keras_model")
        tf.saved_model.save(model, path)
        restored_model = tf.saved_model.load(path)
        self.assertAllClose(
            expected,
            restored_model.signatures["serving_default"](
                tf.convert_to_tensor(inp, dtype=tf.float32)
            )["output_0"],
            rtol=1e-4,
            atol=1e-4,
        )

    def test_fixed_signature_string_dtype(self):
        @object_registration.register_keras_serializable(package="my_package")
        class Adder(models.Model):
            @tf.function(
                input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)]
            )
            def concat(self, x):
                return x + x

        model = Adder()
        path = os.path.join(self.get_temp_dir(), "my_keras_model")
        tf.saved_model.save(model, path)
        restored_model = tf.saved_model.load(path)
        self.assertEqual(model.concat("hello"), restored_model.concat("hello"))

    def test_non_fixed_signature_string_dtype(self):
        @object_registration.register_keras_serializable(package="my_package")
        class Adder(models.Model):
            @tf.function
            def concat(self, x):
                return x + x

        model = Adder()

        no_fn_path = os.path.join(self.get_temp_dir(), "my_keras_model_no_fn")
        tf.saved_model.save(model, no_fn_path)
        restored_model = tf.saved_model.load(no_fn_path)
        with self.assertRaisesRegex(ValueError, "zero restored functions"):
            _ = restored_model.concat("hello")

        path = os.path.join(self.get_temp_dir(), "my_keras_model")
        tf.saved_model.save(
            model,
            path,
            signatures=model.concat.get_concrete_function(
                tf.TensorSpec(shape=[], dtype=tf.string, name="string_input")
            ),
        )
        restored_model = tf.saved_model.load(path)
        self.assertEqual(model.concat("hello"), restored_model.concat("hello"))

    def test_fine_tuning(self):
        model = CustomSignatureModel()
        model_no_signatures_path = os.path.join(
            self.get_temp_dir(), "model_no_signatures"
        )
        _ = model(tf.constant(0.0))

        tf.saved_model.save(model, model_no_signatures_path)
        restored_model = tf.saved_model.load(model_no_signatures_path)

        self.assertLen(list(restored_model.signatures.keys()), 0)
        self.assertEqual(restored_model(tf.constant(3.0)).numpy(), 3)
        restored_model.mutate(tf.constant(2.0))
        self.assertEqual(restored_model(tf.constant(3.0)).numpy(), 6)
        optimizer = optimizers.SGD(0.05)

        def train_step():
            with tf.GradientTape() as tape:
                loss = (10.0 - restored_model(tf.constant(2.0))) ** 2
            variables = tape.watched_variables()
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))
            return loss

        for _ in range(10):
            # "v" approaches 5, "loss" approaches 0
            loss = train_step()

        self.assertAllClose(loss, 0.0, rtol=1e-2, atol=1e-2)
        self.assertAllClose(restored_model.v.numpy(), 5.0, rtol=1e-2, atol=1e-2)

    def test_signatures_path(self):
        model = CustomSignatureModel()
        model_with_signature_path = os.path.join(
            self.get_temp_dir(), "model_with_signature"
        )
        call = model.__call__.get_concrete_function(
            tf.TensorSpec(None, tf.float32)
        )

        tf.saved_model.save(model, model_with_signature_path, signatures=call)
        restored_model = tf.saved_model.load(model_with_signature_path)
        self.assertEqual(
            list(restored_model.signatures.keys()), ["serving_default"]
        )

    def test_multiple_signatures_dict_path(self):
        model = CustomSignatureModel()
        model_multiple_signatures_path = os.path.join(
            self.get_temp_dir(), "model_with_multiple_signatures"
        )
        call = model.__call__.get_concrete_function(
            tf.TensorSpec(None, tf.float32)
        )
        signatures = {
            "serving_default": call,
            "array_input": model.__call__.get_concrete_function(
                tf.TensorSpec([None], tf.float32)
            ),
        }

        tf.saved_model.save(
            model, model_multiple_signatures_path, signatures=signatures
        )
        restored_model = tf.saved_model.load(model_multiple_signatures_path)
        self.assertEqual(
            list(restored_model.signatures.keys()),
            ["serving_default", "array_input"],
        )
