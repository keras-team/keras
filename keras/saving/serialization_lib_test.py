# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for serialization_lib."""

import json

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.saving import serialization_lib
from keras.saving.legacy import serialization as legacy_serialization
from keras.testing_infra import test_utils


def custom_fn(x):
    return x**2


class CustomLayer(keras.layers.Layer):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def call(self, x):
        return x * self.factor

    def get_config(self):
        return {"factor": self.factor}


class NestedCustomLayer(keras.layers.Layer):
    def __init__(self, factor, dense=None, activation=None):
        super().__init__()
        self.factor = factor

        if dense is None:
            self.dense = keras.layers.Dense(1, activation=custom_fn)
        else:
            self.dense = serialization_lib.deserialize_keras_object(dense)
        if activation is None:
            self.activation = keras.layers.Activation("relu")
        else:
            self.activation = serialization_lib.deserialize_keras_object(
                activation
            )

    def call(self, x):
        return self.dense(x * self.factor)

    def get_config(self):
        return {
            "factor": self.factor,
            "dense": self.dense,
            "activation": self.activation,
        }


class WrapperLayer(keras.layers.Layer):
    def __init__(self, layer, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer

    def call(self, x):
        return self.layer(x)

    def get_config(self):
        config = super().get_config()
        return {"layer": self.layer, **config}


@test_utils.run_v2_only
class SerializationLibTest(tf.test.TestCase, parameterized.TestCase):
    def roundtrip(self, obj, custom_objects=None, safe_mode=True):
        serialized = serialization_lib.serialize_keras_object(obj)
        json_data = json.dumps(serialized)
        json_data = json.loads(json_data)
        deserialized = serialization_lib.deserialize_keras_object(
            json_data, custom_objects=custom_objects, safe_mode=safe_mode
        )
        reserialized = serialization_lib.serialize_keras_object(deserialized)
        return serialized, deserialized, reserialized

    @parameterized.named_parameters(
        ("str", "hello"),
        ("bytes", b"hello"),
        ("nparray_int", np.array([0, 1])),
        ("nparray_float", np.array([0.0, 1.0])),
        ("nparray_item", np.float32(1.0)),
        ("plain_types_list", ["hello", 0, "world", 1.0, True]),
        ("plain_types_dict", {"1": "hello", "2": 0, "3": True}),
        ("plain_types_nested_dict", {"1": "hello", "2": [True, False]}),
    )
    def test_simple_objects(self, obj):
        serialized, _, reserialized = self.roundtrip(obj)
        self.assertEqual(serialized, reserialized)

    def test_builtin_layers(self):
        serialized, _, reserialized = self.roundtrip(keras.layers.Dense(3))
        self.assertEqual(serialized, reserialized)

    def test_tensors_and_tensorshape(self):
        x = tf.random.normal((2, 2), dtype="float64")
        obj = {"x": x}
        _, new_obj, _ = self.roundtrip(obj)
        self.assertAllClose(x, new_obj["x"], atol=1e-5)

        obj = {"x.shape": x.shape}
        _, new_obj, _ = self.roundtrip(obj)
        self.assertListEqual(x.shape.as_list(), new_obj["x.shape"])

    def test_custom_fn(self):
        obj = {"activation": custom_fn}
        serialized, _, reserialized = self.roundtrip(
            obj, custom_objects={"custom_fn": custom_fn}
        )
        self.assertEqual(serialized, reserialized)

        # Test inside layer
        dense = keras.layers.Dense(1, activation=custom_fn)
        dense.build((None, 2))
        _, new_dense, _ = self.roundtrip(
            dense, custom_objects={"custom_fn": custom_fn}
        )
        x = tf.random.normal((2, 2))
        y1 = dense(x)
        _ = new_dense(x)
        new_dense.set_weights(dense.get_weights())
        y2 = new_dense(x)
        self.assertAllClose(y1, y2, atol=1e-5)

    def test_custom_layer(self):
        layer = CustomLayer(factor=2)
        x = tf.random.normal((2, 2))
        y1 = layer(x)
        _, new_layer, _ = self.roundtrip(
            layer, custom_objects={"CustomLayer": CustomLayer}
        )
        y2 = new_layer(x)
        self.assertAllClose(y1, y2, atol=1e-5)

        layer = NestedCustomLayer(factor=2)
        x = tf.random.normal((2, 2))
        y1 = layer(x)
        _, new_layer, _ = self.roundtrip(
            layer,
            custom_objects={
                "NestedCustomLayer": NestedCustomLayer,
                "custom_fn": custom_fn,
            },
        )
        _ = new_layer(x)
        new_layer.set_weights(layer.get_weights())
        y2 = new_layer(x)
        self.assertAllClose(y1, y2, atol=1e-5)

    def test_lambda_fn(self):
        obj = {"activation": lambda x: x**2}
        with self.assertRaisesRegex(ValueError, "arbitrary code execution"):
            self.roundtrip(obj, safe_mode=True)

        _, new_obj, _ = self.roundtrip(obj, safe_mode=False)
        self.assertEqual(obj["activation"](3), new_obj["activation"](3))

    def test_lambda_layer(self):
        lmbda = keras.layers.Lambda(lambda x: x**2)
        with self.assertRaisesRegex(ValueError, "arbitrary code execution"):
            self.roundtrip(lmbda, safe_mode=True)

        _, new_lmbda, _ = self.roundtrip(lmbda, safe_mode=False)
        x = tf.random.normal((2, 2))
        y1 = lmbda(x)
        y2 = new_lmbda(x)
        self.assertAllClose(y1, y2, atol=1e-5)

    def test_safe_mode_scope(self):
        lmbda = keras.layers.Lambda(lambda x: x**2)
        with serialization_lib.SafeModeScope(safe_mode=True):
            with self.assertRaisesRegex(ValueError, "arbitrary code execution"):
                self.roundtrip(lmbda)
        with serialization_lib.SafeModeScope(safe_mode=False):
            _, new_lmbda, _ = self.roundtrip(lmbda)
        x = tf.random.normal((2, 2))
        y1 = lmbda(x)
        y2 = new_lmbda(x)
        self.assertAllClose(y1, y2, atol=1e-5)

    def test_tensorspec(self):
        inputs = keras.Input(type_spec=tf.TensorSpec((2, 2), tf.float32))
        outputs = keras.layers.Dense(1)(inputs)
        model = keras.Model(inputs, outputs)
        _, new_model, _ = self.roundtrip(model)
        x = tf.random.normal((2, 2))
        y1 = model(x)
        new_model.set_weights(model.get_weights())
        y2 = new_model(x)
        self.assertAllClose(y1, y2, atol=1e-5)

    def shared_inner_layer(self):
        input_1 = keras.Input((2,))
        input_2 = keras.Input((2,))
        shared_layer = keras.layers.Dense(1)
        output_1 = shared_layer(input_1)
        wrapper_layer = WrapperLayer(shared_layer)
        output_2 = wrapper_layer(input_2)
        model = keras.Model([input_1, input_2], [output_1, output_2])
        _, new_model, _ = self.roundtrip(
            model, custom_objects={"WrapperLayer": WrapperLayer}
        )

        self.assertIs(model.layers[2], model.layers[3].layer)
        self.assertIs(new_model.layers[2], new_model.layers[3].layer)

    def test_functional_subclass(self):
        class PlainFunctionalSubclass(keras.Model):
            pass

        inputs = keras.Input((2,))
        outputs = keras.layers.Dense(1)(inputs)
        model = PlainFunctionalSubclass(inputs, outputs)
        x = tf.random.normal((2, 2))
        y1 = model(x)
        _, new_model, _ = self.roundtrip(
            model,
            custom_objects={"PlainFunctionalSubclass": PlainFunctionalSubclass},
        )
        new_model.set_weights(model.get_weights())
        y2 = new_model(x)
        self.assertAllClose(y1, y2, atol=1e-5)
        self.assertIsInstance(new_model, PlainFunctionalSubclass)

        class FunctionalSubclassWCustomInit(keras.Model):
            def __init__(self, num_units=1, **kwargs):
                inputs = keras.Input((2,))
                outputs = keras.layers.Dense(num_units)(inputs)
                super().__init__(inputs, outputs)

        model = FunctionalSubclassWCustomInit(num_units=2)
        x = tf.random.normal((2, 2))
        y1 = model(x)
        _, new_model, _ = self.roundtrip(
            model,
            custom_objects={
                "FunctionalSubclassWCustomInit": FunctionalSubclassWCustomInit
            },
        )
        new_model.set_weights(model.get_weights())
        y2 = new_model(x)
        self.assertAllClose(y1, y2, atol=1e-5)
        self.assertIsInstance(new_model, FunctionalSubclassWCustomInit)

    def test_shared_object(self):
        class MyLayer(keras.layers.Layer):
            def __init__(self, activation, **kwargs):
                super().__init__(**kwargs)
                if isinstance(activation, dict):
                    self.activation = (
                        serialization_lib.deserialize_keras_object(activation)
                    )
                else:
                    self.activation = activation

            def call(self, x):
                return self.activation(x)

            def get_config(self):
                config = super().get_config()
                config["activation"] = self.activation
                return config

        class SharedActivation:
            def __call__(self, x):
                return x**2

            def get_config(self):
                return {}

            @classmethod
            def from_config(cls, config):
                return cls()

        shared_act = SharedActivation()
        layer_1 = MyLayer(activation=shared_act)
        layer_2 = MyLayer(activation=shared_act)
        layers = [layer_1, layer_2]

        with serialization_lib.ObjectSharingScope():
            serialized, new_layers, reserialized = self.roundtrip(
                layers,
                custom_objects={
                    "MyLayer": MyLayer,
                    "SharedActivation": SharedActivation,
                },
            )
        self.assertIn("shared_object_id", serialized[0]["config"]["activation"])
        obj_id = serialized[0]["config"]["activation"]
        self.assertIn("shared_object_id", serialized[1]["config"]["activation"])
        self.assertEqual(obj_id, serialized[1]["config"]["activation"])
        self.assertIs(layers[0].activation, layers[1].activation)
        self.assertIs(new_layers[0].activation, new_layers[1].activation)

    def test_legacy_internal_object(self):
        from keras.layers.rnn.legacy_cells import (
            LSTMCell,  # pylint: disable=C6204
        )

        # tf.nn.rnn_cell.LSTMCell belongs to keras.__internal__.legacy namespace
        cell = LSTMCell(32)
        x = keras.Input((None, 5))
        layer = keras.layers.RNN(cell)
        y = layer(x)
        model = keras.models.Model(x, y)
        model.compile(optimizer="rmsprop", loss="mse")

        x_in = np.random.random((3, 5, 5))
        y_out_1 = model.predict(x_in)
        weights = model.get_weights()

        # serialize and deserialize
        config = serialization_lib.serialize_keras_object(layer)
        layer = serialization_lib.deserialize_keras_object(
            config,
            custom_objects={"LSTMCell": LSTMCell},
        )

        # Restore RNN cell into model with weights
        y = layer(x)
        model = keras.models.Model(x, y)
        model.set_weights(weights)
        y_out_2 = model.predict(x_in)

        self.assertAllClose(y_out_1, y_out_2, atol=1e-5)


@keras.utils.register_keras_serializable()
class MyDense(keras.layers.Layer):
    def __init__(
        self,
        units,
        *,
        kernel_regularizer=None,
        kernel_initializer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._units = units
        self._kernel_regularizer = kernel_regularizer
        self._kernel_initializer = kernel_initializer

    def get_config(self):
        return dict(
            units=self._units,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            **super().get_config()
        )

    def build(self, input_shape):
        unused_batch_size, input_units = input_shape.as_list()
        self._kernel = self.add_weight(
            "kernel",
            [input_units, self._units],
            dtype=tf.float32,
            regularizer=self._kernel_regularizer,
            initializer=self._kernel_initializer,
        )

    def call(self, inputs):
        return tf.matmul(inputs, self._kernel)


@keras.utils.register_keras_serializable()
class MyWrapper(keras.layers.Layer):
    def __init__(self, wrapped, **kwargs):
        super().__init__(**kwargs)
        self._wrapped = wrapped

    def get_config(self):
        return dict(wrapped=self._wrapped, **super().get_config())

    @classmethod
    def from_config(cls, config):
        config["wrapped"] = keras.utils.deserialize_keras_object(
            config["wrapped"]
        )
        return cls(**config)

    def call(self, inputs):
        return self._wrapped(inputs)


@test_utils.run_v2_only
class JsonSerializationTest(tf.test.TestCase, parameterized.TestCase):
    def test_serialize_deserialize_custom_layer_json(self):
        reg = keras.regularizers.L2(0.101)
        ini = keras.initializers.Constant(1.0)
        dense = MyDense(4, kernel_regularizer=reg, kernel_initializer=ini)
        inputs = keras.layers.Input(shape=[3])
        outputs = dense(inputs)
        model = keras.Model(inputs, outputs)

        model_json = model.to_json()
        model2 = keras.models.model_from_json(model_json)

        self.assertEqual(model_json, model2.to_json())

    def test_serialize_deserialize_custom_layer_with_wrapper_json(self):
        reg = keras.regularizers.L2(0.101)
        ini = keras.initializers.Constant(1.0)
        dense = MyDense(4, kernel_regularizer=reg, kernel_initializer=ini)
        wrapper = MyWrapper(dense)
        inputs = keras.layers.Input(shape=[3])
        outputs = wrapper(inputs)
        model = keras.Model(inputs, outputs)

        model_json = model.to_json()
        model2 = keras.models.model_from_json(model_json)

        self.assertEqual(model_json, model2.to_json())


@test_utils.run_v2_only
class BackwardsCompatibilityTest(tf.test.TestCase, parameterized.TestCase):
    def assert_old_format_can_be_deserialized(self, obj, custom_objects=None):
        old_config = legacy_serialization.serialize_keras_object(obj)
        revived = serialization_lib.deserialize_keras_object(
            old_config, custom_objects=custom_objects
        )
        new_config_1 = serialization_lib.serialize_keras_object(obj)
        new_config_2 = serialization_lib.serialize_keras_object(revived)
        self.assertEqual(new_config_1, new_config_2)

    def test_backwards_compatibility_with_old_serialized_format(self):
        optimizer = keras.optimizers.Adam(learning_rate=0.1)
        self.assert_old_format_can_be_deserialized(
            optimizer, custom_objects=vars(keras.optimizers)
        )
        activation = keras.activations.relu
        self.assert_old_format_can_be_deserialized(
            activation, custom_objects=vars(keras.activations)
        )
        initializer = keras.initializers.VarianceScaling(scale=2.0)
        self.assert_old_format_can_be_deserialized(
            initializer, custom_objects=vars(keras.initializers)
        )
        regularizer = keras.regularizers.L2(0.3)
        self.assert_old_format_can_be_deserialized(
            regularizer, custom_objects=vars(keras.regularizers)
        )
        constraint = keras.constraints.UnitNorm()
        self.assert_old_format_can_be_deserialized(
            constraint, custom_objects=vars(keras.constraints)
        )
        layer = keras.layers.Dense(2)
        self.assert_old_format_can_be_deserialized(
            layer, custom_objects=vars(keras.layers)
        )
        layer = keras.layers.MultiHeadAttention(2, 4)
        self.assert_old_format_can_be_deserialized(
            layer, custom_objects=vars(keras.layers)
        )

        # Custom objects
        layer = CustomLayer(2)
        self.assert_old_format_can_be_deserialized(
            layer, custom_objects={"CustomLayer": CustomLayer}
        )
        layer = keras.layers.Dense(1, activation=custom_fn)
        self.assert_old_format_can_be_deserialized(
            layer, custom_objects={**vars(keras.layers), "custom_fn": custom_fn}
        )


if __name__ == "__main__":
    tf.test.main()
