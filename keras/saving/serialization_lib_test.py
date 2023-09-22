"""Tests for serialization_lib."""

import json

import numpy as np
import pytest

import keras
from keras import ops
from keras import testing
from keras.saving import serialization_lib


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
        self.activation = serialization_lib.deserialize_keras_object(activation)

    def call(self, x):
        return self.dense(x * self.factor)

    def get_config(self):
        return {
            "factor": self.factor,
            "dense": self.dense,
            "activation": self.activation,
        }


class WrapperLayer(keras.layers.Wrapper):
    def call(self, x):
        return self.layer(x)


class SerializationLibTest(testing.TestCase):
    def roundtrip(self, obj, custom_objects=None, safe_mode=True):
        serialized = serialization_lib.serialize_keras_object(obj)
        json_data = json.dumps(serialized)
        json_data = json.loads(json_data)
        deserialized = serialization_lib.deserialize_keras_object(
            json_data, custom_objects=custom_objects, safe_mode=safe_mode
        )
        reserialized = serialization_lib.serialize_keras_object(deserialized)
        return serialized, deserialized, reserialized

    def test_simple_objects(self):
        for obj in [
            "hello",
            b"hello",
            np.array([0, 1]),
            np.array([0.0, 1.0]),
            np.float32(1.0),
            ["hello", 0, "world", 1.0, True],
            {"1": "hello", "2": 0, "3": True},
            {"1": "hello", "2": [True, False]},
            slice(None, 20, 1),
            slice(None, np.array([0, 1]), 1),
        ]:
            serialized, _, reserialized = self.roundtrip(obj)
            self.assertEqual(serialized, reserialized)

    def test_builtin_layers(self):
        layer = keras.layers.Dense(
            3,
            name="foo",
            trainable=False,
            dtype="float16",
        )
        serialized, restored, reserialized = self.roundtrip(layer)
        self.assertEqual(serialized, reserialized)
        self.assertEqual(layer.name, restored.name)
        self.assertEqual(layer.trainable, restored.trainable)
        self.assertEqual(layer.compute_dtype, restored.compute_dtype)

    def test_tensors_and_shapes(self):
        x = ops.random.normal((2, 2), dtype="float64")
        obj = {"x": x}
        _, new_obj, _ = self.roundtrip(obj)
        self.assertAllClose(x, new_obj["x"], atol=1e-5)

        obj = {"x.shape": x.shape}
        _, new_obj, _ = self.roundtrip(obj)
        self.assertEqual(tuple(x.shape), tuple(new_obj["x.shape"]))

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
        x = ops.random.normal((2, 2))
        y1 = dense(x)
        _ = new_dense(x)
        new_dense.set_weights(dense.get_weights())
        y2 = new_dense(x)
        self.assertAllClose(y1, y2, atol=1e-5)

    def test_custom_layer(self):
        layer = CustomLayer(factor=2)
        x = ops.random.normal((2, 2))
        y1 = layer(x)
        _, new_layer, _ = self.roundtrip(
            layer, custom_objects={"CustomLayer": CustomLayer}
        )
        y2 = new_layer(x)
        self.assertAllClose(y1, y2, atol=1e-5)

        layer = NestedCustomLayer(factor=2)
        x = ops.random.normal((2, 2))
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

    # TODO
    # def test_lambda_layer(self):
    #     lmbda = keras.layers.Lambda(lambda x: x**2)
    #     with self.assertRaisesRegex(ValueError, "arbitrary code execution"):
    #         self.roundtrip(lmbda, safe_mode=True)

    #     _, new_lmbda, _ = self.roundtrip(lmbda, safe_mode=False)
    #     x = ops.random.normal((2, 2))
    #     y1 = lmbda(x)
    #     y2 = new_lmbda(x)
    #     self.assertAllClose(y1, y2, atol=1e-5)

    # def test_safe_mode_scope(self):
    #     lmbda = keras.layers.Lambda(lambda x: x**2)
    #     with serialization_lib.SafeModeScope(safe_mode=True):
    #         with self.assertRaisesRegex(
    #             ValueError, "arbitrary code execution"
    #         ):
    #             self.roundtrip(lmbda)
    #     with serialization_lib.SafeModeScope(safe_mode=False):
    #         _, new_lmbda, _ = self.roundtrip(lmbda)
    #     x = ops.random.normal((2, 2))
    #     y1 = lmbda(x)
    #     y2 = new_lmbda(x)
    #     self.assertAllClose(y1, y2, atol=1e-5)

    @pytest.mark.requires_trainable_backend
    def test_dict_inputs_outputs(self):
        input_foo = keras.Input((2,), name="foo")
        input_bar = keras.Input((2,), name="bar")
        dense = keras.layers.Dense(1)
        output_foo = dense(input_foo)
        output_bar = dense(input_bar)
        model = keras.Model(
            {"foo": input_foo, "bar": input_bar},
            {"foo": output_foo, "bar": output_bar},
        )
        _, new_model, _ = self.roundtrip(model)
        original_output = model(
            {"foo": np.zeros((2, 2)), "bar": np.zeros((2, 2))}
        )
        restored_output = model(
            {"foo": np.zeros((2, 2)), "bar": np.zeros((2, 2))}
        )
        self.assertAllClose(original_output["foo"], restored_output["foo"])
        self.assertAllClose(original_output["bar"], restored_output["bar"])

    @pytest.mark.requires_trainable_backend
    def test_shared_inner_layer(self):
        with serialization_lib.ObjectSharingScope():
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

    @pytest.mark.requires_trainable_backend
    def test_functional_subclass(self):
        class PlainFunctionalSubclass(keras.Model):
            pass

        inputs = keras.Input((2,), batch_size=3)
        outputs = keras.layers.Dense(1)(inputs)
        model = PlainFunctionalSubclass(inputs, outputs)
        x = ops.random.normal((2, 2))
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
            def __init__(self, num_units=2):
                inputs = keras.Input((2,), batch_size=3)
                outputs = keras.layers.Dense(num_units)(inputs)
                super().__init__(inputs, outputs)
                self.num_units = num_units

            def get_config(self):
                return {"num_units": self.num_units}

        model = FunctionalSubclassWCustomInit(num_units=3)
        x = ops.random.normal((2, 2))
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


@keras.saving.register_keras_serializable()
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
        _, input_units = input_shape
        self._kernel = self.add_weight(
            name="kernel",
            shape=[input_units, self._units],
            dtype="float32",
            regularizer=self._kernel_regularizer,
            initializer=self._kernel_initializer,
        )

    def call(self, inputs):
        return ops.matmul(inputs, self._kernel)


@keras.saving.register_keras_serializable()
class MyWrapper(keras.layers.Layer):
    def __init__(self, wrapped, **kwargs):
        super().__init__(**kwargs)
        self._wrapped = wrapped

    def get_config(self):
        return dict(wrapped=self._wrapped, **super().get_config())

    @classmethod
    def from_config(cls, config):
        config["wrapped"] = keras.saving.deserialize_keras_object(
            config["wrapped"]
        )
        return cls(**config)

    def call(self, inputs):
        return self._wrapped(inputs)
