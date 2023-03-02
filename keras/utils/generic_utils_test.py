# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Keras generic Python utils."""


import os
import sys
from functools import partial

import numpy as np
import tensorflow.compat.v2 as tf

import keras
from keras.saving import serialization_lib
from keras.saving.legacy import serialization
from keras.utils import generic_utils
from keras.utils import io_utils


class SnakeCaseTest(tf.test.TestCase):
    def test_snake_case(self):
        self.assertEqual(generic_utils.to_snake_case("SomeClass"), "some_class")
        self.assertEqual(generic_utils.to_snake_case("Conv2D"), "conv2d")
        self.assertEqual(
            generic_utils.to_snake_case("ConvLSTM2D"), "conv_lstm2d"
        )


class HasArgTest(tf.test.TestCase):
    def test_has_arg(self):
        def f_x(x):
            return x

        def f_x_args(x, *args):
            _ = args
            return x

        def f_x_kwargs(x, **kwargs):
            _ = kwargs
            return x

        def f(a, b, c):
            return a + b + c

        partial_f = partial(f, b=1)

        self.assertTrue(
            keras.utils.generic_utils.has_arg(f_x, "x", accept_all=False)
        )
        self.assertFalse(
            keras.utils.generic_utils.has_arg(f_x, "y", accept_all=False)
        )
        self.assertTrue(
            keras.utils.generic_utils.has_arg(f_x_args, "x", accept_all=False)
        )
        self.assertFalse(
            keras.utils.generic_utils.has_arg(f_x_args, "y", accept_all=False)
        )
        self.assertTrue(
            keras.utils.generic_utils.has_arg(f_x_kwargs, "x", accept_all=False)
        )
        self.assertFalse(
            keras.utils.generic_utils.has_arg(f_x_kwargs, "y", accept_all=False)
        )
        self.assertTrue(
            keras.utils.generic_utils.has_arg(f_x_kwargs, "y", accept_all=True)
        )
        self.assertTrue(
            keras.utils.generic_utils.has_arg(partial_f, "c", accept_all=True)
        )


class SerializeKerasObjectTest(tf.test.TestCase):
    def test_serialize_none(self):
        serialized = serialization_lib.serialize_keras_object(None)
        self.assertEqual(serialized, None)
        deserialized = serialization_lib.deserialize_keras_object(serialized)
        self.assertEqual(deserialized, None)

    def test_serializable_object(self):
        class SerializableInt(int):
            """A serializable object to pass out of a test layer's config."""

            def __new__(cls, value):
                return int.__new__(cls, value)

            def get_config(self):
                return {"value": int(self)}

            @classmethod
            def from_config(cls, config):
                return cls(**config)

        layer = keras.layers.Dense(
            SerializableInt(3),
            activation="relu",
            kernel_initializer="ones",
            bias_regularizer="l2",
        )
        config = keras.layers.serialize(layer)
        new_layer = keras.layers.deserialize(
            config, custom_objects={"SerializableInt": SerializableInt}
        )
        self.assertEqual(new_layer.activation, keras.activations.relu)
        self.assertEqual(
            new_layer.bias_regularizer.__class__, keras.regularizers.L2
        )
        self.assertEqual(new_layer.units.__class__, SerializableInt)
        self.assertEqual(new_layer.units, 3)

    def test_nested_serializable_object(self):
        class SerializableInt(int):
            """A serializable object to pass out of a test layer's config."""

            def __new__(cls, value):
                return int.__new__(cls, value)

            def get_config(self):
                return {"value": int(self)}

            @classmethod
            def from_config(cls, config):
                return cls(**config)

        class SerializableNestedInt(int):
            """A serializable object containing another serializable object."""

            def __new__(cls, value, int_obj):
                obj = int.__new__(cls, value)
                obj.int_obj = int_obj
                return obj

            def get_config(self):
                return {"value": int(self), "int_obj": self.int_obj}

            @classmethod
            def from_config(cls, config):
                return cls(**config)

        nested_int = SerializableInt(4)
        layer = keras.layers.Dense(
            SerializableNestedInt(3, nested_int),
            name="SerializableNestedInt",
            activation="relu",
            kernel_initializer="ones",
            bias_regularizer="l2",
        )
        config = keras.layers.serialize(layer)
        new_layer = keras.layers.deserialize(
            config,
            custom_objects={
                "SerializableInt": SerializableInt,
                "SerializableNestedInt": SerializableNestedInt,
            },
        )
        # Make sure the string field doesn't get convert to custom object, even
        # they have same value.
        self.assertEqual(new_layer.name, "SerializableNestedInt")
        self.assertEqual(new_layer.activation, keras.activations.relu)
        self.assertEqual(
            new_layer.bias_regularizer.__class__, keras.regularizers.L2
        )
        self.assertEqual(new_layer.units.__class__, SerializableNestedInt)
        self.assertEqual(new_layer.units, 3)
        self.assertEqual(new_layer.units.int_obj.__class__, SerializableInt)
        self.assertEqual(new_layer.units.int_obj, 4)

    def test_nested_serializable_fn(self):
        def serializable_fn(x):
            """A serializable function to pass out of a test layer's config."""
            return x

        class SerializableNestedInt(int):
            """A serializable object containing a serializable function."""

            def __new__(cls, value, fn):
                obj = int.__new__(cls, value)
                obj.fn = fn
                return obj

            def get_config(self):
                return {"value": int(self), "fn": self.fn}

            @classmethod
            def from_config(cls, config):
                return cls(**config)

        layer = keras.layers.Dense(
            SerializableNestedInt(3, serializable_fn),
            activation="relu",
            kernel_initializer="ones",
            bias_regularizer="l2",
        )
        config = keras.layers.serialize(layer)
        new_layer = keras.layers.deserialize(
            config,
            custom_objects={
                "serializable_fn": serializable_fn,
                "SerializableNestedInt": SerializableNestedInt,
            },
        )
        self.assertEqual(new_layer.activation, keras.activations.relu)
        self.assertIsInstance(new_layer.bias_regularizer, keras.regularizers.L2)
        self.assertIsInstance(new_layer.units, SerializableNestedInt)
        self.assertEqual(new_layer.units, 3)
        self.assertIs(new_layer.units.fn, serializable_fn)

    def test_serialize_type_object_initializer(self):
        layer = keras.layers.Dense(
            1,
            kernel_initializer=keras.initializers.ones,
            bias_initializer=keras.initializers.zeros,
        )
        config = keras.layers.serialize(layer)
        self.assertEqual(
            config["config"]["bias_initializer"]["class_name"], "Zeros"
        )
        self.assertEqual(
            config["config"]["kernel_initializer"]["class_name"], "Ones"
        )

    def test_serializable_with_old_config(self):
        # model config generated by tf-1.2.1
        old_model_config = {
            "class_name": "Sequential",
            "config": [
                {
                    "class_name": "Dense",
                    "config": {
                        "name": "dense_1",
                        "trainable": True,
                        "batch_input_shape": [None, 784],
                        "dtype": "float32",
                        "units": 32,
                        "activation": "linear",
                        "use_bias": True,
                        "kernel_initializer": {
                            "class_name": "Ones",
                            "config": {"dtype": "float32"},
                        },
                        "bias_initializer": {
                            "class_name": "Zeros",
                            "config": {"dtype": "float32"},
                        },
                        "kernel_regularizer": None,
                        "bias_regularizer": None,
                        "activity_regularizer": None,
                        "kernel_constraint": None,
                        "bias_constraint": None,
                    },
                }
            ],
        }
        old_model = serialization_lib.deserialize_keras_object(
            old_model_config, module_objects={"Sequential": keras.Sequential}
        )
        new_model = keras.Sequential(
            [
                keras.layers.Dense(
                    32, input_dim=784, kernel_initializer="Ones"
                ),
            ]
        )
        input_data = np.random.normal(2, 1, (5, 784))
        output = old_model.predict(input_data)
        expected_output = new_model.predict(input_data)
        self.assertAllEqual(output, expected_output)

    def test_deserialize_unknown_object(self):
        class CustomLayer(keras.layers.Layer):
            pass

        layer = CustomLayer()
        config = serialization_lib.serialize_keras_object(layer)
        if tf.__internal__.tf2.enabled():
            with self.assertRaisesRegex(
                TypeError,
                "Could not locate class 'CustomLayer'. Make sure custom classes",  # noqa: E501
            ):
                serialization_lib.deserialize_keras_object(config)
        else:
            with self.assertRaisesRegex(
                ValueError, "using a `keras.utils.custom_object_scope`"
            ):
                serialization.deserialize_keras_object(config)
        restored = serialization_lib.deserialize_keras_object(
            config, custom_objects={"CustomLayer": CustomLayer}
        )
        self.assertIsInstance(restored, CustomLayer)


class SliceArraysTest(tf.test.TestCase):
    def test_slice_arrays(self):
        input_a = list([1, 2, 3])
        self.assertEqual(
            keras.utils.generic_utils.slice_arrays(input_a, start=0),
            [None, None, None],
        )
        self.assertEqual(
            keras.utils.generic_utils.slice_arrays(input_a, stop=3),
            [None, None, None],
        )
        self.assertEqual(
            keras.utils.generic_utils.slice_arrays(input_a, start=0, stop=1),
            [None, None, None],
        )


# object() alone isn't compatible with WeakKeyDictionary, which we use to
# track shared configs.
class MaybeSharedObject:
    pass


class SharedObjectScopeTest(tf.test.TestCase):
    def test_shared_object_saving_scope_single_object_doesnt_export_id(self):
        with serialization.SharedObjectSavingScope() as scope:
            single_object = MaybeSharedObject()
            self.assertIsNone(scope.get_config(single_object))
            single_object_config = scope.create_config({}, single_object)
            self.assertIsNotNone(single_object_config)
            self.assertNotIn(
                serialization.SHARED_OBJECT_KEY, single_object_config
            )

    def test_shared_object_saving_scope_shared_object_exports_id(self):
        with serialization.SharedObjectSavingScope() as scope:
            shared_object = MaybeSharedObject()
            self.assertIsNone(scope.get_config(shared_object))
            scope.create_config({}, shared_object)
            first_object_config = scope.get_config(shared_object)
            second_object_config = scope.get_config(shared_object)
            self.assertIn(serialization.SHARED_OBJECT_KEY, first_object_config)
            self.assertIn(serialization.SHARED_OBJECT_KEY, second_object_config)
            self.assertIs(first_object_config, second_object_config)

    def test_shared_object_loading_scope_noop(self):
        # Test that, without a context manager scope, adding configs will do
        # nothing.
        obj_id = 1
        obj = MaybeSharedObject()
        serialization._shared_object_loading_scope().set(obj_id, obj)
        self.assertIsNone(
            serialization._shared_object_loading_scope().get(obj_id)
        )

    def test_shared_object_loading_scope_returns_shared_obj(self):
        obj_id = 1
        obj = MaybeSharedObject()
        with serialization.SharedObjectLoadingScope() as scope:
            scope.set(obj_id, obj)
            self.assertIs(scope.get(obj_id), obj)

    def test_nested_shared_object_saving_scopes(self):
        my_obj = MaybeSharedObject()
        with serialization.SharedObjectSavingScope() as scope_1:
            scope_1.create_config({}, my_obj)
            with serialization.SharedObjectSavingScope() as scope_2:
                # Nesting saving scopes should return the original scope and
                # should not clear any objects we're tracking.
                self.assertIs(scope_1, scope_2)
                self.assertIsNotNone(scope_2.get_config(my_obj))
            self.assertIsNotNone(scope_1.get_config(my_obj))
        self.assertIsNone(serialization._shared_object_saving_scope())

    def test_custom_object_scope_correct_class(self):
        train_step_message = "This is my training step"
        temp_dir = os.path.join(self.get_temp_dir(), "my_model")

        class CustomModelX(keras.Model):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.dense1 = keras.layers.Dense(1)

            def call(self, inputs):
                return self.dense1(inputs)

            def train_step(self, data):
                tf.print(train_step_message)
                x, y = data
                with tf.GradientTape() as tape:
                    y_pred = self(x)
                    loss = self.compiled_loss(y, y_pred)

                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(gradients, self.trainable_variables)
                )
                return {}

            def func_that_returns_one(self):
                return 1

        subclassed_model = CustomModelX()
        subclassed_model.compile(optimizer="adam", loss="mse")

        x = np.random.random((100, 32))
        y = np.random.random((100, 1))
        subclassed_model.fit(x, y, epochs=1)
        subclassed_model.save(temp_dir, save_format="tf")

        with keras.utils.custom_object_scope({"CustomModelX": CustomModelX}):
            loaded_model = keras.models.load_model(temp_dir)

        io_utils.enable_interactive_logging()
        # `tf.print` writes to stderr.
        with self.captureWritesToStream(sys.stderr) as printed:
            loaded_model.fit(x, y, epochs=1)
            if tf.__internal__.tf2.enabled():
                # `tf.print` message is only available in stderr in TF2. Check
                # that custom `train_step` is used.
                self.assertRegex(printed.contents(), train_step_message)

        # Check that the custom class does get used.
        self.assertIsInstance(loaded_model, CustomModelX)
        # Check that the custom method is available.
        self.assertEqual(loaded_model.func_that_returns_one(), 1)


if __name__ == "__main__":
    tf.test.main()
