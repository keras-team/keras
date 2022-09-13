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
from keras.saving.experimental import serialization_lib
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


@test_utils.run_v2_only
class SerializationLibTest(tf.test.TestCase, parameterized.TestCase):
    def roundtrip(self, obj, custom_objects=None):
        serialized = serialization_lib.serialize_keras_object(obj)
        json_data = json.dumps(serialized)
        json_data = json.loads(json_data)
        deserialized = serialization_lib.deserialize_keras_object(
            json_data, custom_objects=custom_objects
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
        serialized, new_dense, reserialized = self.roundtrip(
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
        serialized, new_layer, reserialized = self.roundtrip(
            layer, custom_objects={"CustomLayer": CustomLayer}
        )
        y2 = new_layer(x)
        self.assertAllClose(y1, y2, atol=1e-5)

        layer = NestedCustomLayer(factor=2)
        x = tf.random.normal((2, 2))
        y1 = layer(x)
        serialized, new_layer, reserialized = self.roundtrip(
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


if __name__ == "__main__":
    tf.test.main()
