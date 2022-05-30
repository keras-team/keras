# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for hashed crossing layer."""

import os

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.layers.preprocessing import hashed_crossing
from keras.layers.preprocessing import preprocessing_test_utils
from keras.testing_infra import test_combinations


@test_combinations.run_all_keras_modes(always_skip_v1=True)
class HashedCrossingTest(test_combinations.TestCase):
    @parameterized.named_parameters(
        ("python_value", lambda x: x),
        ("dense", tf.constant),
    )
    def test_cross_scalars(self, data_fn):
        layer = hashed_crossing.HashedCrossing(num_bins=10)
        feat1 = data_fn("A")
        feat2 = data_fn(101)
        outputs = layer((feat1, feat2))
        self.assertAllClose(outputs, 1)
        self.assertAllEqual(outputs.shape.as_list(), [])

    @parameterized.named_parameters(
        ("tuple", tuple),
        ("list", list),
        ("numpy", np.array),
        ("array_like", preprocessing_test_utils.ArrayLike),
        ("dense", tf.constant),
    )
    def test_cross_batch_of_scalars_1d(self, data_fn):
        layer = hashed_crossing.HashedCrossing(num_bins=10)
        feat1 = data_fn(["A", "B", "A", "B", "A"])
        feat2 = data_fn([101, 101, 101, 102, 102])
        outputs = layer((feat1, feat2))
        self.assertAllClose(outputs, [1, 4, 1, 6, 3])
        self.assertAllEqual(outputs.shape.as_list(), [5])

    @parameterized.named_parameters(
        ("tuple", tuple),
        ("list", list),
        ("numpy", np.array),
        ("array_like", preprocessing_test_utils.ArrayLike),
        ("dense", tf.constant),
    )
    def test_cross_batch_of_scalars_2d(self, data_fn):
        layer = hashed_crossing.HashedCrossing(num_bins=10)
        feat1 = data_fn([["A"], ["B"], ["A"], ["B"], ["A"]])
        feat2 = data_fn([[101], [101], [101], [102], [102]])
        outputs = layer((feat1, feat2))
        self.assertAllClose(outputs, [[1], [4], [1], [6], [3]])
        self.assertAllEqual(outputs.shape.as_list(), [5, 1])

    @parameterized.named_parameters(
        ("sparse", True),
        ("dense", False),
    )
    def test_cross_one_hot_output(self, sparse):
        layer = hashed_crossing.HashedCrossing(
            num_bins=5, output_mode="one_hot", sparse=sparse
        )
        feat1 = tf.constant([["A"], ["B"], ["A"], ["B"], ["A"]])
        feat2 = tf.constant([[101], [101], [101], [102], [102]])
        outputs = layer((feat1, feat2))
        if sparse:
            outputs = tf.sparse.to_dense(outputs)
        self.assertAllClose(
            outputs,
            [
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0],
            ],
        )
        self.assertAllEqual(outputs.shape.as_list(), [5, 5])

    def test_cross_output_dtype(self):
        layer = hashed_crossing.HashedCrossing(num_bins=2)
        self.assertAllEqual(layer(([1], [1])).dtype, tf.int64)
        layer = hashed_crossing.HashedCrossing(num_bins=2, dtype=tf.int32)
        self.assertAllEqual(layer(([1], [1])).dtype, tf.int32)
        layer = hashed_crossing.HashedCrossing(
            num_bins=2, output_mode="one_hot"
        )
        self.assertAllEqual(layer(([1], [1])).dtype, tf.float32)
        layer = hashed_crossing.HashedCrossing(
            num_bins=2, output_mode="one_hot", dtype=tf.float64
        )
        self.assertAllEqual(layer(([1], [1])).dtype, tf.float64)

    def test_non_list_input_fails(self):
        with self.assertRaisesRegex(ValueError, "should be called on a list"):
            hashed_crossing.HashedCrossing(num_bins=10)(tf.constant(1))

    def test_single_input_fails(self):
        with self.assertRaisesRegex(ValueError, "at least two inputs"):
            hashed_crossing.HashedCrossing(num_bins=10)([tf.constant(1)])

    def test_sparse_input_fails(self):
        with self.assertRaisesRegex(
            ValueError, "inputs should be dense tensors"
        ):
            sparse_in = tf.sparse.from_dense(tf.constant([1]))
            hashed_crossing.HashedCrossing(num_bins=10)((sparse_in, sparse_in))

    def test_float_input_fails(self):
        with self.assertRaisesRegex(
            ValueError, "should have an integer or string"
        ):
            hashed_crossing.HashedCrossing(num_bins=10)(
                (tf.constant([1.0]), tf.constant([1.0]))
            )

    def test_upsupported_shape_input_fails(self):
        with self.assertRaisesRegex(ValueError, "inputs should have shape"):
            hashed_crossing.HashedCrossing(num_bins=10)(
                (tf.constant([[[1.0]]]), tf.constant([[[1.0]]]))
            )

    def test_from_config(self):
        layer = hashed_crossing.HashedCrossing(
            num_bins=5, output_mode="one_hot", sparse=True
        )
        cloned_layer = hashed_crossing.HashedCrossing.from_config(
            layer.get_config()
        )
        feat1 = tf.constant([["A"], ["B"], ["A"], ["B"], ["A"]])
        feat2 = tf.constant([[101], [101], [101], [102], [102]])
        original_outputs = layer((feat1, feat2))
        cloned_outputs = cloned_layer((feat1, feat2))
        self.assertAllEqual(
            tf.sparse.to_dense(cloned_outputs),
            tf.sparse.to_dense(original_outputs),
        )

    def test_saved_model_keras(self):
        string_in = keras.Input(shape=(1,), dtype=tf.string)
        int_in = keras.Input(shape=(1,), dtype=tf.int64)
        out = hashed_crossing.HashedCrossing(num_bins=10)((string_in, int_in))
        model = keras.Model(inputs=(string_in, int_in), outputs=out)

        string_data = tf.constant([["A"], ["B"], ["A"], ["B"], ["A"]])
        int_data = tf.constant([[101], [101], [101], [102], [102]])
        expected_output = [[1], [4], [1], [6], [3]]

        output_data = model((string_data, int_data))
        self.assertAllClose(output_data, expected_output)

        # Save the model to disk.
        output_path = os.path.join(self.get_temp_dir(), "saved_model")
        model.save(output_path, save_format="tf")
        loaded_model = keras.models.load_model(
            output_path,
            custom_objects={"HashedCrossing": hashed_crossing.HashedCrossing},
        )

        # Validate correctness of the new model.
        new_output_data = loaded_model((string_data, int_data))
        self.assertAllClose(new_output_data, expected_output)


if __name__ == "__main__":
    tf.test.main()
