# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for global max pooling layers."""

import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class GlobalMaxPoolingTest(tf.test.TestCase, parameterized.TestCase):
    def test_global_max_pooling_1d(self):
        test_utils.layer_test(
            keras.layers.GlobalMaxPooling1D, input_shape=(3, 4, 5)
        )
        test_utils.layer_test(
            keras.layers.GlobalMaxPooling1D,
            kwargs={"data_format": "channels_first"},
            input_shape=(3, 4, 5),
        )

    def test_global_max_pooling_2d_with_ragged(self):
        ragged_data = tf.ragged.constant(
            [
                [[[1.0], [1.0]], [[2.0], [2.0]], [[3.0], [3.0]]],
                [[[1.0], [1.0]], [[2.0], [2.0]]],
            ],
            ragged_rank=1,
        )
        dense_data = ragged_data.to_tensor()

        inputs = keras.Input(shape=(None, 2, 1), dtype="float32", ragged=True)
        out = keras.layers.GlobalMaxPooling2D()(inputs)
        model = keras.models.Model(inputs=inputs, outputs=out)
        output_ragged = model.predict(ragged_data, steps=1)

        inputs = keras.Input(shape=(None, 2, 1), dtype="float32")
        out = keras.layers.GlobalMaxPooling2D()(inputs)
        model = keras.models.Model(inputs=inputs, outputs=out)
        output_dense = model.predict(dense_data, steps=1)

        self.assertAllEqual(output_ragged, output_dense)

    def test_global_max_pooling_2d(self):
        test_utils.layer_test(
            keras.layers.GlobalMaxPooling2D,
            kwargs={"data_format": "channels_first"},
            input_shape=(3, 4, 5, 6),
        )
        test_utils.layer_test(
            keras.layers.GlobalMaxPooling2D,
            kwargs={"data_format": "channels_last"},
            input_shape=(3, 5, 6, 4),
        )

    def test_global_maxpooling_3d(self):
        test_utils.layer_test(
            keras.layers.GlobalMaxPooling3D,
            kwargs={"data_format": "channels_first"},
            input_shape=(3, 4, 3, 4, 3),
        )
        test_utils.layer_test(
            keras.layers.GlobalMaxPooling3D,
            kwargs={"data_format": "channels_last"},
            input_shape=(3, 4, 3, 4, 3),
        )

    def test_global_max_pooling_1d_keepdims(self):
        test_utils.layer_test(
            keras.layers.GlobalMaxPooling1D,
            kwargs={"keepdims": True},
            input_shape=(3, 4, 5),
            expected_output_shape=(None, 1, 5),
        )
        test_utils.layer_test(
            keras.layers.GlobalMaxPooling1D,
            kwargs={"data_format": "channels_first", "keepdims": True},
            input_shape=(3, 4, 5),
            expected_output_shape=(None, 4, 1),
        )

    def test_global_max_pooling_2d_keepdims(self):
        test_utils.layer_test(
            keras.layers.GlobalMaxPooling2D,
            kwargs={"data_format": "channels_first", "keepdims": True},
            input_shape=(3, 4, 5, 6),
            expected_output_shape=(None, 4, 1, 1),
        )
        test_utils.layer_test(
            keras.layers.GlobalMaxPooling2D,
            kwargs={"data_format": "channels_last", "keepdims": True},
            input_shape=(3, 4, 5, 6),
            expected_output_shape=(None, 1, 1, 6),
        )

    def test_global_max_pooling_3d_keepdims(self):
        test_utils.layer_test(
            keras.layers.GlobalMaxPooling3D,
            kwargs={"data_format": "channels_first", "keepdims": True},
            input_shape=(3, 4, 3, 4, 3),
            expected_output_shape=(None, 4, 1, 1, 1),
        )
        test_utils.layer_test(
            keras.layers.GlobalMaxPooling3D,
            kwargs={"data_format": "channels_last", "keepdims": True},
            input_shape=(3, 4, 3, 4, 3),
            expected_output_shape=(None, 1, 1, 1, 3),
        )

    def test_global_max_pooling_1d_invalid_input_dimension(self):
        with self.assertRaisesRegex(ValueError, r"""Incorrect input shape"""):
            layer = keras.layers.GlobalMaxPooling1D()
            layer.build((None, 0, 2))

    def test_global_max_pooling_3d_invalid_input_dimension(self):
        with self.assertRaisesRegex(ValueError, r"""Incorrect input shape"""):
            layer = keras.layers.GlobalMaxPooling3D(keepdims=True)
            layer.build((None, 0, 16, 16, 3))


if __name__ == "__main__":
    tf.test.main()
