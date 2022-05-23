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
"""Tests for zero-padding layers."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_combinations.run_all_keras_modes
class ZeroPaddingTest(test_combinations.TestCase):
    def test_zero_padding_1d(self):
        num_samples = 2
        input_dim = 2
        num_steps = 5
        shape = (num_samples, num_steps, input_dim)
        inputs = np.ones(shape)

        with self.cached_session():
            # basic test
            test_utils.layer_test(
                keras.layers.ZeroPadding1D,
                kwargs={"padding": 2},
                input_shape=inputs.shape,
            )
            test_utils.layer_test(
                keras.layers.ZeroPadding1D,
                kwargs={"padding": (1, 2)},
                input_shape=inputs.shape,
            )

            # correctness test
            layer = keras.layers.ZeroPadding1D(padding=2)
            layer.build(shape)
            output = layer(keras.backend.variable(inputs))
            if tf.executing_eagerly():
                np_output = output.numpy()
            else:
                np_output = keras.backend.eval(output)
            for offset in [0, 1, -1, -2]:
                np.testing.assert_allclose(np_output[:, offset, :], 0.0)
            np.testing.assert_allclose(np_output[:, 2:-2, :], 1.0)

            layer = keras.layers.ZeroPadding1D(padding=(1, 2))
            layer.build(shape)
            output = layer(keras.backend.variable(inputs))
            if tf.executing_eagerly():
                np_output = output.numpy()
            else:
                np_output = keras.backend.eval(output)
            for left_offset in [0]:
                np.testing.assert_allclose(np_output[:, left_offset, :], 0.0)
            for right_offset in [-1, -2]:
                np.testing.assert_allclose(np_output[:, right_offset, :], 0.0)
            np.testing.assert_allclose(np_output[:, 1:-2, :], 1.0)
            layer.get_config()

        # test incorrect use
        with self.assertRaises(ValueError):
            keras.layers.ZeroPadding1D(padding=(1, 1, 1))
        with self.assertRaises(ValueError):
            keras.layers.ZeroPadding1D(padding=None)

    @parameterized.named_parameters(
        ("channels_first", "channels_first"), ("channels_last", "channels_last")
    )
    def test_zero_padding_2d(self, data_format):
        num_samples = 2
        stack_size = 2
        input_num_row = 4
        input_num_col = 5
        if data_format == "channels_first":
            inputs = np.ones(
                (num_samples, stack_size, input_num_row, input_num_col)
            )
        elif data_format == "channels_last":
            inputs = np.ones(
                (num_samples, input_num_row, input_num_col, stack_size)
            )

        # basic test
        with self.cached_session():
            test_utils.layer_test(
                keras.layers.ZeroPadding2D,
                kwargs={"padding": (2, 2), "data_format": data_format},
                input_shape=inputs.shape,
            )
            test_utils.layer_test(
                keras.layers.ZeroPadding2D,
                kwargs={
                    "padding": ((1, 2), (3, 4)),
                    "data_format": data_format,
                },
                input_shape=inputs.shape,
            )

        # correctness test
        with self.cached_session():
            layer = keras.layers.ZeroPadding2D(
                padding=(2, 2), data_format=data_format
            )
            layer.build(inputs.shape)
            output = layer(keras.backend.variable(inputs))
            if tf.executing_eagerly():
                np_output = output.numpy()
            else:
                np_output = keras.backend.eval(output)
            if data_format == "channels_last":
                for offset in [0, 1, -1, -2]:
                    np.testing.assert_allclose(np_output[:, offset, :, :], 0.0)
                    np.testing.assert_allclose(np_output[:, :, offset, :], 0.0)
                np.testing.assert_allclose(np_output[:, 2:-2, 2:-2, :], 1.0)
            elif data_format == "channels_first":
                for offset in [0, 1, -1, -2]:
                    np.testing.assert_allclose(np_output[:, :, offset, :], 0.0)
                    np.testing.assert_allclose(np_output[:, :, :, offset], 0.0)
                np.testing.assert_allclose(np_output[:, 2:-2, 2:-2, :], 1.0)

            layer = keras.layers.ZeroPadding2D(
                padding=((1, 2), (3, 4)), data_format=data_format
            )
            layer.build(inputs.shape)
            output = layer(keras.backend.variable(inputs))
            if tf.executing_eagerly():
                np_output = output.numpy()
            else:
                np_output = keras.backend.eval(output)
            if data_format == "channels_last":
                for top_offset in [0]:
                    np.testing.assert_allclose(
                        np_output[:, top_offset, :, :], 0.0
                    )
                for bottom_offset in [-1, -2]:
                    np.testing.assert_allclose(
                        np_output[:, bottom_offset, :, :], 0.0
                    )
                for left_offset in [0, 1, 2]:
                    np.testing.assert_allclose(
                        np_output[:, :, left_offset, :], 0.0
                    )
                for right_offset in [-1, -2, -3, -4]:
                    np.testing.assert_allclose(
                        np_output[:, :, right_offset, :], 0.0
                    )
                np.testing.assert_allclose(np_output[:, 1:-2, 3:-4, :], 1.0)
            elif data_format == "channels_first":
                for top_offset in [0]:
                    np.testing.assert_allclose(
                        np_output[:, :, top_offset, :], 0.0
                    )
                for bottom_offset in [-1, -2]:
                    np.testing.assert_allclose(
                        np_output[:, :, bottom_offset, :], 0.0
                    )
                for left_offset in [0, 1, 2]:
                    np.testing.assert_allclose(
                        np_output[:, :, :, left_offset], 0.0
                    )
                for right_offset in [-1, -2, -3, -4]:
                    np.testing.assert_allclose(
                        np_output[:, :, :, right_offset], 0.0
                    )
                np.testing.assert_allclose(np_output[:, :, 1:-2, 3:-4], 1.0)

        # test incorrect use
        with self.assertRaises(ValueError):
            keras.layers.ZeroPadding2D(padding=(1, 1, 1))
        with self.assertRaises(ValueError):
            keras.layers.ZeroPadding2D(padding=None)

    @parameterized.named_parameters(
        ("channels_first", "channels_first"), ("channels_last", "channels_last")
    )
    def test_zero_padding_3d(self, data_format):
        num_samples = 2
        stack_size = 2
        input_len_dim1 = 4
        input_len_dim2 = 5
        input_len_dim3 = 3

        if data_format == "channels_first":
            inputs = np.ones(
                (
                    num_samples,
                    stack_size,
                    input_len_dim1,
                    input_len_dim2,
                    input_len_dim3,
                )
            )
        elif data_format == "channels_last":
            inputs = np.ones(
                (
                    num_samples,
                    input_len_dim1,
                    input_len_dim2,
                    input_len_dim3,
                    stack_size,
                )
            )

        with self.cached_session():
            # basic test
            test_utils.layer_test(
                keras.layers.ZeroPadding3D,
                kwargs={"padding": (2, 2, 2), "data_format": data_format},
                input_shape=inputs.shape,
            )
            test_utils.layer_test(
                keras.layers.ZeroPadding3D,
                kwargs={
                    "padding": ((1, 2), (3, 4), (0, 2)),
                    "data_format": data_format,
                },
                input_shape=inputs.shape,
            )

        with self.cached_session():
            # correctness test
            layer = keras.layers.ZeroPadding3D(
                padding=(2, 2, 2), data_format=data_format
            )
            layer.build(inputs.shape)
            output = layer(keras.backend.variable(inputs))
            if tf.executing_eagerly():
                np_output = output.numpy()
            else:
                np_output = keras.backend.eval(output)
            if data_format == "channels_last":
                for offset in [0, 1, -1, -2]:
                    np.testing.assert_allclose(
                        np_output[:, offset, :, :, :], 0.0
                    )
                    np.testing.assert_allclose(
                        np_output[:, :, offset, :, :], 0.0
                    )
                    np.testing.assert_allclose(
                        np_output[:, :, :, offset, :], 0.0
                    )
                np.testing.assert_allclose(
                    np_output[:, 2:-2, 2:-2, 2:-2, :], 1.0
                )
            elif data_format == "channels_first":
                for offset in [0, 1, -1, -2]:
                    np.testing.assert_allclose(
                        np_output[:, :, offset, :, :], 0.0
                    )
                    np.testing.assert_allclose(
                        np_output[:, :, :, offset, :], 0.0
                    )
                    np.testing.assert_allclose(
                        np_output[:, :, :, :, offset], 0.0
                    )
                np.testing.assert_allclose(
                    np_output[:, :, 2:-2, 2:-2, 2:-2], 1.0
                )

            layer = keras.layers.ZeroPadding3D(
                padding=((1, 2), (3, 4), (0, 2)), data_format=data_format
            )
            layer.build(inputs.shape)
            output = layer(keras.backend.variable(inputs))
            if tf.executing_eagerly():
                np_output = output.numpy()
            else:
                np_output = keras.backend.eval(output)
            if data_format == "channels_last":
                for offset in [0]:
                    np.testing.assert_allclose(
                        np_output[:, offset, :, :, :], 0.0
                    )
                for offset in [-1, -2]:
                    np.testing.assert_allclose(
                        np_output[:, offset, :, :, :], 0.0
                    )
                for offset in [0, 1, 2]:
                    np.testing.assert_allclose(
                        np_output[:, :, offset, :, :], 0.0
                    )
                for offset in [-1, -2, -3, -4]:
                    np.testing.assert_allclose(
                        np_output[:, :, offset, :, :], 0.0
                    )
                for offset in [-1, -2]:
                    np.testing.assert_allclose(
                        np_output[:, :, :, offset, :], 0.0
                    )
                np.testing.assert_allclose(
                    np_output[:, 1:-2, 3:-4, 0:-2, :], 1.0
                )
            elif data_format == "channels_first":
                for offset in [0]:
                    np.testing.assert_allclose(
                        np_output[:, :, offset, :, :], 0.0
                    )
                for offset in [-1, -2]:
                    np.testing.assert_allclose(
                        np_output[:, :, offset, :, :], 0.0
                    )
                for offset in [0, 1, 2]:
                    np.testing.assert_allclose(
                        np_output[:, :, :, offset, :], 0.0
                    )
                for offset in [-1, -2, -3, -4]:
                    np.testing.assert_allclose(
                        np_output[:, :, :, offset, :], 0.0
                    )
                for offset in [-1, -2]:
                    np.testing.assert_allclose(
                        np_output[:, :, :, :, offset], 0.0
                    )
                np.testing.assert_allclose(
                    np_output[:, :, 1:-2, 3:-4, 0:-2], 1.0
                )

        # test incorrect use
        with self.assertRaises(ValueError):
            keras.layers.ZeroPadding3D(padding=(1, 1))
        with self.assertRaises(ValueError):
            keras.layers.ZeroPadding3D(padding=None)


if __name__ == "__main__":
    tf.test.main()
