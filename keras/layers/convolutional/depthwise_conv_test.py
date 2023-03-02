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
"""Tests for depthwise convolutional layers."""

import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_combinations.run_all_keras_modes
class DepthwiseConv1DTest(test_combinations.TestCase):
    def _run_test(self, kwargs, expected_output_shape=None):
        num_samples = 2
        stack_size = 3
        num_row = 7

        with self.cached_session():
            test_utils.layer_test(
                keras.layers.DepthwiseConv1D,
                kwargs=kwargs,
                input_shape=(num_samples, num_row, stack_size),
                expected_output_shape=expected_output_shape,
            )

    @parameterized.named_parameters(
        ("padding_valid", {"padding": "valid"}),
        ("padding_same", {"padding": "same"}),
        ("strides", {"strides": 2}),
        # Only runs on GPU with CUDA, channels_first is not supported on CPU.
        # TODO(b/62340061): Support channels_first on CPU.
        ("data_format", {"data_format": "channels_first"}),
        ("depth_multiplier_1", {"depth_multiplier": 1}),
        ("depth_multiplier_2", {"depth_multiplier": 2}),
        ("dilation_rate", {"dilation_rate": 2}, (None, 3, 3)),
    )
    def test_depthwise_conv1d(self, kwargs, expected_output_shape=None):
        kwargs["kernel_size"] = 3
        if "data_format" not in kwargs or tf.test.is_gpu_available(
            cuda_only=True
        ):
            self._run_test(kwargs, expected_output_shape)

    def test_depthwise_conv1d_full(self):
        kwargs = {
            "kernel_size": 3,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": 1,
            "activation": None,
            "depthwise_regularizer": "l2",
            "bias_regularizer": "l2",
            "activity_regularizer": "l2",
            "depthwise_constraint": "unit_norm",
            "use_bias": True,
            "strides": 2,
            "depth_multiplier": 1,
        }
        self._run_test(kwargs)

    def test_depthwise_conv1d_invalid_strides_and_dilation_rate(self):
        kwargs = {"strides": 2, "dilation_rate": 2}
        with self.assertRaisesRegex(
            ValueError, r"""`strides > 1` not supported in conjunction"""
        ):
            keras.layers.DepthwiseConv1D(kernel_size=2, **kwargs)


@test_combinations.run_all_keras_modes
class DepthwiseConv2DTest(test_combinations.TestCase):
    def _run_test(self, kwargs, expected_output_shape=None):
        num_samples = 2
        stack_size = 3
        num_row = 7
        num_col = 6

        with self.cached_session():
            test_utils.layer_test(
                keras.layers.DepthwiseConv2D,
                kwargs=kwargs,
                input_shape=(num_samples, num_row, num_col, stack_size),
                expected_output_shape=expected_output_shape,
            )

    @parameterized.named_parameters(
        ("padding_valid", {"padding": "valid"}),
        ("padding_same", {"padding": "same"}),
        ("strides", {"strides": (2, 2)}),
        # Only runs on GPU with CUDA, channels_first is not supported on CPU.
        # TODO(b/62340061): Support channels_first on CPU.
        ("data_format", {"data_format": "channels_first"}),
        ("depth_multiplier_1", {"depth_multiplier": 1}),
        ("depth_multiplier_2", {"depth_multiplier": 2}),
        ("dilation_rate", {"dilation_rate": (2, 2)}, (None, 3, 2, 3)),
    )
    def test_depthwise_conv2d(self, kwargs, expected_output_shape=None):
        kwargs["kernel_size"] = (3, 3)
        if "data_format" not in kwargs or tf.test.is_gpu_available(
            cuda_only=True
        ):
            self._run_test(kwargs, expected_output_shape)

    def test_depthwise_conv2d_full(self):
        kwargs = {
            "kernel_size": 3,
            "padding": "valid",
            "data_format": "channels_last",
            "dilation_rate": (1, 1),
            "activation": None,
            "depthwise_regularizer": "l2",
            "bias_regularizer": "l2",
            "activity_regularizer": "l2",
            "depthwise_constraint": "unit_norm",
            "use_bias": True,
            "strides": (2, 2),
            "depth_multiplier": 1,
        }
        self._run_test(kwargs)

    def test_depthwise_conv2d_invalid_strides_and_dilation_rate(self):
        kwargs = {"strides": [2, 1], "dilation_rate": [2, 1]}
        with self.assertRaisesRegex(
            ValueError, r"""`strides > 1` not supported in conjunction"""
        ):
            keras.layers.DepthwiseConv2D(kernel_size=2, **kwargs)


if __name__ == "__main__":
    tf.test.main()
