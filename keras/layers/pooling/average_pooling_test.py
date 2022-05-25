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
"""Tests for average pooling layers."""

import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class AveragePoolingTest(tf.test.TestCase, parameterized.TestCase):
    def test_average_pooling_1d(self):
        for padding in ["valid", "same"]:
            for stride in [1, 2]:
                test_utils.layer_test(
                    keras.layers.AveragePooling1D,
                    kwargs={"strides": stride, "padding": padding},
                    input_shape=(3, 5, 4),
                )

        test_utils.layer_test(
            keras.layers.AveragePooling1D,
            kwargs={"data_format": "channels_first"},
            input_shape=(3, 2, 6),
        )

    def test_average_pooling_2d(self):
        test_utils.layer_test(
            keras.layers.AveragePooling2D,
            kwargs={"strides": (2, 2), "padding": "same", "pool_size": (2, 2)},
            input_shape=(3, 5, 6, 4),
        )
        test_utils.layer_test(
            keras.layers.AveragePooling2D,
            kwargs={"strides": (2, 2), "padding": "valid", "pool_size": (3, 3)},
            input_shape=(3, 5, 6, 4),
        )

        # This part of the test can only run on GPU but doesn't appear
        # to be properly assigned to a GPU when running in eager mode.
        if not tf.executing_eagerly():
            # Only runs on GPU with CUDA, channels_first is not supported on
            # CPU.
            # TODO(b/62340061): Support channels_first on CPU.
            if tf.test.is_gpu_available(cuda_only=True):
                test_utils.layer_test(
                    keras.layers.AveragePooling2D,
                    kwargs={
                        "strides": (1, 1),
                        "padding": "valid",
                        "pool_size": (2, 2),
                        "data_format": "channels_first",
                    },
                    input_shape=(3, 4, 5, 6),
                )

    def test_average_pooling_3d(self):
        pool_size = (3, 3, 3)
        test_utils.layer_test(
            keras.layers.AveragePooling3D,
            kwargs={"strides": 2, "padding": "valid", "pool_size": pool_size},
            input_shape=(3, 11, 12, 10, 4),
        )
        test_utils.layer_test(
            keras.layers.AveragePooling3D,
            kwargs={
                "strides": 3,
                "padding": "valid",
                "data_format": "channels_first",
                "pool_size": pool_size,
            },
            input_shape=(3, 4, 11, 12, 10),
        )


if __name__ == "__main__":
    tf.test.main()
