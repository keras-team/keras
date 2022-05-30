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
"""Tests for spatial dropout layers."""

import tensorflow.compat.v2 as tf

import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_combinations.run_all_keras_modes
class SpacialDropoutTest(test_combinations.TestCase):
    def test_spatial_dropout_1d(self):
        test_utils.layer_test(
            keras.layers.SpatialDropout1D,
            kwargs={"rate": 0.5},
            input_shape=(2, 3, 4),
        )

    def test_spatial_dropout_2d(self):
        test_utils.layer_test(
            keras.layers.SpatialDropout2D,
            kwargs={"rate": 0.5},
            input_shape=(2, 3, 4, 5),
        )

        test_utils.layer_test(
            keras.layers.SpatialDropout2D,
            kwargs={"rate": 0.5, "data_format": "channels_first"},
            input_shape=(2, 3, 4, 5),
        )

    def test_spatial_dropout_3d(self):
        test_utils.layer_test(
            keras.layers.SpatialDropout3D,
            kwargs={"rate": 0.5},
            input_shape=(2, 3, 4, 4, 5),
        )

        test_utils.layer_test(
            keras.layers.SpatialDropout3D,
            kwargs={"rate": 0.5, "data_format": "channels_first"},
            input_shape=(2, 3, 4, 4, 5),
        )


if __name__ == "__main__":
    tf.test.main()
