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
"""Tests for ThresholdedReLU layer."""

import tensorflow.compat.v2 as tf

import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_combinations.run_all_keras_modes
class ThresholdedReLUTest(test_combinations.TestCase):
    def test_thresholded_relu(self):
        test_utils.layer_test(
            keras.layers.ThresholdedReLU,
            kwargs={"theta": 0.5},
            input_shape=(2, 3, 4),
            supports_masking=True,
        )

    def test_threshold_relu_with_invalid_theta(self):
        with self.assertRaisesRegex(
            ValueError,
            "Theta of a Thresholded ReLU layer cannot "
            "be None, expecting a float. Received: None",
        ):
            test_utils.layer_test(
                keras.layers.ThresholdedReLU,
                kwargs={"theta": None},
                input_shape=(2, 3, 4),
                supports_masking=True,
            )

        with self.assertRaisesRegex(
            ValueError,
            "The theta value of a Thresholded ReLU "
            "layer should be >=0. Received: -10",
        ):
            test_utils.layer_test(
                keras.layers.ThresholdedReLU,
                kwargs={"theta": -10},
                input_shape=(2, 3, 4),
                supports_masking=True,
            )


if __name__ == "__main__":
    tf.test.main()
