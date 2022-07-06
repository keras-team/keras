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
"""Tests for LeakyReLU layer."""

import tensorflow.compat.v2 as tf

import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_combinations.run_all_keras_modes
class LeakyReLUTest(test_combinations.TestCase):
    def test_leaky_relu(self):
        for alpha in [0.0, 0.5]:
            test_utils.layer_test(
                keras.layers.LeakyReLU,
                kwargs={"alpha": alpha},
                input_shape=(2, 3, 4),
                supports_masking=True,
            )

    def test_leaky_relu_with_invalid_alpha(self):
        # Test case for GitHub issue 46993.
        with self.assertRaisesRegex(
            ValueError,
            "The alpha value of a Leaky ReLU layer "
            "cannot be None. Expecting a float. Received: None",
        ):
            test_utils.layer_test(
                keras.layers.LeakyReLU,
                kwargs={"alpha": None},
                input_shape=(2, 3, 4),
                supports_masking=True,
            )


if __name__ == "__main__":
    tf.test.main()
