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
"""Tests for flatten layer."""

import numpy as np
import tensorflow.compat.v2 as tf

import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_combinations.run_all_keras_modes
class FlattenTest(test_combinations.TestCase):
    def test_flatten(self):
        test_utils.layer_test(
            keras.layers.Flatten, kwargs={}, input_shape=(3, 2, 4)
        )

        # Test channels_first
        inputs = np.random.random((10, 3, 5, 5)).astype("float32")
        outputs = test_utils.layer_test(
            keras.layers.Flatten,
            kwargs={"data_format": "channels_first"},
            input_data=inputs,
        )
        target_outputs = np.reshape(
            np.transpose(inputs, (0, 2, 3, 1)), (-1, 5 * 5 * 3)
        )
        self.assertAllClose(outputs, target_outputs)

    def test_flatten_scalar_channels(self):
        test_utils.layer_test(keras.layers.Flatten, kwargs={}, input_shape=(3,))

        # Test channels_first
        inputs = np.random.random((10,)).astype("float32")
        outputs = test_utils.layer_test(
            keras.layers.Flatten,
            kwargs={"data_format": "channels_first"},
            input_data=inputs,
        )
        target_outputs = np.expand_dims(inputs, -1)
        self.assertAllClose(outputs, target_outputs)


if __name__ == "__main__":
    tf.test.main()
