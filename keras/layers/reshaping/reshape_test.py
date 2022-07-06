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
"""Tests for reshape layer."""

import tensorflow.compat.v2 as tf

import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_combinations.run_all_keras_modes
class ReshapeTest(test_combinations.TestCase):
    def test_reshape(self):
        test_utils.layer_test(
            keras.layers.Reshape,
            kwargs={"target_shape": (8, 1)},
            input_shape=(3, 2, 4),
        )

        test_utils.layer_test(
            keras.layers.Reshape,
            kwargs={"target_shape": (-1, 1)},
            input_shape=(3, 2, 4),
        )

        test_utils.layer_test(
            keras.layers.Reshape,
            kwargs={"target_shape": (1, -1)},
            input_shape=(3, 2, 4),
        )

        test_utils.layer_test(
            keras.layers.Reshape,
            kwargs={"target_shape": (-1, 1)},
            input_shape=(None, None, 2),
        )

    def test_reshape_set_static_shape(self):
        input_layer = keras.Input(batch_shape=(1, None))
        reshaped = keras.layers.Reshape((1, 100))(input_layer)
        # Make sure the batch dim is not lost after array_ops.reshape.
        self.assertEqual(reshaped.shape, [1, 1, 100])


if __name__ == "__main__":
    tf.test.main()
