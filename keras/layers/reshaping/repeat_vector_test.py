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
"""Tests for repeat vector layer."""

import numpy as np
import tensorflow.compat.v2 as tf

import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_combinations.run_all_keras_modes
class RepeatVectorTest(test_combinations.TestCase):
    def test_repeat_vector(self):
        test_utils.layer_test(
            keras.layers.RepeatVector, kwargs={"n": 3}, input_shape=(3, 2)
        )

    def test_numpy_inputs(self):
        if tf.executing_eagerly():
            layer = keras.layers.RepeatVector(2)
            x = np.ones((10, 10))
            self.assertAllEqual(np.ones((10, 2, 10)), layer(x))


if __name__ == "__main__":
    tf.test.main()
