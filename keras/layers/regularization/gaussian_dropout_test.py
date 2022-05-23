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
"""Tests for gaussian dropout layer."""

import numpy as np
import tensorflow.compat.v2 as tf

import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


@test_combinations.run_all_keras_modes
class NoiseLayersTest(test_combinations.TestCase):
    def test_GaussianDropout(self):
        test_utils.layer_test(
            keras.layers.GaussianDropout,
            kwargs={"rate": 0.5},
            input_shape=(3, 2, 3),
        )

    def _make_model(self, dtype):
        assert dtype in (tf.float32, tf.float64)
        model = keras.Sequential()
        model.add(keras.layers.Dense(8, input_shape=(32,), dtype=dtype))
        layer = keras.layers.GaussianDropout(0.1, dtype=dtype)
        model.add(layer)
        return model

    def _train_model(self, dtype):
        model = self._make_model(dtype)
        model.compile(
            optimizer="sgd",
            loss="mse",
            run_eagerly=test_utils.should_run_eagerly(),
        )
        model.train_on_batch(np.zeros((8, 32)), np.zeros((8, 8)))

    def test_gaussian_dropout_float32(self):
        self._train_model(tf.float32)

    def test_gaussian_dropout_float64(self):
        self._train_model(tf.float64)


if __name__ == "__main__":
    tf.test.main()
