# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for keras.layers.preprocessing.hashing."""

import tensorflow.compat.v2 as tf

import numpy as np

import keras
from keras import keras_parameterized
from keras.distribute.strategy_combinations import all_strategies
from keras.layers.preprocessing import hashing
from keras.layers.preprocessing import preprocessing_test_utils


@tf.__internal__.distribute.combinations.generate(
    tf.__internal__.test.combinations.combine(
        distribution=all_strategies,
        mode=["eager", "graph"]))
class HashingDistributionTest(keras_parameterized.TestCase,
                              preprocessing_test_utils.PreprocessingLayerTest):

  def test_distribution(self, distribution):
    input_data = np.asarray([["omar"], ["stringer"], ["marlo"], ["wire"]])
    input_dataset = tf.data.Dataset.from_tensor_slices(input_data).batch(
        2, drop_remainder=True)
    expected_output = [[0], [0], [1], [0]]

    tf.config.set_soft_device_placement(True)

    with distribution.scope():
      input_data = keras.Input(shape=(None,), dtype=tf.string)
      layer = hashing.Hashing(num_bins=2)
      int_data = layer(input_data)
      model = keras.Model(inputs=input_data, outputs=int_data)
    output_dataset = model.predict(input_dataset)
    self.assertAllEqual(expected_output, output_dataset)


if __name__ == "__main__":
  tf.test.main()
