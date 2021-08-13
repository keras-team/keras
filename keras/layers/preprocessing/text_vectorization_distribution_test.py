# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Distribution tests for keras.layers.preprocessing.text_vectorization."""

import tensorflow.compat.v2 as tf

import numpy as np

import keras
from keras import backend
from keras import keras_parameterized
from keras.distribute import strategy_combinations
from keras.layers.preprocessing import preprocessing_test_utils
from keras.layers.preprocessing import text_vectorization


@tf.__internal__.distribute.combinations.generate(
    tf.__internal__.test.combinations.combine(
        strategy=strategy_combinations.all_strategies +
        strategy_combinations.multi_worker_mirrored_strategies,
        mode=["eager"]))
class TextVectorizationDistributionTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_distribution_strategy_output(self, strategy):
    # TODO(b/180614455): remove this check when MLIR bridge is always enabled.
    if backend.is_tpu_strategy(strategy):
      self.skipTest("This test needs MLIR bridge on TPU.")

    vocab_data = ["earth", "wind", "and", "fire"]
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    input_dataset = tf.data.Dataset.from_tensor_slices(input_array).batch(
        2, drop_remainder=True)

    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    tf.config.set_soft_device_placement(True)

    with strategy.scope():
      input_data = keras.Input(shape=(None,), dtype=tf.string)
      layer = text_vectorization.TextVectorization(
          max_tokens=None,
          standardize=None,
          split=None,
          output_mode=text_vectorization.INT,
          vocabulary=vocab_data)
      int_data = layer(input_data)
      model = keras.Model(inputs=input_data, outputs=int_data)

    output_dataset = model.predict(input_dataset)
    self.assertAllEqual(expected_output, output_dataset)

  def test_distribution_strategy_output_with_adapt(self, strategy):
    # TODO(b/180614455): remove this check when MLIR bridge is always enabled.
    if backend.is_tpu_strategy(strategy):
      self.skipTest("This test needs MLIR bridge on TPU.")

    vocab_data = [[
        "earth", "earth", "earth", "earth", "wind", "wind", "wind", "and",
        "and", "fire"
    ]]
    vocab_dataset = tf.data.Dataset.from_tensors(vocab_data)
    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    input_dataset = tf.data.Dataset.from_tensor_slices(input_array).batch(
        2, drop_remainder=True)

    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    tf.config.set_soft_device_placement(True)

    with strategy.scope():
      input_data = keras.Input(shape=(None,), dtype=tf.string)
      layer = text_vectorization.TextVectorization(
          max_tokens=None,
          standardize=None,
          split=None,
          output_mode=text_vectorization.INT)
      layer.adapt(vocab_dataset)
      int_data = layer(input_data)
      model = keras.Model(inputs=input_data, outputs=int_data)

    output_dataset = model.predict(input_dataset)
    self.assertAllEqual(expected_output, output_dataset)

if __name__ == "__main__":
  tf.compat.v1.enable_v2_behavior()
  tf.__internal__.distribute.multi_process_runner.test_main()
