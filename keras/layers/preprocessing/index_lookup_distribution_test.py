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
"""Distribution tests for keras.layers.preprocessing.index_lookup."""

import tensorflow.compat.v2 as tf

import os
import numpy as np

import keras
from keras import backend
from keras import keras_parameterized
from keras.distribute import strategy_combinations
from keras.layers.preprocessing import index_lookup
from keras.layers.preprocessing import preprocessing_test_utils


@tf.__internal__.distribute.combinations.generate(
    tf.__internal__.test.combinations.combine(
        strategy=strategy_combinations.all_strategies +
        strategy_combinations.multi_worker_mirrored_strategies,
        mode=["eager"]))  # Eager-only, no graph: b/158793009
class IndexLookupDistributionTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def _write_to_temp_file(self, file_name, vocab_list):
    vocab_path = os.path.join(self.get_temp_dir(), file_name + ".txt")
    with tf.io.gfile.GFile(vocab_path, "w") as writer:
      for vocab in vocab_list:
        writer.write(vocab + "\n")
      writer.flush()
      writer.close()
    return vocab_path

  def test_strategy(self, strategy):
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
      layer = index_lookup.IndexLookup(
          max_tokens=None,
          num_oov_indices=1,
          mask_token="",
          oov_token="[OOV]",
          dtype=tf.string)
      layer.adapt(vocab_dataset)
      int_data = layer(input_data)
      model = keras.Model(inputs=input_data, outputs=int_data)
    model.compile(loss="mse")
    output_dataset = model.predict(input_dataset)
    self.assertAllEqual(expected_output, output_dataset)

  def test_strategy_with_file(self, strategy):
    # TODO(b/180614455): remove this check when MLIR bridge is always enabled.
    if backend.is_tpu_strategy(strategy):
      self.skipTest("This test needs MLIR bridge on TPU.")

    vocab_data = ["earth", "wind", "and", "fire"]
    vocab_file = self._write_to_temp_file("temp", vocab_data)

    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    input_dataset = tf.data.Dataset.from_tensor_slices(input_array).batch(
        2, drop_remainder=True)
    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    tf.config.set_soft_device_placement(True)

    with strategy.scope():
      input_data = keras.Input(shape=(None,), dtype=tf.string)
      layer = index_lookup.IndexLookup(
          max_tokens=None,
          num_oov_indices=1,
          mask_token="",
          oov_token="[OOV]",
          dtype=tf.string,
          vocabulary=vocab_file)
      int_data = layer(input_data)
      model = keras.Model(inputs=input_data, outputs=int_data)
    model.compile(loss="mse")
    output_dataset = model.predict(input_dataset)
    self.assertAllEqual(expected_output, output_dataset)

  def test_tpu_with_multiple_oov(self, strategy):
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
    expected_output = [[3, 4, 5, 6], [6, 5, 3, 1]]

    tf.config.set_soft_device_placement(True)

    with strategy.scope():
      input_data = keras.Input(shape=(None,), dtype=tf.string)
      layer = index_lookup.IndexLookup(
          max_tokens=None,
          num_oov_indices=2,
          mask_token="",
          oov_token="[OOV]",
          dtype=tf.string)
      layer.adapt(vocab_dataset)
      int_data = layer(input_data)
      model = keras.Model(inputs=input_data, outputs=int_data)
    model.compile(loss="mse")
    output_dataset = model.predict(input_dataset)
    self.assertAllEqual(expected_output, output_dataset)


if __name__ == "__main__":
  tf.compat.v1.enable_v2_behavior()
  tf.__internal__.distribute.multi_process_runner.test_main()
