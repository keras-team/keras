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
"""Tests that are common for GRU and LSTM.

See also: lstm_test.py, gru_test.py.
"""

import os

from absl.testing import parameterized
import keras
from keras.layers import embeddings
from keras.layers.rnn import gru
from keras.layers.rnn import lstm
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
import numpy as np
import tensorflow.compat.v2 as tf


@test_combinations.run_all_keras_modes
class RNNV2Test(test_combinations.TestCase):

  @parameterized.parameters([lstm.LSTM, gru.GRU])
  def test_device_placement(self, layer):
    if not tf.test.is_gpu_available():
      self.skipTest('Need GPU for testing.')
    vocab_size = 20
    embedding_dim = 10
    batch_size = 8
    timestep = 12
    units = 5
    x = np.random.randint(0, vocab_size, size=(batch_size, timestep))
    y = np.random.randint(0, vocab_size, size=(batch_size, timestep))

    # Test when GPU is available but not used, the graph should be properly
    # created with CPU ops.
    with test_utils.device(should_use_gpu=False):
      model = keras.Sequential([
          keras.layers.Embedding(vocab_size, embedding_dim,
                                 batch_input_shape=[batch_size, timestep]),
          layer(units, return_sequences=True, stateful=True),
          keras.layers.Dense(vocab_size)
      ])
      model.compile(
          optimizer='adam',
          loss='sparse_categorical_crossentropy',
          run_eagerly=test_utils.should_run_eagerly())
      model.fit(x, y, epochs=1, shuffle=False)

  @parameterized.parameters([lstm.LSTM, gru.GRU])
  def test_reset_dropout_mask_between_batch(self, layer):
    # See https://github.com/tensorflow/tensorflow/issues/29187 for more details
    batch_size = 8
    timestep = 12
    embedding_dim = 10
    units = 5
    layer = layer(units, dropout=0.5, recurrent_dropout=0.5)

    inputs = np.random.random((batch_size, timestep, embedding_dim)).astype(
        np.float32)
    previous_dropout, previous_recurrent_dropout = None, None

    for _ in range(5):
      layer(inputs, training=True)
      dropout = layer.cell.get_dropout_mask_for_cell(inputs, training=True)
      recurrent_dropout = layer.cell.get_recurrent_dropout_mask_for_cell(
          inputs, training=True)
      if previous_dropout is not None:
        self.assertNotAllClose(self.evaluate(previous_dropout),
                               self.evaluate(dropout))
        previous_dropout = dropout
      if previous_recurrent_dropout is not None:
        self.assertNotAllClose(self.evaluate(previous_recurrent_dropout),
                               self.evaluate(recurrent_dropout))
        previous_recurrent_dropout = recurrent_dropout

  @parameterized.parameters([lstm.LSTM, gru.GRU])
  def test_recurrent_dropout_with_stateful_RNN(self, layer):
    # See https://github.com/tensorflow/tensorflow/issues/27829 for details.
    # The issue was caused by using inplace mul for a variable, which was a
    # warning for RefVariable, but an error for ResourceVariable in 2.0
    keras.models.Sequential([
        layer(128, stateful=True, return_sequences=True, dropout=0.2,
              batch_input_shape=[32, None, 5], recurrent_dropout=0.2)
    ])

  @parameterized.parameters([lstm.LSTM, gru.GRU])
  def test_recurrent_dropout_saved_model(self, layer):
    if not tf.executing_eagerly():
      self.skipTest('v2-only test')
    inputs = keras.Input(shape=(784, 3), name='digits')
    x = layer(64, activation='relu', name='RNN', dropout=0.1)(inputs)
    x = keras.layers.Dense(64, activation='relu', name='dense')(x)
    outputs = keras.layers.Dense(
        10, activation='softmax', name='predictions')(
            x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer')
    model.save(os.path.join(self.get_temp_dir(), 'model'), save_format='tf')

  @parameterized.parameters([lstm.LSTM, gru.GRU])
  def test_ragged(self, layer):
    vocab_size = 100
    inputs = tf.ragged.constant(
        np.random.RandomState(0).randint(0, vocab_size, [128, 25]))
    embedder = embeddings.Embedding(input_dim=vocab_size, output_dim=16)
    embedded_inputs = embedder(inputs)
    layer = layer(32)
    layer(embedded_inputs)

  @parameterized.parameters([lstm.LSTM, gru.GRU])
  @test_utils.run_v2_only
  def test_compare_ragged_with_masks(self, layer):
    vocab_size = 100
    timestep = 20
    units = 32
    embedder = embeddings.Embedding(input_dim=vocab_size, output_dim=units)
    layer = layer(units, return_sequences=True)
    data = tf.constant(
        np.random.RandomState(0).randint(0, vocab_size, [timestep, timestep]))
    mask = tf.sequence_mask(tf.range(1, timestep + 1))
    data_ragged = tf.ragged.boolean_mask(data, mask)

    outputs = []
    devices = [test_utils.device(should_use_gpu=False)]
    if tf.test.is_gpu_available():
      devices.append(test_utils.device(should_use_gpu=True))
    for device in devices:
      with device:
        outputs.append(tf.boolean_mask(layer(embedder(data), mask=mask), mask))
        outputs.append(layer(embedder(data_ragged)).values)

    for i in range(len(outputs) - 1):
      self.assertAllClose(outputs[i], outputs[i + 1], atol=1e-4)


if __name__ == '__main__':
  tf.test.main()
