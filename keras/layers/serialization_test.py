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
"""Tests for layer serialization utils."""

import tensorflow.compat.v2 as tf

from absl.testing import parameterized

import keras
from keras.testing_infra import test_combinations
from keras.layers.rnn import gru
from keras.layers.rnn import gru_v1
from keras.layers.rnn import lstm
from keras.layers.rnn import lstm_v1
from keras.layers.normalization import batch_normalization as batchnorm_v2
from keras.layers.normalization import batch_normalization_v1 as batchnorm_v1


class SerializableInt(int):

  def __new__(cls, value):
    return int.__new__(cls, value)

  def get_config(self):
    return {'value': int(self)}

  @classmethod
  def from_config(cls, config):
    return cls(**config)


@test_combinations.generate(test_combinations.combine(mode=['graph', 'eager']))
class LayerSerializationTest(parameterized.TestCase, tf.test.TestCase):

  def test_serialize_deserialize(self):
    layer = keras.layers.Dense(
        3, activation='relu', kernel_initializer='ones', bias_regularizer='l2')
    config = keras.layers.serialize(layer)
    new_layer = keras.layers.deserialize(config)
    self.assertEqual(new_layer.activation, keras.activations.relu)
    self.assertEqual(new_layer.bias_regularizer.__class__,
                     keras.regularizers.L2)
    if tf.__internal__.tf2.enabled():
      self.assertEqual(new_layer.kernel_initializer.__class__,
                       keras.initializers.OnesV2)
    else:
      self.assertEqual(new_layer.kernel_initializer.__class__,
                       keras.initializers.Ones)
    self.assertEqual(new_layer.units, 3)

  def test_implicit_serialize_deserialize_fails_without_object(self):
    layer = keras.layers.Dense(
        SerializableInt(3),
        activation='relu',
        kernel_initializer='ones',
        bias_regularizer='l2')
    config = keras.layers.serialize(layer)
    # Because we're passing an unknown class here, deserialization should fail
    # unless we add SerializableInt to the custom object dict.
    with self.assertRaisesRegex(ValueError,
                                'Unknown config_item: SerializableInt.*'):
      _ = keras.layers.deserialize(config)

  def test_implicit_serialize_deserialize_succeeds_with_object(self):
    layer = keras.layers.Dense(
        SerializableInt(3),
        activation='relu',
        kernel_initializer='ones',
        bias_regularizer='l2')
    config = keras.layers.serialize(layer)
    # Because we're passing an unknown class here, deserialization should fail
    # unless we add SerializableInt to the custom object dict.
    new_layer = keras.layers.deserialize(
        config, custom_objects={'SerializableInt': SerializableInt})
    self.assertEqual(new_layer.activation, keras.activations.relu)
    self.assertEqual(new_layer.bias_regularizer.__class__,
                     keras.regularizers.L2)
    if tf.__internal__.tf2.enabled():
      self.assertEqual(new_layer.kernel_initializer.__class__,
                       keras.initializers.OnesV2)
    else:
      self.assertEqual(new_layer.kernel_initializer.__class__,
                       keras.initializers.Ones)
    self.assertEqual(new_layer.units.__class__, SerializableInt)
    self.assertEqual(new_layer.units, 3)

  @parameterized.parameters(
      [batchnorm_v1.BatchNormalization, batchnorm_v2.BatchNormalization])
  def test_serialize_deserialize_batchnorm(self, batchnorm_layer):
    layer = batchnorm_layer(
        momentum=0.9, beta_initializer='zeros', gamma_regularizer='l2')
    config = keras.layers.serialize(layer)
    self.assertEqual(config['class_name'], 'BatchNormalization')
    new_layer = keras.layers.deserialize(config)
    self.assertEqual(new_layer.momentum, 0.9)
    if tf.__internal__.tf2.enabled():
      self.assertIsInstance(new_layer, batchnorm_v2.BatchNormalization)
      self.assertEqual(new_layer.beta_initializer.__class__,
                       keras.initializers.ZerosV2)
    else:
      self.assertIsInstance(new_layer, batchnorm_v1.BatchNormalization)
      self.assertEqual(new_layer.beta_initializer.__class__,
                       keras.initializers.Zeros)
    self.assertEqual(new_layer.gamma_regularizer.__class__,
                     keras.regularizers.L2)

  @parameterized.parameters(
      [batchnorm_v1.BatchNormalization, batchnorm_v2.BatchNormalization])
  def test_deserialize_batchnorm_backwards_compatibility(self, batchnorm_layer):
    layer = batchnorm_layer(
        momentum=0.9, beta_initializer='zeros', gamma_regularizer='l2')
    config = keras.layers.serialize(layer)
    new_layer = keras.layers.deserialize(config)
    self.assertEqual(new_layer.momentum, 0.9)
    if tf.__internal__.tf2.enabled():
      self.assertIsInstance(new_layer, batchnorm_v2.BatchNormalization)
      self.assertEqual(new_layer.beta_initializer.__class__,
                       keras.initializers.ZerosV2)
    else:
      self.assertIsInstance(new_layer, batchnorm_v1.BatchNormalization)
      self.assertEqual(new_layer.beta_initializer.__class__,
                       keras.initializers.Zeros)
    self.assertEqual(new_layer.gamma_regularizer.__class__,
                     keras.regularizers.L2)

  @parameterized.parameters([lstm_v1.LSTM, lstm.LSTM])
  def test_serialize_deserialize_lstm(self, layer):
    lstm_layer = layer(5, return_sequences=True)
    config = keras.layers.serialize(lstm_layer)
    self.assertEqual(config['class_name'], 'LSTM')
    new_layer = keras.layers.deserialize(config)
    self.assertEqual(new_layer.units, 5)
    self.assertEqual(new_layer.return_sequences, True)
    if tf.__internal__.tf2.enabled():
      self.assertIsInstance(new_layer, lstm.LSTM)
    else:
      self.assertIsInstance(new_layer, lstm_v1.LSTM)
      self.assertNotIsInstance(new_layer, lstm.LSTM)

  @parameterized.parameters([gru_v1.GRU, gru.GRU])
  def test_serialize_deserialize_gru(self, layer):
    gru_layer = layer(5, return_sequences=True)
    config = keras.layers.serialize(gru_layer)
    self.assertEqual(config['class_name'], 'GRU')
    new_layer = keras.layers.deserialize(config)
    self.assertEqual(new_layer.units, 5)
    self.assertEqual(new_layer.return_sequences, True)
    if tf.__internal__.tf2.enabled():
      self.assertIsInstance(new_layer, gru.GRU)
    else:
      self.assertIsInstance(new_layer, gru_v1.GRU)
      self.assertNotIsInstance(new_layer, gru.GRU)


if __name__ == '__main__':
  tf.test.main()
