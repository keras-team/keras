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
"""Tests for GRU V1 layer."""
# pylint: disable=g-direct-tensorflow-import

from absl.testing import parameterized
import keras
from keras.layers.rnn import gru
from keras.layers.rnn import gru_v1
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
from keras.utils import np_utils
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow.core.protobuf import rewriter_config_pb2


# Global config for grappler setting that is used for graph mode test.
_rewrites = rewriter_config_pb2.RewriterConfig()
_rewrites.implementation_selector = rewriter_config_pb2.RewriterConfig.ON
_rewrites.min_graph_nodes = -1
_graph_options = tf.compat.v1.GraphOptions(rewrite_options=_rewrites)
_config = tf.compat.v1.ConfigProto(graph_options=_graph_options)


@test_utils.run_all_without_tensor_float_32('RNN GRU can use TF32 on GPU')
@test_combinations.run_all_keras_modes(config=_config)
class GRUGraphRewriteTest(test_combinations.TestCase):

  @tf.test.disable_with_predicate(
      pred=tf.test.is_built_with_rocm,
      skip_message='Skipping as ROCm MIOpen does not support padded input yet.')
  @test_utils.run_v2_only
  def test_gru_feature_parity_v1_v2(self):
    input_shape = 10
    rnn_state_size = 8
    timestep = 4
    batch = 20

    (x_train, y_train), _ = test_utils.get_test_data(
        train_samples=batch,
        test_samples=0,
        input_shape=(timestep, input_shape),
        num_classes=rnn_state_size,
        random_seed=87654321)
    y_train = np_utils.to_categorical(y_train, rnn_state_size)
    # For the last batch item of the test data, we filter out the last
    # timestep to simulate the variable length sequence and masking test.
    x_train[-2:, -1, :] = 0.0
    y_train[-2:] = 0

    inputs = keras.layers.Input(
        shape=[timestep, input_shape], dtype=tf.float32)
    masked_input = keras.layers.Masking()(inputs)
    gru_layer = gru_v1.GRU(rnn_state_size,
                           recurrent_activation='sigmoid',
                           reset_after=True)
    output = gru_layer(masked_input)
    gru_model = keras.models.Model(inputs, output)
    weights = gru_model.get_weights()
    y_1 = gru_model.predict(x_train)
    gru_model.compile('rmsprop', 'mse')
    gru_model.fit(x_train, y_train)
    y_2 = gru_model.predict(x_train)

    with test_utils.device(should_use_gpu=True):
      cudnn_layer = gru.GRU(rnn_state_size,
                            recurrent_activation='sigmoid',
                            reset_after=True)
      cudnn_model = keras.models.Model(inputs, cudnn_layer(masked_input))
    cudnn_model.set_weights(weights)
    y_3 = cudnn_model.predict(x_train)
    cudnn_model.compile('rmsprop', 'mse')
    cudnn_model.fit(x_train, y_train)
    y_4 = cudnn_model.predict(x_train)

    self.assertAllClose(y_1, y_3, rtol=2e-5, atol=2e-5)
    self.assertAllClose(y_2, y_4, rtol=2e-5, atol=2e-5)

  @parameterized.named_parameters(
      # test_name, time_major, go_backwards
      ('normal', False, False),
      ('time_major', True, False),
      ('go_backwards', False, True),
      ('both', True, True),
  )
  def test_time_major_and_go_backward_v1_v2(self, time_major, go_backwards):
    input_shape = 10
    rnn_state_size = 8
    timestep = 4
    batch = 100

    x_train = np.random.random((batch, timestep, input_shape))

    def build_model(layer_cls):
      inputs = keras.layers.Input(
          shape=[timestep, input_shape], dtype=tf.float32)
      layer = layer_cls(rnn_state_size,
                        recurrent_activation='sigmoid',
                        time_major=time_major,
                        return_sequences=True,
                        go_backwards=go_backwards,
                        reset_after=True)
      if time_major:
        converted_input = keras.layers.Lambda(
            lambda t: tf.transpose(t, [1, 0, 2]))(inputs)
        outputs = layer(converted_input)
        outputs = keras.layers.Lambda(
            lambda t: tf.transpose(t, [1, 0, 2]))(outputs)
      else:
        outputs = layer(inputs)
      return keras.models.Model(inputs, outputs)

    gru_model = build_model(gru_v1.GRU)
    y_ref = gru_model.predict(x_train)
    weights = gru_model.get_weights()

    gru_v2_model = build_model(gru.GRU)
    gru_v2_model.set_weights(weights)
    y = gru_v2_model.predict(x_train)

    self.assertAllClose(y, y_ref)

  @tf.test.disable_with_predicate(
      pred=tf.test.is_built_with_rocm,
      skip_message='Skipping as ROCm MIOpen does not support padded input yet.')
  @test_utils.run_v2_only
  def test_explicit_device_with_go_backward_and_mask_v1(self):
    batch_size = 8
    timestep = 7
    masksteps = 5
    units = 4

    inputs = np.random.randn(batch_size, timestep, units).astype(np.float32)
    mask = np.ones((batch_size, timestep)).astype(np.bool)
    mask[:, masksteps:] = 0

    gru_layer = gru_v1.GRU(
        units, return_sequences=True, go_backwards=True)
    with test_utils.device(should_use_gpu=True):
      outputs_masked = gru_layer(inputs, mask=tf.constant(mask))
      outputs_trimmed = gru_layer(inputs[:, :masksteps])
    self.assertAllClose(outputs_masked[:, -masksteps:], outputs_trimmed)


if __name__ == '__main__':
  tf.test.main()
