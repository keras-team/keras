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
"""Tests for LSTM V1 layer."""
# pylint: disable=g-direct-tensorflow-import

import time

from absl.testing import parameterized
import keras
from keras.layers.rnn import lstm
from keras.layers.rnn import lstm_v1
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
from keras.utils import np_utils
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.platform import tf_logging as logging


# Global config for grappler setting that is used for graph mode test.
_rewrites = rewriter_config_pb2.RewriterConfig()
_rewrites.implementation_selector = rewriter_config_pb2.RewriterConfig.ON
_rewrites.min_graph_nodes = -1
_graph_options = tf.compat.v1.GraphOptions(rewrite_options=_rewrites)
_config = tf.compat.v1.ConfigProto(graph_options=_graph_options)


@test_combinations.run_all_keras_modes(config=_config)
class LSTMGraphRewriteTest(test_combinations.TestCase):

  @tf.test.disable_with_predicate(
      pred=tf.test.is_built_with_rocm,
      skip_message='Skipping as ROCm MIOpen does not support padded input yet.')
  @test_utils.run_v2_only
  def test_lstm_feature_parity_v1_v2(self):
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
    lstm_layer = lstm_v1.LSTM(rnn_state_size, recurrent_activation='sigmoid')
    output = lstm_layer(masked_input)
    lstm_model = keras.models.Model(inputs, output)
    weights = lstm_model.get_weights()
    y_1 = lstm_model.predict(x_train)
    lstm_model.compile('rmsprop', 'mse')
    lstm_model.fit(x_train, y_train)
    y_2 = lstm_model.predict(x_train)

    with test_utils.device(should_use_gpu=True):
      cudnn_layer = lstm.LSTM(rnn_state_size)
      cudnn_model = keras.models.Model(inputs, cudnn_layer(masked_input))
    cudnn_model.set_weights(weights)
    y_3 = cudnn_model.predict(x_train)
    cudnn_model.compile('rmsprop', 'mse')
    cudnn_model.fit(x_train, y_train)
    y_4 = cudnn_model.predict(x_train)

    self.assertAllClose(y_1, y_3, rtol=1e-5, atol=2e-5)
    self.assertAllClose(y_2, y_4, rtol=1e-5, atol=2e-5)

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
                        go_backwards=go_backwards)
      if time_major:
        converted_input = keras.layers.Lambda(
            lambda t: tf.transpose(t, [1, 0, 2]))(inputs)
        outputs = layer(converted_input)
        outputs = keras.layers.Lambda(
            lambda t: tf.transpose(t, [1, 0, 2]))(outputs)
      else:
        outputs = layer(inputs)
      return keras.models.Model(inputs, outputs)

    lstm_model = build_model(lstm_v1.LSTM)
    y_ref = lstm_model.predict(x_train)
    weights = lstm_model.get_weights()

    lstm_v2_model = build_model(lstm.LSTM)
    lstm_v2_model.set_weights(weights)
    y = lstm_v2_model.predict(x_train)

    self.assertAllClose(y, y_ref)

    input_shape = 10
    rnn_state_size = 8
    output_shape = 8
    timestep = 4
    batch = 100
    epoch = 10

    (x_train, y_train), _ = test_utils.get_test_data(
        train_samples=batch,
        test_samples=0,
        input_shape=(timestep, input_shape),
        num_classes=output_shape)
    y_train = np_utils.to_categorical(y_train, output_shape)

    layer = lstm.LSTM(rnn_state_size)

    inputs = keras.layers.Input(
        shape=[timestep, input_shape], dtype=tf.float32)

    outputs = layer(inputs)
    model = keras.models.Model(inputs, outputs)
    model.compile('rmsprop', loss='mse')
    model.fit(x_train, y_train, epochs=epoch)
    model.evaluate(x_train, y_train)
    model.predict(x_train)

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

    lstm_v1_layer = lstm_v1.LSTM(
        units, return_sequences=True, go_backwards=True)
    with test_utils.device(should_use_gpu=True):
      outputs_masked_v1 = lstm_v1_layer(inputs, mask=tf.constant(mask))
      outputs_trimmed_v1 = lstm_v1_layer(inputs[:, :masksteps])
    self.assertAllClose(outputs_masked_v1[:, -masksteps:], outputs_trimmed_v1)


class LSTMPerformanceTest(tf.test.Benchmark):

  def _measure_performance(self, test_config, model, x_train, y_train):
    batch = test_config['batch']
    epoch = test_config['epoch']
    warmup_epoch = test_config['warmup_epoch']

    # warm up the model
    model.fit(x_train, y_train, batch_size=batch, epochs=warmup_epoch)
    start_time = time.time()
    model.fit(x_train, y_train, batch_size=batch, epochs=epoch - warmup_epoch)
    end_time = time.time()
    return (end_time - start_time) / (epoch - warmup_epoch)

  def _time_performance_run_cudnn_lstm(self, test_config, x_train, y_train):
    # Get the performance number for standard Cudnn LSTM
    input_shape = test_config['input_shape']
    rnn_state_size = test_config['rnn_state_size']
    timestep = test_config['timestep']

    cudnn_lstm_layer = keras.layers.CuDNNLSTM(rnn_state_size)
    inputs = keras.layers.Input(
        shape=[timestep, input_shape], dtype=tf.float32)

    outputs = cudnn_lstm_layer(inputs)
    model = keras.models.Model(inputs, outputs)
    model.compile('sgd', 'mse')

    sec_per_epoch = self._measure_performance(
        test_config, model, x_train, y_train)
    logging.info('Average performance for %s per epoch is: %s',
                 'CuDNN LSTM', sec_per_epoch)
    return sec_per_epoch

  def _time_performance_run_unifed_lstm_gpu(
      self, test_config, x_train, y_train):
    # Get performance number for lstm_v2 with grappler swap the impl
    input_shape = test_config['input_shape']
    rnn_state_size = test_config['rnn_state_size']
    timestep = test_config['timestep']

    layer = keras.layers.LSTM(rnn_state_size)
    inputs = keras.layers.Input(
        shape=[timestep, input_shape], dtype=tf.float32)

    outputs = layer(inputs)
    model = keras.models.Model(inputs, outputs)
    model.compile('sgd', 'mse')

    sec_per_epoch = self._measure_performance(
        test_config, model, x_train, y_train)
    logging.info('Average performance for %s per epoch is: %s',
                 'LSTM V2', sec_per_epoch)
    return sec_per_epoch

  def _time_performance_run_normal_lstm(
      self, test_config, x_train, y_train):
    # Get performance number for standard LSTM on GPU.
    input_shape = test_config['input_shape']
    rnn_state_size = test_config['rnn_state_size']
    timestep = test_config['timestep']

    layer = lstm_v1.LSTM(rnn_state_size)
    inputs = keras.layers.Input(
        shape=[timestep, input_shape], dtype=tf.float32)

    outputs = layer(inputs)
    model = keras.models.Model(inputs, outputs)
    model.compile('sgd', 'mse')

    sec_per_epoch = self._measure_performance(
        test_config, model, x_train, y_train)
    logging.info('Average performance for %s per epoch is: %s',
                 'Normal LSTM', sec_per_epoch)
    return sec_per_epoch

  def _benchmark_performance_with_standard_cudnn_impl(self):
    if not tf.test.is_gpu_available():
      self.skipTest('performance test will only run on GPU')

    mode = 'eager' if tf.executing_eagerly() else 'graph'
    batch = 64
    num_batch = 10
    test_config = {
        'input_shape': 128,
        'rnn_state_size': 64,
        'output_shape': 64,
        'timestep': 50,
        'batch': batch,
        'epoch': 20,
        # The performance for warmup epoch is ignored.
        'warmup_epoch': 1,
    }
    (x_train, y_train), _ = test_utils.get_test_data(
        train_samples=(batch * num_batch),
        test_samples=0,
        input_shape=(test_config['timestep'], test_config['input_shape']),
        num_classes=test_config['output_shape'])
    y_train = np_utils.to_categorical(y_train, test_config['output_shape'])

    cudnn_sec_per_epoch = self._time_performance_run_cudnn_lstm(
        test_config, x_train, y_train)
    lstm_v2_sec_per_epoch = self._time_performance_run_unifed_lstm_gpu(
        test_config, x_train, y_train)
    normal_lstm_sec_per_epoch = self._time_performance_run_normal_lstm(
        test_config, x_train, y_train)

    cudnn_vs_v2 = cudnn_sec_per_epoch / lstm_v2_sec_per_epoch
    v2_vs_normal = normal_lstm_sec_per_epoch / lstm_v2_sec_per_epoch

    self.report_benchmark(name='keras_cudnn_lstm_' + mode,
                          wall_time=cudnn_sec_per_epoch,
                          iters=test_config['epoch'],
                          extras=test_config)
    self.report_benchmark(name='keras_lstm_v2_' + mode,
                          wall_time=lstm_v2_sec_per_epoch,
                          iters=test_config['epoch'],
                          extras=test_config)
    self.report_benchmark(name='keras_canonical_lstm_' + mode,
                          wall_time=normal_lstm_sec_per_epoch,
                          iters=test_config['epoch'],
                          extras=test_config)

    logging.info('Expect the performance of LSTM V2 is within 80% of '
                 'cuDNN LSTM, got {0:.2f}%'.format(cudnn_vs_v2 * 100))
    logging.info('Expect the performance of LSTM V2 is more than 5 times'
                 ' of normal LSTM, got {0:.2f}'.format(v2_vs_normal))

  def benchmark_performance_graph(self):
    with tf.compat.v1.get_default_graph().as_default():
      with tf.compat.v1.Session(config=_config):
        self._benchmark_performance_with_standard_cudnn_impl()

  def benchmark_performance_eager(self):
    with tf.__internal__.eager_context.eager_mode():
      self._benchmark_performance_with_standard_cudnn_impl()


if __name__ == '__main__':
  tf.test.main()
