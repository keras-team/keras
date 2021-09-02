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
"""Tests for keras premade models using tf.distribute.Strategy."""

from absl.testing import parameterized

from keras.engine import sequential
from keras.layers import core
from keras.optimizer_v2 import adagrad
from keras.optimizer_v2 import gradient_descent
from keras.premade import linear
from keras.premade import wide_deep
from keras.utils import dataset_creator
import numpy as np
import tensorflow.compat.v2 as tf


def strategy_combinations_eager_data_fn():
  return tf.__internal__.test.combinations.combine(
      distribution=[
          tf.__internal__.distribute.combinations.default_strategy,
          tf.__internal__.distribute.combinations.one_device_strategy,
          tf.__internal__.distribute.combinations.one_device_strategy_gpu,
          tf.__internal__.distribute.combinations
          .mirrored_strategy_with_gpu_and_cpu,
          tf.__internal__.distribute.combinations
          .mirrored_strategy_with_two_gpus,
          tf.__internal__.distribute.combinations
          .mirrored_strategy_with_two_gpus_no_merge_call,
          tf.__internal__.distribute.combinations.multi_worker_mirrored_2x1_cpu,
          tf.__internal__.distribute.combinations.multi_worker_mirrored_2x1_gpu,
          tf.__internal__.distribute.combinations.multi_worker_mirrored_2x2_gpu,
          tf.__internal__.distribute.combinations
          .parameter_server_strategy_1worker_2ps_cpu,
          tf.__internal__.distribute.combinations
          .parameter_server_strategy_1worker_2ps_1gpu,
          # NOTE: TPUStrategy not tested because the models in this test are
          # sparse and do not work with TPUs.
      ],
      use_dataset_creator=[True, False],
      mode=['eager'],
      data_fn=['numpy', 'dataset'])


INPUT_SIZE = 64
BATCH_SIZE = 10


def get_numpy():
  inputs = np.random.uniform(
      low=-5., high=5., size=(INPUT_SIZE, 2)).astype(np.float32)
  output = .3 * inputs[:, 0] + .2 * inputs[:, 1]
  return inputs, output


def get_dataset(input_context=None, batch_size=None):
  inputs, output = get_numpy()
  dataset = tf.data.Dataset.from_tensor_slices((inputs, output))
  if input_context:
    dataset = dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)
  if batch_size is None:
    batch_size = BATCH_SIZE

  dataset = dataset.batch(batch_size).repeat(200)
  return dataset


# A `dataset_fn` is required for `Model.fit` to work across all strategies.
def dataset_fn(input_context):
  batch_size = input_context.get_per_replica_batch_size(
      global_batch_size=BATCH_SIZE)
  return get_dataset(input_context, batch_size)


class KerasPremadeModelsTest(tf.test.TestCase, parameterized.TestCase):

  @tf.__internal__.distribute.combinations.generate(
      strategy_combinations_eager_data_fn())
  def test_linear_model(self, distribution, use_dataset_creator, data_fn):
    if ((not use_dataset_creator) and isinstance(
        distribution, tf.distribute.experimental.ParameterServerStrategy)):
      self.skipTest(
          'Parameter Server strategy requires dataset creator to be used in '
          'model.fit.')
    if (not tf.__internal__.tf2.enabled() and use_dataset_creator
        and isinstance(distribution,
                       tf.distribute.experimental.ParameterServerStrategy)):
      self.skipTest(
          'Parameter Server strategy with dataset creator needs to be run when '
          'eager execution is enabled.')
    with distribution.scope():
      model = linear.LinearModel()
      opt = gradient_descent.SGD(learning_rate=0.1)
      model.compile(opt, 'mse')
      if use_dataset_creator:
        x = dataset_creator.DatasetCreator(dataset_fn)
        hist = model.fit(x, epochs=5, steps_per_epoch=INPUT_SIZE)
      else:
        if data_fn == 'numpy':
          inputs, output = get_numpy()
          hist = model.fit(inputs, output, epochs=5)
        else:
          hist = model.fit(get_dataset(), epochs=5)
        self.assertLess(hist.history['loss'][4], 0.2)

  @tf.__internal__.distribute.combinations.generate(
      strategy_combinations_eager_data_fn())
  def test_wide_deep_model(self, distribution, use_dataset_creator, data_fn):
    if ((not use_dataset_creator) and isinstance(
        distribution, tf.distribute.experimental.ParameterServerStrategy)):
      self.skipTest(
          'Parameter Server strategy requires dataset creator to be used in '
          'model.fit.')
    if (not tf.__internal__.tf2.enabled() and use_dataset_creator
        and isinstance(distribution,
                       tf.distribute.experimental.ParameterServerStrategy)):
      self.skipTest(
          'Parameter Server strategy with dataset creator needs to be run when '
          'eager execution is enabled.')
    with distribution.scope():
      linear_model = linear.LinearModel(units=1)
      dnn_model = sequential.Sequential([core.Dense(units=1)])
      wide_deep_model = wide_deep.WideDeepModel(linear_model, dnn_model)
      linear_opt = gradient_descent.SGD(learning_rate=0.05)
      dnn_opt = adagrad.Adagrad(learning_rate=0.1)
      wide_deep_model.compile(optimizer=[linear_opt, dnn_opt], loss='mse')

      if use_dataset_creator:
        x = dataset_creator.DatasetCreator(dataset_fn)
        hist = wide_deep_model.fit(x, epochs=5, steps_per_epoch=INPUT_SIZE)
      else:
        if data_fn == 'numpy':
          inputs, output = get_numpy()
          hist = wide_deep_model.fit(inputs, output, epochs=5)
        else:
          hist = wide_deep_model.fit(get_dataset(), epochs=5)
      self.assertLess(hist.history['loss'][4], 0.2)


if __name__ == '__main__':
  tf.__internal__.distribute.multi_process_runner.test_main()
