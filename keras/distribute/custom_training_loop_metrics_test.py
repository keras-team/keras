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
"""Tests for custom training loops."""

import tensorflow.compat.v2 as tf

from absl.testing import parameterized
import numpy as np
from tensorflow.python.framework import test_util
from keras import metrics
from keras.distribute import strategy_combinations


class KerasMetricsTest(tf.test.TestCase, parameterized.TestCase):

  @tf.__internal__.distribute.combinations.generate(
      tf.__internal__.test.combinations.combine(
          distribution=strategy_combinations.all_strategies +
          strategy_combinations.multiworker_strategies,
          mode=["eager"]
      ))
  def test_multiple_keras_metrics_experimental_run(self, distribution):
    with distribution.scope():
      loss_metric = metrics.Mean("loss", dtype=np.float32)
      loss_metric_2 = metrics.Mean("loss_2", dtype=np.float32)

    @tf.function
    def train_step():
      def step_fn():
        loss = tf.constant(5.0, dtype=np.float32)
        loss_metric.update_state(loss)
        loss_metric_2.update_state(loss)

      distribution.run(step_fn)

    train_step()
    self.assertEqual(loss_metric.result().numpy(),
                     loss_metric_2.result().numpy())
    self.assertEqual(loss_metric.result().numpy(), 5.0)

  @tf.__internal__.distribute.combinations.generate(
      tf.__internal__.test.combinations.combine(
          distribution=strategy_combinations.all_strategies+
          strategy_combinations.multiworker_strategies,
          mode=["eager"]
      ))
  def test_update_keras_metric_declared_in_strategy_scope(self, distribution):
    with distribution.scope():
      metric = metrics.Mean("test_metric", dtype=np.float32)

    dataset = tf.data.Dataset.range(10).batch(2)
    dataset = distribution.experimental_distribute_dataset(dataset)

    @tf.function
    def step_fn(i):
      metric.update_state(i)

    for i in dataset:
      distribution.run(step_fn, args=(i,))

    # This should be the mean of integers 0-9 which has a sum of 45 and a count
    # of 10 resulting in mean of 4.5.
    self.assertEqual(metric.result().numpy(), 4.5)

  @tf.__internal__.distribute.combinations.generate(
      tf.__internal__.test.combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def test_update_keras_metric_outside_strategy_scope_cross_replica(
      self, distribution):
    metric = metrics.Mean("test_metric", dtype=np.float32)

    with distribution.scope():
      for i in range(10):
        metric.update_state(i)

    # This should be the mean of integers 0-9 which has a sum of 45 and a count
    # of 10 resulting in mean of 4.5.
    self.assertEqual(metric.result().numpy(), 4.5)

  @tf.__internal__.distribute.combinations.generate(
      tf.__internal__.test.combinations.combine(
          distribution=strategy_combinations.all_strategies, mode=["eager"]))
  @test_util.disable_mlir_bridge("TODO(b/168036682): Support dynamic padder")
  def test_update_keras_metrics_dynamic_shape(self, distribution):
    with distribution.scope():
      metric = metrics.Mean("test_metric", dtype=np.float32)

    dataset = tf.data.Dataset.range(10).batch(2, drop_remainder=False)

    @tf.function
    def train_fn(dataset):
      weights = tf.constant([0.1, 0.1])

      def step_fn(i):
        metric.update_state(i, weights)

      for i in dataset:
        distribution.run(step_fn, args=(i,))

    train_fn(dataset)

    # This should be the mean of integers 0-9 which has a sum of 45 and a count
    # of 10 resulting in mean of 4.5.
    self.assertEqual(metric.result().numpy(), 4.5)


if __name__ == "__main__":
  tf.__internal__.distribute.multi_process_runner.test_main()
