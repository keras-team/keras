# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for MirroredStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np
from tensorflow.python.eager import backprop
from keras.engine import training as keras_training
from keras.layers import core as keras_core
from tensorflow.python.training import optimizer as optimizer_lib


class MiniModel(keras_training.Model):
  """Minimal model for mnist.

  Useful for testing and debugging on slow TPU simulators.
  """

  def __init__(self):
    super(MiniModel, self).__init__(name="")
    self.fc = keras_core.Dense(1, name="fc", kernel_initializer="ones",
                               bias_initializer="ones")

  def call(self, inputs, training=True):
    inputs = tf.ones([1, 10])
    return self.fc(inputs)


@tf.__internal__.distribute.combinations.generate(
    tf.__internal__.test.combinations.combine(
        distribution=[
            tf.__internal__.distribute.combinations.mirrored_strategy_with_gpu_and_cpu,
        ],
        mode=["eager"]))
class MirroredStrategyDefunTest(tf.test.TestCase):

  def testTrain(self, distribution):
    with distribution.scope():
      mock_model = MiniModel()
      mock_model.call = tf.function(mock_model.call)

      def loss_fn(ctx):
        del ctx
        return mock_model(tf.ones([1, 10]))

      gradients_fn = backprop.implicit_grad(loss_fn)
      gradients_fn = optimizer_lib.get_filtered_grad_fn(gradients_fn)
      grads_and_vars = distribution.extended.call_for_each_replica(
          gradients_fn, args=(None,))

      optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.25)
      update_ops = optimizer._distributed_apply(distribution, grads_and_vars)  # pylint: disable=protected-access

      if not tf.executing_eagerly():
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.evaluate(update_ops)

      updated_var_values = self.evaluate(mock_model.variables)
      # All variables start at 1.0 and get two updates of 0.25.
      self.assertAllEqual(0.5 * np.ones([10, 1]), updated_var_values[0])
      self.assertAllEqual([0.5], updated_var_values[1])


if __name__ == "__main__":
  tf.test.main()
