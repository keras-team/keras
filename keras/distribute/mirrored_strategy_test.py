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

import tensorflow.compat.v2 as tf

from absl.testing import parameterized
import numpy as np

import keras
from tensorflow.python.eager import backprop
from keras.engine import training as keras_training
from keras.layers import core as keras_core
from keras.optimizer_v2 import rmsprop
from keras.utils import kpl_test_utils
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
class MirroredStrategyDefunTest(tf.test.TestCase, parameterized.TestCase):

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

  def testTrainAndServeWithKPL(self, distribution):
    use_adapt = False
    test_utils_obj = kpl_test_utils.DistributeKplTestUtils()
    with distribution.scope():
      feature_mapper, label_mapper = test_utils_obj.define_kpls_for_training(
          use_adapt)
      model = test_utils_obj.define_model()
      optimizer = rmsprop.RMSprop(learning_rate=0.1)
      accuracy = keras.metrics.Accuracy()

      def dataset_fn(_):
        return test_utils_obj.dataset_fn(feature_mapper, label_mapper)

      @tf.function
      def train_step(iterator):
        """The step function for one training step."""

        def step_fn(inputs):
          """The computation to run on each replica(GPU)."""
          features, labels = inputs
          with tf.GradientTape() as tape:
            pred = model(features, training=True)
            loss = keras.losses.binary_crossentropy(labels, pred)
            loss = tf.nn.compute_average_loss(loss)
          grads = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

          actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
          accuracy.update_state(labels, actual_pred)

        distribution.run(step_fn, args=(next(iterator),))

      distributed_dataset = distribution.distribute_datasets_from_function(
          dataset_fn)
      distributed_iterator = iter(distributed_dataset)
      num_epochs = 4
      num_steps = 7
      for _ in range(num_epochs):
        accuracy.reset_state()
        for _ in range(num_steps):
          train_step(distributed_iterator)

      self.assertGreater(accuracy.result().numpy(), 0.5)
      self.assertEqual(optimizer.iterations.numpy(), num_epochs * num_steps)

    # Test save/load/serving the trained model.
    test_utils_obj.test_save_load_serving_model(
        model, feature_mapper, test_utils_obj.define_reverse_lookup_layer())


if __name__ == "__main__":
  tf.test.main()
