# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for `DatasetCreator` with `Model.fit` across usages and strategies."""

import tensorflow.compat.v2 as tf
from keras import callbacks as callbacks_lib
from keras.distribute import dataset_creator_model_fit_test_base as test_base
from keras.distribute import strategy_combinations


@tf.__internal__.distribute.combinations.generate(
    tf.__internal__.test.combinations.combine(
        strategy=strategy_combinations.all_strategies +
        strategy_combinations.multi_worker_mirrored_strategies +
        ["ParameterServerStrategy"],
        mode="eager"))
class DatasetCreatorModelFitTest(test_base.DatasetCreatorModelFitTestBase):

  def testModelFit(self, strategy):
    model = self._model_fit(strategy)
    self.assertEqual(model.optimizer.iterations, 100)

  def testModelFitWithLookupLayer(self, strategy):
    model = self._model_fit(strategy, use_lookup_layer=True)
    self.assertEqual(model.optimizer.iterations, 100)

  def testModelFitWithNormalizationLayer(self, strategy):
    model = self._model_fit(strategy, with_normalization_layer=True)
    self.assertEqual(model.optimizer.iterations, 100)

  def testModelFitWithStepsPerExecution(self, strategy):
    model = self._model_fit(strategy, steps_per_execution=10)
    self.assertEqual(model.optimizer.iterations, 100)

  def testModelFitWithNoStepsPerEpoch(self, strategy):
    with self.assertRaisesRegex(
        ValueError, "When using a "
        "`tf.keras.utils.experimental.DatasetCreator`, `steps_per_epoch`, "
        "`validation_steps` or `steps` argument must be provided in "
        "`Model.fit` or `Model.evaluate`."):
      self._model_fit(strategy, steps_per_epoch=None)


@tf.__internal__.distribute.combinations.generate(
    tf.__internal__.test.combinations.combine(strategy=["ParameterServerStrategy"], mode="eager"))
class DatasetCreatorModelEvaluateParameterServerStrategyOnlyTest(
    test_base.DatasetCreatorModelFitTestBase):

  def testModelEvaluate(self, strategy):
    self._model_evaluate(strategy)
    self.assertGreaterEqual(self._metric.result(), 0.0)

  def testModelEvaluateWithNormalizationLayer(self, strategy):
    self._model_evaluate(strategy, with_normalization_layer=True)
    self.assertGreaterEqual(self._metric.result(), 0.0)

  def testModelEvaluateWithStepsPerExecution(self, strategy):
    self._model_evaluate(strategy, steps_per_execution=10)
    self.assertGreaterEqual(self._metric.result(), 0.0)

  def testModelEvaluateWithNoStepsPerEpoch(self, strategy):
    with self.assertRaisesRegex(
        ValueError, "When using a "
        "`tf.keras.utils.experimental.DatasetCreator`, `steps_per_epoch`, "
        "`validation_steps` or `steps` argument must be provided in "
        "`Model.fit` or `Model.evaluate`."):
      self._model_evaluate(strategy, steps=None)

  def testModelEvaluateWithDatasetInstance(self, strategy):
    with self.assertRaisesRegex(
        NotImplementedError,
        "Only `tf.keras.utils.experimental.DatasetCreator` input is supported "
        "with `ParameterServerStrategy` at this time. Please see "
        "`tf.keras.utils.experimental.DatasetCreator` class docstring for more "
        "information."
    ):
      self._model_evaluate(
          strategy,
          validation_data=tf.data.Dataset.from_tensor_slices([1, 1]))

  def testModelFitErrorOnBatchLevelCallbacks(self, strategy):

    class BatchLevelCallback(callbacks_lib.Callback):

      def on_train_batch_end(self, batch, logs=None):
        pass

    with self.assertRaisesRegex(ValueError,
                                "Batch-level `Callback`s are not supported"):
      callbacks = [BatchLevelCallback()]
      self._model_evaluate(strategy, callbacks=callbacks)


if __name__ == "__main__":
  tf.compat.v1.enable_v2_behavior()
  tf.__internal__.distribute.multi_process_runner.test_main()
