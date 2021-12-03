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

from keras import callbacks as callbacks_lib
from keras.distribute import dataset_creator_model_fit_test_base as test_base
from keras.distribute import strategy_combinations
import tensorflow.compat.v2 as tf


@tf.__internal__.distribute.combinations.generate(
    tf.__internal__.test.combinations.combine(
        strategy=strategy_combinations.parameter_server_strategies_multi_worker,
        use_dataset_creator=[True, False],
        mode="eager"))
class DatasetCreatorModelFitParameterServerStrategyOnlyTest(
    test_base.DatasetCreatorModelFitTestBase):

  def testModelFitWithRunEagerly(self, strategy, use_dataset_creator):
    with self.assertRaisesRegex(
        ValueError, "When using `Model` with `ParameterServerStrategy`, "
        "`run_eagerly` is not supported."):
      self._model_fit(
          strategy, run_eagerly=True, use_dataset_creator=use_dataset_creator)

  def testModelPredict(self, strategy, use_dataset_creator):
    if use_dataset_creator:
      self.skipTest("Unused option.")
    model, _ = self._model_compile(strategy)
    test_data = tf.data.Dataset.from_tensor_slices(
        [[1.], [2.], [3.], [1.], [5.], [1.]]).repeat().batch(2)
    model.predict(x=test_data, steps=3)

  def testClusterCoordinatorSingleInstance(self, strategy, use_dataset_creator):
    model = self._model_fit(strategy, use_dataset_creator=use_dataset_creator)
    strategy = model.distribute_strategy
    self.assertIs(
        strategy._cluster_coordinator,
        tf.distribute.experimental.coordinator.ClusterCoordinator(strategy))

  def testModelFitErrorOnBatchLevelCallbacks(self, strategy,
                                             use_dataset_creator):

    class BatchLevelCallback(callbacks_lib.Callback):

      def on_train_batch_end(self, batch, logs=None):
        pass

    with self.assertRaisesRegex(ValueError,
                                "Batch-level `Callback`s are not supported"):
      callbacks = [BatchLevelCallback()]
      self._model_fit(
          strategy,
          callbacks=callbacks,
          use_dataset_creator=use_dataset_creator)

  def testModelFitCallbackSupportsTFLogs(self, strategy, use_dataset_creator):

    class MyCallback(callbacks_lib.Callback):

      def __init__(self):
        super(MyCallback, self).__init__()
        # Fetches the RemoteValues if necessary.
        self._supports_tf_logs = True

      def on_train_batch_end(self, batch, logs=None):
        assert isinstance(logs, tf.distribute.experimental.coordinator.RemoteValue)

    my_callback = MyCallback()
    callbacks = [my_callback]
    self._model_fit(
        strategy, callbacks=callbacks, use_dataset_creator=use_dataset_creator)

  def testModelFitVerbosity(self, strategy, use_dataset_creator):

    class MyCallback(callbacks_lib.Callback):
      pass

    my_callback = MyCallback()
    callbacks = [my_callback]
    self._model_fit(
        strategy, callbacks=callbacks, use_dataset_creator=use_dataset_creator)
    # PSStrategy should default to epoch-level logging.
    self.assertEqual(my_callback.params["verbose"], 2)

  def testModelFitTensorBoardEpochLevel(self, strategy, use_dataset_creator):
    log_dir = self.get_temp_dir()
    callbacks = [callbacks_lib.TensorBoard(log_dir)]
    self._model_fit(
        strategy, callbacks=callbacks, use_dataset_creator=use_dataset_creator)
    self.assertTrue(tf.compat.v1.gfile.Exists(log_dir))
    files = tf.compat.v1.gfile.ListDirectory(log_dir)
    self.assertGreaterEqual(len(files), 1)

  def testModelFitVerbose1(self, strategy, use_dataset_creator):
    with self.assertRaisesRegex(ValueError,
                                "`verbose=1` is not allowed with "
                                "`ParameterServerStrategy` for performance "
                                "reasons. Received: `verbose`=1"):
      self._model_fit(
          strategy, use_dataset_creator=use_dataset_creator,
          verbose=1)

  def testModelEvaluateErrorOnBatchLevelCallbacks(self, strategy,
                                                  use_dataset_creator):

    class BatchLevelCallback(callbacks_lib.Callback):

      def on_train_batch_end(self, batch, logs=None):
        pass

    with self.assertRaisesRegex(ValueError,
                                "Batch-level `Callback`s are not supported"):
      callbacks = [BatchLevelCallback()]
      self._model_evaluate(
          strategy,
          callbacks=callbacks,
          use_dataset_creator=use_dataset_creator)


if __name__ == "__main__":
  tf.compat.v1.enable_v2_behavior()
  tf.__internal__.distribute.multi_process_runner.test_main()
