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
"""Tests training metrics with PSS distribution strategy."""

import numpy as np
import tensorflow.compat.v2 as tf

from keras import layers as layers_module
from keras import metrics as metrics_module
from keras.engine import training as training_module
from keras.testing_infra import test_combinations

# isort: off
from tensorflow.python.distribute import (
    multi_process_runner,
    multi_worker_test_base,
)


class ParameterServerTrainingMetricTest(test_combinations.TestCase):
    """Test Parameter Server Distribution strategy with Keras Model Training"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cluster = multi_worker_test_base.create_multi_process_cluster(
            num_workers=2, num_ps=3, rpc_layer="grpc"
        )
        cls.cluster_resolver = cls.cluster.cluster_resolver

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.cluster.stop()

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_pss_fit_metric_batch_counter(self):
        """Verify that metric data is complete during fit when using
        ParameterServerStrategy
        """
        strategy = tf.distribute.ParameterServerStrategy(
            self.cluster_resolver,
            variable_partitioner=None,
        )

        class BatchCount(metrics_module.Sum):
            def __init__(self, name="batch_count", dtype=tf.int64):
                super().__init__(name=name, dtype=dtype)

            def update_state(self, y_true, y_pred, sample_weight=None):
                return super().update_state(1, sample_weight)

        # Build and compile model within strategy scope.
        with strategy.scope():
            inputs = layers_module.Input((1,))
            outputs = layers_module.Dense(1)(inputs)
            model = training_module.Model(inputs, outputs)
            model.compile(
                loss="mse", metrics=[BatchCount()], steps_per_execution=2
            )

        BATCH_SIZE = 10
        x, y = np.ones((400, 1)), np.ones((400, 1))
        val_x, val_y = np.ones((100, 1)), np.ones((100, 1))
        train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        train_dataset = train_dataset.batch(BATCH_SIZE)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        val_dataset = val_dataset.batch(BATCH_SIZE)
        train_batch_count = x.shape[0] // BATCH_SIZE
        val_batch_count = val_x.shape[0] // BATCH_SIZE
        # Verify that Model fit doesn't drop any batches
        hist = model.fit(
            train_dataset,
            steps_per_epoch=train_batch_count,
            validation_data=val_dataset,
            validation_steps=val_batch_count,
            epochs=5,
        )
        # Verify that min and max value of batch count metric is accurate
        self.assertEqual(max(hist.history["batch_count"]), train_batch_count)
        self.assertEqual(min(hist.history["batch_count"]), train_batch_count)
        self.assertEqual(max(hist.history["val_batch_count"]), val_batch_count)
        self.assertEqual(min(hist.history["val_batch_count"]), val_batch_count)

    @test_combinations.run_all_keras_modes(always_skip_v1=True)
    def test_pss_evaluate_metric_batch_counter(self):
        """Verify that metric data is complete during evaluate when using
        ParameterServerStrategy
        """
        strategy = tf.distribute.ParameterServerStrategy(
            self.cluster_resolver,
            variable_partitioner=None,
        )

        class BatchCount(metrics_module.Sum):
            def __init__(self, name="batch_count", dtype=tf.int64):
                super().__init__(name=name, dtype=dtype)

            def update_state(self, y_true, y_pred, sample_weight=None):
                return super().update_state(1, sample_weight)

        # Build and compile model within strategy scope.
        with strategy.scope():
            inputs = layers_module.Input((1,))
            outputs = layers_module.Dense(1)(inputs)
            model = training_module.Model(inputs, outputs)
            model.compile(
                loss="mse", metrics=[BatchCount()], steps_per_execution=2
            )

        BATCH_SIZE = 10
        x, y = np.ones((400, 1)), np.ones((400, 1))
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        batch_count = x.shape[0] // BATCH_SIZE
        # Verify that Model Eval batch counter metric is accurate.
        eval_results = model.evaluate(dataset, steps=batch_count)
        self.assertEqual(eval_results[-1], batch_count)


if __name__ == "__main__":
    tf.enable_v2_behavior()
    multi_process_runner.test_main()
