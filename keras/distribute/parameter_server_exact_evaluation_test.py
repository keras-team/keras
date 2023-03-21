# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for evaluation using Keras model and ParameterServerStrategy."""
import threading
import time

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
from tensorflow.python.platform import tf_logging as logging

import keras
from keras.metrics import base_metric
from keras.testing_infra import test_utils
from keras.utils import dataset_creator
from keras.utils import tf_utils

# isort: off
from tensorflow.python.distribute import (
    multi_worker_test_base,
)
from tensorflow.python.distribute.cluster_resolver import (
    SimpleClusterResolver,
)


def _aggregate_results(coordinator_metrics, results):
    for result in results:
        for metric in coordinator_metrics:
            if metric.name == "loss":
                continue
            assert metric.name in result.keys()
            metric_result = result[metric.name]
            assert len(metric_result) == len(metric.weights)
            for weight, val in zip(metric.weights, metric_result):
                weight.assign_add(val)
    return coordinator_metrics


@test_utils.run_v2_only
class ExactEvaluationTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        super(ExactEvaluationTest, self).setUp()
        self._cluster = multi_worker_test_base.create_multi_process_cluster(
            num_workers=5, num_ps=1, rpc_layer="grpc"
        )
        self._cluster_def = (
            self._cluster.cluster_resolver.cluster_spec().as_dict()
        )
        cluster_resolver = SimpleClusterResolver(
            tf.train.ClusterSpec(self._cluster_def), rpc_layer="grpc"
        )

        self.strategy = tf.distribute.experimental.ParameterServerStrategy(
            cluster_resolver
        )
        self.cluster_coord = (
            tf.distribute.experimental.coordinator.ClusterCoordinator(
                self.strategy
            )
        )

    def tearDown(self):
        super(ExactEvaluationTest, self).tearDown()
        self._cluster.stop()
        self._cluster = None

    def testDistributedMetrics(self):
        coordinator_metrics = [
            keras.metrics.AUC(),
            keras.metrics.MeanAbsoluteError(),
        ]

        def dataset_fn():
            y_true = np.concatenate((np.zeros(512), np.ones(512)))
            y_pred = np.concatenate(
                (np.linspace(0, 1, 512), np.linspace(0, 1, 512))
            )
            return tf.data.Dataset.from_tensor_slices((y_true, y_pred)).batch(1)

        @tf.function
        def eval_shard_fn(total_shard, shard_id, worker_dataset):
            with tf_utils.with_metric_local_vars_scope():
                worker_metrics = []
                for coord_metric in coordinator_metrics:
                    worker_metrics.append(
                        base_metric.clone_metric(coord_metric)
                    )

                dataset_shard = worker_dataset.shard(total_shard, shard_id)

                for value in dataset_shard:
                    for worker_metric in worker_metrics:
                        worker_metric.update_state(*value)

                return {
                    metric.name: metric.weights for metric in worker_metrics
                }

        per_worker_dataset = self.cluster_coord.create_per_worker_dataset(
            dataset_fn()
        )
        # Trigger dataset creation on workers without creating an iterator
        built_dataset = per_worker_dataset.build()

        # needs to be a tf.constant so it doesn't get re-traced each time
        # needs to be int64 because that's what Dataset.shard expects
        total_shards = tf.constant(100, dtype=tf.int64)

        result_remote_values = []
        logging.info("Scheduling eval closures")
        for i in tf.range(total_shards):
            result_remote_values.append(
                self.cluster_coord.schedule(
                    eval_shard_fn,
                    args=(total_shards, i, built_dataset),
                )
            )

        logging.info("Killing 2 workers")
        self._cluster.kill_task("worker", 0)
        self._cluster.kill_task("worker", 1)
        time.sleep(1)
        self._cluster.start_task("worker", 0)
        self._cluster.start_task("worker", 1)

        self.cluster_coord.join()
        results = [r.fetch() for r in result_remote_values]
        coordinator_metrics = _aggregate_results(coordinator_metrics, results)

        expected_results = {"auc": 0.5, "mean_absolute_error": 0.5}
        for metric in coordinator_metrics:
            self.assertAlmostEqual(
                metric.result().numpy(), expected_results[metric.name], places=5
            )

    def testModelAddMetricErrors(self):
        class MyModel(keras.Model):
            def call(self, x):
                self.add_metric(
                    tf.cast(x >= 0, tf.float32),
                    aggregation="sum",
                    name="num_positive",
                )
                return tf.cast(tf.add(x, 1), tf.float32)

        dataset = tf.data.Dataset.zip(
            (tf.data.Dataset.range(-5, 5), tf.data.Dataset.range(-4, 6))
        ).batch(1)
        with self.strategy.scope():
            model = MyModel()
            model.compile(
                metrics=[keras.metrics.Accuracy()],
                loss="binary_crossentropy",
                pss_evaluation_shards="auto",
            )

        # run a single train step to compile metrics
        model.fit(dataset, steps_per_epoch=1)
        with self.assertRaises(ValueError):
            model.evaluate(dataset, return_dict=True)

    def testModelInfiniteDatasetErrors(self):
        dataset = tf.data.Dataset.range(10).repeat()
        with self.strategy.scope():
            model = keras.Model()
            model.compile(pss_evaluation_shards="auto")
        with self.assertRaisesRegex(
            ValueError,
            "When performing exact evaluation, the dataset must "
            "be finite. Make sure not to call `repeat\(\)` on your "
            "dataset.",
        ):
            model.evaluate(dataset)

    def testTrainingWithVariablesCreatedInFunction(self):
        # When metrics are specified via string, they are instantiated in a
        # tf.function in the the first pass of the model when update_state is
        # called. This use case should not be affected by exact visitation
        # guarantee support.

        class MyModel(keras.Model):
            @tf.function
            def worker_fn(self, y_true, y_pred):
                self.compiled_metrics.update_state(y_true, y_pred)

        with self.strategy.scope():
            model = MyModel()
            model.compile(metrics=["accuracy"])

        y_true_0 = tf.convert_to_tensor([[0.0], [0.0], [0.0]])
        y_pred_0 = tf.convert_to_tensor([[0.0], [0.0], [1.0]])
        self.cluster_coord.schedule(model.worker_fn, args=(y_true_0, y_pred_0))

        y_true_1 = tf.convert_to_tensor([[0.0], [0.0], [0.0]])
        y_pred_1 = tf.convert_to_tensor([[0.0], [1.0], [1.0]])
        self.cluster_coord.schedule(model.worker_fn, args=(y_true_1, y_pred_1))

        self.cluster_coord.join()
        for metric in model.compiled_metrics.metrics:
            self.assertAlmostEqual(metric.result().numpy(), 0.5)

    @tf.__internal__.distribute.combinations.generate(
        tf.__internal__.test.combinations.combine(
            input_type=["dataset", "dataset_creator", "distributed_dataset"],
            eval_in_model_fit=[True, False],
            use_auto=[True, False],
            custom_metric=[True, False],
        )
    )
    def testDistributedModelEvaluation(
        self, input_type, eval_in_model_fit, use_auto, custom_metric
    ):

        # Define dataset by batch size, number of shards, and batches per shard
        batch_size = 16
        num_data_shards = 32
        batches_per_shard = 4
        num_examples = batch_size * num_data_shards * batches_per_shard

        # Input dataset x: just the sequence of numbers up to the dataset size
        # Input dataset y: defined such that each shard has index equal to the
        # number of y_i's == True in that shard
        expected_acc = sum(range(num_data_shards)) / num_examples

        # The predictions y_pred from this dummy model are fixed to True. This
        # way we can control the expected accuracy by just modifying y.
        class MyModel(keras.Model):
            def __call__(self, x, training=False):
                return tf.cast(x >= 0, tf.float32)

        def dataset_fn(input_context=None):
            del input_context
            x = np.arange(num_examples)

            def make_batch_with_n_true(n):
                return np.concatenate((np.ones(n), np.zeros(batch_size - n)))

            y = np.zeros(num_examples)
            batch_idxs = np.arange(num_examples // batch_size)
            for shard_idx in range(num_data_shards):
                num_correct = shard_idx
                # Dataset.shard uses mod sharding, so each shard consists of the
                # batches whose index mod (num_data_shards) = shard_idx
                batch_idxs_for_shard = np.where(
                    np.mod(batch_idxs, num_data_shards) == shard_idx
                )[0]
                for batch_idx in batch_idxs_for_shard:
                    # Select the individual data elements for this batch
                    batch_range = range(
                        batch_idx * batch_size, (batch_idx + 1) * batch_size
                    )
                    num_for_batch = min(num_correct, batch_size)
                    y[batch_range] = make_batch_with_n_true(num_for_batch)
                    num_correct -= num_for_batch

            dataset = tf.data.Dataset.from_tensor_slices((x, y))

            dataset = dataset.batch(batch_size)
            return dataset

        class CustomAccuracy(keras.metrics.Metric):
            def __init__(self, name="custom_acc", dtype=None):
                super().__init__(name, dtype)
                self.total = self.add_weight("total", initializer="zeros")
                self.count = self.add_weight("count", initializer="zeros")

            def update_state(self, y_true, y_pred, sample_weight=None):
                y_true = tf.cast(y_true, tf.float32)
                y_pred = tf.cast(y_pred, tf.float32)
                matches = tf.cast(tf.equal(y_true, y_pred), tf.float32)
                count = tf.reduce_sum(matches)
                self.count.assign_add(count)
                total = tf.cast(tf.size(y_true), tf.float32)
                self.total.assign_add(total)

            def result(self):
                return self.count / self.total

            def reset_state(self):
                self.total.assign(0)
                self.count.assign(0)

        def build_metric():
            metric = (
                CustomAccuracy() if custom_metric else keras.metrics.Accuracy()
            )
            return metric

        logging.info("Local evaluation (exact)")
        model = MyModel()
        model.compile(metrics=[build_metric()])
        ground_truth_evaluation = model.evaluate(dataset_fn())
        logging.info(
            "Result local evaluation (exact): %s", ground_truth_evaluation
        )
        self.assertAlmostEqual(ground_truth_evaluation[1], expected_acc)

        logging.info("Distributed evaluation (exact)")
        if use_auto:
            num_shards = "auto"
        else:
            num_shards = 5 * self.strategy._extended._num_workers

        with self.strategy.scope():
            model = MyModel()
            model.compile(
                metrics=[build_metric()],
                loss="binary_crossentropy",
                pss_evaluation_shards=num_shards,
            )

        if input_type == "dataset":
            train_dataset = dataset_fn()
            val_dataset = dataset_fn()
        elif input_type == "dataset_creator":
            train_dataset = dataset_creator.DatasetCreator(dataset_fn)
            val_dataset = dataset_creator.DatasetCreator(dataset_fn)
        elif input_type == "distributed_dataset":
            train_dataset = self.strategy.experimental_distribute_dataset(
                dataset_fn()
            )
            val_dataset = self.strategy.experimental_distribute_dataset(
                dataset_fn()
            )

        metric_name = "custom_acc" if custom_metric else "accuracy"
        expected_results = {metric_name: expected_acc}

        def kill_and_revive_in_thread(wait_secs=2):
            def _kill_and_revive_fn():
                time.sleep(wait_secs)
                logging.info("Killing 2 workers")
                self._cluster.kill_task("worker", 0)
                self._cluster.kill_task("worker", 1)
                time.sleep(1)
                self._cluster.start_task("worker", 0)
                self._cluster.start_task("worker", 1)

            restart_thread = threading.Thread(target=_kill_and_revive_fn)
            restart_thread.start()
            return restart_thread

        eval_results = {}
        if eval_in_model_fit:
            kill_and_revive_in_thread()
            history = model.fit(
                train_dataset,
                steps_per_epoch=1,
                validation_data=val_dataset,
            )
            logging.info(
                "History: params (%r), history (%r)",
                history.params,
                history.history,
            )
            eval_results = {
                metric.split("val_")[1]: val[-1]
                for metric, val in history.history.items()
                if metric.startswith("val_")
            }
        else:
            # run a single train step to compile metrics
            model.fit(train_dataset, steps_per_epoch=1)
            kill_and_revive_in_thread()
            eval_results = model.evaluate(val_dataset, return_dict=True)
            eval_results = {
                metric: val.numpy() for metric, val in eval_results.items()
            }
        for metric, val in eval_results.items():
            if "loss" not in metric:
                self.assertIn(metric, expected_results)
                self.assertAlmostEqual(val, expected_results[metric], places=5)


if __name__ == "__main__":
    tf.__internal__.distribute.multi_process_runner.test_main()
