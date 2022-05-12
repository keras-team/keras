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

import time

import tensorflow.compat.v2 as tf
from typing import List, Tuple

import numpy as np
import keras
from keras.metrics import base_metric
from keras.testing_infra import test_utils
from tensorflow.python.platform import tf_logging as logging

# isort: off
from tensorflow.python.distribute import (
    multi_worker_test_base,)
from tensorflow.python.distribute.cluster_resolver import (
    SimpleClusterResolver,)
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.distribute.coordinator import coordinator_context


# TODO(yuefengz): move the following implementation to Keras core.
class CompositeMetricWrapperSpec(tf.TypeSpec):

  def __init__(self, serialize_config, weights):
    self._serialize_config = serialize_config
    self._weights = weights

  def _serialize(self):
    return (self._serialize_config, self._weights)

  @property
  def value_type(self):
    return CompositeMetricWrapper

  @property
  def _component_specs(self):
    return self._weights

  def _to_components(self, value):
    return value.metric.weights

  def _from_components(self, weights):
    counter = [0]

    def fetch_variable(next_creator, **kwargs):
      del next_creator, kwargs
      # TODO(yuefengz): verify the var creation order matches the weights
      # property
      var = weights[counter[0]]
      counter[0] += 1
      return var

    with tf.variable_creator_scope(fetch_variable):
      ret = keras.metrics.deserialize(self._serialize_config)

    assert len(weights) == len(ret.weights)
    return CompositeMetricWrapper(ret)


class CompositeMetricWrapper(tf.__internal__.CompositeTensor):

  def __init__(self, metric: keras.metrics.Metric):
    self.metric = metric

  def element_spec(self):
    raise NotImplementedError("element_spec not implemented")

  @property
  def _type_spec(self):

    def get_spec(w):
      if isinstance(w, tf.Tensor):
        return tf.TensorSpec.from_tensor(w)
      else:
        return resource_variable_ops.VariableSpec.from_value(w)

    weight_specs = [get_spec(w) for w in self.metric.weights]
    return CompositeMetricWrapperSpec(
        keras.metrics.serialize(self.metric), weight_specs)

  def merge_state(self, metrics: List["CompositeMetricWrapper"]) -> None:
    self.metric.merge_state([x.metric for x in metrics])

  def result(self):
    return self.metric.result()

  def update_state(self, values, sample_weight=None):
    return self.metric.update_state(values, sample_weight)

  def reset_states(self):
    self.metric.reset_states()


@test_utils.run_v2_only
class EvaluationTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(EvaluationTest, cls).setUpClass()
    cls._cluster = multi_worker_test_base.create_multi_process_cluster(
        num_workers=5, num_ps=1, rpc_layer="grpc")
    cls._cluster_def = cls._cluster.cluster_resolver.cluster_spec().as_dict()
    cluster_resolver = SimpleClusterResolver(
        tf.train.ClusterSpec(cls._cluster_def), rpc_layer="grpc")

    cls.strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver)
    cls.cluster_coord = tf.distribute.experimental.coordinator.ClusterCoordinator(
        cls.strategy)

  @classmethod
  def tearDownClass(cls):
    cls._cluster.stop()
    cls._cluster = None
    super(EvaluationTest, cls).tearDownClass()

  def testPassMetricToTfFunction(self):
    metric1 = CompositeMetricWrapper(keras.metrics.Mean())
    metric2 = CompositeMetricWrapper(keras.metrics.Mean())

    self.assertEqual(metric1.result(), 0.0)
    self.assertEqual(metric2.result(), 0.0)

    tf.nest.assert_same_structure(
        metric1, metric2._type_spec, expand_composites=True)
    tf.nest.assert_same_structure(
        metric1._type_spec, metric2, expand_composites=True)

    @tf.function
    def func(m):
      m.update_state([1.0, 2.0])

    func(metric1)
    self.assertEqual(metric1.result(), 1.5)
    self.assertEqual(metric2.result(), 0.0)

    concrete_f = func.get_concrete_function(metric1._type_spec)
    concrete_f(metric2)
    self.assertEqual(metric1.result(), 1.5)
    self.assertEqual(metric2.result(), 1.5)

  def testDistributedMean(self):

    def metric_fn():
      # TODO(gbm): Test with AUC.
      return CompositeMetricWrapper(keras.metrics.Mean())

    # TODO(yuefengz): make _create_per_worker_resources public and get rid
    # of the type_spec hack.
    per_worker_metric = self.cluster_coord._create_per_worker_resources(
        metric_fn)

    metric_on_coordinator = metric_fn()
    for metric_remote_value in per_worker_metric._values:
      metric_remote_value._type_spec = metric_on_coordinator._type_spec

    def dataset_fn():
      return tf.data.Dataset.range(1024)

    # TODO(yuefengz): integrate it into model.evaluate.
    @tf.function
    def eval_fn(total_shard, shard_id, metric):

      metric.reset_states()
      dataset_shard = dataset_fn().shard(total_shard, shard_id)
      for value in dataset_shard:
        metric.update_state(value)

      return metric

    total_shards = 100
    result_remote_values = []
    for i in range(total_shards):
      result_remote_values.append(
          self.cluster_coord.schedule(
              eval_fn, args=(total_shards, i, per_worker_metric)))

    self._cluster.kill_task("worker", 0)
    self._cluster.kill_task("worker", 1)
    time.sleep(1)
    self._cluster.start_task("worker", 0)
    self._cluster.start_task("worker", 1)

    results = [r.fetch() for r in result_remote_values]
    metric_on_coordinator.merge_state(results)
    self.assertEqual(metric_on_coordinator.result(), 511.5)

  def testDistributedModelEvaluation(self):

    class MyModel(keras.Model):

      def __call__(self, x, training=False):
        return x >= 0.5

    num_examples = 1000
    batch_size = 10

    def dataset_fn(context=None, repeat=False):
      x = np.linspace(0.0, 1.0, num_examples)
      y = x >= 0.75
      dataset = tf.data.Dataset.from_tensor_slices((x, y))

      if context is not None:
        # Split the dataset among the workers.
        worker_idx, num_workers = _get_worker_idx_and_num_workers(context)
        assert num_workers == 5
        dataset = dataset.shard(num_shards=num_workers, index=worker_idx)

      dataset = dataset.batch(batch_size)
      if repeat:
        dataset = dataset.repeat(None)
      return dataset

    def build_metric():
      return keras.metrics.Accuracy()

    logging.info("Local evaluation (exact)")
    model = MyModel()
    model.compile(metrics=[build_metric()])
    ground_truth_evaluation = model.evaluate(dataset_fn())
    logging.info("Result local evaluation (exact): %s", ground_truth_evaluation)
    # The accuracy should be exactly 75%.

    logging.info("Distributed evaluation (non exact v1)")
    with self.strategy.scope():
      model = MyModel()
      model.compile(metrics=[build_metric()])
      dataset = self.strategy.distribute_datasets_from_function(
          lambda context: dataset_fn(None))
    evaluation = model.evaluate(dataset, steps=num_examples // batch_size)
    logging.info("Result distributed evaluation (non exact v1): %s", evaluation)

    logging.info("Distributed evaluation (non exact v2)")
    with self.strategy.scope():
      model = MyModel()
      model.compile(metrics=[build_metric()])
      dataset = self.strategy.distribute_datasets_from_function(
          lambda context: dataset_fn(context, repeat=True))
    evaluation = model.evaluate(dataset, steps=num_examples // batch_size)
    logging.info("Result distributed evaluation (non exact v2): %s", evaluation)

    logging.info("Distributed evaluation (non exact v3)")
    with self.strategy.scope():
      model = MyModel()
      model.compile(metrics=[build_metric()])
      dataset = self.strategy.distribute_datasets_from_function(
          lambda context: dataset_fn(context, repeat=True))
    evaluation = model.evaluate(dataset, steps=num_examples // batch_size)
    logging.info("Result distributed evaluation (non exact v3): %s", evaluation)

    logging.info("Distributed evaluation (exact)")
    with self.strategy.scope():
      model = MyModel()
      # TODO(gbm): Figure why model.compile does not register the metrics in
      # model.metrics here. When figured, remove the metric argument from
      # "_evaluate_exact_distributed" and extract the metric from the model.
      model.compile(metrics=[build_metric()])
      logging.info("Model metrics: %s", model.metrics)

      # V1
      # dataset = self.strategy.distribute_datasets_from_function(
      #     lambda context: dataset_fn(None))

      # V2
      dataset = dataset_fn(None)

    evaluation = _evaluate_exact_distributed(model, dataset, build_metric())
    logging.info("Result distributed evaluation (exact): %s", evaluation)


def _evaluate_exact_distributed(model, dataset, metric):
  """Prototype version of keras' evaluate for this specific case."""

  def metric_fn():
    return CompositeMetricWrapper(base_metric.clone_metric(metric))

  cluster_coord = (
      tf.distribute.experimental.coordinator.ClusterCoordinator(
          model.distribute_strategy))

  # TODO(yuefengz): make _create_per_worker_resources public and get rid of
  # the type_spec hack.
  per_worker_metric = cluster_coord._create_per_worker_resources(metric_fn)

  metric_on_coordinator = metric_fn()
  for metric_remote_value in per_worker_metric._values:
    metric_remote_value._type_spec = metric_on_coordinator._type_spec

  @tf.function
  def eval_shard_fn(total_shard, shard_id,
                    worker_metric):  # ,worker_dataset (V1)

    worker_metric.reset_states()

    # V1
    # worker_dataset._dataset_fn = lambda: worker_dataset._dataset_fn().shard(
    #    total_shard, shard_id)
    # dataset_shard = worker_dataset
    # Current error:
    # RuntimeError: __iter__() is not supported inside of tf.function or in graph mode.

    # V2
    dataset_shard = dataset.shard(total_shard, shard_id)
    # Current error:
    # Additional GRPC error information from remote target /job:worker/replica:0/task:3:
    # :{"created":"@1653298017.499700382","description":"Error received from peer ipv6:[::1]:26623","file":"third_party/grpc/src/core/lib/surface/call.cc","file_line":967,"grpc_message":"Unable to parse tensor proto","grpc_status":3} [Op:__inference_eval_shard_fn_3082]

    for batch in dataset_shard:
      # TODO(gbm): Use "data_adapter".
      x, y = batch

      # TODO(gbm): Make sure the model does not rely on the PS server at ever
      # step.
      y_pred = model(x)

      worker_metric.update_state(y, y_pred)
    return worker_metric

  # V1
  # with model.distribute_strategy.scope():
  #   per_worker_dataset = cluster_coord.create_per_worker_dataset(dataset)
  # per_worker_iter = iter(per_worker_dataset)

  # TODO(gbm): 10 shards per workers. Make this value adaptative.
  total_shards = 10 * model.distribute_strategy._extended._num_workers
  result_remote_values = []
  for i in range(total_shards):
    result_remote_values.append(
        cluster_coord.schedule(
            eval_shard_fn,
            args=(
                total_shards, i, per_worker_metric
                #, per_worker_dataset (V1)
                #,dataset._dataset_fn(None) (V3)
            )))

  results = [r.fetch() for r in result_remote_values]
  metric_on_coordinator.merge_state(results)
  return [0.0, metric_on_coordinator.result()]


def _get_worker_idx_and_num_workers(context) -> Tuple[int, int]:

  # TODO(gbm): Use the context worker index and num when populated.
  if not tf.inside_function():
    raise ValueError("Should be called in a function")

  def call_time_worker_index():
    dispatch_context = coordinator_context.get_current_dispatch_context()
    return dispatch_context.worker_index

  worker_index = tf.compat.v1.get_default_graph().capture_call_time_value(
      call_time_worker_index, tf.TensorSpec([], dtype=tf.dtypes.int64))
  worker_index.op._set_attr(  # pylint: disable=protected-access
      "_user_specified_name",
      tf.compat.v1.AttrValue(s=tf.compat.as_bytes("worker_index")))

  return worker_index, tf.distribute.get_strategy()._extended._num_workers


if __name__ == "__main__":
  tf.__internal__.distribute.multi_process_runner.test_main()
