# Lint as: python3
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
"""Tests for ClusterCoordinator and Keras models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

import random
import tempfile
from absl import logging
from absl.testing import parameterized
import numpy as np

import keras
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from keras import callbacks as callbacks_lib
from keras.engine import sequential
from keras.layers import core as core_layers
from keras.layers.preprocessing import string_lookup
from keras.optimizer_v2 import gradient_descent
from keras.optimizer_v2 import rmsprop
from keras.utils import dataset_creator
from keras.utils import losses_utils
from tensorflow.python.training.server_lib import ClusterSpec


# These vocabularies usually come from TFT or a Beam pipeline.
FEATURE_VOCAB = [
    "avenger", "ironman", "batman", "hulk", "spiderman", "kingkong",
    "wonder_woman"
]
LABEL_VOCAB = ["yes", "no"]


def make_cluster(num_workers, num_ps):
  cluster_def = multi_worker_test_base.create_in_process_cluster(
      num_workers=num_workers, num_ps=num_ps, rpc_layer="grpc")
  cluster_def["chief"] = [
      "localhost:%d" % multi_worker_test_base.pick_unused_port()
  ]
  return SimpleClusterResolver(ClusterSpec(cluster_def), rpc_layer="grpc")


def make_coordinator(num_workers, num_ps, variable_partitioner=None):
  return tf.distribute.experimental.coordinator.ClusterCoordinator(
      tf.distribute.experimental.ParameterServerStrategy(
          make_cluster(num_workers, num_ps),
          variable_partitioner=variable_partitioner))


# TODO(yuefengz): move this to keras/integration_tests.
class KPLTest(tf.test.TestCase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(KPLTest, cls).setUpClass()
    cls.coordinator = make_coordinator(num_workers=3, num_ps=2)

  def define_kpls_for_training(self, use_adapt):
    # Define KPLs under strategy's scope. Right now, if they have look up
    # tables, they will be created on the client. Their variables will be
    # created on PS. Ideally they should be cached on each worker since they
    # will not be changed in a training step.
    if use_adapt:
      feature_lookup_layer = string_lookup.StringLookup(num_oov_indices=1)
      feature_lookup_layer.adapt(FEATURE_VOCAB)
      label_lookup_layer = string_lookup.StringLookup(
          num_oov_indices=0, mask_token=None)
      label_lookup_layer.adapt(LABEL_VOCAB)
    else:
      # Do vocab shuffling.
      shuffled_vocab = FEATURE_VOCAB.copy()
      random.shuffle(shuffled_vocab)
      feature_lookup_layer = string_lookup.StringLookup(
          vocabulary=shuffled_vocab, num_oov_indices=1)
      label_lookup_layer = string_lookup.StringLookup(
          vocabulary=LABEL_VOCAB, num_oov_indices=0, mask_token=None)

    raw_feature_input = keras.layers.Input(
        shape=(3,), dtype=tf.string, name="feature", ragged=True)
    feature_id_input = feature_lookup_layer(raw_feature_input)

    # Model creates variables as well.
    feature_ps = keras.Model({"features": raw_feature_input}, feature_id_input)

    raw_label_input = keras.layers.Input(
        shape=(1,), dtype=tf.string, name="label")
    label_id_input = label_lookup_layer(raw_label_input)
    label_ps = keras.Model({"label": raw_label_input}, label_id_input)

    return feature_ps, label_ps

  def define_reverse_lookup_layer(self):
    # Only needed for serving.
    label_inverse_lookup_layer = string_lookup.StringLookup(
        num_oov_indices=0, mask_token=None, vocabulary=LABEL_VOCAB, invert=True)
    return label_inverse_lookup_layer

  @tf.__internal__.distribute.combinations.generate(
      tf.__internal__.test.combinations.combine(mode=["eager"], use_adapt=[True, False]))
  def testTrainAndServe(self, use_adapt):

    with self.coordinator.strategy.scope():

      feature_ps, label_ps = self.define_kpls_for_training(use_adapt)

      def dataset_fn():

        def feature_and_label_gen():
          while True:
            features = random.sample(FEATURE_VOCAB, 3)
            label = ["yes"] if "avenger" in features else ["no"]
            yield {"features": features, "label": label}

        # The dataset will be created on the coordinator.
        raw_dataset = tf.data.Dataset.from_generator(
            feature_and_label_gen,
            output_signature={
                "features": tf.TensorSpec([3], tf.string),
                "label": tf.TensorSpec([1], tf.string)
            }).shuffle(100).batch(32)

        train_dataset = raw_dataset.map(lambda x: (  # pylint: disable=g-long-lambda
            {
                "features": feature_ps(x["features"])
            }, label_ps(x["label"])))
        return train_dataset

      # Create the model. The input needs to be compatible with KPLs.
      model_input = keras.layers.Input(
          shape=(3,), dtype=tf.int64, name="model_input")

      # input_dim includes a mask token and an oov token.
      emb_output = keras.layers.Embedding(
          input_dim=len(FEATURE_VOCAB) + 2, output_dim=20)(
              model_input)
      emb_output = tf.reduce_mean(emb_output, axis=1)
      dense_output = keras.layers.Dense(
          units=1, activation="sigmoid")(
              emb_output)
      model = keras.Model({"features": model_input}, dense_output)

      optimizer = rmsprop.RMSprop(learning_rate=0.1)
      accuracy = keras.metrics.Accuracy()

    @tf.function
    def worker_fn(iterator):

      def replica_fn(iterator):
        batch_data, labels = next(iterator)
        with tf.GradientTape() as tape:
          pred = model(batch_data, training=True)
          loss = tf.nn.compute_average_loss(
              keras.losses.BinaryCrossentropy(
                  reduction=losses_utils.ReductionV2.NONE)(labels, pred))
          gradients = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
        accuracy.update_state(labels, actual_pred)

      self.coordinator.strategy.run(replica_fn, args=(iterator,))

    distributed_dataset = self.coordinator.create_per_worker_dataset(dataset_fn)
    distributed_iterator = iter(distributed_dataset)
    for _ in range(4):
      accuracy.reset_states()
      for _ in range(7):
        self.coordinator.schedule(worker_fn, args=(distributed_iterator,))
      self.coordinator.join()
    self.assertGreater(accuracy.result().numpy(), 0.5)

    # Create a saved model.
    model.feature_ps = feature_ps
    model.label_ps = label_ps
    model.label_inverse_lookup_layer = self.define_reverse_lookup_layer()

    def create_serving_signature(model):

      @tf.function
      def serve_fn(raw_features):
        raw_features = tf.compat.v1.expand_dims(raw_features, axis=0)
        transformed_features = model.feature_ps(raw_features)
        outputs = model(transformed_features)
        outputs = tf.compat.v1.squeeze(outputs, axis=0)
        outputs = tf.cast(tf.greater(outputs, 0.5), tf.int64)
        decoded_outputs = model.label_inverse_lookup_layer(outputs)
        return tf.compat.v1.squeeze(decoded_outputs, axis=0)

      # serving does NOT have batch dimension
      return serve_fn.get_concrete_function(
          tf.TensorSpec(
              shape=(3), dtype=tf.string, name="example"))

    serving_fn = create_serving_signature(model)

    saved_model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    model.save(saved_model_dir, signatures={"serving_default": serving_fn})

    # Test the saved_model.
    loaded_serving_fn = keras.saving.save.load_model(
        saved_model_dir).signatures["serving_default"]

    # check the result w/ and w/o avenger.
    prediction0 = loaded_serving_fn(
        tf.constant(["avenger", "ironman", "avenger"]))["output_0"]
    self.assertIn(prediction0, ("yes", "no"))

    prediction1 = loaded_serving_fn(
        tf.constant(["ironman", "ironman", "unkonwn"]))["output_0"]
    self.assertIn(prediction1, ("yes", "no"))


class ModelFitTest(tf.test.TestCase, parameterized.TestCase):

  def _model_compile(self,
                     steps_per_execution=1,
                     run_eagerly=False,
                     with_normalization_layer=False):

    class ResultAssertingCallback(callbacks_lib.Callback):

      def __init__(self):
        self._prev_epoch = -1

      def on_epoch_end(self, epoch, logs=None):
        logging.info("testModelFit: epoch=%r, logs=%r", epoch, logs)
        if epoch <= self._prev_epoch:
          raise RuntimeError("Epoch is supposed to be larger than previous.")
        self._prev_epoch = epoch
        if (logs.get("loss", None) is None or
            not isinstance(logs["loss"], np.floating)):
          raise RuntimeError("loss is supposed to be in the logs and float.")

    strategy = tf.distribute.experimental.ParameterServerStrategy(
        make_cluster(3, 2),
        variable_partitioner=tf.distribute.experimental.partitioners.FixedShardsPartitioner(2))
    with strategy.scope():
      model = sequential.Sequential([core_layers.Dense(10)])
      if with_normalization_layer:
        norm = keras.layers.BatchNormalization(
            axis=-1, input_shape=(4, 4, 3), momentum=0.8)
        model.add(norm)

    model.compile(
        gradient_descent.SGD(),
        loss="mse",
        steps_per_execution=steps_per_execution,
        run_eagerly=run_eagerly)
    return model, [ResultAssertingCallback()]

  def _model_fit(self,
                 steps_per_execution=1,
                 validation_data=None,
                 x=None,
                 steps_per_epoch=10,
                 run_eagerly=False,
                 with_normalization_layer=False):
    model, callbacks = self._model_compile(steps_per_execution, run_eagerly,
                                           with_normalization_layer)

    def dataset_fn(input_context):
      del input_context
      x = tf.random.uniform((10, 10))
      y = tf.random.uniform((10,))
      return tf.data.Dataset.from_tensor_slices(
          (x, y)).shuffle(10).repeat().batch(2)

    x = x or dataset_creator.DatasetCreator(dataset_fn)

    model.fit(
        x,
        epochs=10,
        steps_per_epoch=steps_per_epoch,
        verbose=0,
        callbacks=callbacks,
        validation_data=validation_data)
    return model

  @tf.__internal__.distribute.combinations.generate(tf.__internal__.test.combinations.combine(mode=["eager"]))
  def testModelFit(self):
    model = self._model_fit()
    self.assertEqual(model.optimizer.iterations, 100)
    return model

  @tf.__internal__.distribute.combinations.generate(tf.__internal__.test.combinations.combine(mode=["eager"]))
  def testModelFitWithNormalizationLayer(self):
    model = self._model_fit(with_normalization_layer=True)
    self.assertEqual(model.optimizer.iterations, 100)

  @tf.__internal__.distribute.combinations.generate(tf.__internal__.test.combinations.combine(mode=["eager"]))
  def testModelFitWithStepsPerExecution(self):
    model = self._model_fit(steps_per_execution=10)
    self.assertEqual(model.optimizer.iterations, 100)

  @tf.__internal__.distribute.combinations.generate(tf.__internal__.test.combinations.combine(mode=["eager"]))
  def testModelFitWithNoStepsPerEpoch(self):
    with self.assertRaisesRegex(
        ValueError, "`steps_per_epoch` must be specified with "
        "`ParameterServerStrategy`."):
      self._model_fit(steps_per_epoch=None)

  @tf.__internal__.distribute.combinations.generate(tf.__internal__.test.combinations.combine(mode=["eager"]))
  def testModelFitWithRunEagerly(self):
    with self.assertRaisesRegex(
        ValueError, "When using `Model` with `ParameterServerStrategy`, "
        "`run_eagerly` is not supported."):
      self._model_fit(run_eagerly=True)

  @tf.__internal__.distribute.combinations.generate(tf.__internal__.test.combinations.combine(mode=["eager"]))
  def testModelFitWithValidationData(self):
    with self.assertRaisesRegex(
        NotImplementedError, "Evaluation in `model.fit` with "
        "`ParameterServerStrategy` is not yet supported."):
      self._model_fit(
          validation_data=tf.data.Dataset.from_tensor_slices([1, 1]))

  @tf.__internal__.distribute.combinations.generate(tf.__internal__.test.combinations.combine(mode=["eager"]))
  def testModelFitWithDatasetInstance(self):
    with self.assertRaisesRegex(
        NotImplementedError, "Only `DatasetCreator` input is supported in "
        "`ParameterServerStrategy` at this time."):
      self._model_fit(x=tf.data.Dataset.from_tensor_slices([1, 1]))

  @tf.__internal__.distribute.combinations.generate(tf.__internal__.test.combinations.combine(mode=["eager"]))
  def testModelEvaluate(self):
    model, _ = self._model_compile()
    with self.assertRaisesRegex(
        NotImplementedError, "`model.evaluate` is not yet supported with "
        "`ParameterServerStrategy`."):
      model.evaluate(x=tf.data.Dataset.from_tensor_slices([1, 1]))

  @tf.__internal__.distribute.combinations.generate(tf.__internal__.test.combinations.combine(mode=["eager"]))
  def testModelPredict(self):
    model, _ = self._model_compile()
    with self.assertRaisesRegex(
        NotImplementedError, "`model.predict` is not yet supported with "
        "`ParameterServerStrategy`."):
      model.predict(x=tf.data.Dataset.from_tensor_slices([1, 1]))

  @tf.__internal__.distribute.combinations.generate(tf.__internal__.test.combinations.combine(mode=["eager"]))
  def testClusterCoordinatorSingleInstance(self):
    model = self._model_fit()
    strategy = model.distribute_strategy
    self.assertIs(strategy._cluster_coordinator,
                  tf.distribute.experimental.coordinator.ClusterCoordinator(strategy))


if __name__ == "__main__":
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
