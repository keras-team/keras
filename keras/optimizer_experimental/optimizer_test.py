"""Tests for the reworked optimizer.

More context in go/new-keras-optimizer
"""

import os

from absl.testing import parameterized
import keras
from keras.optimizer_experimental import adam as adam_new
from keras.optimizer_v2 import adam as adam_old
from keras.optimizer_v2 import learning_rate_schedule
from keras.utils import losses_utils
import numpy as np
import tensorflow.compat.v2 as tf

ds_combinations = tf.__internal__.distribute.combinations

STRATEGIES = [
    # TODO(b/202992598): Add PSS strategy once the XLA issues is resolved.
    ds_combinations.one_device_strategy,
    ds_combinations.mirrored_strategy_with_cpu_1_and_2,
    ds_combinations.mirrored_strategy_with_two_gpus,
    ds_combinations.tpu_strategy,
    ds_combinations.cloud_tpu_strategy,
    ds_combinations.multi_worker_mirrored_2x1_cpu,
    ds_combinations.multi_worker_mirrored_2x2_gpu,
    ds_combinations.central_storage_strategy_with_two_gpus,
]

adam_new_fn = tf.__internal__.test.combinations.NamedObject(
    "AdamExperimental", adam_new.Adam)

OPTIMIZER_FN = [
    adam_new_fn,
]


class OptimizerFuntionalityTest(tf.test.TestCase):
  """Test the functionality of optimizer."""

  def testAddVariableFromReference(self):
    optimizer = adam_new.Adam()
    variable = optimizer.add_variable_from_reference(
        tf.Variable(1.0, name="tmp"), "test")
    self.assertEqual(variable._shared_name, "test/tmp")
    self.assertEqual(self.evaluate(variable), 0)

  def testBuildIndexDict(self):
    optimizer = adam_new.Adam()
    var_list = [tf.Variable(0, name=f"var{i}") for i in range(10)]
    optimizer._build_index_dict(var_list)
    self.assertEqual(optimizer._index_dict[optimizer._var_key(var_list[7])], 7)

  def testGetConfig(self):
    optimizer = adam_new.Adam(
        learning_rate=np.float64(0.05),
        beta_1=0.7,
        beta_2=0.77,
        amsgrad=True,
        epsilon=0.001)
    config = optimizer.get_config()
    self.assertDictEqual(
        config, {
            "learning_rate": np.float32(0.05),
            "beta_1": 0.7,
            "beta_2": 0.77,
            "epsilon": 0.001,
            "amsgrad": True,
            "clipnorm": None,
            "clipvalue": None,
        })

  def testCheckpointOptimizer(self):
    x = tf.Variable([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    lr_schedule = learning_rate_schedule.ExponentialDecay(
        initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9)
    optimizer_1 = adam_new.Adam(
        learning_rate=lr_schedule, beta_1=0.8, beta_2=0.888)
    grads = tf.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])

    for _ in range(1):
      optimizer_1.apply_gradients(zip([grads], [x]))

    # Then save the variable and optimizer to a checkpoint.
    checkpoint_1 = tf.train.Checkpoint(var=x, optimizer=optimizer_1)
    checkpoint_path = checkpoint_1.save(self.get_temp_dir())

    # Create a new optimizer and call restore on it (and x)
    x2 = tf.Variable([[0., 0.], [0., 0.]], dtype=x.dtype)
    optimizer_2 = adam_new.Adam(learning_rate=0.02, beta_1=0.7, beta_2=0.777)
    optimizer_2.build([x2])
    checkpoint_2 = tf.train.Checkpoint(var=x2, optimizer=optimizer_2)
    checkpoint_2.restore(checkpoint_path)

    self.assertTrue(
        (self.evaluate(optimizer_1._momentums._storage[0]) == self.evaluate(
            optimizer_2._momentums._storage[0])).all())
    self.assertEqual(
        self.evaluate(optimizer_1._iterations),
        self.evaluate(optimizer_2._iterations))

  def disabletestSaveAndLoadOptimizerWithModel(self):
    model = keras.Sequential(
        [keras.layers.Input(shape=(1,)),
         keras.layers.Dense(1)])
    optimizer = adam_new.Adam(
        learning_rate=0.02, beta_1=0.8, beta_2=0.888)
    x = tf.expand_dims(tf.convert_to_tensor([1, 1, 1, 0, 0, 0]), axis=1)
    y = tf.expand_dims(tf.convert_to_tensor([1, 1, 1, 0, 0, 0]), axis=1)
    model.compile(loss="mse", optimizer=optimizer)
    model.fit(x, y)
    path = os.path.join(self.get_temp_dir(), "model")
    model.save(path)
    loaded_model = keras.models.load_model(path)
    loaded_model.load_weights(path)
    self.assertEqual(loaded_model.optimizer.learning_rate, 0.02)


class OptimizerRegressionTest(tf.test.TestCase, parameterized.TestCase):
  """Test optimizer outputs the same numerical results as optimizer_v2."""

  def _compare_numerical(self, old_optimizer, new_optimizer):
    tf.config.run_functions_eagerly(True)
    x1 = tf.Variable(np.ones([10]), dtype=tf.float64)
    x2 = tf.Variable(np.ones([10]), dtype=tf.float64)
    grads = tf.convert_to_tensor(np.arange(0.1, 1.1, 0.1))
    sparse_grads = tf.IndexedSlices(
        tf.convert_to_tensor([0, 0.2, 0.4, 0.8], dtype=tf.float64),
        [0, 2, 4, 6], dense_shape=[len(grads)])

    for _ in range(5):
      self.assertAllClose(x1, x2)
      old_optimizer.apply_gradients(zip([grads], [x1]))
      new_optimizer.apply_gradients(zip([grads], [x2]))

    for _ in range(5):
      self.assertAllClose(x1, x2)
      old_optimizer.apply_gradients(zip([sparse_grads], [x1]))
      new_optimizer.apply_gradients(zip([sparse_grads], [x2]))

  def testAdam(self):
    self._compare_numerical(
        adam_old.Adam(amsgrad=True), adam_new.Adam(amsgrad=True))


class DistributedTrainingTest(tf.test.TestCase, parameterized.TestCase):

  @ds_combinations.generate(
      tf.__internal__.test.combinations.combine(
          strategy=STRATEGIES,
          optimizer_fn=OPTIMIZER_FN))
  def testGetGradientsInModel(self, strategy, optimizer_fn):
    with strategy.scope():
      model = keras.Sequential(
          [keras.layers.Input(shape=(1,)),
           keras.layers.Dense(1)])
      optimizer = optimizer_fn()
      x = tf.expand_dims(tf.convert_to_tensor([1, 1, 1, 0, 0, 0]), axis=1)
      y = tf.expand_dims(tf.convert_to_tensor([1, 1, 1, 0, 0, 0]), axis=1)
      model.compile(loss="mse", optimizer=optimizer)
    model.fit(x, y, epochs=1, steps_per_epoch=5)
    # Assert the momentum variable is not 0.
    self.assertNotEqual(self.evaluate(optimizer._momentums._storage[0]), 0)

  @ds_combinations.generate(
      tf.__internal__.test.combinations.combine(
          strategy=STRATEGIES,
          optimizer_fn=OPTIMIZER_FN))
  def testGetGradientsInCustomTrainingLoop(self, strategy, optimizer_fn):
    with strategy.scope():
      model = keras.Sequential(
          [keras.layers.Input(shape=(1,)),
           keras.layers.Dense(1)])
      optimizer = optimizer_fn()

      def per_worker_dataset_fn():
        def dataset_fn(_):
          x, y = [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]
          ds = tf.data.Dataset.from_tensor_slices((x, y))
          ds = ds.repeat().batch(6)
          return ds
        return strategy.distribute_datasets_from_function(dataset_fn)
      ds = per_worker_dataset_fn()

      @tf.function
      def train_step(ds):
        def replica_fn(data):
          features, labels = data
          with tf.GradientTape() as tape:
            output = model(tf.expand_dims(features, axis=1))
            loss = keras.losses.MeanSquaredError(
                reduction=losses_utils.ReductionV2.NONE)(labels, output)
          grads = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(grads, model.trainable_variables))

        strategy.run(replica_fn, args=(next(iter(ds)),))

      for _ in range(3):
        train_step(ds)
    self.assertEqual(self.evaluate(optimizer.iterations), 3)


if __name__ == "__main__":
  tf.__internal__.distribute.multi_process_runner.test_main()
