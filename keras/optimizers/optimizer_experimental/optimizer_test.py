"""Tests for the reworked optimizer.

More context in go/new-keras-optimizer
"""

import os
import re

from absl import logging
from absl.testing import parameterized
import keras
from keras.optimizers import learning_rate_schedule
from keras.optimizers.optimizer_experimental import adadelta as adadelta_new
from keras.optimizers.optimizer_experimental import adagrad as adagrad_new
from keras.optimizers.optimizer_experimental import adam as adam_new
from keras.optimizers.optimizer_experimental import adamax as adamax_new
from keras.optimizers.optimizer_experimental import adamw as adamw_new
from keras.optimizers.optimizer_experimental import ftrl as ftrl_new
from keras.optimizers.optimizer_experimental import rmsprop as rmsprop_new
from keras.optimizers.optimizer_experimental import sgd as sgd_new
from keras.optimizers.optimizer_v2 import adadelta as adadelta_old
from keras.optimizers.optimizer_v2 import adagrad as adagrad_old
from keras.optimizers.optimizer_v2 import adam as adam_old
from keras.optimizers.optimizer_v2 import ftrl as ftrl_old
from keras.optimizers.optimizer_v2 import gradient_descent as sgd_old
from keras.optimizers.optimizer_v2 import rmsprop as rmsprop_old
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

adadelta_new_fn = tf.__internal__.test.combinations.NamedObject(
    "experimentaladadelta",
    lambda: adadelta_new.Adadelta(  # pylint: disable=g-long-lambda
        0.002,
        use_ema=True,
        ema_overwrite_frequency=None))
adagrad_new_fn = tf.__internal__.test.combinations.NamedObject(
    "experimentaladagrad", lambda: adagrad_new.Adagrad(0.002))
adam_new_fn = tf.__internal__.test.combinations.NamedObject(
    "experimentaladam", lambda: adam_new.Adam(0.002))
adamax_new_fn = tf.__internal__.test.combinations.NamedObject(
    "experimentaladamax", lambda: adamax_new.Adamax(0.002))
adamw_new_fn = tf.__internal__.test.combinations.NamedObject(
    "experimentaladamw", lambda: adamw_new.AdamW(0.002, weight_decay=0.004))
ftrl_new_fn = tf.__internal__.test.combinations.NamedObject(
    "experimentalftrl", lambda: ftrl_new.Ftrl(0.002))
rmsprop_new_fn = tf.__internal__.test.combinations.NamedObject(
    "experimentalrmsprop", lambda: rmsprop_new.RMSprop(0.002))
sgd_new_fn = tf.__internal__.test.combinations.NamedObject(
    "experimentalsgdaverage",
    lambda: sgd_new.SGD(  # pylint: disable=g-long-lambda
        0.002,
        use_ema=True,
        ema_overwrite_frequency=1))

OPTIMIZER_FN = [
    adadelta_new_fn,
    adagrad_new_fn,
    adam_new_fn,
    adamax_new_fn,
    adamw_new_fn,
    ftrl_new_fn,
    rmsprop_new_fn,
    sgd_new_fn,
]


class OptimizerFuntionalityTest(tf.test.TestCase, parameterized.TestCase):
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

  def testClipNorm(self):
    optimizer = adam_new.Adam(clipnorm=1)
    grad = [tf.convert_to_tensor([100.0, 100.0])]
    clipped_grad = optimizer._clip_gradients(grad)
    self.assertAllClose(clipped_grad[0], [2**0.5 / 2, 2**0.5 / 2])

  def testClipValue(self):
    optimizer = adam_new.Adam(clipvalue=1)
    grad = [tf.convert_to_tensor([100.0, 100.0])]
    clipped_grad = optimizer._clip_gradients(grad)
    self.assertAllEqual(clipped_grad[0], [1.0, 1.0])

  def testClipGlobalNorm(self):
    optimizer = adam_new.Adam(global_clipnorm=1)
    grad = [
        tf.cast([100.0, 100.0], dtype=tf.float32),
        tf.cast([100.0, 100.0], dtype=tf.float32)
    ]
    clipped_grad = optimizer._clip_gradients(grad)
    self.assertAllClose(clipped_grad[0], [0.5, 0.5])

  def testPassingLegacyArgsRaiseWarning(self):
    with self.assertLogs(level="WARNING") as log_output:
      logging.set_verbosity(logging.WARNING)
      _ = adam_new.Adam(clipnorm=1, decay=0.5)
      expected_log = "decay is deprecated in"
      output = log_output[0][0].message

      self.assertTrue(re.search(expected_log, output))

  def testPassingLegacyClipnorm(self):
    optimizer = adam_new.Adam(clipnorm=1)
    self.assertEqual(optimizer.clipnorm, 1)

  def testReturnAllOptimizerVariables(self):
    x = tf.Variable([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    optimizer = adam_new.Adam()
    grads = tf.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
    optimizer.apply_gradients(zip([grads], [x]))
    optimizer_variables = optimizer.variables
    all_names = [var._shared_name for var in optimizer_variables]
    self.assertLen(optimizer_variables, 4)
    self.assertCountEqual(
        all_names,
        ["iteration", "learning_rate", "Adam/m/Variable", "Adam/v/Variable"])

  def testSetLearningRate(self):
    optimizer = adam_new.Adam(learning_rate=1.0)
    self.assertIsInstance(optimizer._learning_rate, tf.Variable)
    self.assertEqual(self.evaluate(optimizer.learning_rate), 1.0)
    optimizer.learning_rate = 2.0
    self.assertEqual(self.evaluate(optimizer.learning_rate), 2.0)
    # Test the legacy setter.
    optimizer.lr = 3.0
    self.assertEqual(self.evaluate(optimizer.learning_rate), 3.0)

    lr_schedule = learning_rate_schedule.ExponentialDecay(
        initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9)
    optimizer = adam_new.Adam(learning_rate=lr_schedule)
    self.assertIsInstance(optimizer._learning_rate,
                          learning_rate_schedule.ExponentialDecay)
    self.assertEqual(optimizer.learning_rate, 0.01)
    # Test the legacy property.
    self.assertEqual(optimizer.lr, 0.01)

    x = tf.Variable([1.0, 2.0], dtype=tf.float32)
    grads = tf.convert_to_tensor([1.0, 2.0])
    for _ in range(2):
      optimizer.apply_gradients(zip([grads], [x]))
    self.assertTrue(optimizer.learning_rate < 0.01 and
                    optimizer.learning_rate > 0.00999)
    with self.assertRaisesRegex(TypeError, "This optimizer was created with*"):
      optimizer.learning_rate = 2.0

  def testSetIterations(self):
    optimizer = adam_new.Adam()
    optimizer.iterations = tf.Variable(2, dtype=tf.int32)
    self.assertEqual(optimizer.iterations, 2)
    var_list = [tf.Variable(2.0), tf.Variable(2.0)]
    grads = tf.convert_to_tensor([1.0, 1.0])
    optimizer.apply_gradients(zip(grads, var_list))
    self.assertEqual(optimizer.iterations, 3)
    with self.assertRaisesRegex(RuntimeError, "Cannot set*"):
      optimizer.iterations = 2

  def testPassingMissingWDError(self):
    with self.assertRaises(ValueError):
      _ = adamw_new.AdamW(0.01, weight_decay=None)

    with self.assertRaisesRegex(ValueError, "Missing value of"):
      _ = adamw_new.AdamW(0.01, weight_decay=None)

  def testMovingAverageOptimizer(self):
    optimizer = sgd_new.SGD(
        learning_rate=1,
        use_ema=True,
        ema_momentum=0.5,
        ema_overwrite_frequency=3)

    var1, var2 = tf.Variable(2.0), tf.Variable(2.0)
    with tf.GradientTape() as tape:
      loss = var1 + var2
    grads = tape.gradient(loss, [var1, var2])
    # First iteration: [var1, var2] = [1.0, 1.0]
    optimizer.apply_gradients(zip(grads, [var1, var2]))
    self.assertAllEqual([var1.numpy(), var2.numpy()], [1.0, 1.0])

    # Second iteration: [var1, var2] = [0.0, 0.0]
    optimizer.apply_gradients(zip(grads, [var1, var2]))
    self.assertAllEqual([var1.numpy(), var2.numpy()], [0.0, 0.0])

    # Third iteration, without EMA, we should see [var1, var2] = [-1.0, -1.0],
    # but overwriting results in [var1, var2] = [-0.125, -0.125].
    optimizer.apply_gradients(zip(grads, [var1, var2]))
    self.assertAllEqual([var1.numpy(), var2.numpy()], [-0.125, -0.125])

  def testGetAndFromConfig(self):
    optimizer = adam_new.Adam(
        learning_rate=np.float64(0.05),
        beta_1=0.7,
        beta_2=0.77,
        amsgrad=True,
        epsilon=0.001,
        clipnorm=0.5,
        use_ema=True,
        ema_momentum=0.5,
        ema_overwrite_frequency=50)
    config = optimizer.get_config()
    self.assertDictEqual(
        config, {
            "learning_rate": np.float32(0.05),
            "beta_1": 0.7,
            "beta_2": 0.77,
            "epsilon": 0.001,
            "amsgrad": True,
            "clipnorm": 0.5,
            "global_clipnorm": None,
            "clipvalue": None,
            "use_ema": True,
            "ema_momentum": 0.5,
            "ema_overwrite_frequency": 50,
            "jit_compile": False,
        })
    restored_optimizer = adam_new.Adam.from_config(config)
    self.assertDictEqual(restored_optimizer.get_config(),
                         optimizer.get_config())

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

  @parameterized.product(optimizer_fn=OPTIMIZER_FN)
  def testSaveAndLoadOptimizerWithModel(self, optimizer_fn):
    model = keras.Sequential(
        [keras.layers.Input(shape=(1,)),
         keras.layers.Dense(1)])
    optimizer = optimizer_fn()
    optimizer.clipnorm = 0.1
    x = tf.expand_dims(tf.convert_to_tensor([1, 1, 1, 0, 0, 0]), axis=1)
    y = tf.expand_dims(tf.convert_to_tensor([1, 1, 1, 0, 0, 0]), axis=1)
    model.compile(loss="mse", optimizer=optimizer)
    model.fit(x, y)

    # Save in h5 format.
    path = os.path.join(self.get_temp_dir(), "model.h5")
    model.save(path)
    loaded_model = keras.models.load_model(path)
    loaded_model.load_weights(path)
    loaded_optimizer = loaded_model.optimizer
    self.assertEqual(type(optimizer), type(loaded_optimizer))
    self.assertEqual(loaded_optimizer.learning_rate, 0.002)
    self.assertEqual(loaded_optimizer.clipnorm, 0.1)

    # Save in Keras SavedModel format.
    model.fit(x, y)
    path = os.path.join(self.get_temp_dir(), "model")
    model.save(path)
    loaded_model = keras.models.load_model(path)
    loaded_model.load_weights(path)
    loaded_optimizer = loaded_model.optimizer
    self.assertEqual(type(optimizer), type(loaded_optimizer))
    self.assertEqual(loaded_optimizer.learning_rate, 0.002)
    self.assertEqual(loaded_optimizer.clipnorm, 0.1)

  @parameterized.product(optimizer_fn=OPTIMIZER_FN)
  def testSparseGradientsWorkAsExpected(self, optimizer_fn):
    optimizer_1 = optimizer_fn()
    optimizer_2 = optimizer_fn()
    x1 = tf.Variable(np.ones([5]), dtype=tf.float64)
    x2 = tf.Variable(np.ones([5]), dtype=tf.float64)
    grads = tf.convert_to_tensor([0, 1., 1.5, 0, 0], dtype=tf.float64)
    sparse_grads = tf.IndexedSlices(
        tf.convert_to_tensor([1., 1.5], dtype=tf.float64),
        [1, 2],
        dense_shape=[len(grads)])
    for _ in range(5):
      optimizer_1.apply_gradients(zip([grads], [x1]))
      optimizer_2.apply_gradients(zip([sparse_grads], [x2]))
      self.assertAllClose(x1, x2)


class OptimizerRegressionTest(tf.test.TestCase, parameterized.TestCase):
  """Test optimizer outputs the same numerical results as optimizer_v2."""

  def _compare_numerical(self, old_optimizer, new_optimizer):
    x1 = tf.Variable(np.ones([10]), dtype=tf.float64)
    x2 = tf.Variable(np.ones([10]), dtype=tf.float64)
    grads = tf.convert_to_tensor(np.arange(0.1, 1.1, 0.1))
    sparse_grads = tf.IndexedSlices(
        tf.convert_to_tensor([0, 0.2, 0.4, 0.8], dtype=tf.float64),
        [0, 2, 4, 6],
        dense_shape=[len(grads)])

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

  def testAdadelta(self):
    self._compare_numerical(adadelta_old.Adadelta(), adadelta_new.Adadelta())

  def testAdagrad(self):
    self._compare_numerical(adagrad_old.Adagrad(), adagrad_new.Adagrad())

  def testFtrl(self):
    self._compare_numerical(ftrl_old.Ftrl(), ftrl_new.Ftrl())

  def testRMSprop(self):
    self._compare_numerical(rmsprop_old.RMSprop(), rmsprop_new.RMSprop())

  @parameterized.product(nesterov=[True, False])
  def testSgd(self, nesterov):
    self._compare_numerical(
        sgd_old.SGD(nesterov=nesterov), sgd_new.SGD(nesterov=nesterov))


class DistributedTrainingTest(tf.test.TestCase, parameterized.TestCase):

  @ds_combinations.generate(
      tf.__internal__.test.combinations.combine(
          strategy=STRATEGIES, optimizer_fn=OPTIMIZER_FN))
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
    if optimizer.name == "Adam":
      # Assert the momentum variable is not 0.
      self.assertNotEqual(self.evaluate(optimizer._momentums._storage[0]), 0)
    elif optimizer.name == "Adadelta":
      # Assert the accumulated variable is not 0.
      self.assertNotEqual(
          self.evaluate(optimizer._accumulated_grads._storage[0]), 0)
    elif optimizer.name == "Adagrad":
      # Assert the accumulated variable is not 0.
      self.assertNotEqual(self.evaluate(optimizer._accumulators._storage[0]), 0)

  @ds_combinations.generate(
      tf.__internal__.test.combinations.combine(
          strategy=STRATEGIES, optimizer_fn=OPTIMIZER_FN))
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

  @ds_combinations.generate(
      tf.__internal__.test.combinations.combine(strategy=[
          ds_combinations.mirrored_strategy_with_two_gpus,
          ds_combinations.tpu_strategy,
          ds_combinations.multi_worker_mirrored_2x2_gpu,
          ds_combinations.central_storage_strategy_with_two_gpus,
      ]))
  def testJitCompile(self, strategy):
    # Test the optimizer yields same numerical results when jit_compile is
    # on and off.
    with strategy.scope():
      optimizer_1 = adam_new.Adam(use_ema=True, ema_overwrite_frequency=1)
      optimizer_2 = adam_new.Adam(
          jit_compile=True, use_ema=True, ema_overwrite_frequency=1)
      model_1 = keras.Sequential([
          keras.layers.Input(shape=(2,)),
          keras.layers.Dense(5),
          keras.layers.Dense(1)
      ])
      model_2 = keras.models.clone_model(model_1)
      model_2.set_weights(model_1.get_weights())

      def per_worker_dataset_fn():

        def dataset_fn(_):
          x = np.random.rand(6, 2)
          y = [1, 1, 1, 0, 0, 0]
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
            output_1 = model_1(features)
            loss_1 = keras.losses.MeanSquaredError(
                reduction=losses_utils.ReductionV2.NONE)(labels, output_1)
          grads_1 = tape.gradient(loss_1, model_1.trainable_variables)
          optimizer_1.apply_gradients(zip(grads_1, model_1.trainable_variables))

          with tf.GradientTape() as tape:
            output_2 = model_2(features)
            loss_2 = keras.losses.MeanSquaredError(
                reduction=losses_utils.ReductionV2.NONE)(labels, output_2)
          grads_2 = tape.gradient(loss_2, model_2.trainable_variables)
          optimizer_2.apply_gradients(zip(grads_2, model_2.trainable_variables))

        strategy.run(replica_fn, args=(next(iter(ds)),))

      for _ in range(3):
        train_step(ds)
        self.assertAllClose(model_1.trainable_variables[0][0],
                            model_2.trainable_variables[0][0])


if __name__ == "__main__":
  tf.__internal__.distribute.multi_process_runner.test_main()
