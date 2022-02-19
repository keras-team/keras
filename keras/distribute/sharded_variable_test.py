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

import keras
from keras.distribute import multi_worker_testing_utils
from keras.engine import base_layer
import tensorflow.compat.v2 as tf


class ShardedVariableTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.strategy = tf.distribute.experimental.ParameterServerStrategy(
        multi_worker_testing_utils.make_parameter_server_cluster(3, 2),
        variable_partitioner=tf.distribute.experimental.partitioners
        .FixedShardsPartitioner(2))

  def assert_list_all_equal(self, list1, list2):
    """Used in lieu of `assertAllEqual`.

    This is used to replace standard `assertAllEqual` for the cases where
    `list1` and `list2` contain `AggregatingVariable`. Lists with
    `AggregatingVariable` are not convertible to numpy array via `np.array`
    calls as numpy would raise `ValueError: setting an array element with a
    sequence.`

    Args:
      list1: The first list to compare equality.
      list2: The second list to compare equality.
    """
    for lhs, rhs in zip(list1, list2):
      self.assertEqual(lhs, rhs)

  def test_keras_layer_setattr(self):

    class Layer(base_layer.Layer):

      def __init__(self):
        super().__init__()
        self.w = tf.Variable([0, 1])
        self.b = tf.Variable([2, 3], trainable=False)

    with self.strategy.scope():
      layer = Layer()

    self.assertLen(layer.trainable_weights, 2)
    self.assertEqual(layer.trainable_weights[0], [0])
    self.assertEqual(layer.trainable_weights[1], [1])
    self.assertLen(layer.non_trainable_weights, 2)
    self.assertEqual(layer.non_trainable_weights[0], [2])
    self.assertEqual(layer.non_trainable_weights[1], [3])
    self.assert_list_all_equal(
        layer.weights, layer.trainable_weights + layer.non_trainable_weights)
    self.assert_list_all_equal(layer.trainable_weights,
                               layer.trainable_variables)
    self.assert_list_all_equal(layer.weights, layer.variables)

    checkpoint_deps = set(layer._trackable_children().values())
    self.assertEqual(checkpoint_deps, set([layer.w, layer.b]))

  def test_keras_layer_add_weight(self):

    class Layer(base_layer.Layer):

      def __init__(self):
        super().__init__()
        self.w = self.add_weight(
            shape=(2,),
            initializer=lambda shape, dtype: tf.constant([0., 1.],),
            trainable=True)
        self.b = self.add_weight(
            shape=(2,),
            initializer=lambda shape, dtype: tf.constant([2., 3.]),
            trainable=False)

    with self.strategy.scope():
      layer = Layer()

    self.assertLen(layer.trainable_weights, 2)
    self.assertEqual(layer.trainable_weights[0], [0.])
    self.assertEqual(layer.trainable_weights[1], [1.])
    self.assertLen(layer.non_trainable_weights, 2)
    self.assertEqual(layer.non_trainable_weights[0], [2.])
    self.assertEqual(layer.non_trainable_weights[1], [3.])
    self.assert_list_all_equal(
        layer.weights, layer.trainable_weights + layer.non_trainable_weights)
    self.assert_list_all_equal(layer.trainable_weights,
                               layer.trainable_variables)
    self.assert_list_all_equal(layer.weights, layer.variables)

    checkpoint_deps = set(layer._trackable_children().values())
    self.assertEqual(checkpoint_deps, set([layer.w, layer.b]))

  def test_keras_metrics(self):
    with self.strategy.scope():
      fp = keras.metrics.FalsePositives(thresholds=[0.2, 0.5, 0.7, 0.8])
      auc = keras.metrics.AUC(num_thresholds=10)

    @tf.function
    def update():
      fp.update_state([0., 1., 0., 0.], [0., 0., 0.3, 0.9])
      auc.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])

    @tf.function
    def reset():
      fp.reset_state()
      auc.reset_state()

    update()
    self.assertEqual(auc.result(), 0.75)
    self.assertAllEqual(fp.result(), [2., 1., 1., 1.])
    reset()
    self.assertEqual(auc.result(), 0.0)
    self.assertAllEqual(fp.result(), [0., 0., 0., 0.])

    self.assertTrue(hasattr(auc.true_positives, 'variables'))
    self.assertTrue(hasattr(fp.accumulator, 'variables'))

  def test_saved_model(self):

    def create_model():
      inputs = keras.layers.Input(shape=(4,))
      outputs = keras.layers.Dense(2)(inputs)
      model = keras.Model(inputs, outputs)
      model.compile(optimizer='adam', loss='mean_squared_error')
      return model

    with self.strategy.scope():
      model = create_model()

    inputs = tf.random.normal(shape=(8, 4))
    expect = model(inputs)
    saved_dir = self.get_temp_dir()
    model.save(saved_dir)

    loaded_model = keras.models.load_model(saved_dir)
    got = loaded_model(inputs)
    self.assertAllClose(got, expect)
    self.assertGreater(len(model.variables), len(loaded_model.variables))

    with self.assertRaises(ValueError):
      with self.strategy.scope():
        keras.models.load_model(saved_dir)

  def test_slot_variable_checkpointing(self):

    with self.strategy.scope():
      # Set a name so the ShardedVariable is well-named for slot var keying
      var = tf.Variable([1., 2., 3., 4., 5., 6.], name='test')

    opt = keras.optimizers.optimizer_v2.adam.Adam()

    # Run once to trigger apply_gradients to populate optimizer slot variables.
    def train_step():
      with tf.GradientTape() as tape:
        loss = sum(var)
      opt.minimize(loss, var.variables, tape=tape)

    self.strategy.run(train_step)

    # Check that we can call get_slot using each slot, before and after
    # Checkpointing, and get the same results
    pre_ckpt_slots = []
    for slot in opt.get_slot_names():
      pre_ckpt_slots.extend([v.numpy() for v in opt.get_slot(var, slot)])

    ckpt = tf.train.Checkpoint(var=var, opt=opt)

    # Assert that checkpoint has slots for each shard and the ShardedVariable
    self.assertLen(ckpt.opt._slots, 3)
    for var_name in ckpt.opt._slots.keys():
      self.assertLen(ckpt.opt._slots[var_name], 2)
      self.assertEqual(ckpt.opt._slots[var_name].keys(), {'m', 'v'})
      if hasattr(ckpt.opt._slots[var_name]['m'], 'variables'):
        self.assertLen(ckpt.opt._slots[var_name]['m'].variables, 2)
        self.assertLen(ckpt.opt._slots[var_name]['v'].variables, 2)

    saved_dir = self.get_temp_dir()
    ckpt_prefix = f'{saved_dir}/ckpt'
    ckpt.save(ckpt_prefix)

    # Run once more to alter slot variables and ensure checkpoint restores
    # the earlier values.
    self.strategy.run(train_step)

    changed_ckpt_slots = []
    for slot in opt.get_slot_names():
      changed_ckpt_slots.extend([v.numpy() for v in opt.get_slot(var, slot)])
    self.assertNotAllClose(pre_ckpt_slots, changed_ckpt_slots)

    ckpt.restore(tf.train.latest_checkpoint(saved_dir))

    post_ckpt_slots = []
    for slot in opt.get_slot_names():
      post_ckpt_slots.extend([v.numpy() for v in opt.get_slot(var, slot)])

    self.assertAllClose(pre_ckpt_slots, post_ckpt_slots)

  def test_slot_variable_checkpoint_load_with_diff_shards(self):

    with self.strategy.scope():
      # Set a name so the ShardedVariable is well-named for slot var keying
      var = tf.Variable([1., 2., 3., 4., 5., 6.], name='test')

    opt = keras.optimizers.optimizer_v2.adam.Adam()

    # Run once to trigger apply_gradients to populate optimizer slot variables.
    def train_step():
      with tf.GradientTape() as tape:
        loss = sum(var)
      opt.minimize(loss, var.variables, tape=tape)

    self.strategy.run(train_step)

    # Check that we can call get_slot using each slot, before and after
    # Checkpointing, and get the same results
    pre_ckpt_slots = []
    for slot in opt.get_slot_names():
      pre_ckpt_slots.extend(
          tf.concat(list(opt.get_slot(var, slot)), axis=0).numpy())

    ckpt = tf.train.Checkpoint(var=var, opt=opt)
    saved_dir = self.get_temp_dir()
    ckpt_prefix = f'{saved_dir}/ckpt'
    ckpt.save(ckpt_prefix)

    # Create new strategy with different number of shards
    strategy2 = tf.distribute.experimental.ParameterServerStrategy(
        multi_worker_testing_utils.make_parameter_server_cluster(3, 2),
        variable_partitioner=tf.distribute.experimental.partitioners
        .FixedShardsPartitioner(3))

    # Create new variable with different values, to be overwritten by ckpt.
    with strategy2.scope():
      var = tf.Variable([0., 1., 2., 3., 4., 5.], name='test')

    opt = keras.optimizers.optimizer_v2.adam.Adam()
    # Run once to trigger apply_gradients to populate optimizer slot variables.
    strategy2.run(train_step)

    new_ckpt = tf.train.Checkpoint(var=var, opt=opt)
    new_ckpt.restore(tf.train.latest_checkpoint(saved_dir))
    post_ckpt_slots = []
    for slot in new_ckpt.opt.get_slot_names():
      post_ckpt_slots.extend(
          tf.concat(list(new_ckpt.opt.get_slot(var, slot)),
                    axis=0).numpy())
    self.assertAllClose(pre_ckpt_slots, post_ckpt_slots)

if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
