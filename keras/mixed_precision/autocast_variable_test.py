# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for AutoCastVariable."""

import tensorflow.compat.v2 as tf

import os
import threading

from absl.testing import parameterized
import numpy as np
from keras.mixed_precision import autocast_variable
from keras.optimizer_v2 import adadelta
from keras.optimizer_v2 import adagrad
from keras.optimizer_v2 import adam
from keras.optimizer_v2 import adamax
from keras.optimizer_v2 import ftrl
from keras.optimizer_v2 import gradient_descent as gradient_descent_v2
from keras.optimizer_v2 import nadam
from keras.optimizer_v2 import rmsprop

maybe_distribute = tf.__internal__.test.combinations.combine(distribution=[
    tf.__internal__.distribute.combinations.default_strategy,
    tf.__internal__.distribute.combinations.mirrored_strategy_with_cpu_1_and_2
])


def get_var(val, dtype, name=None):
  return tf.Variable(val, dtype=dtype, name=name)


def set_cpu_logical_devices_to_at_least(num):
  """Create cpu logical devices of at least a given number."""
  physical_devices = tf.config.list_physical_devices('CPU')
  if not physical_devices:
    raise RuntimeError('No CPU found')
  if len(physical_devices) >= num:
    return
  # By default each physical device corresponds to one logical device. We create
  # multiple logical devices for the last physical device so that we have `num`
  # logical devices.
  num = num - len(physical_devices) + 1
  logical_devices = []
  for _ in range(num):
    logical_devices.append(tf.config.LogicalDeviceConfiguration())
  # Create logical devices from the last device since sometimes the first GPU
  # is the primary graphic card and may have less memory available.
  tf.config.set_logical_device_configuration(physical_devices[-1], logical_devices)


@tf.__internal__.distribute.combinations.generate(tf.__internal__.test.combinations.combine(mode=['graph', 'eager']))
class AutoCastVariableTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    set_cpu_logical_devices_to_at_least(3)
    super(AutoCastVariableTest, self).setUp()

  @tf.__internal__.distribute.combinations.generate(maybe_distribute)
  def test_read(self, distribution):
    with distribution.scope():
      x = get_var(1., tf.float32)
      x = autocast_variable.create_autocast_variable(x)
      self.evaluate(x.initializer)

      # outside of auto cast scope.
      self.assertEqual(x.dtype, tf.float32)
      self.assertEqual(x.value().dtype, tf.float32)
      self.assertEqual(x.read_value().dtype, tf.float32)
      self.assertEqual(tf.identity(x).dtype, tf.float32)

      # within auto cast scope of different dtype
      with autocast_variable.enable_auto_cast_variables(tf.float16):
        self.assertEqual(x.dtype, tf.float32)
        self.assertEqual(x.value().dtype, tf.float16)
        self.assertEqual(x.read_value().dtype, tf.float16)
        self.assertEqual(tf.identity(x).dtype, tf.float16)

      # within auto cast scope of same dtype
      with autocast_variable.enable_auto_cast_variables(tf.float32):
        self.assertEqual(x.dtype, tf.float32)
        self.assertEqual(x.value().dtype, tf.float32)
        self.assertEqual(x.read_value().dtype, tf.float32)
        self.assertEqual(tf.identity(x).dtype, tf.float32)

  def test_sparse_reads(self):
    x = get_var([1., 2], tf.float32)
    # DistributedVariables do not support sparse_read or gather_nd, so we pass
    # distribute=False
    x = autocast_variable.create_autocast_variable(x)
    self.evaluate(x.initializer)

    self.assertEqual(x.sparse_read([0]).dtype, tf.float32)
    self.assertEqual(x.gather_nd([0]).dtype, tf.float32)

    with autocast_variable.enable_auto_cast_variables(tf.float16):
      self.assertEqual(x.sparse_read([0]).dtype, tf.float16)
      self.assertEqual(x.gather_nd([0]).dtype, tf.float16)

  @tf.__internal__.distribute.combinations.generate(maybe_distribute)
  def test_read_nested_scopes(self, distribution):
    with distribution.scope():
      x = get_var(1., tf.float32)
      x = autocast_variable.create_autocast_variable(x)
      self.evaluate(x.initializer)

      with autocast_variable.enable_auto_cast_variables(tf.float16):
        self.assertEqual(x.read_value().dtype, tf.float16)

        with autocast_variable.enable_auto_cast_variables(tf.float32):
          self.assertEqual(x.read_value().dtype, tf.float32)

        self.assertEqual(x.read_value().dtype, tf.float16)

  @tf.__internal__.distribute.combinations.generate(maybe_distribute)
  def test_dtype_is_not_string(self, distribution):
    with distribution.scope():
      x = get_var(1., tf.float32)
      x = autocast_variable.create_autocast_variable(x)
      self.assertEqual(x.dtype, tf.float32)
      self.assertIsInstance(x.dtype, tf.DType)
      self.assertEqual(x.true_dtype, tf.float32)
      self.assertIsInstance(x.true_dtype, tf.DType)

      dtype = tf.float16
      with autocast_variable.enable_auto_cast_variables(dtype):
        self.assertEqual(x.dtype, tf.float32)
        self.assertIsInstance(x.dtype, tf.DType)
        self.assertEqual(x.true_dtype, tf.float32)
        self.assertIsInstance(x.true_dtype, tf.DType)

  @tf.__internal__.distribute.combinations.generate(maybe_distribute)
  def test_method_delegations(self, distribution):
    # Test AutoCastVariable correctly delegates Variable methods to the
    # underlying variable.
    with self.test_session(), distribution.scope():
      for read_dtype in (tf.float32, tf.float16):
        if tf.distribute.has_strategy() and not tf.executing_eagerly():
          # MirroredVariable.assign will (incorrectly) return a Mirrored value
          # instead of a MirroredVariable in graph mode.
          # So we cannot properly wrap it in an AutoCastVariable.
          evaluate = self.evaluate
        else:

          def evaluate(var):
            self.assertIsInstance(var, autocast_variable.AutoCastVariable)
            self.assertEqual(tf.identity(var).dtype, read_dtype)  # pylint: disable=cell-var-from-loop
            return self.evaluate(var)

        x = get_var(7., tf.float32)
        x = autocast_variable.create_autocast_variable(x)
        with autocast_variable.enable_auto_cast_variables(read_dtype):
          self.evaluate(x.initializer)
          self.assertEqual(self.evaluate(x.value()), 7)
          self.assertEqual(self.evaluate(x.read_value()), 7)
          self.assertTrue(x.trainable)
          self.assertEqual(x.synchronization, x._variable.synchronization)
          self.assertEqual(x.aggregation, x._variable.aggregation)
          self.assertEqual(self.evaluate(x.initialized_value()), 7)
          if not tf.executing_eagerly():
            if not tf.distribute.has_strategy():
              # These functions are not supported for DistributedVariables
              x.load(9)
              self.assertEqual(x.eval(), 9)
            self.assertEqual(self.evaluate(x.initial_value), 7)
            self.assertEqual(x.op, x._variable.op)
            self.assertEqual(x.graph, x._variable.graph)
          if not tf.distribute.has_strategy():
            # These attributes are not supported for DistributedVariables
            self.assertIsNone(x.constraint)
            self.assertEqual(x.initializer, x._variable.initializer)
          self.assertEqual(evaluate(x.assign(8)), 8)
          self.assertEqual(evaluate(x.assign_add(2)), 10)
          self.assertEqual(evaluate(x.assign_sub(3)), 7)
          self.assertEqual(x.name, x._variable.name)
          self.assertEqual(x.device, x._variable.device)
          self.assertEqual(x.shape, ())
          self.assertEqual(x.get_shape(), ())

        if not tf.distribute.has_strategy():
          # Test scatter_* methods. These are not supported for
          # DistributedVariables
          x = get_var([7, 8], tf.float32)
          x = autocast_variable.create_autocast_variable(x)
          with autocast_variable.enable_auto_cast_variables(read_dtype):
            self.evaluate(x.initializer)
            self.assertAllEqual(self.evaluate(x.value()), [7, 8])

            def slices(val, index):
              return tf.IndexedSlices(
                  values=tf.constant(val, dtype=tf.float32),
                  indices=tf.constant(index, dtype=tf.int32),
                  dense_shape=tf.constant([2], dtype=tf.int32))

            self.assertAllEqual(evaluate(x.scatter_sub(slices(1., 0))), [6, 8])
            self.assertAllEqual(evaluate(x.scatter_add(slices(1., 0))), [7, 8])
            self.assertAllEqual(evaluate(x.scatter_max(slices(9., 1))), [7, 9])
            self.assertAllEqual(evaluate(x.scatter_min(slices(8., 1))), [7, 8])
            self.assertAllEqual(evaluate(x.scatter_mul(slices(2., 1))), [7, 16])
            self.assertAllEqual(evaluate(x.scatter_div(slices(2., 1))), [7, 8])
            self.assertAllEqual(
                evaluate(x.scatter_update(slices(4., 1))), [7, 4])
            self.assertAllEqual(
                evaluate(x.scatter_nd_sub([[0], [1]], [1., 2.])), [6, 2])
            self.assertAllEqual(
                evaluate(x.scatter_nd_add([[0], [1]], [1., 2.])), [7, 4])
            self.assertAllEqual(
                evaluate(x.scatter_nd_update([[0], [1]], [1., 2.])), [1, 2])

  @tf.__internal__.distribute.combinations.generate(maybe_distribute)
  def test_operator_overloads(self, distribution):
    with distribution.scope():
      for read_dtype in (tf.float32, tf.float16):
        x = get_var(7., tf.float32)
        x = autocast_variable.create_autocast_variable(x)
        with autocast_variable.enable_auto_cast_variables(read_dtype):
          self.evaluate(x.initializer)
          self.assertAlmostEqual(8, self.evaluate(x + 1))
          self.assertAlmostEqual(10, self.evaluate(3 + x))
          self.assertAlmostEqual(14, self.evaluate(x + x))
          self.assertAlmostEqual(5, self.evaluate(x - 2))
          self.assertAlmostEqual(6, self.evaluate(13 - x))
          self.assertAlmostEqual(0, self.evaluate(x - x))
          self.assertAlmostEqual(14, self.evaluate(x * 2))
          self.assertAlmostEqual(21, self.evaluate(3 * x))
          self.assertAlmostEqual(49, self.evaluate(x * x))
          self.assertAlmostEqual(3.5, self.evaluate(x / 2))
          self.assertAlmostEqual(1.5, self.evaluate(10.5 / x))
          self.assertAlmostEqual(3, self.evaluate(x // 2))
          self.assertAlmostEqual(2, self.evaluate(15 // x))
          if read_dtype == tf.float32:
            # The "mod" operator does not support float16
            self.assertAlmostEqual(1, self.evaluate(x % 2))
            self.assertAlmostEqual(2, self.evaluate(16 % x))
          self.assertTrue(self.evaluate(x < 12))
          self.assertTrue(self.evaluate(x <= 12))
          self.assertFalse(self.evaluate(x > 12))
          self.assertFalse(self.evaluate(x >= 12))
          self.assertFalse(self.evaluate(12 < x))
          self.assertFalse(self.evaluate(12 <= x))
          self.assertTrue(self.evaluate(12 > x))
          self.assertTrue(self.evaluate(12 >= x))
          self.assertAlmostEqual(343, self.evaluate(pow(x, 3)), places=4)
          self.assertAlmostEqual(128, self.evaluate(pow(2, x)), places=4)
          self.assertAlmostEqual(-7, self.evaluate(-x))
          self.assertAlmostEqual(7, self.evaluate(abs(x)))

          x = get_var([7, 8, 9], tf.float32)
          x = autocast_variable.create_autocast_variable(x)
          self.evaluate(x.initializer)
          self.assertEqual(self.evaluate(x[1]), 8)
          if tf.__internal__.tf2.enabled() and tf.executing_eagerly():
            self.assertAllEqual(x == [7., 8., 10.], [True, True, False])
            self.assertAllEqual(x != [7., 8., 10.], [False, False, True])

  @tf.__internal__.distribute.combinations.generate(maybe_distribute)
  def test_assign(self, distribution):
    with distribution.scope():
      x = get_var(0., tf.float32)
      x = autocast_variable.create_autocast_variable(x)
      self.evaluate(x.initializer)

      # outside of auto cast scope.
      v1 = tf.constant(3., dtype=tf.float32)
      v2 = tf.constant(3., dtype=tf.float16)

      def run_and_check():
        # Assign float32 values
        self.assertAllClose(3., self.evaluate(x.assign(v1)))
        self.assertAllClose(3. * 2, self.evaluate(x.assign_add(v1)))
        self.assertAllClose(3., self.evaluate(x.assign_sub(v1)))

        # Attempt to assign float16 values
        with self.assertRaisesRegex(
            ValueError,
            'conversion requested dtype float32 for Tensor with dtype float16'):
          self.evaluate(x.assign(v2))
        with self.assertRaisesRegex(
            ValueError,
            'conversion requested dtype float32 for Tensor with dtype float16'):
          self.evaluate(x.assign_add(v2))
        with self.assertRaisesRegex(
            ValueError,
            'conversion requested dtype float32 for Tensor with dtype float16'):
          self.evaluate(x.assign_sub(v2))

        # Assign Python floats
        self.assertAllClose(0., self.evaluate(x.assign(0.)))
        self.assertAllClose(3., self.evaluate(x.assign(3.)))
        self.assertAllClose(3. * 2, self.evaluate(x.assign_add(3.)))
        self.assertAllClose(3., self.evaluate(x.assign_sub(3.)))

        # Assign multiple times
        # This currently doesn't work in graph mode if a strategy is used
        if not tf.distribute.has_strategy() or tf.executing_eagerly():
          assign = x.assign(1.)
          self.assertAllClose(1., self.evaluate(assign))
          self.assertAllClose(0., self.evaluate(assign.assign(0.)))
          assign_add = x.assign_add(3.)
          self.assertAllClose(3., self.evaluate(assign_add))
          self.assertAllClose(3. * 3,
                              self.evaluate(x.assign_add(3.).assign_add(3.)))
          self.assertAllClose(3. * 3, x)
          assign_sub = x.assign_sub(3.)
          self.assertAllClose(3. * 2, self.evaluate(assign_sub))
          self.assertAllClose(0.,
                              self.evaluate(x.assign_sub(3.).assign_sub(3.)))

        # Assign with read_value=False
        self.assertIsNone(self.evaluate(x.assign(1., read_value=False)))
        self.assertAllClose(1., self.evaluate(x))
        self.assertIsNone(self.evaluate(x.assign_add(2., read_value=False)))
        self.assertAllClose(3., self.evaluate(x))
        self.assertIsNone(self.evaluate(x.assign_sub(3., read_value=False)))
        self.assertAllClose(0., self.evaluate(x))

        # Use the tf.assign functions instead of the var.assign methods.
        self.assertAllClose(0., self.evaluate(tf.compat.v1.assign(x, 0.)))
        self.assertAllClose(3., self.evaluate(tf.compat.v1.assign(x, 3.)))
        self.assertAllClose(3. * 2,
                            self.evaluate(tf.compat.v1.assign_add(x, 3.)))
        self.assertAllClose(3., self.evaluate(tf.compat.v1.assign_sub(x, 3.)))

      run_and_check()
      # reset x
      self.evaluate(x.assign(0.))
      # within auto cast scope.
      with autocast_variable.enable_auto_cast_variables(tf.float16):
        # assign still expect float32 value even if in float16 scope
        run_and_check()

  @tf.__internal__.distribute.combinations.generate(maybe_distribute)
  def test_assign_tf_function(self, distribution):
    if not tf.executing_eagerly():
      self.skipTest('Test is not compatible with graph mode')

    with distribution.scope():
      x = get_var(0., tf.float32)
      x = autocast_variable.create_autocast_variable(x)

      @tf.function
      def run_assign():
        return x.assign(1.).assign_add(3.).assign_add(3.).assign_sub(2.)

      with autocast_variable.enable_auto_cast_variables(tf.float16):
        self.assertAllClose(5., self.evaluate(run_assign()))

  @tf.__internal__.distribute.combinations.generate(maybe_distribute)
  def test_op_attribute(self, distribution):
    with distribution.scope():
      x = get_var(0., tf.float32)
      x = autocast_variable.create_autocast_variable(x)

      # Variable.op raises an AttributeError in Eager mode and is an op in graph
      # mode. Variable.assign(...).op is None in Eager mode and an op in Graph
      # mode or a tf.function. We test this is also true of AutoCastVariable.
      if tf.executing_eagerly():
        with self.assertRaises(AttributeError):
          x.op  # pylint: disable=pointless-statement
        self.assertIsNone(x.assign(1.0).op)
        self.assertIsNone(x.assign_add(1.0).op)
        self.assertIsNone(x.assign_sub(1.0).op)
      else:
        self.assertIsNotNone(x.op)
        self.assertIsNotNone(x.assign(1.0).op)
        self.assertIsNotNone(x.assign_add(1.0).op)
        self.assertIsNotNone(x.assign_sub(1.0).op)

      @tf.function
      def func():
        self.assertIsNotNone(x.assign(1.0).op)
        self.assertIsNotNone(x.assign_add(1.0).op)
        self.assertIsNotNone(x.assign_sub(1.0).op)

      func()

  @tf.__internal__.distribute.combinations.generate(maybe_distribute)
  def test_tf_function_control_dependencies(self, distribution):
    if not tf.executing_eagerly():
      self.skipTest('Test is not compatible with graph mode')

    with distribution.scope():
      x = get_var(0., tf.float32)
      x = autocast_variable.create_autocast_variable(x)

      @tf.function
      def func():
        update = x.assign_add(1.)
        with tf.control_dependencies([update]):
          x.assign_add(1.)

      func()
      self.assertAllClose(2., self.evaluate(x))

  @tf.__internal__.distribute.combinations.generate(maybe_distribute)
  def test_assign_stays_in_true_dtype(self, distribution):
    with distribution.scope():
      x = get_var(1., tf.float32)
      x = autocast_variable.create_autocast_variable(x)
      self.evaluate(x.initializer)
      # small_val is a value such that 1.0 + small_val == 1.0 in fp16, but not
      # in fp32
      small_val = np.finfo('float16').eps / 2
      small_tensor = tf.constant(small_val, dtype=tf.float32)
      with autocast_variable.enable_auto_cast_variables(tf.float16):
        # Variable should be increased, despite it appearing to be the same
        # float16 value.
        self.evaluate(x.assign(1. + small_tensor))
        self.assertEqual(1., self.evaluate(x.value()))
      self.assertEqual(1. + small_val, self.evaluate(x))

      self.evaluate(x.assign(1.))
      with autocast_variable.enable_auto_cast_variables(tf.float16):
        self.evaluate(x.assign_add(small_tensor))
        self.assertEqual(1., self.evaluate(x.value()))
      self.assertEqual(1. + small_val, self.evaluate(x))

  def test_thread_local_autocast_dtype(self):
    x = get_var(1., tf.float32)
    x = autocast_variable.create_autocast_variable(x)
    self.evaluate(x.initializer)

    with autocast_variable.enable_auto_cast_variables(tf.float16):
      self.assertEqual(tf.identity(x).dtype, tf.float16)

      # New threads should not see the modified value of the autocast dtype.
      var_dtype = None
      def f():
        nonlocal var_dtype
        var_dtype = x._cast_dtype
      thread = threading.Thread(target=f)
      thread.start()
      thread.join()
      self.assertEqual(var_dtype, tf.float32)

  @tf.__internal__.distribute.combinations.generate(maybe_distribute)
  def test_checkpoint(self, distribution):
    with self.test_session():
      with distribution.scope():
        x = get_var(1., tf.float32)
        x = autocast_variable.create_autocast_variable(x)
      self.evaluate(x.initializer)
      self.evaluate(x.assign(123.))

      checkpoint = tf.train.Checkpoint(x=x)
      prefix = os.path.join(self.get_temp_dir(), 'ckpt')
      save_path = checkpoint.save(prefix)
      self.evaluate(x.assign(234.))
      checkpoint.restore(save_path).assert_consumed().run_restore_ops()
      self.assertEqual(self.evaluate(x), 123.)

  @tf.__internal__.distribute.combinations.generate(maybe_distribute)
  def test_invalid_wrapped_variable(self, distribution):
    with distribution.scope():
      # Wrap a non-variable
      with self.assertRaisesRegex(ValueError, 'variable must be of type'):
        x = tf.constant([1.], dtype=tf.float32)
        autocast_variable.create_autocast_variable(x)

      # Wrap a non-floating point variable
      with self.assertRaisesRegex(ValueError,
                                  'variable must be a floating point'):
        x = get_var(1, tf.int32)
        autocast_variable.create_autocast_variable(x)

  def test_repr(self):
    # We do not test with DistributionStrategy because we do not want to rely on
    # the exact __repr__ output of a DistributedVariable.
    x = get_var(1., tf.float32, name='x')
    x = autocast_variable.create_autocast_variable(x)
    if tf.executing_eagerly():
      self.assertStartsWith(
          repr(x),
          "<AutoCastVariable 'x:0' shape=() dtype=float32 "
          "dtype_to_cast_to=float32, numpy="
      )
      with autocast_variable.enable_auto_cast_variables(tf.float16):
        self.assertStartsWith(
            repr(x),
            "<AutoCastVariable 'x:0' shape=() dtype=float32 "
            "dtype_to_cast_to=float16, numpy="
        )
    else:
      self.assertEqual(
          repr(x),
          "<AutoCastVariable 'x:0' shape=() dtype=float32 "
          "dtype_to_cast_to=float32>"
      )
      with autocast_variable.enable_auto_cast_variables(tf.float16):
        self.assertEqual(
            repr(x),
            "<AutoCastVariable 'x:0' shape=() dtype=float32 "
            "dtype_to_cast_to=float16>"
        )

  def test_repr_distributed(self):
    strategy = tf.distribute.MirroredStrategy(['/cpu:1', '/cpu:2'])
    with strategy.scope():
      x = get_var(1., tf.float32)
      x = autocast_variable.create_autocast_variable(x)
      use_policy = getattr(strategy.extended, '_use_var_policy', False)
      if use_policy:
        self.assertRegex(
            repr(x).replace('\n', ' '),
            '<AutoCastDistributedVariable dtype=float32 '
            'dtype_to_cast_to=float32 '
            'inner_variable=DistributedVariable.*>')
      else:
        self.assertRegex(
            repr(x).replace('\n', ' '),
            '<AutoCastDistributedVariable dtype=float32 '
            'dtype_to_cast_to=float32 '
            'inner_variable=MirroredVariable.*>')

  @tf.__internal__.distribute.combinations.generate(tf.__internal__.test.combinations.combine(
      optimizer_class=[
          adadelta.Adadelta,
          adagrad.Adagrad,
          adam.Adam,
          adamax.Adamax,
          ftrl.Ftrl,
          gradient_descent_v2.SGD,
          nadam.Nadam,
          rmsprop.RMSprop,
          tf.compat.v1.train.GradientDescentOptimizer
      ],
      use_tf_function=[False, True]))
  def test_optimizer(self, optimizer_class, use_tf_function):
    if use_tf_function and not tf.executing_eagerly():
      self.skipTest('Test does not support graph mode with tf.function')
    x = get_var(1., tf.float32)
    x = autocast_variable.create_autocast_variable(x)
    y = get_var(1., tf.float32)
    opt = optimizer_class(learning_rate=1.)

    def f():
      # Minimize both the AutoCastVariable and the normal tf.Variable. Both
      # variables should be updated to the same value.
      op = opt.minimize(lambda: x + y, var_list=[x, y])
      return None if tf.compat.v1.executing_eagerly_outside_functions() else op

    if use_tf_function:
      f = tf.function(f)

    if tf.executing_eagerly():
      f()
    else:
      op = f()
      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(op)
    # Assert the AutoCastVariable has changed from its initial value
    self.assertNotEqual(self.evaluate(x), 1.)
    # Assert AutoCastVariable is updated correctly by comparing it to the normal
    # variable
    self.assertAlmostEqual(self.evaluate(x), self.evaluate(y))
    if optimizer_class in (gradient_descent_v2.SGD,
                           tf.compat.v1.train.GradientDescentOptimizer):
      # With SGD, the variables decreases by exactly 1
      self.assertEqual(self.evaluate(x), 0)


if __name__ == '__main__':
  tf.test.main()
