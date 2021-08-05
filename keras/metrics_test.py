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
"""Tests for Keras metrics functions."""

import tensorflow.compat.v2 as tf

import json
import math
import os

from absl.testing import parameterized
import numpy as np
from keras import backend
from keras import combinations
from keras import keras_parameterized
from keras import layers
from keras import metrics
from keras import Model
from keras import testing_utils
from keras.engine import base_layer
from keras.engine import training as training_module


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class KerasSumTest(tf.test.TestCase, parameterized.TestCase):

  def test_sum(self):
    with self.test_session():
      m = metrics.Sum(name='my_sum')

      # check config
      self.assertEqual(m.name, 'my_sum')
      self.assertTrue(m.stateful)
      self.assertEqual(m.dtype, tf.float32)
      self.assertLen(m.variables, 1)
      self.evaluate(tf.compat.v1.variables_initializer(m.variables))

      # check initial state
      self.assertEqual(self.evaluate(m.total), 0)

      # check __call__()
      self.assertEqual(self.evaluate(m(100)), 100)
      self.assertEqual(self.evaluate(m.total), 100)

      # check update_state() and result() + state accumulation + tensor input
      update_op = m.update_state(tf.convert_to_tensor([1, 5]))
      self.evaluate(update_op)
      self.assertAlmostEqual(self.evaluate(m.result()), 106)
      self.assertEqual(self.evaluate(m.total), 106)  # 100 + 1 + 5

      # check reset_state()
      m.reset_state()
      self.assertEqual(self.evaluate(m.total), 0)

  def test_sum_with_sample_weight(self):
    m = metrics.Sum(dtype=tf.float64)
    self.assertEqual(m.dtype, tf.float64)
    self.evaluate(tf.compat.v1.variables_initializer(m.variables))

    # check scalar weight
    result_t = m(100, sample_weight=0.5)
    self.assertEqual(self.evaluate(result_t), 50)
    self.assertEqual(self.evaluate(m.total), 50)

    # check weights not scalar and weights rank matches values rank
    result_t = m([1, 5], sample_weight=[1, 0.2])
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 52., 4)  # 50 + 1 + 5 * 0.2
    self.assertAlmostEqual(self.evaluate(m.total), 52., 4)

    # check weights broadcast
    result_t = m([1, 2], sample_weight=0.5)
    self.assertAlmostEqual(self.evaluate(result_t), 53.5, 1)  # 52 + 0.5 + 1
    self.assertAlmostEqual(self.evaluate(m.total), 53.5, 1)

    # check weights squeeze
    result_t = m([1, 5], sample_weight=[[1], [0.2]])
    self.assertAlmostEqual(self.evaluate(result_t), 55.5, 1)  # 53.5 + 1 + 1
    self.assertAlmostEqual(self.evaluate(m.total), 55.5, 1)

    # check weights expand
    result_t = m([[1], [5]], sample_weight=[1, 0.2])
    self.assertAlmostEqual(self.evaluate(result_t), 57.5, 2)  # 55.5 + 1 + 1
    self.assertAlmostEqual(self.evaluate(m.total), 57.5, 1)

    # check values reduced to the dimensions of weight
    result_t = m([[[1., 2.], [3., 2.], [0.5, 4.]]], sample_weight=[0.5])
    result = np.round(self.evaluate(result_t), decimals=2)
    # result = (prev: 57.5) + 0.5 + 1 + 1.5 + 1 + 0.25 + 2
    self.assertAlmostEqual(result, 63.75, 2)
    self.assertAlmostEqual(self.evaluate(m.total), 63.75, 2)

  def test_sum_graph_with_placeholder(self):
    with tf.compat.v1.get_default_graph().as_default(), self.cached_session() as sess:
      m = metrics.Sum()
      v = tf.compat.v1.placeholder(tf.float32)
      w = tf.compat.v1.placeholder(tf.float32)
      self.evaluate(tf.compat.v1.variables_initializer(m.variables))

      # check __call__()
      result_t = m(v, sample_weight=w)
      result = sess.run(result_t, feed_dict=({v: 100, w: 0.5}))
      self.assertEqual(result, 50)
      self.assertEqual(self.evaluate(m.total), 50)

      # check update_state() and result()
      result = sess.run(result_t, feed_dict=({v: [1, 5], w: [1, 0.2]}))
      self.assertAlmostEqual(result, 52., 2)  # 50 + 1 + 5 * 0.2
      self.assertAlmostEqual(self.evaluate(m.total), 52., 2)

  def test_save_restore(self):
    with self.test_session():
      checkpoint_directory = self.get_temp_dir()
      checkpoint_prefix = os.path.join(checkpoint_directory, 'ckpt')
      m = metrics.Sum()
      checkpoint = tf.train.Checkpoint(sum=m)
      self.evaluate(tf.compat.v1.variables_initializer(m.variables))

      # update state
      self.evaluate(m(100.))
      self.evaluate(m(200.))

      # save checkpoint and then add an update
      save_path = checkpoint.save(checkpoint_prefix)
      self.evaluate(m(1000.))

      # restore to the same checkpoint sum object (= 300)
      checkpoint.restore(save_path).assert_consumed().run_restore_ops()
      self.evaluate(m(300.))
      self.assertEqual(600., self.evaluate(m.result()))

      # restore to a different checkpoint sum object
      restore_sum = metrics.Sum()
      restore_checkpoint = tf.train.Checkpoint(sum=restore_sum)
      status = restore_checkpoint.restore(save_path)
      restore_update = restore_sum(300.)
      status.assert_consumed().run_restore_ops()
      self.evaluate(restore_update)
      self.assertEqual(600., self.evaluate(restore_sum.result()))


class MeanTest(keras_parameterized.TestCase):

  # TODO(b/120949004): Re-enable garbage collection check
  # @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  @keras_parameterized.run_all_keras_modes
  def test_mean(self):
    m = metrics.Mean(name='my_mean')

    # check config
    self.assertEqual(m.name, 'my_mean')
    self.assertTrue(m.stateful)
    self.assertEqual(m.dtype, tf.float32)
    self.assertEqual(len(m.variables), 2)
    self.evaluate(tf.compat.v1.variables_initializer(m.variables))

    # check initial state
    self.assertEqual(self.evaluate(m.total), 0)
    self.assertEqual(self.evaluate(m.count), 0)

    # check __call__()
    self.assertEqual(self.evaluate(m(100)), 100)
    self.assertEqual(self.evaluate(m.total), 100)
    self.assertEqual(self.evaluate(m.count), 1)

    # check update_state() and result() + state accumulation + tensor input
    update_op = m.update_state([
        tf.convert_to_tensor(1),
        tf.convert_to_tensor(5)
    ])
    self.evaluate(update_op)
    self.assertAlmostEqual(self.evaluate(m.result()), 106 / 3, 2)
    self.assertEqual(self.evaluate(m.total), 106)  # 100 + 1 + 5
    self.assertEqual(self.evaluate(m.count), 3)

    # check reset_state()
    m.reset_state()
    self.assertEqual(self.evaluate(m.total), 0)
    self.assertEqual(self.evaluate(m.count), 0)

    # Check save and restore config
    m2 = metrics.Mean.from_config(m.get_config())
    self.assertEqual(m2.name, 'my_mean')
    self.assertTrue(m2.stateful)
    self.assertEqual(m2.dtype, tf.float32)
    self.assertEqual(len(m2.variables), 2)

  @testing_utils.run_v2_only
  def test_function_wrapped_reset_state(self):
    m = metrics.Mean(name='my_mean')

    # check reset_state in function.
    @tf.function
    def reset_in_fn():
      m.reset_state()
      return m.update_state(100)

    for _ in range(5):
      self.evaluate(reset_in_fn())
    self.assertEqual(self.evaluate(m.count), 1)

  @keras_parameterized.run_all_keras_modes
  def test_mean_with_sample_weight(self):
    m = metrics.Mean(dtype=tf.float64)
    self.assertEqual(m.dtype, tf.float64)
    self.evaluate(tf.compat.v1.variables_initializer(m.variables))

    # check scalar weight
    result_t = m(100, sample_weight=0.5)
    self.assertEqual(self.evaluate(result_t), 50 / 0.5)
    self.assertEqual(self.evaluate(m.total), 50)
    self.assertEqual(self.evaluate(m.count), 0.5)

    # check weights not scalar and weights rank matches values rank
    result_t = m([1, 5], sample_weight=[1, 0.2])
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 52 / 1.7, 2)
    self.assertAlmostEqual(self.evaluate(m.total), 52, 2)  # 50 + 1 + 5 * 0.2
    self.assertAlmostEqual(self.evaluate(m.count), 1.7, 2)  # 0.5 + 1.2

    # check weights broadcast
    result_t = m([1, 2], sample_weight=0.5)
    self.assertAlmostEqual(self.evaluate(result_t), 53.5 / 2.7, 2)
    self.assertAlmostEqual(self.evaluate(m.total), 53.5, 2)  # 52 + 0.5 + 1
    self.assertAlmostEqual(self.evaluate(m.count), 2.7, 2)  # 1.7 + 0.5 + 0.5

    # check weights squeeze
    result_t = m([1, 5], sample_weight=[[1], [0.2]])
    self.assertAlmostEqual(self.evaluate(result_t), 55.5 / 3.9, 2)
    self.assertAlmostEqual(self.evaluate(m.total), 55.5, 2)  # 53.5 + 1 + 1
    self.assertAlmostEqual(self.evaluate(m.count), 3.9, 2)  # 2.7 + 1.2

    # check weights expand
    result_t = m([[1], [5]], sample_weight=[1, 0.2])
    self.assertAlmostEqual(self.evaluate(result_t), 57.5 / 5.1, 2)
    self.assertAlmostEqual(self.evaluate(m.total), 57.5, 2)  # 55.5 + 1 + 1
    self.assertAlmostEqual(self.evaluate(m.count), 5.1, 2)  # 3.9 + 1.2

    # check values reduced to the dimensions of weight
    result_t = m([[[1., 2.], [3., 2.], [0.5, 4.]]], sample_weight=[0.5])
    result = np.round(self.evaluate(result_t), decimals=2)  # 58.5 / 5.6
    self.assertEqual(result, 10.45)
    self.assertEqual(np.round(self.evaluate(m.total), decimals=2), 58.54)
    self.assertEqual(np.round(self.evaluate(m.count), decimals=2), 5.6)

  @keras_parameterized.run_all_keras_modes
  def test_mean_graph_with_placeholder(self):
    with tf.compat.v1.get_default_graph().as_default(), self.cached_session() as sess:
      m = metrics.Mean()
      v = tf.compat.v1.placeholder(tf.float32)
      w = tf.compat.v1.placeholder(tf.float32)
      self.evaluate(tf.compat.v1.variables_initializer(m.variables))

      # check __call__()
      result_t = m(v, sample_weight=w)
      result = sess.run(result_t, feed_dict=({v: 100, w: 0.5}))
      self.assertEqual(self.evaluate(m.total), 50)
      self.assertEqual(self.evaluate(m.count), 0.5)
      self.assertEqual(result, 50 / 0.5)

      # check update_state() and result()
      result = sess.run(result_t, feed_dict=({v: [1, 5], w: [1, 0.2]}))
      self.assertAlmostEqual(self.evaluate(m.total), 52, 2)  # 50 + 1 + 5 * 0.2
      self.assertAlmostEqual(self.evaluate(m.count), 1.7, 2)  # 0.5 + 1.2
      self.assertAlmostEqual(result, 52 / 1.7, 2)

  @keras_parameterized.run_all_keras_modes
  def test_save_restore(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, 'ckpt')
    m = metrics.Mean()
    checkpoint = tf.train.Checkpoint(mean=m)
    self.evaluate(tf.compat.v1.variables_initializer(m.variables))

    # update state
    self.evaluate(m(100.))
    self.evaluate(m(200.))

    # save checkpoint and then add an update
    save_path = checkpoint.save(checkpoint_prefix)
    self.evaluate(m(1000.))

    # restore to the same checkpoint mean object
    checkpoint.restore(save_path).assert_consumed().run_restore_ops()
    self.evaluate(m(300.))
    self.assertEqual(200., self.evaluate(m.result()))

    # restore to a different checkpoint mean object
    restore_mean = metrics.Mean()
    restore_checkpoint = tf.train.Checkpoint(mean=restore_mean)
    status = restore_checkpoint.restore(save_path)
    restore_update = restore_mean(300.)
    status.assert_consumed().run_restore_ops()
    self.evaluate(restore_update)
    self.assertEqual(200., self.evaluate(restore_mean.result()))
    self.assertEqual(3, self.evaluate(restore_mean.count))

  @keras_parameterized.run_all_keras_modes
  def test_multiple_instances(self):
    m = metrics.Mean()
    m2 = metrics.Mean()

    self.assertEqual(m.name, 'mean')
    self.assertEqual(m2.name, 'mean')

    self.assertEqual([v.name for v in m.variables],
                     testing_utils.get_expected_metric_variable_names(
                         ['total', 'count']))
    self.assertEqual([v.name for v in m2.variables],
                     testing_utils.get_expected_metric_variable_names(
                         ['total', 'count'], name_suffix='_1'))

    self.evaluate(tf.compat.v1.variables_initializer(m.variables))
    self.evaluate(tf.compat.v1.variables_initializer(m2.variables))

    # check initial state
    self.assertEqual(self.evaluate(m.total), 0)
    self.assertEqual(self.evaluate(m.count), 0)
    self.assertEqual(self.evaluate(m2.total), 0)
    self.assertEqual(self.evaluate(m2.count), 0)

    # check __call__()
    self.assertEqual(self.evaluate(m(100)), 100)
    self.assertEqual(self.evaluate(m.total), 100)
    self.assertEqual(self.evaluate(m.count), 1)
    self.assertEqual(self.evaluate(m2.total), 0)
    self.assertEqual(self.evaluate(m2.count), 0)

    self.assertEqual(self.evaluate(m2([63, 10])), 36.5)
    self.assertEqual(self.evaluate(m2.total), 73)
    self.assertEqual(self.evaluate(m2.count), 2)
    self.assertEqual(self.evaluate(m.result()), 100)
    self.assertEqual(self.evaluate(m.total), 100)
    self.assertEqual(self.evaluate(m.count), 1)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class KerasAccuracyTest(tf.test.TestCase):

  def test_accuracy(self):
    acc_obj = metrics.Accuracy(name='my_acc')

    # check config
    self.assertEqual(acc_obj.name, 'my_acc')
    self.assertTrue(acc_obj.stateful)
    self.assertEqual(len(acc_obj.variables), 2)
    self.assertEqual(acc_obj.dtype, tf.float32)
    self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))

    # verify that correct value is returned
    update_op = acc_obj.update_state([[1], [2], [3], [4]], [[1], [2], [3], [4]])
    self.evaluate(update_op)
    result = self.evaluate(acc_obj.result())
    self.assertEqual(result, 1)  # 2/2

    # Check save and restore config
    a2 = metrics.Accuracy.from_config(acc_obj.get_config())
    self.assertEqual(a2.name, 'my_acc')
    self.assertTrue(a2.stateful)
    self.assertEqual(len(a2.variables), 2)
    self.assertEqual(a2.dtype, tf.float32)

    # check with sample_weight
    result_t = acc_obj([[2], [1]], [[2], [0]], sample_weight=[[0.5], [0.2]])
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 0.96, 2)  # 4.5/4.7

  def test_accuracy_ragged(self):
    acc_obj = metrics.Accuracy(name='my_acc')
    self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))

    # verify that correct value is returned
    rt1 = tf.ragged.constant([[1], [2], [3], [4]])
    rt2 = tf.ragged.constant([[1], [2], [3], [4]])
    update_op = acc_obj.update_state(rt1, rt2)
    self.evaluate(update_op)
    result = self.evaluate(acc_obj.result())
    self.assertEqual(result, 1)  # 2/2

    # check with sample_weight
    rt1 = tf.ragged.constant([[2], [1]])
    rt2 = tf.ragged.constant([[2], [0]])
    sw_ragged = tf.ragged.constant([[0.5], [0.2]])
    result_t = acc_obj(rt1, rt2, sample_weight=sw_ragged)
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 0.96, 2)  # 4.5/4.7

  def test_binary_accuracy(self):
    acc_obj = metrics.BinaryAccuracy(name='my_acc')

    # check config
    self.assertEqual(acc_obj.name, 'my_acc')
    self.assertTrue(acc_obj.stateful)
    self.assertEqual(len(acc_obj.variables), 2)
    self.assertEqual(acc_obj.dtype, tf.float32)
    self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))

    # verify that correct value is returned
    update_op = acc_obj.update_state([[1], [0]], [[1], [0]])
    self.evaluate(update_op)
    result = self.evaluate(acc_obj.result())
    self.assertEqual(result, 1)  # 2/2

    # check y_pred squeeze
    update_op = acc_obj.update_state([[1], [1]], [[[1]], [[0]]])
    self.evaluate(update_op)
    result = self.evaluate(acc_obj.result())
    self.assertAlmostEqual(result, 0.75, 2)  # 3/4

    # check y_true squeeze
    result_t = acc_obj([[[1]], [[1]]], [[1], [0]])
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 0.67, 2)  # 4/6

    # check with sample_weight
    result_t = acc_obj([[1], [1]], [[1], [0]], [[0.5], [0.2]])
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 0.67, 2)  # 4.5/6.7

  def test_binary_accuracy_ragged(self):
    acc_obj = metrics.BinaryAccuracy(name='my_acc')
    self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))

    # verify that correct value is returned
    rt1 = tf.ragged.constant([[1], [0]])
    rt2 = tf.ragged.constant([[1], [0]])
    update_op = acc_obj.update_state(rt1, rt2)
    self.evaluate(update_op)
    result = self.evaluate(acc_obj.result())
    self.assertEqual(result, 1)  # 2/2

    # check y_true squeeze only supported for dense tensors and is
    # not supported by ragged tensor (different ranks). --> error
    rt1 = tf.ragged.constant([[[1], [1]]])
    rt2 = tf.ragged.constant([[1], [0]])
    with self.assertRaises(ValueError):
      result_t = acc_obj(rt1, rt2)
      result = self.evaluate(result_t)

  def test_binary_accuracy_threshold(self):
    acc_obj = metrics.BinaryAccuracy(threshold=0.7)
    self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))
    result_t = acc_obj([[1], [1], [0], [0]], [[0.9], [0.6], [0.4], [0.8]])
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 0.5, 2)

  def test_binary_accuracy_threshold_ragged(self):
    acc_obj = metrics.BinaryAccuracy(threshold=0.7)
    self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))
    rt1 = tf.ragged.constant([[1], [1], [0], [0]])
    rt2 = tf.ragged.constant([[0.9], [0.6], [0.4], [0.8]])
    result_t = acc_obj(rt1, rt2)
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 0.5, 2)

  def test_categorical_accuracy(self):
    acc_obj = metrics.CategoricalAccuracy(name='my_acc')

    # check config
    self.assertEqual(acc_obj.name, 'my_acc')
    self.assertTrue(acc_obj.stateful)
    self.assertEqual(len(acc_obj.variables), 2)
    self.assertEqual(acc_obj.dtype, tf.float32)
    self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))

    # verify that correct value is returned
    update_op = acc_obj.update_state([[0, 0, 1], [0, 1, 0]],
                                     [[0.1, 0.1, 0.8], [0.05, 0.95, 0]])
    self.evaluate(update_op)
    result = self.evaluate(acc_obj.result())
    self.assertEqual(result, 1)  # 2/2

    # check with sample_weight
    result_t = acc_obj([[0, 0, 1], [0, 1, 0]],
                       [[0.1, 0.1, 0.8], [0.05, 0, 0.95]], [[0.5], [0.2]])
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 0.93, 2)  # 2.5/2.7

  def test_categorical_accuracy_ragged(self):
    acc_obj = metrics.CategoricalAccuracy(name='my_acc')
    self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))

    # verify that correct value is returned
    rt1 = tf.ragged.constant([[0, 0, 1], [0, 1, 0]])
    rt2 = tf.ragged.constant([[0.1, 0.1, 0.8], [0.05, 0.95, 0]])
    update_op = acc_obj.update_state(rt1, rt2)
    self.evaluate(update_op)
    result = self.evaluate(acc_obj.result())
    self.assertEqual(result, 1)  # 2/2

    # check with sample_weight
    rt1 = tf.ragged.constant([[0, 0, 1], [0, 1, 0]])
    rt2 = tf.ragged.constant([[0.1, 0.1, 0.8], [0.05, 0, 0.95]])
    sample_weight = tf.ragged.constant([[0.5], [0.2]])
    with self.assertRaises(tf.errors.InvalidArgumentError):
      result_t = acc_obj(rt1, rt2, sample_weight)
      result = self.evaluate(result_t)

  def test_sparse_categorical_accuracy(self):
    acc_obj = metrics.SparseCategoricalAccuracy(name='my_acc')

    # check config
    self.assertEqual(acc_obj.name, 'my_acc')
    self.assertTrue(acc_obj.stateful)
    self.assertEqual(len(acc_obj.variables), 2)
    self.assertEqual(acc_obj.dtype, tf.float32)
    self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))

    # verify that correct value is returned
    update_op = acc_obj.update_state([[2], [1]],
                                     [[0.1, 0.1, 0.8], [0.05, 0.95, 0]])
    self.evaluate(update_op)
    result = self.evaluate(acc_obj.result())
    self.assertEqual(result, 1)  # 2/2

    # check with sample_weight
    result_t = acc_obj([[2], [1]], [[0.1, 0.1, 0.8], [0.05, 0, 0.95]],
                       [[0.5], [0.2]])
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 0.93, 2)  # 2.5/2.7

  def test_sparse_categorical_accuracy_ragged(self):
    acc_obj = metrics.SparseCategoricalAccuracy(name='my_acc')

    # verify that correct value is returned
    rt1 = tf.ragged.constant([[2], [1]])
    rt2 = tf.ragged.constant([[0.1, 0.1, 0.8], [0.05, 0.95, 0]])

    with self.assertRaises(tf.errors.InvalidArgumentError):
      # sparse_categorical_accuracy is not supported for composite/ragged
      # tensors.
      update_op = acc_obj.update_state(rt1, rt2)
      self.evaluate(update_op)

  def test_sparse_categorical_accuracy_mismatched_dims(self):
    acc_obj = metrics.SparseCategoricalAccuracy(name='my_acc')

    # check config
    self.assertEqual(acc_obj.name, 'my_acc')
    self.assertTrue(acc_obj.stateful)
    self.assertEqual(len(acc_obj.variables), 2)
    self.assertEqual(acc_obj.dtype, tf.float32)
    self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))

    # verify that correct value is returned
    update_op = acc_obj.update_state([2, 1], [[0.1, 0.1, 0.8], [0.05, 0.95, 0]])
    self.evaluate(update_op)
    result = self.evaluate(acc_obj.result())
    self.assertEqual(result, 1)  # 2/2

    # check with sample_weight
    result_t = acc_obj([2, 1], [[0.1, 0.1, 0.8], [0.05, 0, 0.95]],
                       [[0.5], [0.2]])
    result = self.evaluate(result_t)
    self.assertAlmostEqual(result, 0.93, 2)  # 2.5/2.7

  def test_sparse_categorical_accuracy_mismatched_dims_dynamic(self):
    with tf.compat.v1.get_default_graph().as_default(), self.cached_session() as sess:
      acc_obj = metrics.SparseCategoricalAccuracy(name='my_acc')
      self.evaluate(tf.compat.v1.variables_initializer(acc_obj.variables))

      t = tf.compat.v1.placeholder(tf.float32)
      p = tf.compat.v1.placeholder(tf.float32)
      w = tf.compat.v1.placeholder(tf.float32)

      result_t = acc_obj(t, p, w)
      result = sess.run(
          result_t,
          feed_dict=({
              t: [2, 1],
              p: [[0.1, 0.1, 0.8], [0.05, 0, 0.95]],
              w: [[0.5], [0.2]]
          }))
      self.assertAlmostEqual(result, 0.71, 2)  # 2.5/2.7

  def test_get_acc(self):
    acc_fn = metrics.get('acc')
    self.assertEqual(acc_fn, metrics.accuracy)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class CosineSimilarityTest(tf.test.TestCase):

  def l2_norm(self, x, axis):
    epsilon = 1e-12
    square_sum = np.sum(np.square(x), axis=axis, keepdims=True)
    x_inv_norm = 1 / np.sqrt(np.maximum(square_sum, epsilon))
    return np.multiply(x, x_inv_norm)

  def setup(self, axis=1):
    self.np_y_true = np.asarray([[1, 9, 2], [-5, -2, 6]], dtype=np.float32)
    self.np_y_pred = np.asarray([[4, 8, 12], [8, 1, 3]], dtype=np.float32)

    y_true = self.l2_norm(self.np_y_true, axis)
    y_pred = self.l2_norm(self.np_y_pred, axis)
    self.expected_loss = np.sum(np.multiply(y_true, y_pred), axis=(axis,))

    self.y_true = tf.constant(self.np_y_true)
    self.y_pred = tf.constant(self.np_y_pred)

  def test_config(self):
    cosine_obj = metrics.CosineSimilarity(
        axis=2, name='my_cos', dtype=tf.int32)
    self.assertEqual(cosine_obj.name, 'my_cos')
    self.assertEqual(cosine_obj._dtype, tf.int32)

    # Check save and restore config
    cosine_obj2 = metrics.CosineSimilarity.from_config(cosine_obj.get_config())
    self.assertEqual(cosine_obj2.name, 'my_cos')
    self.assertEqual(cosine_obj2._dtype, tf.int32)

  def test_unweighted(self):
    self.setup()
    cosine_obj = metrics.CosineSimilarity()
    self.evaluate(tf.compat.v1.variables_initializer(cosine_obj.variables))
    loss = cosine_obj(self.y_true, self.y_pred)
    expected_loss = np.mean(self.expected_loss)
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_weighted(self):
    self.setup()
    cosine_obj = metrics.CosineSimilarity()
    self.evaluate(tf.compat.v1.variables_initializer(cosine_obj.variables))
    sample_weight = np.asarray([1.2, 3.4])
    loss = cosine_obj(
        self.y_true,
        self.y_pred,
        sample_weight=tf.constant(sample_weight))
    expected_loss = np.sum(
        self.expected_loss * sample_weight) / np.sum(sample_weight)
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)

  def test_axis(self):
    self.setup(axis=1)
    cosine_obj = metrics.CosineSimilarity(axis=1)
    self.evaluate(tf.compat.v1.variables_initializer(cosine_obj.variables))
    loss = cosine_obj(self.y_true, self.y_pred)
    expected_loss = np.mean(self.expected_loss)
    self.assertAlmostEqual(self.evaluate(loss), expected_loss, 3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class MeanAbsoluteErrorTest(tf.test.TestCase):

  def test_config(self):
    mae_obj = metrics.MeanAbsoluteError(name='my_mae', dtype=tf.int32)
    self.assertEqual(mae_obj.name, 'my_mae')
    self.assertEqual(mae_obj._dtype, tf.int32)

    # Check save and restore config
    mae_obj2 = metrics.MeanAbsoluteError.from_config(mae_obj.get_config())
    self.assertEqual(mae_obj2.name, 'my_mae')
    self.assertEqual(mae_obj2._dtype, tf.int32)

  def test_unweighted(self):
    mae_obj = metrics.MeanAbsoluteError()
    self.evaluate(tf.compat.v1.variables_initializer(mae_obj.variables))
    y_true = tf.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = tf.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))

    update_op = mae_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = mae_obj.result()
    self.assertAllClose(0.5, result, atol=1e-5)

  def test_weighted(self):
    mae_obj = metrics.MeanAbsoluteError()
    self.evaluate(tf.compat.v1.variables_initializer(mae_obj.variables))
    y_true = tf.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = tf.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))
    sample_weight = tf.constant((1., 1.5, 2., 2.5))
    result = mae_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(0.54285, self.evaluate(result), atol=1e-5)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class MeanAbsolutePercentageErrorTest(tf.test.TestCase):

  def test_config(self):
    mape_obj = metrics.MeanAbsolutePercentageError(
        name='my_mape', dtype=tf.int32)
    self.assertEqual(mape_obj.name, 'my_mape')
    self.assertEqual(mape_obj._dtype, tf.int32)

    # Check save and restore config
    mape_obj2 = metrics.MeanAbsolutePercentageError.from_config(
        mape_obj.get_config())
    self.assertEqual(mape_obj2.name, 'my_mape')
    self.assertEqual(mape_obj2._dtype, tf.int32)

  def test_unweighted(self):
    mape_obj = metrics.MeanAbsolutePercentageError()
    self.evaluate(tf.compat.v1.variables_initializer(mape_obj.variables))
    y_true = tf.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = tf.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))

    update_op = mape_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = mape_obj.result()
    self.assertAllClose(35e7, result, atol=1e-5)

  def test_weighted(self):
    mape_obj = metrics.MeanAbsolutePercentageError()
    self.evaluate(tf.compat.v1.variables_initializer(mape_obj.variables))
    y_true = tf.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = tf.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))
    sample_weight = tf.constant((1., 1.5, 2., 2.5))
    result = mape_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(40e7, self.evaluate(result), atol=1e-5)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class MeanSquaredErrorTest(tf.test.TestCase):

  def test_config(self):
    mse_obj = metrics.MeanSquaredError(name='my_mse', dtype=tf.int32)
    self.assertEqual(mse_obj.name, 'my_mse')
    self.assertEqual(mse_obj._dtype, tf.int32)

    # Check save and restore config
    mse_obj2 = metrics.MeanSquaredError.from_config(mse_obj.get_config())
    self.assertEqual(mse_obj2.name, 'my_mse')
    self.assertEqual(mse_obj2._dtype, tf.int32)

  def test_unweighted(self):
    mse_obj = metrics.MeanSquaredError()
    self.evaluate(tf.compat.v1.variables_initializer(mse_obj.variables))
    y_true = tf.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = tf.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))

    update_op = mse_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = mse_obj.result()
    self.assertAllClose(0.5, result, atol=1e-5)

  def test_weighted(self):
    mse_obj = metrics.MeanSquaredError()
    self.evaluate(tf.compat.v1.variables_initializer(mse_obj.variables))
    y_true = tf.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = tf.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))
    sample_weight = tf.constant((1., 1.5, 2., 2.5))
    result = mse_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(0.54285, self.evaluate(result), atol=1e-5)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class MeanSquaredLogarithmicErrorTest(tf.test.TestCase):

  def test_config(self):
    msle_obj = metrics.MeanSquaredLogarithmicError(
        name='my_msle', dtype=tf.int32)
    self.assertEqual(msle_obj.name, 'my_msle')
    self.assertEqual(msle_obj._dtype, tf.int32)

    # Check save and restore config
    msle_obj2 = metrics.MeanSquaredLogarithmicError.from_config(
        msle_obj.get_config())
    self.assertEqual(msle_obj2.name, 'my_msle')
    self.assertEqual(msle_obj2._dtype, tf.int32)

  def test_unweighted(self):
    msle_obj = metrics.MeanSquaredLogarithmicError()
    self.evaluate(tf.compat.v1.variables_initializer(msle_obj.variables))
    y_true = tf.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = tf.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))

    update_op = msle_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = msle_obj.result()
    self.assertAllClose(0.24022, result, atol=1e-5)

  def test_weighted(self):
    msle_obj = metrics.MeanSquaredLogarithmicError()
    self.evaluate(tf.compat.v1.variables_initializer(msle_obj.variables))
    y_true = tf.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = tf.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))
    sample_weight = tf.constant((1., 1.5, 2., 2.5))
    result = msle_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(0.26082, self.evaluate(result), atol=1e-5)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class HingeTest(tf.test.TestCase):

  def test_config(self):
    hinge_obj = metrics.Hinge(name='hinge', dtype=tf.int32)
    self.assertEqual(hinge_obj.name, 'hinge')
    self.assertEqual(hinge_obj._dtype, tf.int32)

    # Check save and restore config
    hinge_obj2 = metrics.Hinge.from_config(hinge_obj.get_config())
    self.assertEqual(hinge_obj2.name, 'hinge')
    self.assertEqual(hinge_obj2._dtype, tf.int32)

  def test_unweighted(self):
    hinge_obj = metrics.Hinge()
    self.evaluate(tf.compat.v1.variables_initializer(hinge_obj.variables))
    y_true = tf.constant([[0, 1, 0, 1], [0, 0, 1, 1]])
    y_pred = tf.constant([[-0.3, 0.2, -0.1, 1.6],
                                   [-0.25, -1., 0.5, 0.6]])

    # metric = max(0, 1-y_true * y_pred), where y_true is -1/1

    # y_true = [[-1, 1, -1, 1], [-1, -1, 1, 1]]
    # y_true * y_pred = [[0.3, 0.2, 0.1, 1.6], [0.25, 1, 0.5, 0.6]]
    # 1 - y_true * y_pred = [[0.7, 0.8, 0.9, -0.6], [0.75, 0, 0.5, 0.4]]
    # metric = [(0.7 + 0.8 + 0.9 + 0) / 4, (0.75 + 0 + 0.5 + 0.4) / 4]
    #        = [0.6, 0.4125]
    # reduced metric = (0.6 + 0.4125) / 2

    update_op = hinge_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = hinge_obj.result()
    self.assertAllClose(0.506, result, atol=1e-3)

  def test_weighted(self):
    hinge_obj = metrics.Hinge()
    self.evaluate(tf.compat.v1.variables_initializer(hinge_obj.variables))
    y_true = tf.constant([[-1, 1, -1, 1], [-1, -1, 1, 1]])
    y_pred = tf.constant([[-0.3, 0.2, -0.1, 1.6],
                                   [-0.25, -1., 0.5, 0.6]])
    sample_weight = tf.constant([1.5, 2.])

    # metric = max(0, 1-y_true * y_pred), where y_true is -1/1

    # y_true * y_pred = [[0.3, 0.2, 0.1, 1.6], [0.25, 1, 0.5, 0.6]]
    # 1 - y_true * y_pred = [[0.7, 0.8, 0.9, -0.6], [0.75, 0, 0.5, 0.4]]
    # metric = [(0.7 + 0.8 + 0.9 + 0) / 4, (0.75 + 0 + 0.5 + 0.4) / 4]
    #        = [0.6, 0.4125]
    # weighted metric = [0.6 * 1.5, 0.4125 * 2]
    # reduced metric = (0.6 * 1.5 + 0.4125 * 2) / (1.5 + 2)

    result = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(0.493, self.evaluate(result), atol=1e-3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class SquaredHingeTest(tf.test.TestCase):

  def test_config(self):
    sq_hinge_obj = metrics.SquaredHinge(name='sq_hinge', dtype=tf.int32)
    self.assertEqual(sq_hinge_obj.name, 'sq_hinge')
    self.assertEqual(sq_hinge_obj._dtype, tf.int32)

    # Check save and restore config
    sq_hinge_obj2 = metrics.SquaredHinge.from_config(sq_hinge_obj.get_config())
    self.assertEqual(sq_hinge_obj2.name, 'sq_hinge')
    self.assertEqual(sq_hinge_obj2._dtype, tf.int32)

  def test_unweighted(self):
    sq_hinge_obj = metrics.SquaredHinge()
    self.evaluate(tf.compat.v1.variables_initializer(sq_hinge_obj.variables))
    y_true = tf.constant([[0, 1, 0, 1], [0, 0, 1, 1]])
    y_pred = tf.constant([[-0.3, 0.2, -0.1, 1.6],
                                   [-0.25, -1., 0.5, 0.6]])

    # metric = max(0, 1-y_true * y_pred), where y_true is -1/1

    # y_true = [[-1, 1, -1, 1], [-1, -1, 1, 1]]
    # y_true * y_pred = [[0.3, 0.2, 0.1, 1.6], [0.25, 1, 0.5, 0.6]]
    # 1 - y_true * y_pred = [[0.7, 0.8, 0.9, -0.6], [0.75, 0, 0.5, 0.4]]
    # max(0, 1 - y_true * y_pred) = [[0.7, 0.8, 0.9, 0], [0.75, 0, 0.5, 0.4]]
    # squared(max(0, 1 - y_true * y_pred)) = [[0.49, 0.64, 0.81, 0],
    #                                         [0.5625, 0, 0.25, 0.16]]
    # metric = [(0.49 + 0.64 + 0.81 + 0) / 4, (0.5625 + 0 + 0.25 + 0.16) / 4]
    #        = [0.485, 0.2431]
    # reduced metric = (0.485 + 0.2431) / 2

    update_op = sq_hinge_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = sq_hinge_obj.result()
    self.assertAllClose(0.364, result, atol=1e-3)

  def test_weighted(self):
    sq_hinge_obj = metrics.SquaredHinge()
    self.evaluate(tf.compat.v1.variables_initializer(sq_hinge_obj.variables))
    y_true = tf.constant([[-1, 1, -1, 1], [-1, -1, 1, 1]])
    y_pred = tf.constant([[-0.3, 0.2, -0.1, 1.6],
                                   [-0.25, -1., 0.5, 0.6]])
    sample_weight = tf.constant([1.5, 2.])

    # metric = max(0, 1-y_true * y_pred), where y_true is -1/1

    # y_true * y_pred = [[0.3, 0.2, 0.1, 1.6], [0.25, 1, 0.5, 0.6]]
    # 1 - y_true * y_pred = [[0.7, 0.8, 0.9, -0.6], [0.75, 0, 0.5, 0.4]]
    # max(0, 1 - y_true * y_pred) = [[0.7, 0.8, 0.9, 0], [0.75, 0, 0.5, 0.4]]
    # squared(max(0, 1 - y_true * y_pred)) = [[0.49, 0.64, 0.81, 0],
    #                                         [0.5625, 0, 0.25, 0.16]]
    # metric = [(0.49 + 0.64 + 0.81 + 0) / 4, (0.5625 + 0 + 0.25 + 0.16) / 4]
    #        = [0.485, 0.2431]
    # weighted metric = [0.485 * 1.5, 0.2431 * 2]
    # reduced metric = (0.485 * 1.5 + 0.2431 * 2) / (1.5 + 2)

    result = sq_hinge_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(0.347, self.evaluate(result), atol=1e-3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class CategoricalHingeTest(tf.test.TestCase):

  def test_config(self):
    cat_hinge_obj = metrics.CategoricalHinge(
        name='cat_hinge', dtype=tf.int32)
    self.assertEqual(cat_hinge_obj.name, 'cat_hinge')
    self.assertEqual(cat_hinge_obj._dtype, tf.int32)

    # Check save and restore config
    cat_hinge_obj2 = metrics.CategoricalHinge.from_config(
        cat_hinge_obj.get_config())
    self.assertEqual(cat_hinge_obj2.name, 'cat_hinge')
    self.assertEqual(cat_hinge_obj2._dtype, tf.int32)

  def test_unweighted(self):
    cat_hinge_obj = metrics.CategoricalHinge()
    self.evaluate(tf.compat.v1.variables_initializer(cat_hinge_obj.variables))
    y_true = tf.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = tf.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))

    update_op = cat_hinge_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = cat_hinge_obj.result()
    self.assertAllClose(0.5, result, atol=1e-5)

  def test_weighted(self):
    cat_hinge_obj = metrics.CategoricalHinge()
    self.evaluate(tf.compat.v1.variables_initializer(cat_hinge_obj.variables))
    y_true = tf.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                                   (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
    y_pred = tf.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                                   (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))
    sample_weight = tf.constant((1., 1.5, 2., 2.5))
    result = cat_hinge_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(0.5, self.evaluate(result), atol=1e-5)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class RootMeanSquaredErrorTest(tf.test.TestCase):

  def test_config(self):
    rmse_obj = metrics.RootMeanSquaredError(name='rmse', dtype=tf.int32)
    self.assertEqual(rmse_obj.name, 'rmse')
    self.assertEqual(rmse_obj._dtype, tf.int32)

    rmse_obj2 = metrics.RootMeanSquaredError.from_config(rmse_obj.get_config())
    self.assertEqual(rmse_obj2.name, 'rmse')
    self.assertEqual(rmse_obj2._dtype, tf.int32)

  def test_unweighted(self):
    rmse_obj = metrics.RootMeanSquaredError()
    self.evaluate(tf.compat.v1.variables_initializer(rmse_obj.variables))
    y_true = tf.constant((2, 4, 6))
    y_pred = tf.constant((1, 3, 2))

    update_op = rmse_obj.update_state(y_true, y_pred)
    self.evaluate(update_op)
    result = rmse_obj.result()
    # error = [-1, -1, -4], square(error) = [1, 1, 16], mean = 18/3 = 6
    self.assertAllClose(math.sqrt(6), result, atol=1e-3)

  def test_weighted(self):
    rmse_obj = metrics.RootMeanSquaredError()
    self.evaluate(tf.compat.v1.variables_initializer(rmse_obj.variables))
    y_true = tf.constant((2, 4, 6, 8))
    y_pred = tf.constant((1, 3, 2, 3))
    sample_weight = tf.constant((0, 1, 0, 1))
    result = rmse_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(math.sqrt(13), self.evaluate(result), atol=1e-3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class TopKCategoricalAccuracyTest(tf.test.TestCase):

  def test_config(self):
    a_obj = metrics.TopKCategoricalAccuracy(name='topkca', dtype=tf.int32)
    self.assertEqual(a_obj.name, 'topkca')
    self.assertEqual(a_obj._dtype, tf.int32)

    a_obj2 = metrics.TopKCategoricalAccuracy.from_config(a_obj.get_config())
    self.assertEqual(a_obj2.name, 'topkca')
    self.assertEqual(a_obj2._dtype, tf.int32)

  def test_correctness(self):
    a_obj = metrics.TopKCategoricalAccuracy()
    self.evaluate(tf.compat.v1.variables_initializer(a_obj.variables))
    y_true = tf.constant([[0, 0, 1], [0, 1, 0]])
    y_pred = tf.constant([[0.1, 0.9, 0.8], [0.05, 0.95, 0]])

    result = a_obj(y_true, y_pred)
    self.assertEqual(1, self.evaluate(result))  # both the samples match

    # With `k` < 5.
    a_obj = metrics.TopKCategoricalAccuracy(k=1)
    self.evaluate(tf.compat.v1.variables_initializer(a_obj.variables))
    result = a_obj(y_true, y_pred)
    self.assertEqual(0.5, self.evaluate(result))  # only sample #2 matches

    # With `k` > 5.
    y_true = tf.constant([[0, 0, 1, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0, 0, 0]])
    y_pred = tf.constant([[0.5, 0.9, 0.1, 0.7, 0.6, 0.5, 0.4],
                                   [0.05, 0.95, 0, 0, 0, 0, 0]])
    a_obj = metrics.TopKCategoricalAccuracy(k=6)
    self.evaluate(tf.compat.v1.variables_initializer(a_obj.variables))
    result = a_obj(y_true, y_pred)
    self.assertEqual(0.5, self.evaluate(result))  # only 1 sample matches.

  def test_weighted(self):
    a_obj = metrics.TopKCategoricalAccuracy(k=2)
    self.evaluate(tf.compat.v1.variables_initializer(a_obj.variables))
    y_true = tf.constant([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    y_pred = tf.constant([[0, 0.9, 0.1], [0, 0.9, 0.1], [0, 0.9, 0.1]])
    sample_weight = tf.constant((1.0, 0.0, 1.0))
    result = a_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(1.0, self.evaluate(result), atol=1e-5)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class SparseTopKCategoricalAccuracyTest(tf.test.TestCase):

  def test_config(self):
    a_obj = metrics.SparseTopKCategoricalAccuracy(
        name='stopkca', dtype=tf.int32)
    self.assertEqual(a_obj.name, 'stopkca')
    self.assertEqual(a_obj._dtype, tf.int32)

    a_obj2 = metrics.SparseTopKCategoricalAccuracy.from_config(
        a_obj.get_config())
    self.assertEqual(a_obj2.name, 'stopkca')
    self.assertEqual(a_obj2._dtype, tf.int32)

  def test_correctness(self):
    a_obj = metrics.SparseTopKCategoricalAccuracy()
    self.evaluate(tf.compat.v1.variables_initializer(a_obj.variables))
    y_true = tf.constant([2, 1])
    y_pred = tf.constant([[0.1, 0.9, 0.8], [0.05, 0.95, 0]])

    result = a_obj(y_true, y_pred)
    self.assertEqual(1, self.evaluate(result))  # both the samples match

    # With `k` < 5.
    a_obj = metrics.SparseTopKCategoricalAccuracy(k=1)
    self.evaluate(tf.compat.v1.variables_initializer(a_obj.variables))
    result = a_obj(y_true, y_pred)
    self.assertEqual(0.5, self.evaluate(result))  # only sample #2 matches

    # With `k` > 5.
    y_pred = tf.constant([[0.5, 0.9, 0.1, 0.7, 0.6, 0.5, 0.4],
                                   [0.05, 0.95, 0, 0, 0, 0, 0]])
    a_obj = metrics.SparseTopKCategoricalAccuracy(k=6)
    self.evaluate(tf.compat.v1.variables_initializer(a_obj.variables))
    result = a_obj(y_true, y_pred)
    self.assertEqual(0.5, self.evaluate(result))  # only 1 sample matches.

  def test_weighted(self):
    a_obj = metrics.SparseTopKCategoricalAccuracy(k=2)
    self.evaluate(tf.compat.v1.variables_initializer(a_obj.variables))
    y_true = tf.constant([1, 0, 2])
    y_pred = tf.constant([[0, 0.9, 0.1], [0, 0.9, 0.1], [0, 0.9, 0.1]])
    sample_weight = tf.constant((1.0, 0.0, 1.0))
    result = a_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertAllClose(1.0, self.evaluate(result), atol=1e-5)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class LogCoshErrorTest(tf.test.TestCase):

  def setup(self):
    y_pred = np.asarray([1, 9, 2, -5, -2, 6]).reshape((2, 3))
    y_true = np.asarray([4, 8, 12, 8, 1, 3]).reshape((2, 3))

    self.batch_size = 6
    error = y_pred - y_true
    self.expected_results = np.log((np.exp(error) + np.exp(-error)) / 2)

    self.y_pred = tf.constant(y_pred, dtype=tf.float32)
    self.y_true = tf.constant(y_true)

  def test_config(self):
    logcosh_obj = metrics.LogCoshError(name='logcosh', dtype=tf.int32)
    self.assertEqual(logcosh_obj.name, 'logcosh')
    self.assertEqual(logcosh_obj._dtype, tf.int32)

  def test_unweighted(self):
    self.setup()
    logcosh_obj = metrics.LogCoshError()
    self.evaluate(tf.compat.v1.variables_initializer(logcosh_obj.variables))

    update_op = logcosh_obj.update_state(self.y_true, self.y_pred)
    self.evaluate(update_op)
    result = logcosh_obj.result()
    expected_result = np.sum(self.expected_results) / self.batch_size
    self.assertAllClose(result, expected_result, atol=1e-3)

  def test_weighted(self):
    self.setup()
    logcosh_obj = metrics.LogCoshError()
    self.evaluate(tf.compat.v1.variables_initializer(logcosh_obj.variables))
    sample_weight = tf.constant([1.2, 3.4], shape=(2, 1))
    result = logcosh_obj(self.y_true, self.y_pred, sample_weight=sample_weight)

    sample_weight = np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3))
    expected_result = np.multiply(self.expected_results, sample_weight)
    expected_result = np.sum(expected_result) / np.sum(sample_weight)
    self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class PoissonTest(tf.test.TestCase):

  def setup(self):
    y_pred = np.asarray([1, 9, 2, 5, 2, 6]).reshape((2, 3))
    y_true = np.asarray([4, 8, 12, 8, 1, 3]).reshape((2, 3))

    self.batch_size = 6
    self.expected_results = y_pred - np.multiply(y_true, np.log(y_pred))

    self.y_pred = tf.constant(y_pred, dtype=tf.float32)
    self.y_true = tf.constant(y_true)

  def test_config(self):
    poisson_obj = metrics.Poisson(name='poisson', dtype=tf.int32)
    self.assertEqual(poisson_obj.name, 'poisson')
    self.assertEqual(poisson_obj._dtype, tf.int32)

    poisson_obj2 = metrics.Poisson.from_config(poisson_obj.get_config())
    self.assertEqual(poisson_obj2.name, 'poisson')
    self.assertEqual(poisson_obj2._dtype, tf.int32)

  def test_unweighted(self):
    self.setup()
    poisson_obj = metrics.Poisson()
    self.evaluate(tf.compat.v1.variables_initializer(poisson_obj.variables))

    update_op = poisson_obj.update_state(self.y_true, self.y_pred)
    self.evaluate(update_op)
    result = poisson_obj.result()
    expected_result = np.sum(self.expected_results) / self.batch_size
    self.assertAllClose(result, expected_result, atol=1e-3)

  def test_weighted(self):
    self.setup()
    poisson_obj = metrics.Poisson()
    self.evaluate(tf.compat.v1.variables_initializer(poisson_obj.variables))
    sample_weight = tf.constant([1.2, 3.4], shape=(2, 1))

    result = poisson_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
    sample_weight = np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3))
    expected_result = np.multiply(self.expected_results, sample_weight)
    expected_result = np.sum(expected_result) / np.sum(sample_weight)
    self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class KLDivergenceTest(tf.test.TestCase):

  def setup(self):
    y_pred = np.asarray([.4, .9, .12, .36, .3, .4]).reshape((2, 3))
    y_true = np.asarray([.5, .8, .12, .7, .43, .8]).reshape((2, 3))

    self.batch_size = 2
    self.expected_results = np.multiply(y_true, np.log(y_true / y_pred))

    self.y_pred = tf.constant(y_pred, dtype=tf.float32)
    self.y_true = tf.constant(y_true)

  def test_config(self):
    k_obj = metrics.KLDivergence(name='kld', dtype=tf.int32)
    self.assertEqual(k_obj.name, 'kld')
    self.assertEqual(k_obj._dtype, tf.int32)

    k_obj2 = metrics.KLDivergence.from_config(k_obj.get_config())
    self.assertEqual(k_obj2.name, 'kld')
    self.assertEqual(k_obj2._dtype, tf.int32)

  def test_unweighted(self):
    self.setup()
    k_obj = metrics.KLDivergence()
    self.evaluate(tf.compat.v1.variables_initializer(k_obj.variables))

    update_op = k_obj.update_state(self.y_true, self.y_pred)
    self.evaluate(update_op)
    result = k_obj.result()
    expected_result = np.sum(self.expected_results) / self.batch_size
    self.assertAllClose(result, expected_result, atol=1e-3)

  def test_weighted(self):
    self.setup()
    k_obj = metrics.KLDivergence()
    self.evaluate(tf.compat.v1.variables_initializer(k_obj.variables))

    sample_weight = tf.constant([1.2, 3.4], shape=(2, 1))
    result = k_obj(self.y_true, self.y_pred, sample_weight=sample_weight)

    sample_weight = np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3))
    expected_result = np.multiply(self.expected_results, sample_weight)
    expected_result = np.sum(expected_result) / (1.2 + 3.4)
    self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class MeanRelativeErrorTest(tf.test.TestCase):

  def test_config(self):
    normalizer = tf.constant([1, 3], dtype=tf.float32)
    mre_obj = metrics.MeanRelativeError(normalizer=normalizer, name='mre')
    self.assertEqual(mre_obj.name, 'mre')
    self.assertArrayNear(self.evaluate(mre_obj.normalizer), [1, 3], 1e-1)

    mre_obj2 = metrics.MeanRelativeError.from_config(mre_obj.get_config())
    self.assertEqual(mre_obj2.name, 'mre')
    self.assertArrayNear(self.evaluate(mre_obj2.normalizer), [1, 3], 1e-1)

  def test_unweighted(self):
    np_y_pred = np.asarray([2, 4, 6, 8], dtype=np.float32)
    np_y_true = np.asarray([1, 3, 2, 3], dtype=np.float32)
    expected_error = np.mean(
        np.divide(np.absolute(np_y_pred - np_y_true), np_y_true))

    y_pred = tf.constant(np_y_pred, shape=(1, 4), dtype=tf.float32)
    y_true = tf.constant(np_y_true, shape=(1, 4))

    mre_obj = metrics.MeanRelativeError(normalizer=y_true)
    self.evaluate(tf.compat.v1.variables_initializer(mre_obj.variables))

    result = mre_obj(y_true, y_pred)
    self.assertAllClose(self.evaluate(result), expected_error, atol=1e-3)

  def test_weighted(self):
    np_y_pred = np.asarray([2, 4, 6, 8], dtype=np.float32)
    np_y_true = np.asarray([1, 3, 2, 3], dtype=np.float32)
    sample_weight = np.asarray([0.2, 0.3, 0.5, 0], dtype=np.float32)
    rel_errors = np.divide(np.absolute(np_y_pred - np_y_true), np_y_true)
    expected_error = np.sum(rel_errors * sample_weight)

    y_pred = tf.constant(np_y_pred, dtype=tf.float32)
    y_true = tf.constant(np_y_true)

    mre_obj = metrics.MeanRelativeError(normalizer=y_true)
    self.evaluate(tf.compat.v1.variables_initializer(mre_obj.variables))

    result = mre_obj(
        y_true, y_pred, sample_weight=tf.constant(sample_weight))
    self.assertAllClose(self.evaluate(result), expected_error, atol=1e-3)

  def test_zero_normalizer(self):
    y_pred = tf.constant([2, 4], dtype=tf.float32)
    y_true = tf.constant([1, 3])

    mre_obj = metrics.MeanRelativeError(normalizer=tf.zeros_like(y_true))
    self.evaluate(tf.compat.v1.variables_initializer(mre_obj.variables))

    result = mre_obj(y_true, y_pred)
    self.assertEqual(self.evaluate(result), 0)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class MeanIoUTest(tf.test.TestCase):

  def test_config(self):
    m_obj = metrics.MeanIoU(num_classes=2, name='mean_iou')
    self.assertEqual(m_obj.name, 'mean_iou')
    self.assertEqual(m_obj.num_classes, 2)

    m_obj2 = metrics.MeanIoU.from_config(m_obj.get_config())
    self.assertEqual(m_obj2.name, 'mean_iou')
    self.assertEqual(m_obj2.num_classes, 2)

  def test_unweighted(self):
    y_pred = [0, 1, 0, 1]
    y_true = [0, 0, 1, 1]

    m_obj = metrics.MeanIoU(num_classes=2)
    self.evaluate(tf.compat.v1.variables_initializer(m_obj.variables))

    result = m_obj(y_true, y_pred)

    # cm = [[1, 1],
    #       [1, 1]]
    # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
    # iou = true_positives / (sum_row + sum_col - true_positives))
    expected_result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2
    self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

  def test_weighted(self):
    y_pred = tf.constant([0, 1, 0, 1], dtype=tf.float32)
    y_true = tf.constant([0, 0, 1, 1])
    sample_weight = tf.constant([0.2, 0.3, 0.4, 0.1])

    m_obj = metrics.MeanIoU(num_classes=2)
    self.evaluate(tf.compat.v1.variables_initializer(m_obj.variables))

    result = m_obj(y_true, y_pred, sample_weight=sample_weight)

    # cm = [[0.2, 0.3],
    #       [0.4, 0.1]]
    # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5], true_positives = [0.2, 0.1]
    # iou = true_positives / (sum_row + sum_col - true_positives))
    expected_result = (0.2 / (0.6 + 0.5 - 0.2) + 0.1 / (0.4 + 0.5 - 0.1)) / 2
    self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

  def test_multi_dim_input(self):
    y_pred = tf.constant([[0, 1], [0, 1]], dtype=tf.float32)
    y_true = tf.constant([[0, 0], [1, 1]])
    sample_weight = tf.constant([[0.2, 0.3], [0.4, 0.1]])

    m_obj = metrics.MeanIoU(num_classes=2)
    self.evaluate(tf.compat.v1.variables_initializer(m_obj.variables))

    result = m_obj(y_true, y_pred, sample_weight=sample_weight)

    # cm = [[0.2, 0.3],
    #       [0.4, 0.1]]
    # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5], true_positives = [0.2, 0.1]
    # iou = true_positives / (sum_row + sum_col - true_positives))
    expected_result = (0.2 / (0.6 + 0.5 - 0.2) + 0.1 / (0.4 + 0.5 - 0.1)) / 2
    self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

  def test_zero_valid_entries(self):
    m_obj = metrics.MeanIoU(num_classes=2)
    self.evaluate(tf.compat.v1.variables_initializer(m_obj.variables))
    self.assertAllClose(self.evaluate(m_obj.result()), 0, atol=1e-3)

  def test_zero_and_non_zero_entries(self):
    y_pred = tf.constant([1], dtype=tf.float32)
    y_true = tf.constant([1])

    m_obj = metrics.MeanIoU(num_classes=2)
    self.evaluate(tf.compat.v1.variables_initializer(m_obj.variables))
    result = m_obj(y_true, y_pred)

    # cm = [[0, 0],
    #       [0, 1]]
    # sum_row = [0, 1], sum_col = [0, 1], true_positives = [0, 1]
    # iou = true_positives / (sum_row + sum_col - true_positives))
    expected_result = (0 + 1 / (1 + 1 - 1)) / 1
    self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)


class MeanTensorTest(tf.test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_config(self):
    with self.test_session():
      m = metrics.MeanTensor(name='mean_by_element')

      # check config
      self.assertEqual(m.name, 'mean_by_element')
      self.assertTrue(m.stateful)
      self.assertEqual(m.dtype, tf.float32)
      self.assertEmpty(m.variables)

      with self.assertRaisesRegex(ValueError, 'does not have any value yet'):
        m.result()

      self.evaluate(m([[3], [5], [3]]))
      self.assertAllEqual(m._shape, [3, 1])

      m2 = metrics.MeanTensor.from_config(m.get_config())
      self.assertEqual(m2.name, 'mean_by_element')
      self.assertTrue(m2.stateful)
      self.assertEqual(m2.dtype, tf.float32)
      self.assertEmpty(m2.variables)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_unweighted(self):
    with self.test_session():
      m = metrics.MeanTensor(dtype=tf.float64)

      # check __call__()
      self.assertAllClose(self.evaluate(m([100, 40])), [100, 40])
      self.assertAllClose(self.evaluate(m.total), [100, 40])
      self.assertAllClose(self.evaluate(m.count), [1, 1])

      # check update_state() and result() + state accumulation + tensor input
      update_op = m.update_state([
          tf.convert_to_tensor(1),
          tf.convert_to_tensor(5)
      ])
      self.evaluate(update_op)
      self.assertAllClose(self.evaluate(m.result()), [50.5, 22.5])
      self.assertAllClose(self.evaluate(m.total), [101, 45])
      self.assertAllClose(self.evaluate(m.count), [2, 2])

      # check reset_state()
      m.reset_state()
      self.assertAllClose(self.evaluate(m.total), [0, 0])
      self.assertAllClose(self.evaluate(m.count), [0, 0])

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_weighted(self):
    with self.test_session():
      m = metrics.MeanTensor(dtype=tf.float64)
      self.assertEqual(m.dtype, tf.float64)

      # check scalar weight
      result_t = m([100, 30], sample_weight=0.5)
      self.assertAllClose(self.evaluate(result_t), [100, 30])
      self.assertAllClose(self.evaluate(m.total), [50, 15])
      self.assertAllClose(self.evaluate(m.count), [0.5, 0.5])

      # check weights not scalar and weights rank matches values rank
      result_t = m([1, 5], sample_weight=[1, 0.2])
      result = self.evaluate(result_t)
      self.assertAllClose(result, [51 / 1.5, 16 / 0.7], 2)
      self.assertAllClose(self.evaluate(m.total), [51, 16])
      self.assertAllClose(self.evaluate(m.count), [1.5, 0.7])

      # check weights broadcast
      result_t = m([1, 2], sample_weight=0.5)
      self.assertAllClose(self.evaluate(result_t), [51.5 / 2, 17 / 1.2])
      self.assertAllClose(self.evaluate(m.total), [51.5, 17])
      self.assertAllClose(self.evaluate(m.count), [2, 1.2])

      # check weights squeeze
      result_t = m([1, 5], sample_weight=[[1], [0.2]])
      self.assertAllClose(self.evaluate(result_t), [52.5 / 3, 18 / 1.4])
      self.assertAllClose(self.evaluate(m.total), [52.5, 18])
      self.assertAllClose(self.evaluate(m.count), [3, 1.4])

      # check weights expand
      m = metrics.MeanTensor(dtype=tf.float64)
      self.evaluate(tf.compat.v1.variables_initializer(m.variables))
      result_t = m([[1], [5]], sample_weight=[1, 0.2])
      self.assertAllClose(self.evaluate(result_t), [[1], [5]])
      self.assertAllClose(self.evaluate(m.total), [[1], [1]])
      self.assertAllClose(self.evaluate(m.count), [[1], [0.2]])

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_invalid_value_shape(self):
    m = metrics.MeanTensor(dtype=tf.float64)
    m([1])
    with self.assertRaisesRegex(
        ValueError, 'MeanTensor input values must always have the same shape'):
      m([1, 5])

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_build_in_tf_function(self):
    """Ensure that variables are created correctly in a tf function."""
    m = metrics.MeanTensor(dtype=tf.float64)

    @tf.function
    def call_metric(x):
      return m(x)

    with self.test_session():
      self.assertAllClose(self.evaluate(call_metric([100, 40])), [100, 40])
      self.assertAllClose(self.evaluate(m.total), [100, 40])
      self.assertAllClose(self.evaluate(m.count), [1, 1])
      self.assertAllClose(self.evaluate(call_metric([20, 2])), [60, 21])

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_in_keras_model(self):
    class ModelWithMetric(Model):

      def __init__(self):
        super(ModelWithMetric, self).__init__()
        self.dense1 = layers.Dense(
            3, activation='relu', kernel_initializer='ones')
        self.dense2 = layers.Dense(
            1, activation='sigmoid', kernel_initializer='ones')
        self.mean_tensor = metrics.MeanTensor()

      def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        self.mean_tensor(self.dense1.kernel)
        return x

    model = ModelWithMetric()
    model.compile(
        loss='mae',
        optimizer='rmsprop',
        run_eagerly=True)

    x = np.ones((100, 4))
    y = np.zeros((100, 1))
    model.evaluate(x, y, batch_size=50)
    self.assertAllClose(self.evaluate(model.mean_tensor.result()),
                        np.ones((4, 3)))
    self.assertAllClose(self.evaluate(model.mean_tensor.total),
                        np.full((4, 3), 2))
    self.assertAllClose(self.evaluate(model.mean_tensor.count),
                        np.full((4, 3), 2))

    model.evaluate(x, y, batch_size=25)
    self.assertAllClose(self.evaluate(model.mean_tensor.result()),
                        np.ones((4, 3)))
    self.assertAllClose(self.evaluate(model.mean_tensor.total),
                        np.full((4, 3), 4))
    self.assertAllClose(self.evaluate(model.mean_tensor.count),
                        np.full((4, 3), 4))


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class BinaryCrossentropyTest(tf.test.TestCase):

  def test_config(self):
    bce_obj = metrics.BinaryCrossentropy(
        name='bce', dtype=tf.int32, label_smoothing=0.2)
    self.assertEqual(bce_obj.name, 'bce')
    self.assertEqual(bce_obj._dtype, tf.int32)

    old_config = bce_obj.get_config()
    self.assertAllClose(old_config['label_smoothing'], 0.2, 1e-3)

    # Check save and restore config
    bce_obj2 = metrics.BinaryCrossentropy.from_config(old_config)
    self.assertEqual(bce_obj2.name, 'bce')
    self.assertEqual(bce_obj2._dtype, tf.int32)
    new_config = bce_obj2.get_config()
    self.assertDictEqual(old_config, new_config)

  def test_unweighted(self):
    bce_obj = metrics.BinaryCrossentropy()
    self.evaluate(tf.compat.v1.variables_initializer(bce_obj.variables))
    y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
    y_pred = np.asarray([1, 1, 1, 0], dtype=np.float32).reshape([2, 2])
    result = bce_obj(y_true, y_pred)

    # EPSILON = 1e-7, y = y_true, y` = y_pred, Y_MAX = 0.9999999
    # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
    # y` = [Y_MAX, Y_MAX, Y_MAX, EPSILON]

    # Metric = -(y log(y` + EPSILON) + (1 - y) log(1 - y` + EPSILON))
    #        = [-log(Y_MAX + EPSILON), -log(1 - Y_MAX + EPSILON),
    #           -log(Y_MAX + EPSILON), -log(1)]
    #        = [(0 + 15.33) / 2, (0 + 0) / 2]
    # Reduced metric = 7.665 / 2

    self.assertAllClose(self.evaluate(result), 3.833, atol=1e-3)

  def test_unweighted_with_logits(self):
    bce_obj = metrics.BinaryCrossentropy(from_logits=True)
    self.evaluate(tf.compat.v1.variables_initializer(bce_obj.variables))

    y_true = tf.constant([[1, 0, 1], [0, 1, 1]])
    y_pred = tf.constant([[100.0, -100.0, 100.0],
                                   [100.0, 100.0, -100.0]])
    result = bce_obj(y_true, y_pred)

    # Metric = max(x, 0) - x * z + log(1 + exp(-abs(x)))
    #              (where x = logits and z = y_true)
    #        = [((100 - 100 * 1 + log(1 + exp(-100))) +
    #            (0 + 100 * 0 + log(1 + exp(-100))) +
    #            (100 - 100 * 1 + log(1 + exp(-100))),
    #           ((100 - 100 * 0 + log(1 + exp(-100))) +
    #            (100 - 100 * 1 + log(1 + exp(-100))) +
    #            (0 + 100 * 1 + log(1 + exp(-100))))]
    #        = [(0 + 0 + 0) / 3, 200 / 3]
    # Reduced metric = (0 + 66.666) / 2

    self.assertAllClose(self.evaluate(result), 33.333, atol=1e-3)

  def test_weighted(self):
    bce_obj = metrics.BinaryCrossentropy()
    self.evaluate(tf.compat.v1.variables_initializer(bce_obj.variables))
    y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
    y_pred = np.asarray([1, 1, 1, 0], dtype=np.float32).reshape([2, 2])
    sample_weight = tf.constant([1.5, 2.])
    result = bce_obj(y_true, y_pred, sample_weight=sample_weight)

    # EPSILON = 1e-7, y = y_true, y` = y_pred, Y_MAX = 0.9999999
    # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
    # y` = [Y_MAX, Y_MAX, Y_MAX, EPSILON]

    # Metric = -(y log(y` + EPSILON) + (1 - y) log(1 - y` + EPSILON))
    #        = [-log(Y_MAX + EPSILON), -log(1 - Y_MAX + EPSILON),
    #           -log(Y_MAX + EPSILON), -log(1)]
    #        = [(0 + 15.33) / 2, (0 + 0) / 2]
    # Weighted metric = [7.665 * 1.5, 0]
    # Reduced metric = 7.665 * 1.5 / (1.5 + 2)

    self.assertAllClose(self.evaluate(result), 3.285, atol=1e-3)

  def test_weighted_from_logits(self):
    bce_obj = metrics.BinaryCrossentropy(from_logits=True)
    self.evaluate(tf.compat.v1.variables_initializer(bce_obj.variables))
    y_true = tf.constant([[1, 0, 1], [0, 1, 1]])
    y_pred = tf.constant([[100.0, -100.0, 100.0],
                                   [100.0, 100.0, -100.0]])
    sample_weight = tf.constant([2., 2.5])
    result = bce_obj(y_true, y_pred, sample_weight=sample_weight)

    # Metric = max(x, 0) - x * z + log(1 + exp(-abs(x)))
    #              (where x = logits and z = y_true)
    #        = [(0 + 0 + 0) / 3, 200 / 3]
    # Weighted metric = [0, 66.666 * 2.5]
    # Reduced metric = 66.666 * 2.5 / (2 + 2.5)

    self.assertAllClose(self.evaluate(result), 37.037, atol=1e-3)

  def test_label_smoothing(self):
    logits = tf.constant(((100., -100., -100.)))
    y_true = tf.constant(((1, 0, 1)))
    label_smoothing = 0.1
    # Metric: max(x, 0) - x * z + log(1 + exp(-abs(x)))
    #             (where x = logits and z = y_true)
    # Label smoothing: z' = z * (1 - L) + 0.5L
    # After label smoothing, label 1 becomes 1 - 0.5L
    #                        label 0 becomes 0.5L
    # Applying the above two fns to the given input:
    # (100 - 100 * (1 - 0.5 L)  + 0 +
    #  0   + 100 * (0.5 L)      + 0 +
    #  0   + 100 * (1 - 0.5 L)  + 0) * (1/3)
    #  = (100 + 50L) * 1/3
    bce_obj = metrics.BinaryCrossentropy(
        from_logits=True, label_smoothing=label_smoothing)
    self.evaluate(tf.compat.v1.variables_initializer(bce_obj.variables))
    result = bce_obj(y_true, logits)
    expected_value = (100.0 + 50.0 * label_smoothing) / 3.0
    self.assertAllClose(expected_value, self.evaluate(result), atol=1e-3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class CategoricalCrossentropyTest(tf.test.TestCase):

  def test_config(self):
    cce_obj = metrics.CategoricalCrossentropy(
        name='cce', dtype=tf.int32, label_smoothing=0.2)
    self.assertEqual(cce_obj.name, 'cce')
    self.assertEqual(cce_obj._dtype, tf.int32)

    old_config = cce_obj.get_config()
    self.assertAllClose(old_config['label_smoothing'], 0.2, 1e-3)

    # Check save and restore config
    cce_obj2 = metrics.CategoricalCrossentropy.from_config(old_config)
    self.assertEqual(cce_obj2.name, 'cce')
    self.assertEqual(cce_obj2._dtype, tf.int32)
    new_config = cce_obj2.get_config()
    self.assertDictEqual(old_config, new_config)

  def test_unweighted(self):
    cce_obj = metrics.CategoricalCrossentropy()
    self.evaluate(tf.compat.v1.variables_initializer(cce_obj.variables))

    y_true = np.asarray([[0, 1, 0], [0, 0, 1]])
    y_pred = np.asarray([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    result = cce_obj(y_true, y_pred)

    # EPSILON = 1e-7, y = y_true, y` = y_pred
    # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
    # y` = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]

    # Metric = -sum(y * log(y'), axis = -1)
    #        = -((log 0.95), (log 0.1))
    #        = [0.051, 2.302]
    # Reduced metric = (0.051 + 2.302) / 2

    self.assertAllClose(self.evaluate(result), 1.176, atol=1e-3)

  def test_unweighted_from_logits(self):
    cce_obj = metrics.CategoricalCrossentropy(from_logits=True)
    self.evaluate(tf.compat.v1.variables_initializer(cce_obj.variables))

    y_true = np.asarray([[0, 1, 0], [0, 0, 1]])
    logits = np.asarray([[1, 9, 0], [1, 8, 1]], dtype=np.float32)
    result = cce_obj(y_true, logits)

    # softmax = exp(logits) / sum(exp(logits), axis=-1)
    # xent = -sum(labels * log(softmax), 1)

    # exp(logits) = [[2.718, 8103.084, 1], [2.718, 2980.958, 2.718]]
    # sum(exp(logits), axis=-1) = [8106.802, 2986.394]
    # softmax = [[0.00033, 0.99954, 0.00012], [0.00091, 0.99817, 0.00091]]
    # log(softmax) = [[-8.00045, -0.00045, -9.00045],
    #                 [-7.00182, -0.00182, -7.00182]]
    # labels * log(softmax) = [[0, -0.00045, 0], [0, 0, -7.00182]]
    # xent = [0.00045, 7.00182]
    # Reduced xent = (0.00045 + 7.00182) / 2

    self.assertAllClose(self.evaluate(result), 3.5011, atol=1e-3)

  def test_weighted(self):
    cce_obj = metrics.CategoricalCrossentropy()
    self.evaluate(tf.compat.v1.variables_initializer(cce_obj.variables))

    y_true = np.asarray([[0, 1, 0], [0, 0, 1]])
    y_pred = np.asarray([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    sample_weight = tf.constant([1.5, 2.])
    result = cce_obj(y_true, y_pred, sample_weight=sample_weight)

    # EPSILON = 1e-7, y = y_true, y` = y_pred
    # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
    # y` = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]

    # Metric = -sum(y * log(y'), axis = -1)
    #        = -((log 0.95), (log 0.1))
    #        = [0.051, 2.302]
    # Weighted metric = [0.051 * 1.5, 2.302 * 2.]
    # Reduced metric = (0.051 * 1.5 + 2.302 * 2.) / 3.5

    self.assertAllClose(self.evaluate(result), 1.338, atol=1e-3)

  def test_weighted_from_logits(self):
    cce_obj = metrics.CategoricalCrossentropy(from_logits=True)
    self.evaluate(tf.compat.v1.variables_initializer(cce_obj.variables))

    y_true = np.asarray([[0, 1, 0], [0, 0, 1]])
    logits = np.asarray([[1, 9, 0], [1, 8, 1]], dtype=np.float32)
    sample_weight = tf.constant([1.5, 2.])
    result = cce_obj(y_true, logits, sample_weight=sample_weight)

    # softmax = exp(logits) / sum(exp(logits), axis=-1)
    # xent = -sum(labels * log(softmax), 1)
    # xent = [0.00045, 7.00182]
    # weighted xent = [0.000675, 14.00364]
    # Reduced xent = (0.000675 + 14.00364) / (1.5 + 2)

    self.assertAllClose(self.evaluate(result), 4.0012, atol=1e-3)

  def test_label_smoothing(self):
    y_true = np.asarray([[0, 1, 0], [0, 0, 1]])
    logits = np.asarray([[1, 9, 0], [1, 8, 1]], dtype=np.float32)
    label_smoothing = 0.1

    # Label smoothing: z' = z * (1 - L) + L/n,
    #     where L = label smoothing value and n = num classes
    # Label value 1 becomes: 1 - L + L/n
    # Label value 0 becomes: L/n
    # y_true with label_smoothing = [[0.0333, 0.9333, 0.0333],
    #                               [0.0333, 0.0333, 0.9333]]

    # softmax = exp(logits) / sum(exp(logits), axis=-1)
    # xent = -sum(labels * log(softmax), 1)
    # log(softmax) = [[-8.00045, -0.00045, -9.00045],
    #                 [-7.00182, -0.00182, -7.00182]]
    # labels * log(softmax) = [[-0.26641, -0.00042, -0.29971],
    #                          [-0.23316, -0.00006, -6.53479]]
    # xent = [0.56654, 6.76801]
    # Reduced xent = (0.56654 + 6.76801) / 2

    cce_obj = metrics.CategoricalCrossentropy(
        from_logits=True, label_smoothing=label_smoothing)
    self.evaluate(tf.compat.v1.variables_initializer(cce_obj.variables))
    loss = cce_obj(y_true, logits)
    self.assertAllClose(self.evaluate(loss), 3.667, atol=1e-3)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class SparseCategoricalCrossentropyTest(tf.test.TestCase):

  def test_config(self):
    scce_obj = metrics.SparseCategoricalCrossentropy(
        name='scce', dtype=tf.int32)
    self.assertEqual(scce_obj.name, 'scce')
    self.assertEqual(scce_obj.dtype, tf.int32)
    old_config = scce_obj.get_config()
    self.assertDictEqual(old_config, json.loads(json.dumps(old_config)))

    # Check save and restore config
    scce_obj2 = metrics.SparseCategoricalCrossentropy.from_config(old_config)
    self.assertEqual(scce_obj2.name, 'scce')
    self.assertEqual(scce_obj2.dtype, tf.int32)
    new_config = scce_obj2.get_config()
    self.assertDictEqual(old_config, new_config)

  def test_unweighted(self):
    scce_obj = metrics.SparseCategoricalCrossentropy()
    self.evaluate(tf.compat.v1.variables_initializer(scce_obj.variables))

    y_true = np.asarray([1, 2])
    y_pred = np.asarray([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    result = scce_obj(y_true, y_pred)

    # EPSILON = 1e-7, y = y_true, y` = y_pred
    # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
    # y` = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
    # logits = log(y`) =  [[-2.9957, -0.0513, -16.1181],
    #                      [-2.3026, -0.2231, -2.3026]]

    # softmax = exp(logits) / sum(exp(logits), axis=-1)
    # y = one_hot(y) = [[0, 1, 0], [0, 0, 1]]
    # xent = -sum(y * log(softmax), 1)

    # exp(logits) = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
    # sum(exp(logits), axis=-1) = [1, 1]
    # softmax = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
    # log(softmax) = [[-2.9957, -0.0513, -16.1181],
    #                 [-2.3026, -0.2231, -2.3026]]
    # y * log(softmax) = [[0, -0.0513, 0], [0, 0, -2.3026]]
    # xent = [0.0513, 2.3026]
    # Reduced xent = (0.0513 + 2.3026) / 2

    self.assertAllClose(self.evaluate(result), 1.176, atol=1e-3)

  def test_unweighted_from_logits(self):
    scce_obj = metrics.SparseCategoricalCrossentropy(from_logits=True)
    self.evaluate(tf.compat.v1.variables_initializer(scce_obj.variables))

    y_true = np.asarray([1, 2])
    logits = np.asarray([[1, 9, 0], [1, 8, 1]], dtype=np.float32)
    result = scce_obj(y_true, logits)

    # softmax = exp(logits) / sum(exp(logits), axis=-1)
    # y_true = one_hot(y_true) = [[0, 1, 0], [0, 0, 1]]
    # xent = -sum(y_true * log(softmax), 1)

    # exp(logits) = [[2.718, 8103.084, 1], [2.718, 2980.958, 2.718]]
    # sum(exp(logits), axis=-1) = [8106.802, 2986.394]
    # softmax = [[0.00033, 0.99954, 0.00012], [0.00091, 0.99817, 0.00091]]
    # log(softmax) = [[-8.00045, -0.00045, -9.00045],
    #                 [-7.00182, -0.00182, -7.00182]]
    # y_true * log(softmax) = [[0, -0.00045, 0], [0, 0, -7.00182]]
    # xent = [0.00045, 7.00182]
    # Reduced xent = (0.00045 + 7.00182) / 2

    self.assertAllClose(self.evaluate(result), 3.5011, atol=1e-3)

  def test_weighted(self):
    scce_obj = metrics.SparseCategoricalCrossentropy()
    self.evaluate(tf.compat.v1.variables_initializer(scce_obj.variables))

    y_true = np.asarray([1, 2])
    y_pred = np.asarray([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    sample_weight = tf.constant([1.5, 2.])
    result = scce_obj(y_true, y_pred, sample_weight=sample_weight)

    # EPSILON = 1e-7, y = y_true, y` = y_pred
    # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
    # y` = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
    # logits = log(y`) =  [[-2.9957, -0.0513, -16.1181],
    #                      [-2.3026, -0.2231, -2.3026]]

    # softmax = exp(logits) / sum(exp(logits), axis=-1)
    # y = one_hot(y) = [[0, 1, 0], [0, 0, 1]]
    # xent = -sum(y * log(softmax), 1)

    # exp(logits) = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
    # sum(exp(logits), axis=-1) = [1, 1]
    # softmax = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
    # log(softmax) = [[-2.9957, -0.0513, -16.1181],
    #                 [-2.3026, -0.2231, -2.3026]]
    # y * log(softmax) = [[0, -0.0513, 0], [0, 0, -2.3026]]
    # xent = [0.0513, 2.3026]
    # Weighted xent = [0.051 * 1.5, 2.302 * 2.]
    # Reduced xent = (0.051 * 1.5 + 2.302 * 2.) / 3.5

    self.assertAllClose(self.evaluate(result), 1.338, atol=1e-3)

  def test_weighted_from_logits(self):
    scce_obj = metrics.SparseCategoricalCrossentropy(from_logits=True)
    self.evaluate(tf.compat.v1.variables_initializer(scce_obj.variables))

    y_true = np.asarray([1, 2])
    logits = np.asarray([[1, 9, 0], [1, 8, 1]], dtype=np.float32)
    sample_weight = tf.constant([1.5, 2.])
    result = scce_obj(y_true, logits, sample_weight=sample_weight)

    # softmax = exp(logits) / sum(exp(logits), axis=-1)
    # y_true = one_hot(y_true) = [[0, 1, 0], [0, 0, 1]]
    # xent = -sum(y_true * log(softmax), 1)
    # xent = [0.00045, 7.00182]
    # weighted xent = [0.000675, 14.00364]
    # Reduced xent = (0.000675 + 14.00364) / (1.5 + 2)

    self.assertAllClose(self.evaluate(result), 4.0012, atol=1e-3)

  def test_axis(self):
    scce_obj = metrics.SparseCategoricalCrossentropy(axis=0)
    self.evaluate(tf.compat.v1.variables_initializer(scce_obj.variables))

    y_true = np.asarray([1, 2])
    y_pred = np.asarray([[0.05, 0.1], [0.95, 0.8], [0, 0.1]])
    result = scce_obj(y_true, y_pred)

    # EPSILON = 1e-7, y = y_true, y` = y_pred
    # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
    # y` = [[0.05, 0.1], [0.95, 0.8], [EPSILON, 0.1]]
    # logits = log(y`) =  [[-2.9957, -2.3026],
    #                      [-0.0513, -0.2231],
    #                      [-16.1181, -2.3026]]

    # softmax = exp(logits) / sum(exp(logits), axis=-1)
    # y = one_hot(y) = [[0, 0], [1, 0], [0, 1]]
    # xent = -sum(y * log(softmax), 1)

    # exp(logits) = [[0.05, 0.1], [0.95, 0.8], [EPSILON, 0.1]]
    # sum(exp(logits)) = [1, 1]
    # softmax = [[0.05, 0.1], [0.95, 0.8], [EPSILON, 0.1]]
    # log(softmax) = [[-2.9957, -2.3026],
    #                 [-0.0513, -0.2231],
    #                 [-16.1181, -2.3026]]
    # y * log(softmax) = [[0, 0], [-0.0513, 0], [0, -2.3026]]
    # xent = [0.0513, 2.3026]
    # Reduced xent = (0.0513 + 2.3026) / 2

    self.assertAllClose(self.evaluate(result), 1.176, atol=1e-3)


class BinaryTruePositives(metrics.Metric):

  def __init__(self, name='binary_true_positives', **kwargs):
    super(BinaryTruePositives, self).__init__(name=name, **kwargs)
    self.true_positives = self.add_weight(name='tp', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)

    values = tf.logical_and(
        tf.equal(y_true, True), tf.equal(y_pred, True))
    values = tf.cast(values, self.dtype)
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, dtype=self.dtype)
      sample_weight = tf.__internal__.ops.broadcast_weights(
          sample_weight, values)
      values = tf.multiply(values, sample_weight)
    self.true_positives.assign_add(tf.reduce_sum(values))

  def result(self):
    return self.true_positives


class BinaryTruePositivesViaControlFlow(metrics.Metric):

  def __init__(self, name='binary_true_positives', **kwargs):
    super(BinaryTruePositivesViaControlFlow, self).__init__(name=name, **kwargs)
    self.true_positives = self.add_weight(name='tp', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)

    for i in range(len(y_true)):
      for j in range(len(y_true[i])):
        if y_true[i][j] and y_pred[i][j]:
          if sample_weight is None:
            self.true_positives.assign_add(1)
          else:
            self.true_positives.assign_add(sample_weight[i][0])

  def result(self):
    if tf.constant(True):
      return self.true_positives
    return 0.0


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class CustomMetricsTest(tf.test.TestCase):

  def test_config(self):
    btp_obj = BinaryTruePositives(name='btp', dtype=tf.int32)
    self.assertEqual(btp_obj.name, 'btp')
    self.assertEqual(btp_obj.dtype, tf.int32)

    # Check save and restore config
    btp_obj2 = BinaryTruePositives.from_config(btp_obj.get_config())
    self.assertEqual(btp_obj2.name, 'btp')
    self.assertEqual(btp_obj2.dtype, tf.int32)

  def test_unweighted(self):
    btp_obj = BinaryTruePositives()
    self.evaluate(tf.compat.v1.variables_initializer(btp_obj.variables))
    y_true = tf.constant([[0, 0.9, 0, 1, 0], [0, 0, 1, 1, 1],
                                   [1, 1, 1, 1, 0], [0, 0, 0, 0, 1.5]])
    y_pred = tf.constant([[0, 0, 1, 5, 0], [1, 1, 1, 1, 1],
                                   [0, 1, 0, 1, 0], [1, 10, 1, 1, 1]])

    update_op = btp_obj.update_state(y_true, y_pred)  # pylint: disable=assignment-from-no-return
    self.evaluate(update_op)
    result = btp_obj.result()
    self.assertEqual(7, self.evaluate(result))

  def test_weighted(self):
    btp_obj = BinaryTruePositives()
    self.evaluate(tf.compat.v1.variables_initializer(btp_obj.variables))
    y_true = tf.constant([[0, 0.9, 0, 1, 0], [0, 0, 1, 1, 1],
                                   [1, 1, 1, 1, 0], [0, 0, 0, 0, 1.5]])
    y_pred = tf.constant([[0, 0, 1, 5, 0], [1, 1, 1, 1, 1],
                                   [0, 1, 0, 1, 0], [1, 10, 1, 1, 1]])
    sample_weight = tf.constant([[1.], [1.5], [2.], [2.5]])
    result = btp_obj(y_true, y_pred, sample_weight=sample_weight)
    self.assertEqual(12, self.evaluate(result))

  def test_autograph(self):
    metric = BinaryTruePositivesViaControlFlow()
    self.evaluate(tf.compat.v1.variables_initializer(metric.variables))
    y_true = tf.constant([[0, 0.9, 0, 1, 0], [0, 0, 1, 1, 1],
                                   [1, 1, 1, 1, 0], [0, 0, 0, 0, 1.5]])
    y_pred = tf.constant([[0, 0, 1, 5, 0], [1, 1, 1, 1, 1],
                                   [0, 1, 0, 1, 0], [1, 10, 1, 1, 1]])
    sample_weight = tf.constant([[1.], [1.5], [2.], [2.5]])

    @tf.function
    def compute_metric(y_true, y_pred, sample_weight):
      metric(y_true, y_pred, sample_weight)
      return metric.result()

    result = compute_metric(y_true, y_pred, sample_weight)
    self.assertEqual(12, self.evaluate(result))

  def test_metric_wrappers_autograph(self):
    def metric_fn(y_true, y_pred):
      x = tf.constant(0.0)
      for i in range(len(y_true)):
        for j in range(len(y_true[i])):
          if tf.equal(y_true[i][j], y_pred[i][j]) and y_true[i][j] > 0:
            x += 1.0
      return x

    mean_metric = metrics.MeanMetricWrapper(metric_fn)
    sum_metric = metrics.SumOverBatchSizeMetricWrapper(metric_fn)
    self.evaluate(tf.compat.v1.variables_initializer(mean_metric.variables))
    self.evaluate(tf.compat.v1.variables_initializer(sum_metric.variables))

    y_true = tf.constant([[0, 0, 0, 1, 0],
                                   [0, 0, 1, 1, 1],
                                   [1, 1, 1, 1, 0],
                                   [1, 1, 1, 0, 1]])
    y_pred = tf.constant([[0, 0, 1, 1, 0],
                                   [1, 1, 1, 1, 1],
                                   [0, 1, 0, 1, 0],
                                   [1, 1, 1, 1, 1]])

    @tf.function
    def tf_functioned_metric_fn(metric, y_true, y_pred):
      return metric(y_true, y_pred)

    metric_result = tf_functioned_metric_fn(mean_metric, y_true, y_pred)
    self.assertAllClose(self.evaluate(metric_result), 10, 1e-2)
    metric_result = tf_functioned_metric_fn(sum_metric, y_true, y_pred)
    self.assertAllClose(self.evaluate(metric_result), 10, 1e-2)

  def test_metric_not_tracked_as_sublayer_in_layer(self):

    class MyLayer(base_layer.Layer):

      def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.mean_obj = metrics.Mean(name='my_mean_obj')

      def call(self, x):
        self.add_metric(
            tf.reduce_sum(x), aggregation='mean', name='my_mean_tensor')
        self.add_metric(self.mean_obj(x))
        return x

    layer = MyLayer()
    x = np.ones((1, 1))
    layer(x)
    self.assertLen(list(layer._flatten_layers(include_self=False)), 0)
    self.assertLen(layer.metrics, 2)

  def test_metric_not_tracked_as_sublayer_in_model(self):

    class MyModel(training_module.Model):

      def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.mean_obj = metrics.Mean(name='my_mean_obj')

      def call(self, x):
        self.add_metric(
            tf.reduce_sum(x), aggregation='mean', name='my_mean_tensor')
        self.add_metric(self.mean_obj(x))
        return x

    model = MyModel()
    x = np.ones((1, 1))
    model(x)
    self.assertLen(list(model._flatten_layers(include_self=False)), 0)
    self.assertLen(model.layers, 0)
    self.assertLen(model.metrics, 2)

  def test_invalid_custom_metric_class_error_msg(self):
    x = layers.Input(shape=(2,))
    y = layers.Dense(3)(x)
    model = training_module.Model(x, y)

    class BadMetric(metrics.Metric):

      def update_state(self, y_true, y_pred, sample_weight=None):
        return

      def result(self):
        return

    with self.assertRaisesRegex(RuntimeError,
                                'can only be a single'):
      model.compile('sgd',
                    'mse',
                    metrics=[BadMetric()])
      model.fit(np.ones((10, 2)), np.ones((10, 3)))

  def test_invalid_custom_metric_fn_error_msg(self):
    x = layers.Input(shape=(2,))
    y = layers.Dense(3)(x)
    model = training_module.Model(x, y)

    def bad_metric(y_true, y_pred, sample_weight=None):  # pylint: disable=unused-argument
      return None

    def dict_metric(y_true, y_pred, sample_weight=None):  # pylint: disable=unused-argument
      return {'value': 0.}

    with self.assertRaisesRegex(RuntimeError,
                                'The output of a metric function can only be'):
      model.compile('sgd',
                    'mse',
                    metrics=[bad_metric])
      model.fit(np.ones((10, 2)), np.ones((10, 3)))
    with self.assertRaisesRegex(RuntimeError,
                                'To return a dict of values, implement'):
      model.compile('sgd',
                    'mse',
                    metrics=[dict_metric])
      model.fit(np.ones((10, 2)), np.ones((10, 3)))


def _get_model(compile_metrics):
  model_layers = [
      layers.Dense(3, activation='relu', kernel_initializer='ones'),
      layers.Dense(1, activation='sigmoid', kernel_initializer='ones')]

  model = testing_utils.get_model_from_layers(model_layers, input_shape=(4,))
  model.compile(
      loss='mae',
      metrics=compile_metrics,
      optimizer='rmsprop',
      run_eagerly=testing_utils.should_run_eagerly())
  return model


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes
class ResetStatesTest(keras_parameterized.TestCase):

  def test_reset_state_false_positives(self):
    fp_obj = metrics.FalsePositives()
    model = _get_model([fp_obj])
    x = np.ones((100, 4))
    y = np.zeros((100, 1))
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(fp_obj.accumulator), 100.)
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(fp_obj.accumulator), 100.)

  def test_reset_state_false_negatives(self):
    fn_obj = metrics.FalseNegatives()
    model = _get_model([fn_obj])
    x = np.zeros((100, 4))
    y = np.ones((100, 1))
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(fn_obj.accumulator), 100.)
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(fn_obj.accumulator), 100.)

  def test_reset_state_true_negatives(self):
    tn_obj = metrics.TrueNegatives()
    model = _get_model([tn_obj])
    x = np.zeros((100, 4))
    y = np.zeros((100, 1))
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(tn_obj.accumulator), 100.)
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(tn_obj.accumulator), 100.)

  def test_reset_state_true_positives(self):
    tp_obj = metrics.TruePositives()
    model = _get_model([tp_obj])
    x = np.ones((100, 4))
    y = np.ones((100, 1))
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(tp_obj.accumulator), 100.)
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(tp_obj.accumulator), 100.)

  def test_reset_state_precision(self):
    p_obj = metrics.Precision()
    model = _get_model([p_obj])
    x = np.concatenate((np.ones((50, 4)), np.ones((50, 4))))
    y = np.concatenate((np.ones((50, 1)), np.zeros((50, 1))))
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(p_obj.true_positives), 50.)
    self.assertEqual(self.evaluate(p_obj.false_positives), 50.)
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(p_obj.true_positives), 50.)
    self.assertEqual(self.evaluate(p_obj.false_positives), 50.)

  def test_reset_state_recall(self):
    r_obj = metrics.Recall()
    model = _get_model([r_obj])
    x = np.concatenate((np.ones((50, 4)), np.zeros((50, 4))))
    y = np.concatenate((np.ones((50, 1)), np.ones((50, 1))))
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(r_obj.true_positives), 50.)
    self.assertEqual(self.evaluate(r_obj.false_negatives), 50.)
    model.evaluate(x, y)
    self.assertEqual(self.evaluate(r_obj.true_positives), 50.)
    self.assertEqual(self.evaluate(r_obj.false_negatives), 50.)

  def test_reset_state_sensitivity_at_specificity(self):
    s_obj = metrics.SensitivityAtSpecificity(0.5, num_thresholds=1)
    model = _get_model([s_obj])
    x = np.concatenate((np.ones((25, 4)), np.zeros((25, 4)), np.zeros((25, 4)),
                        np.ones((25, 4))))
    y = np.concatenate((np.ones((25, 1)), np.zeros((25, 1)), np.ones((25, 1)),
                        np.zeros((25, 1))))

    for _ in range(2):
      model.evaluate(x, y)
      self.assertEqual(self.evaluate(s_obj.true_positives), 25.)
      self.assertEqual(self.evaluate(s_obj.false_positives), 25.)
      self.assertEqual(self.evaluate(s_obj.false_negatives), 25.)
      self.assertEqual(self.evaluate(s_obj.true_negatives), 25.)

  def test_reset_state_specificity_at_sensitivity(self):
    s_obj = metrics.SpecificityAtSensitivity(0.5, num_thresholds=1)
    model = _get_model([s_obj])
    x = np.concatenate((np.ones((25, 4)), np.zeros((25, 4)), np.zeros((25, 4)),
                        np.ones((25, 4))))
    y = np.concatenate((np.ones((25, 1)), np.zeros((25, 1)), np.ones((25, 1)),
                        np.zeros((25, 1))))

    for _ in range(2):
      model.evaluate(x, y)
      self.assertEqual(self.evaluate(s_obj.true_positives), 25.)
      self.assertEqual(self.evaluate(s_obj.false_positives), 25.)
      self.assertEqual(self.evaluate(s_obj.false_negatives), 25.)
      self.assertEqual(self.evaluate(s_obj.true_negatives), 25.)

  def test_reset_state_precision_at_recall(self):
    s_obj = metrics.PrecisionAtRecall(recall=0.5, num_thresholds=1)
    model = _get_model([s_obj])
    x = np.concatenate((np.ones((25, 4)), np.zeros((25, 4)), np.zeros((25, 4)),
                        np.ones((25, 4))))
    y = np.concatenate((np.ones((25, 1)), np.zeros((25, 1)), np.ones((25, 1)),
                        np.zeros((25, 1))))

    for _ in range(2):
      model.evaluate(x, y)
      self.assertEqual(self.evaluate(s_obj.true_positives), 25.)
      self.assertEqual(self.evaluate(s_obj.false_positives), 25.)
      self.assertEqual(self.evaluate(s_obj.false_negatives), 25.)
      self.assertEqual(self.evaluate(s_obj.true_negatives), 25.)

  def test_reset_state_recall_at_precision(self):
    s_obj = metrics.RecallAtPrecision(precision=0.5, num_thresholds=1)
    model = _get_model([s_obj])
    x = np.concatenate((np.ones((25, 4)), np.zeros((25, 4)), np.zeros((25, 4)),
                        np.ones((25, 4))))
    y = np.concatenate((np.ones((25, 1)), np.zeros((25, 1)), np.ones((25, 1)),
                        np.zeros((25, 1))))

    for _ in range(2):
      model.evaluate(x, y)
      self.assertEqual(self.evaluate(s_obj.true_positives), 25.)
      self.assertEqual(self.evaluate(s_obj.false_positives), 25.)
      self.assertEqual(self.evaluate(s_obj.false_negatives), 25.)
      self.assertEqual(self.evaluate(s_obj.true_negatives), 25.)

  def test_reset_state_auc(self):
    auc_obj = metrics.AUC(num_thresholds=3)
    model = _get_model([auc_obj])
    x = np.concatenate((np.ones((25, 4)), np.zeros((25, 4)), np.zeros((25, 4)),
                        np.ones((25, 4))))
    y = np.concatenate((np.ones((25, 1)), np.zeros((25, 1)), np.ones((25, 1)),
                        np.zeros((25, 1))))

    for _ in range(2):
      model.evaluate(x, y)
      self.assertEqual(self.evaluate(auc_obj.true_positives[1]), 25.)
      self.assertEqual(self.evaluate(auc_obj.false_positives[1]), 25.)
      self.assertEqual(self.evaluate(auc_obj.false_negatives[1]), 25.)
      self.assertEqual(self.evaluate(auc_obj.true_negatives[1]), 25.)

  def test_reset_state_auc_from_logits(self):
    auc_obj = metrics.AUC(num_thresholds=3, from_logits=True)

    model_layers = [layers.Dense(1, kernel_initializer='ones', use_bias=False)]
    model = testing_utils.get_model_from_layers(model_layers, input_shape=(4,))
    model.compile(
        loss='mae',
        metrics=[auc_obj],
        optimizer='rmsprop',
        run_eagerly=testing_utils.should_run_eagerly())

    x = np.concatenate((np.ones((25, 4)), -np.ones((25, 4)), -np.ones(
        (25, 4)), np.ones((25, 4))))
    y = np.concatenate((np.ones((25, 1)), np.zeros((25, 1)), np.ones(
        (25, 1)), np.zeros((25, 1))))

    for _ in range(2):
      model.evaluate(x, y)
      self.assertEqual(self.evaluate(auc_obj.true_positives[1]), 25.)
      self.assertEqual(self.evaluate(auc_obj.false_positives[1]), 25.)
      self.assertEqual(self.evaluate(auc_obj.false_negatives[1]), 25.)
      self.assertEqual(self.evaluate(auc_obj.true_negatives[1]), 25.)

  def test_reset_state_auc_manual_thresholds(self):
    auc_obj = metrics.AUC(thresholds=[0.5])
    model = _get_model([auc_obj])
    x = np.concatenate((np.ones((25, 4)), np.zeros((25, 4)), np.zeros((25, 4)),
                        np.ones((25, 4))))
    y = np.concatenate((np.ones((25, 1)), np.zeros((25, 1)), np.ones((25, 1)),
                        np.zeros((25, 1))))

    for _ in range(2):
      model.evaluate(x, y)
      self.assertEqual(self.evaluate(auc_obj.true_positives[1]), 25.)
      self.assertEqual(self.evaluate(auc_obj.false_positives[1]), 25.)
      self.assertEqual(self.evaluate(auc_obj.false_negatives[1]), 25.)
      self.assertEqual(self.evaluate(auc_obj.true_negatives[1]), 25.)

  def test_reset_state_mean_iou(self):
    m_obj = metrics.MeanIoU(num_classes=2)
    model = _get_model([m_obj])
    x = np.asarray([[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
                   dtype=np.float32)
    y = np.asarray([[0], [1], [1], [1]], dtype=np.float32)
    model.evaluate(x, y)
    self.assertArrayNear(self.evaluate(m_obj.total_cm)[0], [1, 0], 1e-1)
    self.assertArrayNear(self.evaluate(m_obj.total_cm)[1], [3, 0], 1e-1)
    model.evaluate(x, y)
    self.assertArrayNear(self.evaluate(m_obj.total_cm)[0], [1, 0], 1e-1)
    self.assertArrayNear(self.evaluate(m_obj.total_cm)[1], [3, 0], 1e-1)

  def test_reset_state_recall_float64(self):
    # Test case for GitHub issue 36790.
    try:
      backend.set_floatx('float64')
      r_obj = metrics.Recall()
      model = _get_model([r_obj])
      x = np.concatenate((np.ones((50, 4)), np.zeros((50, 4))))
      y = np.concatenate((np.ones((50, 1)), np.ones((50, 1))))
      model.evaluate(x, y)
      self.assertEqual(self.evaluate(r_obj.true_positives), 50.)
      self.assertEqual(self.evaluate(r_obj.false_negatives), 50.)
      model.evaluate(x, y)
      self.assertEqual(self.evaluate(r_obj.true_positives), 50.)
      self.assertEqual(self.evaluate(r_obj.false_negatives), 50.)
    finally:
      backend.set_floatx('float32')


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class MergeStateTest(keras_parameterized.TestCase):

  def test_merge_state_incompatible_metrics(self):
    with self.assertRaisesRegex(ValueError,
                                'Metric .* is not compatible with .*'):
      obj1 = metrics.FalsePositives()
      self.evaluate(tf.compat.v1.variables_initializer(obj1.variables))
      obj2 = metrics.Accuracy()
      self.evaluate(tf.compat.v1.variables_initializer(obj2.variables))
      self.evaluate(obj1.merge_state([obj2]))

  def test_merge_state_accuracy(self):
    a_objs = []
    for y_true, y_pred in zip([[[1], [2]], [[3], [4]]],
                              [[[0], [2]], [[3], [4]]]):
      a_obj = metrics.Accuracy()
      a_objs.append(a_obj)
      self.evaluate(tf.compat.v1.variables_initializer(a_obj.variables))
      self.evaluate(a_obj.update_state(y_true, y_pred))
    self.evaluate(a_objs[0].merge_state(a_objs[1:]))
    self.assertEqual(self.evaluate(a_objs[0].total), 3.)
    self.assertEqual(self.evaluate(a_objs[0].count), 4.)
    self.assertEqual(self.evaluate(a_objs[0].result()), 0.75)

  def test_merge_state_false_positives(self):
    fp_objs = []
    for _ in range(4):
      fp_obj = metrics.FalsePositives()
      fp_objs.append(fp_obj)
      self.evaluate(tf.compat.v1.variables_initializer(fp_obj.variables))
      y_true = np.zeros((25, 1))
      y_pred = np.ones((25, 1))
      self.evaluate(fp_obj.update_state(y_true, y_pred))
    self.evaluate(fp_objs[0].merge_state(fp_objs[1:]))
    self.assertEqual(self.evaluate(fp_objs[0].accumulator), 100.)

  def test_merge_state_false_negatives(self):
    fn_objs = []
    for _ in range(4):
      fn_obj = metrics.FalseNegatives()
      fn_objs.append(fn_obj)
      self.evaluate(tf.compat.v1.variables_initializer(fn_obj.variables))
      y_true = np.ones((25, 1))
      y_pred = np.zeros((25, 1))
      self.evaluate(fn_obj.update_state(y_true, y_pred))
    self.evaluate(fn_objs[0].merge_state(fn_objs[1:]))
    self.assertEqual(self.evaluate(fn_objs[0].accumulator), 100.)

  def test_merge_state_true_negatives(self):
    tn_objs = []
    for _ in range(4):
      tn_obj = metrics.TrueNegatives()
      tn_objs.append(tn_obj)
      self.evaluate(tf.compat.v1.variables_initializer(tn_obj.variables))
      y_true = np.zeros((25, 1))
      y_pred = np.zeros((25, 1))
      self.evaluate(tn_obj.update_state(y_true, y_pred))
    self.evaluate(tn_objs[0].merge_state(tn_objs[1:]))
    self.assertEqual(self.evaluate(tn_objs[0].accumulator), 100.)

  def test_merge_state_true_positives(self):
    tp_objs = []
    for _ in range(4):
      tp_obj = metrics.TruePositives()
      tp_objs.append(tp_obj)
      self.evaluate(tf.compat.v1.variables_initializer(tp_obj.variables))
      y_true = np.ones((25, 1))
      y_pred = np.ones((25, 1))
      self.evaluate(tp_obj.update_state(y_true, y_pred))
    self.evaluate(tp_objs[0].merge_state(tp_objs[1:]))
    self.assertEqual(self.evaluate(tp_objs[0].accumulator), 100.)

  def test_merge_state_precision(self):
    p_objs = []
    for _ in range(5):
      p_obj = metrics.Precision()
      p_objs.append(p_obj)
      self.evaluate(tf.compat.v1.variables_initializer(p_obj.variables))
      y_true = np.concatenate((np.ones((10, 1)), np.zeros((10, 1))))
      y_pred = np.concatenate((np.ones((10, 1)), np.ones((10, 1))))
      self.evaluate(p_obj.update_state(y_true, y_pred))
    self.evaluate(p_objs[0].merge_state(p_objs[1:]))
    self.assertEqual(self.evaluate(p_objs[0].true_positives), 50.)
    self.assertEqual(self.evaluate(p_objs[0].false_positives), 50.)

  def test_merge_state_recall(self):
    r_objs = []
    for _ in range(5):
      r_obj = metrics.Recall()
      r_objs.append(r_obj)
      self.evaluate(tf.compat.v1.variables_initializer(r_obj.variables))
      y_true = np.concatenate((np.ones((10, 1)), np.ones((10, 1))))
      y_pred = np.concatenate((np.ones((10, 1)), np.zeros((10, 1))))
      self.evaluate(r_obj.update_state(y_true, y_pred))
    self.evaluate(r_objs[0].merge_state(r_objs[1:]))
    self.assertEqual(self.evaluate(r_objs[0].true_positives), 50.)
    self.assertEqual(self.evaluate(r_objs[0].false_negatives), 50.)

  def test_merge_state_sensitivity_at_specificity(self):
    sas_objs = []
    for _ in range(5):
      sas_obj = metrics.SensitivityAtSpecificity(0.5, num_thresholds=1)
      sas_objs.append(sas_obj)
      self.evaluate(tf.compat.v1.variables_initializer(sas_obj.variables))
      y_true = np.concatenate((np.ones((5, 1)), np.zeros((5, 1)), np.ones(
          (5, 1)), np.zeros((5, 1))))
      y_pred = np.concatenate((np.ones((5, 1)), np.zeros(
          (5, 1)), np.zeros((5, 1)), np.ones((5, 1))))
      self.evaluate(sas_obj.update_state(y_true, y_pred))
    self.evaluate(sas_objs[0].merge_state(sas_objs[1:]))
    self.assertEqual(self.evaluate(sas_objs[0].true_positives), 25.)
    self.assertEqual(self.evaluate(sas_objs[0].false_positives), 25.)
    self.assertEqual(self.evaluate(sas_objs[0].false_negatives), 25.)
    self.assertEqual(self.evaluate(sas_objs[0].true_negatives), 25.)

  def test_merge_state_specificity_at_sensitivity(self):
    sas_objs = []
    for _ in range(5):
      sas_obj = metrics.SpecificityAtSensitivity(0.5, num_thresholds=1)
      sas_objs.append(sas_obj)
      self.evaluate(tf.compat.v1.variables_initializer(sas_obj.variables))
      y_true = np.concatenate((np.ones((5, 1)), np.zeros((5, 1)), np.ones(
          (5, 1)), np.zeros((5, 1))))
      y_pred = np.concatenate((np.ones((5, 1)), np.zeros(
          (5, 1)), np.zeros((5, 1)), np.ones((5, 1))))
      self.evaluate(sas_obj.update_state(y_true, y_pred))
    self.evaluate(sas_objs[0].merge_state(sas_objs[1:]))
    self.assertEqual(self.evaluate(sas_objs[0].true_positives), 25.)
    self.assertEqual(self.evaluate(sas_objs[0].false_positives), 25.)
    self.assertEqual(self.evaluate(sas_objs[0].false_negatives), 25.)
    self.assertEqual(self.evaluate(sas_objs[0].true_negatives), 25.)

  def test_merge_state_precision_at_recall(self):
    par_objs = []
    for _ in range(5):
      par_obj = metrics.PrecisionAtRecall(recall=0.5, num_thresholds=1)
      par_objs.append(par_obj)
      self.evaluate(tf.compat.v1.variables_initializer(par_obj.variables))
      y_true = np.concatenate((np.ones((5, 1)), np.zeros((5, 1)), np.ones(
          (5, 1)), np.zeros((5, 1))))
      y_pred = np.concatenate((np.ones((5, 1)), np.zeros(
          (5, 1)), np.zeros((5, 1)), np.ones((5, 1))))
      self.evaluate(par_obj.update_state(y_true, y_pred))
    self.evaluate(par_objs[0].merge_state(par_objs[1:]))
    self.assertEqual(self.evaluate(par_objs[0].true_positives), 25.)
    self.assertEqual(self.evaluate(par_objs[0].false_positives), 25.)
    self.assertEqual(self.evaluate(par_objs[0].false_negatives), 25.)
    self.assertEqual(self.evaluate(par_objs[0].true_negatives), 25.)

  def test_merge_state_recall_at_precision(self):
    rap_objs = []
    for _ in range(5):
      rap_obj = metrics.PrecisionAtRecall(recall=0.5, num_thresholds=1)
      rap_objs.append(rap_obj)
      self.evaluate(tf.compat.v1.variables_initializer(rap_obj.variables))
      y_true = np.concatenate((np.ones((5, 1)), np.zeros((5, 1)), np.ones(
          (5, 1)), np.zeros((5, 1))))
      y_pred = np.concatenate((np.ones((5, 1)), np.zeros(
          (5, 1)), np.zeros((5, 1)), np.ones((5, 1))))
      self.evaluate(rap_obj.update_state(y_true, y_pred))
    self.evaluate(rap_objs[0].merge_state(rap_objs[1:]))
    self.assertEqual(self.evaluate(rap_objs[0].true_positives), 25.)
    self.assertEqual(self.evaluate(rap_objs[0].false_positives), 25.)
    self.assertEqual(self.evaluate(rap_objs[0].false_negatives), 25.)
    self.assertEqual(self.evaluate(rap_objs[0].true_negatives), 25.)

  def test_merge_state_auc(self):
    auc_objs = []
    for _ in range(5):
      auc_obj = metrics.AUC(num_thresholds=3)
      auc_objs.append(auc_obj)
      self.evaluate(tf.compat.v1.variables_initializer(auc_obj.variables))
      y_true = np.concatenate((np.ones((5, 1)), np.zeros((5, 1)), np.ones(
          (5, 1)), np.zeros((5, 1))))
      y_pred = np.concatenate((np.ones((5, 1)), np.zeros(
          (5, 1)), np.zeros((5, 1)), np.ones((5, 1))))
      self.evaluate(auc_obj.update_state(y_true, y_pred))
    self.evaluate(auc_objs[0].merge_state(auc_objs[1:]))
    self.assertEqual(self.evaluate(auc_objs[0].true_positives[1]), 25.)
    self.assertEqual(self.evaluate(auc_objs[0].false_positives[1]), 25.)
    self.assertEqual(self.evaluate(auc_objs[0].false_negatives[1]), 25.)
    self.assertEqual(self.evaluate(auc_objs[0].true_negatives[1]), 25.)

  def test_merge_state_mean_iou(self):
    m_objs = []
    for y_true, y_pred in zip([[0], [1], [1], [1]],
                              [[0.5], [1.0], [1.0], [1.0]]):
      m_obj = metrics.MeanIoU(num_classes=2)
      m_objs.append(m_obj)
      self.evaluate(tf.compat.v1.variables_initializer(m_obj.variables))
      self.evaluate(m_obj.update_state(y_true, y_pred))
    self.evaluate(m_objs[0].merge_state(m_objs[1:]))
    self.assertArrayNear(self.evaluate(m_objs[0].total_cm)[0], [1, 0], 1e-1)
    self.assertArrayNear(self.evaluate(m_objs[0].total_cm)[1], [0, 3], 1e-1)


if __name__ == '__main__':
  tf.test.main()
