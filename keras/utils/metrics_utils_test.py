# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for metrics_utils."""

import tensorflow.compat.v2 as tf

from absl.testing import parameterized
from keras import combinations
from keras.utils import metrics_utils


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class RaggedSizeOpTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      {
          'x_list': [1],
          'y_list': [2]
      },
      {
          'x_list': [1, 2],
          'y_list': [2, 3]
      },
      {
          'x_list': [1, 2, 4],
          'y_list': [2, 3, 5]
      },
      {
          'x_list': [[1, 2], [3, 4]],
          'y_list': [[2, 3], [5, 6]]
      },
  ])
  def test_passing_dense_tensors(self, x_list, y_list):
    x = tf.constant(x_list)
    y = tf.constant(y_list)
    [x,
     y], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values([x, y])
    x.shape.assert_is_compatible_with(y.shape)

  @parameterized.parameters([
      {
          'x_list': [1],
      },
      {
          'x_list': [1, 2],
      },
      {
          'x_list': [1, 2, 4],
      },
      {
          'x_list': [[1, 2], [3, 4]],
      },
  ])
  def test_passing_one_dense_tensor(self, x_list):
    x = tf.constant(x_list)
    [x], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values([x])

  @parameterized.parameters([
      {
          'x_list': [1],
          'y_list': [2]
      },
      {
          'x_list': [1, 2],
          'y_list': [2, 3]
      },
      {
          'x_list': [1, 2, 4],
          'y_list': [2, 3, 5]
      },
      {
          'x_list': [[1, 2], [3, 4]],
          'y_list': [[2, 3], [5, 6]]
      },
      {
          'x_list': [[1, 2], [3, 4], [1]],
          'y_list': [[2, 3], [5, 6], [3]]
      },
      {
          'x_list': [[1, 2], [], [1]],
          'y_list': [[2, 3], [], [3]]
      },
  ])
  def test_passing_both_ragged(self, x_list, y_list):
    x = tf.ragged.constant(x_list)
    y = tf.ragged.constant(y_list)
    [x,
     y], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values([x, y])
    x.shape.assert_is_compatible_with(y.shape)

  @parameterized.parameters([
      {
          'x_list': [1],
      },
      {
          'x_list': [1, 2],
      },
      {
          'x_list': [1, 2, 4],
      },
      {
          'x_list': [[1, 2], [3, 4]],
      },
      {
          'x_list': [[1, 2], [3, 4], [1]],
      },
      {
          'x_list': [[1, 2], [], [1]],
      },
  ])
  def test_passing_one_ragged(self, x_list):
    x = tf.ragged.constant(x_list)
    [x], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values([x])

  @parameterized.parameters([
      {
          'x_list': [1],
          'y_list': [2],
          'mask_list': [0]
      },
      {
          'x_list': [1, 2],
          'y_list': [2, 3],
          'mask_list': [0, 1]
      },
      {
          'x_list': [1, 2, 4],
          'y_list': [2, 3, 5],
          'mask_list': [1, 1, 1]
      },
      {
          'x_list': [[1, 2], [3, 4]],
          'y_list': [[2, 3], [5, 6]],
          'mask_list': [[1, 1], [0, 1]]
      },
      {
          'x_list': [[1, 2], [3, 4], [1]],
          'y_list': [[2, 3], [5, 6], [3]],
          'mask_list': [[1, 1], [0, 0], [1]]
      },
      {
          'x_list': [[1, 2], [], [1]],
          'y_list': [[2, 3], [], [3]],
          'mask_list': [[1, 1], [], [0]]
      },
  ])
  def test_passing_both_ragged_with_mask(self, x_list, y_list, mask_list):
    x = tf.ragged.constant(x_list)
    y = tf.ragged.constant(y_list)
    mask = tf.ragged.constant(mask_list)
    [x, y], mask = \
        metrics_utils.ragged_assert_compatible_and_get_flat_values([x, y], mask)
    x.shape.assert_is_compatible_with(y.shape)
    y.shape.assert_is_compatible_with(mask.shape)

  @parameterized.parameters([
      {
          'x_list': [1],
          'mask_list': [0]
      },
      {
          'x_list': [1, 2],
          'mask_list': [0, 1]
      },
      {
          'x_list': [1, 2, 4],
          'mask_list': [1, 1, 1]
      },
      {
          'x_list': [[1, 2], [3, 4]],
          'mask_list': [[1, 1], [0, 1]]
      },
      {
          'x_list': [[1, 2], [3, 4], [1]],
          'mask_list': [[1, 1], [0, 0], [1]]
      },
      {
          'x_list': [[1, 2], [], [1]],
          'mask_list': [[1, 1], [], [0]]
      },
  ])
  def test_passing_one_ragged_with_mask(self, x_list, mask_list):
    x = tf.ragged.constant(x_list)
    mask = tf.ragged.constant(mask_list)
    [x], mask = \
        metrics_utils.ragged_assert_compatible_and_get_flat_values([x], mask)
    x.shape.assert_is_compatible_with(mask.shape)

  @parameterized.parameters([
      {
          'x_list': [[[1, 3]]],
          'y_list': [[2, 3]]
      },
  ])
  def test_failing_different_ragged_and_dense_ranks(self, x_list, y_list):
    x = tf.ragged.constant(x_list)
    y = tf.ragged.constant(y_list)
    with self.assertRaises(ValueError):  # pylint: disable=g-error-prone-assert-raises
      [x, y
      ], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values([x, y])

  @parameterized.parameters([
      {
          'x_list': [[[1, 3]]],
          'y_list': [[[2, 3]]],
          'mask_list': [[0, 1]]
      },
  ])
  def test_failing_different_mask_ranks(self, x_list, y_list, mask_list):
    x = tf.ragged.constant(x_list)
    y = tf.ragged.constant(y_list)
    mask = tf.ragged.constant(mask_list)
    with self.assertRaises(ValueError):  # pylint: disable=g-error-prone-assert-raises
      [x, y
      ], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values([x, y],
                                                                        mask)

  # we do not support such cases that ragged_ranks are different but overall
  # dimension shapes and sizes are identical due to adding too much performance
  # overheads to the overall use cases.
  def test_failing_different_ragged_ranks(self):
    dt = tf.constant([[[1, 2]]])
    # adding a ragged dimension
    x = tf.RaggedTensor.from_row_splits(dt, row_splits=[0, 1])
    y = tf.ragged.constant([[[[1, 2]]]])
    with self.assertRaises(ValueError):  # pylint: disable=g-error-prone-assert-raises
      [x, y], _ = \
          metrics_utils.ragged_assert_compatible_and_get_flat_values([x, y])


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class FilterTopKTest(tf.test.TestCase, parameterized.TestCase):

  def test_one_dimensional(self):
    x = tf.constant([.3, .1, .2, -.5, 42.])
    top_1 = self.evaluate(metrics_utils._filter_top_k(x=x, k=1))
    top_2 = self.evaluate(metrics_utils._filter_top_k(x=x, k=2))
    top_3 = self.evaluate(metrics_utils._filter_top_k(x=x, k=3))

    self.assertAllClose(top_1, [
        metrics_utils.NEG_INF, metrics_utils.NEG_INF, metrics_utils.NEG_INF,
        metrics_utils.NEG_INF, 42.
    ])
    self.assertAllClose(top_2, [
        .3, metrics_utils.NEG_INF, metrics_utils.NEG_INF, metrics_utils.NEG_INF,
        42.
    ])
    self.assertAllClose(
        top_3, [.3, metrics_utils.NEG_INF, .2, metrics_utils.NEG_INF, 42.])

  def test_three_dimensional(self):
    x = tf.constant([[[.3, .1, .2], [-.3, -.2, -.1]],
                              [[5., .2, 42.], [-.3, -.6, -.99]]])
    top_2 = self.evaluate(metrics_utils._filter_top_k(x=x, k=2))

    self.assertAllClose(
        top_2,
        [[[.3, metrics_utils.NEG_INF, .2], [metrics_utils.NEG_INF, -.2, -.1]],
         [[5., metrics_utils.NEG_INF, 42.], [-.3, -.6, metrics_utils.NEG_INF]]])

  def test_handles_dynamic_shapes(self):
    # See b/150281686.  # GOOGLE_INTERNAL

    def _identity(x):
      return x

    def _filter_top_k(x):
      # This loses the static shape.
      x = tf.numpy_function(_identity, (x,), tf.float32)

      return metrics_utils._filter_top_k(x=x, k=2)

    x = tf.constant([.3, .1, .2, -.5, 42.])
    top_2 = self.evaluate(_filter_top_k(x))
    self.assertAllClose(top_2, [
        .3, metrics_utils.NEG_INF, metrics_utils.NEG_INF, metrics_utils.NEG_INF,
        42.
    ])


if __name__ == '__main__':
  tf.test.main()
