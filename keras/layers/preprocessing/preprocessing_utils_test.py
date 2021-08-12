# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for preprocessing utils."""

from absl.testing import parameterized
from keras import keras_parameterized
from keras.layers.preprocessing import preprocessing_utils
import numpy as np
import tensorflow.compat.v2 as tf


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class ListifyTensorsTest(keras_parameterized.TestCase):

  def test_tensor_input(self):
    inputs = tf.constant([0, 1, 2, 3, 4])
    outputs = preprocessing_utils.listify_tensors(inputs)
    self.assertAllEqual([0, 1, 2, 3, 4], outputs)
    self.assertIsInstance(outputs, list)

  def test_numpy_input(self):
    inputs = np.array([0, 1, 2, 3, 4])
    outputs = preprocessing_utils.listify_tensors(inputs)
    self.assertAllEqual([0, 1, 2, 3, 4], outputs)
    self.assertIsInstance(outputs, list)


@keras_parameterized.run_all_keras_modes
class EncodeCategoricalInputsTest(keras_parameterized.TestCase):

  def test_int_encoding(self):
    inputs = tf.constant([0, 1, 2])
    outputs = preprocessing_utils.encode_categorical_inputs(
        inputs, output_mode='int', depth=4)
    self.assertAllEqual([0, 1, 2], outputs)

  @parameterized.named_parameters(
      ('sparse', True),
      ('dense', False),
  )
  def test_one_hot_encoding(self, sparse):
    inputs = tf.constant([0, 1, 2])
    outputs = preprocessing_utils.encode_categorical_inputs(
        inputs, output_mode='one_hot', depth=4, sparse=sparse)
    if sparse:
      outputs = tf.sparse.to_dense(outputs)
    self.assertAllEqual([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], outputs)

  @parameterized.named_parameters(
      ('sparse', True),
      ('dense', False),
  )
  def test_multi_hot_encoding(self, sparse):
    inputs = tf.constant([0, 1, 2])
    outputs = preprocessing_utils.encode_categorical_inputs(
        inputs, output_mode='multi_hot', depth=4, sparse=sparse)
    if sparse:
      outputs = tf.sparse.to_dense(outputs)
    self.assertAllEqual([1, 1, 1, 0], outputs)

  @parameterized.named_parameters(
      ('sparse', True),
      ('dense', False),
  )
  def test_count_encoding(self, sparse):
    inputs = tf.constant([0, 1, 1, 2, 2, 2])
    outputs = preprocessing_utils.encode_categorical_inputs(
        inputs, output_mode='count', depth=4, sparse=sparse)
    if sparse:
      outputs = tf.sparse.to_dense(outputs)
    self.assertAllEqual([1, 2, 3, 0], outputs)

  @parameterized.named_parameters(
      ('sparse', True),
      ('dense', False),
  )
  def test_tf_idf_encoding(self, sparse):
    inputs = tf.constant([0, 1, 1, 2, 2, 2])
    outputs = preprocessing_utils.encode_categorical_inputs(
        inputs,
        output_mode='tf_idf',
        depth=4,
        sparse=sparse,
        idf_weights=[0.1, 1.0, 10.0, 0])
    if sparse:
      outputs = tf.sparse.to_dense(outputs)
    self.assertAllClose([.1, 2, 30, 0], outputs)

  def test_rank_3_output_fails(self):
    inputs = tf.constant([[[0]], [[1]], [[2]]])
    with self.assertRaisesRegex(ValueError,
                                'maximum supported output rank is 2'):
      preprocessing_utils.encode_categorical_inputs(inputs, 'multi_hot', 4)

  def test_tf_idf_output_with_no_weights_fails(self):
    inputs = tf.constant([0, 1, 2])
    with self.assertRaisesRegex(ValueError,
                                'idf_weights must be provided'):
      preprocessing_utils.encode_categorical_inputs(inputs, 'tf_idf', 4)


if __name__ == '__main__':
  tf.test.main()
