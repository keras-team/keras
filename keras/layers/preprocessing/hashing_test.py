# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for hashing layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np
from keras import keras_parameterized
from keras import testing_utils
from keras.engine import input_layer
from keras.engine import training
from keras.layers.preprocessing import hashing


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class HashingTest(keras_parameterized.TestCase):

  def test_hash_single_bin(self):
    layer = hashing.Hashing(num_bins=1)
    inp = np.asarray([['A'], ['B'], ['C'], ['D'], ['E']])
    output = layer(inp)
    self.assertAllClose([[0], [0], [0], [0], [0]], output)

  def test_hash_dense_input_farmhash(self):
    layer = hashing.Hashing(num_bins=2)
    inp = np.asarray([['omar'], ['stringer'], ['marlo'], ['wire'],
                      ['skywalker']])
    output = layer(inp)
    # Assert equal for hashed output that should be true on all platforms.
    self.assertAllClose([[0], [0], [1], [0], [0]], output)

  def test_hash_dense_multi_inputs_farmhash(self):
    layer = hashing.Hashing(num_bins=2)
    inp_1 = np.asarray([['omar'], ['stringer'], ['marlo'], ['wire'],
                        ['skywalker']])
    inp_2 = np.asarray([['A'], ['B'], ['C'], ['D'], ['E']])
    output = layer([inp_1, inp_2])
    # Assert equal for hashed output that should be true on all platforms.
    self.assertAllClose([[0], [0], [1], [1], [0]], output)

  def test_hash_dense_list_input_farmhash(self):
    layer = hashing.Hashing(num_bins=2)
    inp = [['omar'], ['stringer'], ['marlo'], ['wire'], ['skywalker']]
    output = layer(inp)
    # Assert equal for hashed output that should be true on all platforms.
    self.assertAllClose([[0], [0], [1], [0], [0]], output)

    inp = ['omar', 'stringer', 'marlo', 'wire', 'skywalker']
    output = layer(inp)
    # Assert equal for hashed output that should be true on all platforms.
    self.assertAllClose([0, 0, 1, 0, 0], output)

  def test_hash_dense_list_inputs_mixed_int_string_farmhash(self):
    layer = hashing.Hashing(num_bins=2)
    inp_1 = np.asarray([['omar'], ['stringer'], ['marlo'], ['wire'],
                        ['skywalker']])
    inp_2 = np.asarray([[1], [2], [3], [4], [5]]).astype(np.int64)
    output = layer([inp_1, inp_2])
    # Assert equal for hashed output that should be true on all platforms.
    self.assertAllClose([[0], [1], [1], [1], [0]], output)

  def test_hash_dense_int_input_farmhash(self):
    layer = hashing.Hashing(num_bins=3)
    inp = np.asarray([[0], [1], [2], [3], [4]])
    output = layer(inp)
    # Assert equal for hashed output that should be true on all platforms.
    self.assertAllClose([[1], [0], [1], [0], [2]], output)

  def test_hash_dense_input_siphash(self):
    layer = hashing.Hashing(num_bins=2, salt=[133, 137])
    inp = np.asarray([['omar'], ['stringer'], ['marlo'], ['wire'],
                      ['skywalker']])
    output = layer(inp)
    # Assert equal for hashed output that should be true on all platforms.
    # Note the result is different from FarmHash.
    self.assertAllClose([[0], [1], [0], [1], [0]], output)

    layer_2 = hashing.Hashing(num_bins=2, salt=[211, 137])
    output_2 = layer_2(inp)
    # Note the result is different from (133, 137).
    self.assertAllClose([[1], [0], [1], [0], [1]], output_2)

  def test_hash_dense_multi_inputs_siphash(self):
    layer = hashing.Hashing(num_bins=2, salt=[133, 137])
    inp_1 = np.asarray([['omar'], ['stringer'], ['marlo'], ['wire'],
                        ['skywalker']])
    inp_2 = np.asarray([['A'], ['B'], ['C'], ['D'], ['E']])
    output = layer([inp_1, inp_2])
    # Assert equal for hashed output that should be true on all platforms.
    # Note the result is different from FarmHash.
    self.assertAllClose([[0], [1], [0], [0], [1]], output)

    layer_2 = hashing.Hashing(num_bins=2, salt=[211, 137])
    output_2 = layer_2([inp_1, inp_2])
    # Note the result is different from (133, 137).
    self.assertAllClose([[1], [1], [1], [0], [1]], output_2)

  def test_hash_dense_int_input_siphash(self):
    layer = hashing.Hashing(num_bins=3, salt=[133, 137])
    inp = np.asarray([[0], [1], [2], [3], [4]])
    output = layer(inp)
    # Assert equal for hashed output that should be true on all platforms.
    self.assertAllClose([[1], [1], [2], [0], [1]], output)

  def test_hash_sparse_input_farmhash(self):
    layer = hashing.Hashing(num_bins=2)
    indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]
    inp = tf.SparseTensor(
        indices=indices,
        values=['omar', 'stringer', 'marlo', 'wire', 'skywalker'],
        dense_shape=[3, 2])
    output = layer(inp)
    self.assertAllClose(indices, output.indices)
    self.assertAllClose([0, 0, 1, 0, 0], output.values)

  def test_hash_sparse_multi_inputs_farmhash(self):
    layer = hashing.Hashing(num_bins=2)
    indices = [[0, 0], [1, 0], [2, 0]]
    inp_1 = tf.SparseTensor(
        indices=indices,
        values=['omar', 'stringer', 'marlo'],
        dense_shape=[3, 1])
    inp_2 = tf.SparseTensor(
        indices=indices, values=['A', 'B', 'C'], dense_shape=[3, 1])
    output = layer([inp_1, inp_2])
    self.assertAllClose(indices, output.indices)
    self.assertAllClose([0, 0, 1], output.values)

  def test_hash_sparse_int_input_farmhash(self):
    layer = hashing.Hashing(num_bins=3)
    indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]
    inp = tf.SparseTensor(
        indices=indices, values=[0, 1, 2, 3, 4], dense_shape=[3, 2])
    output = layer(inp)
    self.assertAllClose(indices, output.indices)
    self.assertAllClose([1, 0, 1, 0, 2], output.values)

  def test_hash_sparse_input_siphash(self):
    layer = hashing.Hashing(num_bins=2, salt=[133, 137])
    indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]
    inp = tf.SparseTensor(
        indices=indices,
        values=['omar', 'stringer', 'marlo', 'wire', 'skywalker'],
        dense_shape=[3, 2])
    output = layer(inp)
    self.assertAllClose(output.indices, indices)
    # The result should be same with test_hash_dense_input_siphash.
    self.assertAllClose([0, 1, 0, 1, 0], output.values)

    layer_2 = hashing.Hashing(num_bins=2, salt=[211, 137])
    output = layer_2(inp)
    # The result should be same with test_hash_dense_input_siphash.
    self.assertAllClose([1, 0, 1, 0, 1], output.values)

  def test_hash_sparse_multi_inputs_siphash(self):
    layer = hashing.Hashing(num_bins=2, salt=[133, 137])
    indices = [[0, 0], [1, 0], [2, 0]]
    inp_1 = tf.SparseTensor(
        indices=indices,
        values=['omar', 'stringer', 'marlo'],
        dense_shape=[3, 1])
    inp_2 = tf.SparseTensor(
        indices=indices, values=['A', 'B', 'C'], dense_shape=[3, 1])
    output = layer([inp_1, inp_2])
    # The result should be same with test_hash_dense_input_siphash.
    self.assertAllClose(indices, output.indices)
    self.assertAllClose([0, 1, 0], output.values)

    layer_2 = hashing.Hashing(num_bins=2, salt=[211, 137])
    output = layer_2([inp_1, inp_2])
    # The result should be same with test_hash_dense_input_siphash.
    self.assertAllClose([1, 1, 1], output.values)

  def test_hash_sparse_int_input_siphash(self):
    layer = hashing.Hashing(num_bins=3, salt=[133, 137])
    indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]
    inp = tf.SparseTensor(
        indices=indices, values=[0, 1, 2, 3, 4], dense_shape=[3, 2])
    output = layer(inp)
    self.assertAllClose(indices, output.indices)
    self.assertAllClose([1, 1, 2, 0, 1], output.values)

  def test_hash_ragged_string_input_farmhash(self):
    layer = hashing.Hashing(num_bins=2)
    inp_data = tf.ragged.constant(
        [['omar', 'stringer', 'marlo', 'wire'], ['marlo', 'skywalker', 'wire']],
        dtype=tf.string)
    out_data = layer(inp_data)
    # Same hashed output as test_hash_sparse_input_farmhash
    expected_output = [[0, 0, 1, 0], [1, 0, 0]]
    self.assertAllEqual(expected_output, out_data)

    inp_t = input_layer.Input(shape=(None,), ragged=True, dtype=tf.string)
    out_t = layer(inp_t)
    model = training.Model(inputs=inp_t, outputs=out_t)
    self.assertAllClose(out_data, model.predict(inp_data))

  def test_hash_ragged_string_multi_inputs_farmhash(self):
    layer = hashing.Hashing(num_bins=2)
    inp_data_1 = tf.ragged.constant(
        [['omar', 'stringer', 'marlo', 'wire'], ['marlo', 'skywalker', 'wire']],
        dtype=tf.string)
    inp_data_2 = tf.ragged.constant(
        [['omar', 'stringer', 'marlo', 'wire'], ['marlo', 'skywalker', 'wire']],
        dtype=tf.string)
    with self.assertRaisesRegex(ValueError, 'not supported yet'):
      _ = layer([inp_data_1, inp_data_2])

  def test_hash_ragged_int_input_farmhash(self):
    layer = hashing.Hashing(num_bins=3)
    inp_data = tf.ragged.constant([[0, 1, 3, 4], [2, 1, 0]],
                                           dtype=tf.int64)
    out_data = layer(inp_data)
    # Same hashed output as test_hash_sparse_input_farmhash
    expected_output = [[1, 0, 0, 2], [1, 0, 1]]
    self.assertAllEqual(expected_output, out_data)

    inp_t = input_layer.Input(shape=(None,), ragged=True, dtype=tf.int64)
    out_t = layer(inp_t)
    model = training.Model(inputs=inp_t, outputs=out_t)
    self.assertAllClose(out_data, model.predict(inp_data))

  def test_hash_ragged_string_input_siphash(self):
    layer = hashing.Hashing(num_bins=2, salt=[133, 137])
    inp_data = tf.ragged.constant(
        [['omar', 'stringer', 'marlo', 'wire'], ['marlo', 'skywalker', 'wire']],
        dtype=tf.string)
    out_data = layer(inp_data)
    # Same hashed output as test_hash_dense_input_siphash
    expected_output = [[0, 1, 0, 1], [0, 0, 1]]
    self.assertAllEqual(expected_output, out_data)

    inp_t = input_layer.Input(shape=(None,), ragged=True, dtype=tf.string)
    out_t = layer(inp_t)
    model = training.Model(inputs=inp_t, outputs=out_t)
    self.assertAllClose(out_data, model.predict(inp_data))

    layer_2 = hashing.Hashing(num_bins=2, salt=[211, 137])
    out_data = layer_2(inp_data)
    expected_output = [[1, 0, 1, 0], [1, 1, 0]]
    self.assertAllEqual(expected_output, out_data)

    out_t = layer_2(inp_t)
    model = training.Model(inputs=inp_t, outputs=out_t)
    self.assertAllClose(out_data, model.predict(inp_data))

  def test_hash_ragged_string_multi_inputs_siphash(self):
    layer = hashing.Hashing(num_bins=2, salt=[133, 137])
    inp_data_1 = tf.ragged.constant(
        [['omar', 'stringer', 'marlo', 'wire'], ['marlo', 'skywalker', 'wire']],
        dtype=tf.string)
    inp_data_2 = tf.ragged.constant(
        [['omar', 'stringer', 'marlo', 'wire'], ['marlo', 'skywalker', 'wire']],
        dtype=tf.string)
    with self.assertRaisesRegex(ValueError, 'not supported yet'):
      _ = layer([inp_data_1, inp_data_2])

  def test_hash_ragged_int_input_siphash(self):
    layer = hashing.Hashing(num_bins=3, salt=[133, 137])
    inp_data = tf.ragged.constant([[0, 1, 3, 4], [2, 1, 0]],
                                           dtype=tf.int64)
    out_data = layer(inp_data)
    # Same hashed output as test_hash_sparse_input_farmhash
    expected_output = [[1, 1, 0, 1], [2, 1, 1]]
    self.assertAllEqual(expected_output, out_data)

    inp_t = input_layer.Input(shape=(None,), ragged=True, dtype=tf.int64)
    out_t = layer(inp_t)
    model = training.Model(inputs=inp_t, outputs=out_t)
    self.assertAllClose(out_data, model.predict(inp_data))

  def test_invalid_inputs(self):
    with self.assertRaisesRegex(ValueError, 'cannot be `None`'):
      _ = hashing.Hashing(num_bins=None)
    with self.assertRaisesRegex(ValueError, 'cannot be `None`'):
      _ = hashing.Hashing(num_bins=-1)
    with self.assertRaisesRegex(ValueError, 'can only be a tuple of size 2'):
      _ = hashing.Hashing(num_bins=2, salt='string')
    with self.assertRaisesRegex(ValueError, 'can only be a tuple of size 2'):
      _ = hashing.Hashing(num_bins=2, salt=[1])
    with self.assertRaisesRegex(ValueError, 'can only be a tuple of size 2'):
      _ = hashing.Hashing(num_bins=1, salt=tf.constant([133, 137]))

  def test_hash_compute_output_signature(self):
    input_shape = tf.TensorShape([2, 3])
    input_spec = tf.TensorSpec(input_shape, tf.string)
    layer = hashing.Hashing(num_bins=2)
    output_spec = layer.compute_output_signature(input_spec)
    self.assertEqual(output_spec.shape.dims, input_shape.dims)
    self.assertEqual(output_spec.dtype, tf.int64)

  @testing_utils.run_v2_only
  def test_config_with_custom_name(self):
    layer = hashing.Hashing(num_bins=2, name='hashing')
    config = layer.get_config()
    layer_1 = hashing.Hashing.from_config(config)
    self.assertEqual(layer_1.name, layer.name)


if __name__ == '__main__':
  tf.test.main()
