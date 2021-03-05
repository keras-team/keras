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
"""Tests for dense_features_v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

import numpy as np
from tensorflow.python.eager import backprop
from keras import combinations
from keras import keras_parameterized
from keras.feature_column import dense_features_v2 as df


def _initialized_session(config=None):
  sess = tf.compat.v1.Session(config=config)
  sess.run(tf.compat.v1.global_variables_initializer())
  sess.run(tf.compat.v1.tables_initializer())
  return sess


class DenseFeaturesTest(keras_parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_retrieving_input(self):
    features = {'a': [0.]}
    dense_features = df.DenseFeatures(tf.feature_column.numeric_column('a'))
    inputs = self.evaluate(dense_features(features))
    self.assertAllClose([[0.]], inputs)

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_reuses_variables(self):
    sparse_input = tf.SparseTensor(
        indices=((0, 0), (1, 0), (2, 0)),
        values=(0, 1, 2),
        dense_shape=(3, 3))

    # Create feature columns (categorical and embedding).
    categorical_column = tf.feature_column.categorical_column_with_identity(
        key='a', num_buckets=3)
    embedding_dimension = 2

    def _embedding_column_initializer(shape, dtype, partition_info=None):
      del shape  # unused
      del dtype  # unused
      del partition_info  # unused
      embedding_values = (
          (1, 0),  # id 0
          (0, 1),  # id 1
          (1, 1))  # id 2
      return embedding_values

    embedding_column = tf.feature_column.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_embedding_column_initializer)

    dense_features = df.DenseFeatures([embedding_column])
    features = {'a': sparse_input}

    inputs = dense_features(features)
    variables = dense_features.variables

    # Sanity check: test that the inputs are correct.
    self.assertAllEqual([[1, 0], [0, 1], [1, 1]], inputs)

    # Check that only one variable was created.
    self.assertEqual(1, len(variables))

    # Check that invoking dense_features on the same features does not create
    # additional variables
    _ = dense_features(features)
    self.assertEqual(1, len(variables))
    self.assertIs(variables[0], dense_features.variables[0])

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_feature_column_dense_features_gradient(self):
    sparse_input = tf.SparseTensor(
        indices=((0, 0), (1, 0), (2, 0)),
        values=(0, 1, 2),
        dense_shape=(3, 3))

    # Create feature columns (categorical and embedding).
    categorical_column = tf.feature_column.categorical_column_with_identity(
        key='a', num_buckets=3)
    embedding_dimension = 2

    def _embedding_column_initializer(shape, dtype, partition_info=None):
      del shape  # unused
      del dtype  # unused
      del partition_info  # unused
      embedding_values = (
          (1, 0),  # id 0
          (0, 1),  # id 1
          (1, 1))  # id 2
      return embedding_values

    embedding_column = tf.feature_column.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_embedding_column_initializer)

    dense_features = df.DenseFeatures([embedding_column])
    features = {'a': sparse_input}

    def scale_matrix():
      matrix = dense_features(features)
      return 2 * matrix

    # Sanity check: Verify that scale_matrix returns the correct output.
    self.assertAllEqual([[2, 0], [0, 2], [2, 2]], scale_matrix())

    # Check that the returned gradient is correct.
    grad_function = backprop.implicit_grad(scale_matrix)
    grads_and_vars = grad_function()
    indexed_slice = grads_and_vars[0][0]
    gradient = grads_and_vars[0][0].values

    self.assertAllEqual([0, 1, 2], indexed_slice.indices)
    self.assertAllEqual([[2, 2], [2, 2], [2, 2]], gradient)

  def test_dense_feature_with_training_arg(self):
    price1 = tf.feature_column.numeric_column('price1', shape=2)
    price2 = tf.feature_column.numeric_column('price2')

    # Monkey patch the second numeric column to simulate a column that has
    # different behavior by mode.
    def training_aware_get_dense_tensor(transformation_cache,
                                        state_manager,
                                        training=None):
      return transformation_cache.get(price2, state_manager, training=training)

    def training_aware_transform_feature(transformation_cache,
                                         state_manager,
                                         training=None):
      input_tensor = transformation_cache.get(
          price2.key, state_manager, training=training)
      if training:
        return input_tensor * 10.0
      else:
        return input_tensor * 20.0

    price2.get_dense_tensor = training_aware_get_dense_tensor
    price2.transform_feature = training_aware_transform_feature
    with tf.Graph().as_default():
      features = {'price1': [[1., 2.], [5., 6.]], 'price2': [[3.], [4.]]}
      train_mode = df.DenseFeatures([price1, price2])(features, training=True)
      predict_mode = df.DenseFeatures([price1, price2
                                      ])(features, training=False)

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.tables_initializer())

      self.assertAllClose([[1., 2., 30.], [5., 6., 40.]],
                          self.evaluate(train_mode))
      self.assertAllClose([[1., 2., 60.], [5., 6., 80.]],
                          self.evaluate(predict_mode))

  def test_raises_if_empty_feature_columns(self):
    with self.assertRaisesRegex(ValueError,
                                'feature_columns must not be empty'):
      df.DenseFeatures(feature_columns=[])(features={})

  def test_should_be_dense_column(self):
    with self.assertRaisesRegex(ValueError, 'must be a .*DenseColumn'):
      df.DenseFeatures(feature_columns=[
          tf.feature_column.categorical_column_with_hash_bucket('wire_cast', 4)
      ])(
          features={
              'a': [[0]]
          })

  def test_does_not_support_dict_columns(self):
    with self.assertRaisesRegex(
        ValueError, 'Expected feature_columns to be iterable, found dict.'):
      df.DenseFeatures(feature_columns={'a': tf.feature_column.numeric_column('a')})(
          features={
              'a': [[0]]
          })

  def test_bare_column(self):
    with tf.Graph().as_default():
      features = features = {'a': [0.]}
      net = df.DenseFeatures(tf.feature_column.numeric_column('a'))(features)

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.tables_initializer())

      self.assertAllClose([[0.]], self.evaluate(net))

  def test_column_generator(self):
    with tf.Graph().as_default():
      features = features = {'a': [0.], 'b': [1.]}
      columns = (tf.feature_column.numeric_column(key) for key in features)
      net = df.DenseFeatures(columns)(features)

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.tables_initializer())

      self.assertAllClose([[0., 1.]], self.evaluate(net))

  def test_raises_if_duplicate_name(self):
    with self.assertRaisesRegex(
        ValueError, 'Duplicate feature column name found for columns'):
      df.DenseFeatures(
          feature_columns=[tf.feature_column.numeric_column('a'),
                           tf.feature_column.numeric_column('a')])(
                               features={
                                   'a': [[0]]
                               })

  def test_one_column(self):
    price = tf.feature_column.numeric_column('price')
    with tf.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      net = df.DenseFeatures([price])(features)

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.tables_initializer())

      self.assertAllClose([[1.], [5.]], self.evaluate(net))

  def test_multi_dimension(self):
    price = tf.feature_column.numeric_column('price', shape=2)
    with tf.Graph().as_default():
      features = {'price': [[1., 2.], [5., 6.]]}
      net = df.DenseFeatures([price])(features)

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.tables_initializer())

      self.assertAllClose([[1., 2.], [5., 6.]], self.evaluate(net))

  def test_compute_output_shape(self):
    price1 = tf.feature_column.numeric_column('price1', shape=2)
    price2 = tf.feature_column.numeric_column('price2', shape=4)
    with tf.Graph().as_default():
      features = {
          'price1': [[1., 2.], [5., 6.]],
          'price2': [[3., 4., 5., 6.], [7., 8., 9., 10.]]
      }
      dense_features = df.DenseFeatures([price1, price2])
      self.assertEqual((None, 6), dense_features.compute_output_shape((None,)))
      net = dense_features(features)

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.tables_initializer())

      self.assertAllClose([[1., 2., 3., 4., 5., 6.], [5., 6., 7., 8., 9., 10.]],
                          self.evaluate(net))

  def test_raises_if_shape_mismatch(self):
    price = tf.feature_column.numeric_column('price', shape=2)
    with tf.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      with self.assertRaisesRegex(
          Exception,
          r'Cannot reshape a tensor with 2 elements to shape \[2,2\]'):
        df.DenseFeatures([price])(features)

  def test_reshaping(self):
    price = tf.feature_column.numeric_column('price', shape=[1, 2])
    with tf.Graph().as_default():
      features = {'price': [[[1., 2.]], [[5., 6.]]]}
      net = df.DenseFeatures([price])(features)

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.tables_initializer())

      self.assertAllClose([[1., 2.], [5., 6.]], self.evaluate(net))

  def test_multi_column(self):
    price1 = tf.feature_column.numeric_column('price1', shape=2)
    price2 = tf.feature_column.numeric_column('price2')
    with tf.Graph().as_default():
      features = {'price1': [[1., 2.], [5., 6.]], 'price2': [[3.], [4.]]}
      net = df.DenseFeatures([price1, price2])(features)

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.tables_initializer())

      self.assertAllClose([[1., 2., 3.], [5., 6., 4.]], self.evaluate(net))

  def test_cols_to_output_tensors(self):
    price1 = tf.feature_column.numeric_column('price1', shape=2)
    price2 = tf.feature_column.numeric_column('price2')
    with tf.Graph().as_default():
      cols_dict = {}
      features = {'price1': [[1., 2.], [5., 6.]], 'price2': [[3.], [4.]]}
      dense_features = df.DenseFeatures([price1, price2])
      net = dense_features(features, cols_dict)

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.tables_initializer())

      self.assertAllClose([[1., 2.], [5., 6.]],
                          self.evaluate(cols_dict[price1]))
      self.assertAllClose([[3.], [4.]], self.evaluate(cols_dict[price2]))
      self.assertAllClose([[1., 2., 3.], [5., 6., 4.]], self.evaluate(net))

  def test_column_order(self):
    price_a = tf.feature_column.numeric_column('price_a')
    price_b = tf.feature_column.numeric_column('price_b')
    with tf.Graph().as_default():
      features = {
          'price_a': [[1.]],
          'price_b': [[3.]],
      }
      net1 = df.DenseFeatures([price_a, price_b])(features)
      net2 = df.DenseFeatures([price_b, price_a])(features)

      self.evaluate(tf.compat.v1.global_variables_initializer())
      self.evaluate(tf.compat.v1.tables_initializer())

      self.assertAllClose([[1., 3.]], self.evaluate(net1))
      self.assertAllClose([[1., 3.]], self.evaluate(net2))

  def test_fails_for_categorical_column(self):
    animal = tf.feature_column.categorical_column_with_identity('animal', num_buckets=4)
    with tf.Graph().as_default():
      features = {
          'animal':
              tf.SparseTensor(
                  indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
      }
      with self.assertRaisesRegex(Exception, 'must be a .*DenseColumn'):
        df.DenseFeatures([animal])(features)

  def test_static_batch_size_mismatch(self):
    price1 = tf.feature_column.numeric_column('price1')
    price2 = tf.feature_column.numeric_column('price2')
    with tf.Graph().as_default():
      features = {
          'price1': [[1.], [5.], [7.]],  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
      with self.assertRaisesRegex(
          ValueError,
          r'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
        df.DenseFeatures([price1, price2])(features)

  def test_subset_of_static_batch_size_mismatch(self):
    price1 = tf.feature_column.numeric_column('price1')
    price2 = tf.feature_column.numeric_column('price2')
    price3 = tf.feature_column.numeric_column('price3')
    with tf.Graph().as_default():
      features = {
          'price1': tf.compat.v1.placeholder(dtype=tf.int64),  # batchsize = 3
          'price2': [[3.], [4.]],  # batchsize = 2
          'price3': [[3.], [4.], [5.]]  # batchsize = 3
      }
      with self.assertRaisesRegex(
          ValueError,
          r'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
        df.DenseFeatures([price1, price2, price3])(features)

  def test_runtime_batch_size_mismatch(self):
    price1 = tf.feature_column.numeric_column('price1')
    price2 = tf.feature_column.numeric_column('price2')
    with tf.Graph().as_default():
      features = {
          'price1': tf.compat.v1.placeholder(dtype=tf.int64),  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
      net = df.DenseFeatures([price1, price2])(features)
      with _initialized_session() as sess:
        with self.assertRaisesRegex(tf.errors.OpError,
                                    'Dimensions of inputs should match'):
          sess.run(net, feed_dict={features['price1']: [[1.], [5.], [7.]]})

  def test_runtime_batch_size_matches(self):
    price1 = tf.feature_column.numeric_column('price1')
    price2 = tf.feature_column.numeric_column('price2')
    with tf.Graph().as_default():
      features = {
          'price1': tf.compat.v1.placeholder(dtype=tf.int64),  # batchsize = 2
          'price2': tf.compat.v1.placeholder(dtype=tf.int64),  # batchsize = 2
      }
      net = df.DenseFeatures([price1, price2])(features)
      with _initialized_session() as sess:
        sess.run(
            net,
            feed_dict={
                features['price1']: [[1.], [5.]],
                features['price2']: [[1.], [5.]],
            })

  def test_multiple_layers_with_same_embedding_column(self):
    some_sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
        'sparse_feature', hash_bucket_size=5)
    some_embedding_column = tf.feature_column.embedding_column(
        some_sparse_column, dimension=10)

    with tf.Graph().as_default():
      features = {
          'sparse_feature': [['a'], ['x']],
      }
      all_cols = [some_embedding_column]
      df.DenseFeatures(all_cols)(features)
      df.DenseFeatures(all_cols)(features)
      # Make sure that 2 variables get created in this case.
      self.assertEqual(2,
                       len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)))
      expected_var_names = [
          'dense_features/sparse_feature_embedding/embedding_weights:0',
          'dense_features_1/sparse_feature_embedding/embedding_weights:0'
      ]
      self.assertItemsEqual(
          expected_var_names,
          [v.name for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)])

  def test_multiple_layers_with_same_shared_embedding_column(self):
    categorical_column_a = tf.feature_column.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    categorical_column_b = tf.feature_column.categorical_column_with_identity(
        key='bbb', num_buckets=3)
    embedding_dimension = 2

    # feature_column.shared_embeddings is not supported in eager.
    with tf.Graph().as_default():
      embedding_column_b, embedding_column_a = tf.feature_column.shared_embeddings(
          [categorical_column_b, categorical_column_a],
          dimension=embedding_dimension)
      features = {
          'aaa':
              tf.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 1, 0),
                  dense_shape=(2, 2)),
          'bbb':
              tf.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(1, 2, 1),
                  dense_shape=(2, 2)),
      }
      all_cols = [embedding_column_a, embedding_column_b]
      df.DenseFeatures(all_cols)(features)
      df.DenseFeatures(all_cols)(features)
      # Make sure that only 1 variable gets created in this case.
      self.assertEqual(1,
                       len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)))
      self.assertItemsEqual(
          ['aaa_bbb_shared_embedding:0'],
          [v.name for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)])

  def test_multiple_layers_with_same_shared_embedding_column_diff_graphs(self):
    categorical_column_a = tf.feature_column.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    categorical_column_b = tf.feature_column.categorical_column_with_identity(
        key='bbb', num_buckets=3)
    embedding_dimension = 2

    # feature_column.shared_embeddings is not supported in eager.
    with tf.Graph().as_default():
      embedding_column_b, embedding_column_a = tf.feature_column.shared_embeddings(
          [categorical_column_b, categorical_column_a],
          dimension=embedding_dimension)
      all_cols = [embedding_column_a, embedding_column_b]
      features = {
          'aaa':
              tf.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 1, 0),
                  dense_shape=(2, 2)),
          'bbb':
              tf.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(1, 2, 1),
                  dense_shape=(2, 2)),
      }
      df.DenseFeatures(all_cols)(features)
      # Make sure that only 1 variable gets created in this case.
      self.assertEqual(1,
                       len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)))

    with tf.Graph().as_default():
      features1 = {
          'aaa':
              tf.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 1, 0),
                  dense_shape=(2, 2)),
          'bbb':
              tf.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(1, 2, 1),
                  dense_shape=(2, 2)),
      }

      df.DenseFeatures(all_cols)(features1)
      # Make sure that only 1 variable gets created in this case.
      self.assertEqual(1,
                       len(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)))
      self.assertItemsEqual(
          ['aaa_bbb_shared_embedding:0'],
          [v.name for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)])

  def test_with_1d_sparse_tensor(self):
    embedding_values = (
        (1., 2., 3., 4., 5.),  # id 0
        (6., 7., 8., 9., 10.),  # id 1
        (11., 12., 13., 14., 15.)  # id 2
    )

    def _initializer(shape, dtype, partition_info=None):
      del shape, dtype, partition_info
      return embedding_values

    # price has 1 dimension in dense_features
    price = tf.feature_column.numeric_column('price')

    # one_hot_body_style has 3 dims in dense_features.
    body_style = tf.feature_column.categorical_column_with_vocabulary_list(
        'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])
    one_hot_body_style = tf.feature_column.indicator_column(body_style)

    # embedded_body_style has 5 dims in dense_features.
    country = tf.feature_column.categorical_column_with_vocabulary_list(
        'country', vocabulary_list=['US', 'JP', 'CA'])
    embedded_country = tf.feature_column.embedding_column(
        country, dimension=5, initializer=_initializer)

    with tf.Graph().as_default():
      # Provides 1-dim tensor and dense tensor.
      features = {
          'price':
              tf.constant([
                  11.,
                  12.,
              ]),
          'body-style':
              tf.SparseTensor(
                  indices=((0,), (1,)),
                  values=('sedan', 'hardtop'),
                  dense_shape=(2,)),
          # This is dense tensor for the categorical_column.
          'country':
              tf.constant(['CA', 'US']),
      }
      self.assertEqual(1, features['price'].shape.ndims)
      self.assertEqual(1, features['body-style'].dense_shape.get_shape()[0])
      self.assertEqual(1, features['country'].shape.ndims)

      net = df.DenseFeatures([price, one_hot_body_style, embedded_country])(
          features)
      self.assertEqual(1 + 3 + 5, net.shape[1])
      with _initialized_session() as sess:

        # Each row is formed by concatenating `embedded_body_style`,
        # `one_hot_body_style`, and `price` in order.
        self.assertAllEqual([[0., 0., 1., 11., 12., 13., 14., 15., 11.],
                             [1., 0., 0., 1., 2., 3., 4., 5., 12.]],
                            sess.run(net))

  def test_with_1d_unknown_shape_sparse_tensor(self):
    embedding_values = (
        (1., 2.),  # id 0
        (6., 7.),  # id 1
        (11., 12.)  # id 2
    )

    def _initializer(shape, dtype, partition_info=None):
      del shape, dtype, partition_info
      return embedding_values

    # price has 1 dimension in dense_features
    price = tf.feature_column.numeric_column('price')

    # one_hot_body_style has 3 dims in dense_features.
    body_style = tf.feature_column.categorical_column_with_vocabulary_list(
        'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])
    one_hot_body_style = tf.feature_column.indicator_column(body_style)

    # embedded_body_style has 5 dims in dense_features.
    country = tf.feature_column.categorical_column_with_vocabulary_list(
        'country', vocabulary_list=['US', 'JP', 'CA'])
    embedded_country = tf.feature_column.embedding_column(
        country, dimension=2, initializer=_initializer)

    # Provides 1-dim tensor and dense tensor.
    with tf.Graph().as_default():
      features = {
          'price': tf.compat.v1.placeholder(tf.float32),
          'body-style': tf.compat.v1.sparse_placeholder(tf.string),
          # This is dense tensor for the categorical_column.
          'country': tf.compat.v1.placeholder(tf.string),
      }
      self.assertIsNone(features['price'].shape.ndims)
      self.assertIsNone(features['body-style'].get_shape().ndims)
      self.assertIsNone(features['country'].shape.ndims)

      price_data = np.array([11., 12.])
      body_style_data = tf.compat.v1.SparseTensorValue(
          indices=((0,), (1,)), values=('sedan', 'hardtop'), dense_shape=(2,))
      country_data = np.array([['US'], ['CA']])

      net = df.DenseFeatures([price, one_hot_body_style, embedded_country])(
          features)
      self.assertEqual(1 + 3 + 2, net.shape[1])
      with _initialized_session() as sess:

        # Each row is formed by concatenating `embedded_body_style`,
        # `one_hot_body_style`, and `price` in order.
        self.assertAllEqual(
            [[0., 0., 1., 1., 2., 11.], [1., 0., 0., 11., 12., 12.]],
            sess.run(
                net,
                feed_dict={
                    features['price']: price_data,
                    features['body-style']: body_style_data,
                    features['country']: country_data
                }))

  def test_with_rank_0_feature(self):
    # price has 1 dimension in dense_features
    price = tf.feature_column.numeric_column('price')
    features = {
        'price': tf.constant(0),
    }
    self.assertEqual(0, features['price'].shape.ndims)

    # Static rank 0 should fail
    with self.assertRaisesRegex(ValueError, 'Feature .* cannot have rank 0'):
      df.DenseFeatures([price])(features)

    with tf.Graph().as_default():
      # Dynamic rank 0 should fail
      features = {
          'price': tf.compat.v1.placeholder(tf.float32),
      }
      net = df.DenseFeatures([price])(features)
      self.assertEqual(1, net.shape[1])
      with _initialized_session() as sess:
        with self.assertRaisesOpError('Feature .* cannot have rank 0'):
          sess.run(net, feed_dict={features['price']: np.array(1)})


if __name__ == '__main__':
  tf.test.main()
