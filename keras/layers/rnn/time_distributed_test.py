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
"""Tests for TimeDistributed wrapper."""
# pylint: disable=g-direct-tensorflow-import

from absl.testing import parameterized
import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow.python.training.tracking import util as trackable_util


class TimeDistributedTest(test_combinations.TestCase):

  @test_combinations.generate(
      test_combinations.combine(mode=['graph', 'eager']))
  def test_timedistributed_dense(self):
    model = keras.models.Sequential()
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(2), input_shape=(3, 4)))
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(
        np.random.random((10, 3, 4)),
        np.random.random((10, 3, 2)),
        epochs=1,
        batch_size=10)

    # test config
    model.get_config()

    # check whether the model variables are present in the
    # trackable list of objects
    checkpointed_object_ids = {
        id(o) for o in trackable_util.list_objects(model)
    }
    for v in model.variables:
      self.assertIn(id(v), checkpointed_object_ids)

  def test_timedistributed_static_batch_size(self):
    model = keras.models.Sequential()
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(2), input_shape=(3, 4), batch_size=10))
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(
        np.random.random((10, 3, 4)),
        np.random.random((10, 3, 2)),
        epochs=1,
        batch_size=10)

  def test_timedistributed_invalid_init(self):
    x = tf.constant(np.zeros((1, 1)).astype('float32'))
    with self.assertRaisesRegex(
        ValueError, 'Please initialize `TimeDistributed` layer with a '
        '`tf.keras.layers.Layer` instance.'):
      keras.layers.TimeDistributed(x)

  def test_timedistributed_conv2d(self):
    with self.cached_session():
      model = keras.models.Sequential()
      model.add(
          keras.layers.TimeDistributed(
              keras.layers.Conv2D(5, (2, 2), padding='same'),
              input_shape=(2, 4, 4, 3)))
      model.add(keras.layers.Activation('relu'))
      model.compile(optimizer='rmsprop', loss='mse')
      model.train_on_batch(
          np.random.random((1, 2, 4, 4, 3)), np.random.random((1, 2, 4, 4, 5)))

      model = keras.models.model_from_json(model.to_json())
      model.summary()

  def test_timedistributed_stacked(self):
    with self.cached_session():
      model = keras.models.Sequential()
      model.add(
          keras.layers.TimeDistributed(
              keras.layers.Dense(2), input_shape=(3, 4)))
      model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
      model.add(keras.layers.Activation('relu'))
      model.compile(optimizer='rmsprop', loss='mse')

      model.fit(
          np.random.random((10, 3, 4)),
          np.random.random((10, 3, 3)),
          epochs=1,
          batch_size=10)

  def test_regularizers(self):
    with self.cached_session():
      model = keras.models.Sequential()
      model.add(
          keras.layers.TimeDistributed(
              keras.layers.Dense(2, kernel_regularizer='l1',
                                 activity_regularizer='l1'),
              input_shape=(3, 4)))
      model.add(keras.layers.Activation('relu'))
      model.compile(optimizer='rmsprop', loss='mse')
      self.assertEqual(len(model.losses), 2)

  def test_TimeDistributed_learning_phase(self):
    with self.cached_session():
      # test layers that need learning_phase to be set
      np.random.seed(1234)
      x = keras.layers.Input(shape=(3, 2))
      y = keras.layers.TimeDistributed(keras.layers.Dropout(.999))(
          x, training=True)
      model = keras.models.Model(x, y)
      y = model.predict(np.random.random((10, 3, 2)))
      self.assertAllClose(np.mean(y), 0., atol=1e-1, rtol=1e-1)

  def test_TimeDistributed_batchnorm(self):
    with self.cached_session():
      # test that wrapped BN updates still work.
      model = keras.models.Sequential()
      model.add(keras.layers.TimeDistributed(
          keras.layers.BatchNormalization(center=True, scale=True),
          name='bn',
          input_shape=(10, 2)))
      model.compile(optimizer='rmsprop', loss='mse')
      # Assert that mean and variance are 0 and 1.
      td = model.layers[0]
      self.assertAllClose(td.get_weights()[2], np.array([0, 0]))
      assert np.array_equal(td.get_weights()[3], np.array([1, 1]))
      # Train
      model.train_on_batch(np.random.normal(loc=2, scale=2, size=(1, 10, 2)),
                           np.broadcast_to(np.array([0, 1]), (1, 10, 2)))
      # Assert that mean and variance changed.
      assert not np.array_equal(td.get_weights()[2], np.array([0, 0]))
      assert not np.array_equal(td.get_weights()[3], np.array([1, 1]))

  def test_TimeDistributed_trainable(self):
    # test layers that need learning_phase to be set
    x = keras.layers.Input(shape=(3, 2))
    layer = keras.layers.TimeDistributed(keras.layers.BatchNormalization())
    _ = layer(x)
    self.assertEqual(len(layer.trainable_weights), 2)
    layer.trainable = False
    assert not layer.trainable_weights
    layer.trainable = True
    assert len(layer.trainable_weights) == 2

  def test_TimeDistributed_with_masked_embedding_and_unspecified_shape(self):
    with self.cached_session():
      # test with unspecified shape and Embeddings with mask_zero
      model = keras.models.Sequential()
      model.add(keras.layers.TimeDistributed(
          keras.layers.Embedding(5, 6, mask_zero=True),
          input_shape=(None, None)))  # N by t_1 by t_2 by 6
      model.add(keras.layers.TimeDistributed(
          keras.layers.SimpleRNN(7, return_sequences=True)))
      model.add(keras.layers.TimeDistributed(
          keras.layers.SimpleRNN(8, return_sequences=False)))
      model.add(keras.layers.SimpleRNN(1, return_sequences=False))
      model.compile(optimizer='rmsprop', loss='mse')
      model_input = np.random.randint(low=1, high=5, size=(10, 3, 4),
                                      dtype='int32')
      for i in range(4):
        model_input[i, i:, i:] = 0
      model.fit(model_input,
                np.random.random((10, 1)), epochs=1, batch_size=10)
      mask_outputs = [model.layers[0].compute_mask(model.input)]
      for layer in model.layers[1:]:
        mask_outputs.append(layer.compute_mask(layer.input, mask_outputs[-1]))
      func = keras.backend.function([model.input], mask_outputs[:-1])
      mask_outputs_val = func([model_input])
      ref_mask_val_0 = model_input > 0         # embedding layer
      ref_mask_val_1 = ref_mask_val_0          # first RNN layer
      ref_mask_val_2 = np.any(ref_mask_val_1, axis=-1)     # second RNN layer
      ref_mask_val = [ref_mask_val_0, ref_mask_val_1, ref_mask_val_2]
      for i in range(3):
        self.assertAllEqual(mask_outputs_val[i], ref_mask_val[i])
      self.assertIs(mask_outputs[-1], None)  # final layer

  @test_combinations.generate(
      test_combinations.combine(mode=['graph', 'eager']))
  def test_TimeDistributed_with_masking_layer(self):
    # test with Masking layer
    model = keras.models.Sequential()
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Masking(mask_value=0.,), input_shape=(None, 4)))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(5)))
    model.compile(optimizer='rmsprop', loss='mse')
    model_input = np.random.randint(low=1, high=5, size=(10, 3, 4))
    for i in range(4):
      model_input[i, i:, :] = 0.
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(model_input, np.random.random((10, 3, 5)), epochs=1, batch_size=6)
    mask_outputs = [model.layers[0].compute_mask(model.input)]
    mask_outputs += [
        model.layers[1].compute_mask(model.layers[1].input, mask_outputs[-1])
    ]
    func = keras.backend.function([model.input], mask_outputs)
    mask_outputs_val = func([model_input])
    self.assertEqual((mask_outputs_val[0]).all(), model_input.all())
    self.assertEqual((mask_outputs_val[1]).all(), model_input.all())

  def test_TimeDistributed_with_different_time_shapes(self):
    time_dist = keras.layers.TimeDistributed(keras.layers.Dense(5))
    ph_1 = keras.backend.placeholder(shape=(None, 10, 13))
    out_1 = time_dist(ph_1)
    self.assertEqual(out_1.shape.as_list(), [None, 10, 5])

    ph_2 = keras.backend.placeholder(shape=(None, 1, 13))
    out_2 = time_dist(ph_2)
    self.assertEqual(out_2.shape.as_list(), [None, 1, 5])

    ph_3 = keras.backend.placeholder(shape=(None, 1, 18))
    with self.assertRaisesRegex(ValueError, 'is incompatible with'):
      time_dist(ph_3)

  def test_TimeDistributed_with_invalid_dimensions(self):
    time_dist = keras.layers.TimeDistributed(keras.layers.Dense(5))
    ph = keras.backend.placeholder(shape=(None, 10))
    with self.assertRaisesRegex(
        ValueError,
        '`TimeDistributed` Layer should be passed an `input_shape `'):
      time_dist(ph)

  @test_combinations.generate(
      test_combinations.combine(mode=['graph', 'eager']))
  def test_TimeDistributed_reshape(self):

    class NoReshapeLayer(keras.layers.Layer):

      def call(self, inputs):
        return inputs

    # Built-in layers that aren't stateful use the reshape implementation.
    td1 = keras.layers.TimeDistributed(keras.layers.Dense(5))
    self.assertTrue(td1._always_use_reshape)

    # Built-in layers that are stateful don't use the reshape implementation.
    td2 = keras.layers.TimeDistributed(
        keras.layers.RNN(keras.layers.SimpleRNNCell(10), stateful=True))
    self.assertFalse(td2._always_use_reshape)

    # Custom layers are not allowlisted for the fast reshape implementation.
    td3 = keras.layers.TimeDistributed(NoReshapeLayer())
    self.assertFalse(td3._always_use_reshape)

  @test_combinations.generate(
      test_combinations.combine(mode=['graph', 'eager']))
  def test_TimeDistributed_output_shape_return_types(self):

    class TestLayer(keras.layers.Layer):

      def call(self, inputs):
        return tf.concat([inputs, inputs], axis=-1)

      def compute_output_shape(self, input_shape):
        output_shape = tf.TensorShape(input_shape).as_list()
        output_shape[-1] = output_shape[-1] * 2
        output_shape = tf.TensorShape(output_shape)
        return output_shape

    class TestListLayer(TestLayer):

      def compute_output_shape(self, input_shape):
        shape = super(TestListLayer, self).compute_output_shape(input_shape)
        return shape.as_list()

    class TestTupleLayer(TestLayer):

      def compute_output_shape(self, input_shape):
        shape = super(TestTupleLayer, self).compute_output_shape(input_shape)
        return tuple(shape.as_list())

    # Layers can specify output shape as list/tuple/TensorShape
    test_layers = [TestLayer, TestListLayer, TestTupleLayer]
    for layer in test_layers:
      input_layer = keras.layers.TimeDistributed(layer())
      inputs = keras.backend.placeholder(shape=(None, 2, 4))
      output = input_layer(inputs)
      self.assertEqual(output.shape.as_list(), [None, 2, 8])
      self.assertEqual(
          input_layer.compute_output_shape([None, 2, 4]).as_list(),
          [None, 2, 8])

  @test_combinations.run_all_keras_modes(always_skip_v1=True)
  # TODO(scottzhu): check why v1 session failed.
  def test_TimeDistributed_with_mask_first_implementation(self):
    np.random.seed(100)
    rnn_layer = keras.layers.LSTM(4, return_sequences=True, stateful=True)

    data = np.array([[[[1.0], [1.0]], [[0.0], [1.0]]],
                     [[[1.0], [0.0]], [[1.0], [1.0]]],
                     [[[1.0], [0.0]], [[1.0], [1.0]]]])
    x = keras.layers.Input(shape=(2, 2, 1), batch_size=3)
    x_masking = keras.layers.Masking()(x)
    y = keras.layers.TimeDistributed(rnn_layer)(x_masking)
    model_1 = keras.models.Model(x, y)
    model_1.compile(
        'rmsprop',
        'mse',
        run_eagerly=test_utils.should_run_eagerly())
    output_with_mask = model_1.predict(data, steps=1)

    y = keras.layers.TimeDistributed(rnn_layer)(x)
    model_2 = keras.models.Model(x, y)
    model_2.compile(
        'rmsprop',
        'mse',
        run_eagerly=test_utils.should_run_eagerly())
    output = model_2.predict(data, steps=1)

    self.assertNotAllClose(output_with_mask, output, atol=1e-7)

  @test_combinations.run_all_keras_modes
  @parameterized.named_parameters(
      *test_utils.generate_combinations_with_testcase_name(
          layer=[keras.layers.LSTM,
                 keras.layers.Dense]))
  def test_TimeDistributed_with_ragged_input(self, layer):
    if tf.executing_eagerly():
      self.skipTest('b/143103634')
    np.random.seed(100)
    layer = layer(4)
    ragged_data = tf.ragged.constant(
        [[[[1.0], [1.0]], [[2.0], [2.0]]],
         [[[4.0], [4.0]], [[5.0], [5.0]], [[6.0], [6.0]]],
         [[[7.0], [7.0]], [[8.0], [8.0]], [[9.0], [9.0]]]],
        ragged_rank=1)

    x_ragged = keras.Input(shape=(None, 2, 1), dtype='float32', ragged=True)
    y_ragged = keras.layers.TimeDistributed(layer)(x_ragged)
    model_1 = keras.models.Model(x_ragged, y_ragged)
    model_1._run_eagerly = test_utils.should_run_eagerly()
    output_ragged = model_1.predict(ragged_data, steps=1)

    x_dense = keras.Input(shape=(None, 2, 1), dtype='float32')
    masking = keras.layers.Masking()(x_dense)
    y_dense = keras.layers.TimeDistributed(layer)(masking)
    model_2 = keras.models.Model(x_dense, y_dense)
    dense_data = ragged_data.to_tensor()
    model_2._run_eagerly = test_utils.should_run_eagerly()
    output_dense = model_2.predict(dense_data, steps=1)

    output_ragged = convert_ragged_tensor_value(output_ragged)
    self.assertAllEqual(output_ragged.to_tensor(), output_dense)

  @test_combinations.run_all_keras_modes
  def test_TimeDistributed_with_ragged_input_with_batch_size(self):
    np.random.seed(100)
    layer = keras.layers.Dense(16)

    ragged_data = tf.ragged.constant(
        [[[[1.0], [1.0]], [[2.0], [2.0]]],
         [[[4.0], [4.0]], [[5.0], [5.0]], [[6.0], [6.0]]],
         [[[7.0], [7.0]], [[8.0], [8.0]], [[9.0], [9.0]]]],
        ragged_rank=1)

    # Use the first implementation by specifying batch_size
    x_ragged = keras.Input(shape=(None, 2, 1), batch_size=3, dtype='float32',
                           ragged=True)
    y_ragged = keras.layers.TimeDistributed(layer)(x_ragged)
    model_1 = keras.models.Model(x_ragged, y_ragged)
    output_ragged = model_1.predict(ragged_data, steps=1)

    x_dense = keras.Input(shape=(None, 2, 1), batch_size=3, dtype='float32')
    masking = keras.layers.Masking()(x_dense)
    y_dense = keras.layers.TimeDistributed(layer)(masking)
    model_2 = keras.models.Model(x_dense, y_dense)
    dense_data = ragged_data.to_tensor()
    output_dense = model_2.predict(dense_data, steps=1)

    output_ragged = convert_ragged_tensor_value(output_ragged)
    self.assertAllEqual(output_ragged.to_tensor(), output_dense)

  def test_TimeDistributed_set_static_shape(self):
    layer = keras.layers.TimeDistributed(keras.layers.Conv2D(16, (3, 3)))
    inputs = keras.Input(batch_shape=(1, None, 32, 32, 1))
    outputs = layer(inputs)
    # Make sure the batch dim is not lost after array_ops.reshape.
    self.assertListEqual(outputs.shape.as_list(), [1, None, 30, 30, 16])

  @test_combinations.run_all_keras_modes
  def test_TimeDistributed_with_mimo(self):
    dense_1 = keras.layers.Dense(8)
    dense_2 = keras.layers.Dense(16)

    class TestLayer(keras.layers.Layer):

      def __init__(self):
        super(TestLayer, self).__init__()
        self.dense_1 = dense_1
        self.dense_2 = dense_2

      def call(self, inputs):
        return self.dense_1(inputs[0]), self.dense_2(inputs[1])

      def compute_output_shape(self, input_shape):
        output_shape_1 = self.dense_1.compute_output_shape(input_shape[0])
        output_shape_2 = self.dense_2.compute_output_shape(input_shape[1])
        return output_shape_1, output_shape_2

    np.random.seed(100)
    layer = TestLayer()

    data_1 = tf.constant([[[[1.0], [1.0]], [[2.0], [2.0]]],
                          [[[4.0], [4.0]], [[5.0], [5.0]]],
                          [[[7.0], [7.0]], [[8.0], [8.0]]]])

    data_2 = tf.constant([[[[1.0], [1.0]], [[2.0], [2.0]]],
                          [[[4.0], [4.0]], [[5.0], [5.0]]],
                          [[[7.0], [7.0]], [[8.0], [8.0]]]])

    x1 = keras.Input(shape=(None, 2, 1), dtype='float32')
    x2 = keras.Input(shape=(None, 2, 1), dtype='float32')
    y1, y2 = keras.layers.TimeDistributed(layer)([x1, x2])
    model_1 = keras.models.Model([x1, x2], [y1, y2])
    model_1.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=test_utils.should_run_eagerly())
    output_1 = model_1.predict((data_1, data_2), steps=1)

    y1 = dense_1(x1)
    y2 = dense_2(x2)
    model_2 = keras.models.Model([x1, x2], [y1, y2])
    output_2 = model_2.predict((data_1, data_2), steps=1)

    self.assertAllClose(output_1, output_2)

    model_1.fit(
        x=[np.random.random((10, 2, 2, 1)),
           np.random.random((10, 2, 2, 1))],
        y=[np.random.random((10, 2, 2, 8)),
           np.random.random((10, 2, 2, 16))],
        epochs=1,
        batch_size=3)

  def test_TimeDistributed_Attention(self):
    query_input = keras.layers.Input(shape=(None, 1, 10), dtype='float32')
    value_input = keras.layers.Input(shape=(None, 4, 10), dtype='float32')

    # Query-value attention of shape [batch_size, Tq, filters].
    query_value_attention_seq = keras.layers.TimeDistributed(
        keras.layers.Attention())([query_input, value_input])
    model = keras.models.Model([query_input, value_input],
                               query_value_attention_seq)
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(
        [np.random.random((10, 8, 1, 10)),
         np.random.random((10, 8, 4, 10))],
        np.random.random((10, 8, 1, 10)),
        epochs=1,
        batch_size=10)

    # test config and serialization/deserialization
    model.get_config()
    model = keras.models.model_from_json(model.to_json())
    model.summary()


def convert_ragged_tensor_value(inputs):
  if isinstance(inputs, tf.compat.v1.ragged.RaggedTensorValue):
    flat_values = tf.convert_to_tensor(
        value=inputs.flat_values,
        name='flat_values')
    return tf.RaggedTensor.from_nested_row_splits(
        flat_values, inputs.nested_row_splits, validate=False)
  return inputs


if __name__ == '__main__':
  tf.test.main()
