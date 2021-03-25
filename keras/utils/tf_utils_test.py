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
"""Tests for Keras TF utils."""

import tensorflow.compat.v2 as tf

from absl.testing import parameterized

import keras
from keras import combinations
from keras.utils import tf_utils

try:
  import attr  # pylint:disable=g-import-not-at-top
except ImportError:
  attr = None


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class TestIsSymbolicTensor(tf.test.TestCase, parameterized.TestCase):

  def test_default_behavior(self):
    if tf.executing_eagerly():
      self.assertFalse(tf_utils.is_symbolic_tensor(
          tf.Variable(name='blah', initial_value=0.)))
      self.assertFalse(
          tf_utils.is_symbolic_tensor(
              tf.convert_to_tensor(0.)))
      self.assertFalse(tf_utils.is_symbolic_tensor(
          tf.SparseTensor(
              indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])))
    else:
      self.assertTrue(tf_utils.is_symbolic_tensor(
          tf.Variable(name='blah', initial_value=0.)))
      self.assertTrue(
          tf_utils.is_symbolic_tensor(
              tf.convert_to_tensor(0.)))
      self.assertTrue(tf_utils.is_symbolic_tensor(
          tf.SparseTensor(
              indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])))

  def test_works_with_registered(self):

    class CustomClass(object):

      def value(self):
        return tf.convert_to_tensor(42.)

    tf.register_tensor_conversion_function(
        CustomClass, lambda value, **_: value.value())

    tf_utils.register_symbolic_tensor_type(CustomClass)

    if tf.executing_eagerly():
      self.assertFalse(tf_utils.is_symbolic_tensor(
          tf.Variable(name='blah', initial_value=0.)))
      self.assertFalse(
          tf_utils.is_symbolic_tensor(
              tf.convert_to_tensor(0.)))
      self.assertFalse(tf_utils.is_symbolic_tensor(
          tf.SparseTensor(
              indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])))
      self.assertFalse(tf_utils.is_symbolic_tensor(CustomClass()))
    else:
      self.assertTrue(tf_utils.is_symbolic_tensor(
          tf.Variable(name='blah', initial_value=0.)))
      self.assertTrue(
          tf_utils.is_symbolic_tensor(
              tf.convert_to_tensor(0.)))
      self.assertTrue(tf_utils.is_symbolic_tensor(
          tf.SparseTensor(
              indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])))
      self.assertTrue(tf_utils.is_symbolic_tensor(CustomClass()))

  def test_enables_nontensor_plumbing(self):
    if tf.executing_eagerly():
      self.skipTest('`compile` functionality changed.')
    # Setup.

    class Foo(object):

      def __init__(self, input_):
        self._input = input_
        self.value = tf.convert_to_tensor([[42.]])

      @property
      def dtype(self):
        return self.value.dtype

    tf.register_tensor_conversion_function(
        Foo, lambda x, *args, **kwargs: x.value)
    tf_utils.register_symbolic_tensor_type(Foo)

    class PlumbingLayer(keras.layers.Lambda):

      def __init__(self, fn, **kwargs):
        def _fn(*fargs, **fkwargs):
          d = fn(*fargs, **fkwargs)
          x = tf.convert_to_tensor(d)
          d.shape = x.shape
          d.get_shape = x.get_shape
          return d, x
        super(PlumbingLayer, self).__init__(_fn, **kwargs)
        self._enter_dunder_call = False

      def __call__(self, inputs, *args, **kwargs):
        self._enter_dunder_call = True
        d, _ = super(PlumbingLayer, self).__call__(inputs, *args, **kwargs)
        self._enter_dunder_call = False
        return d

      def call(self, inputs, *args, **kwargs):
        d, v = super(PlumbingLayer, self).call(inputs, *args, **kwargs)
        if self._enter_dunder_call:
          return d, v
        return d

    # User-land.
    model = keras.Sequential([
        keras.layers.InputLayer((1,)),
        PlumbingLayer(Foo),  # Makes a `Foo` object.
    ])
    # Let's ensure Keras graph history is preserved by composing the models.
    model = keras.Model(model.inputs, model(model.outputs))
    # Now we instantiate the model and verify we have a `Foo` object, not a
    # `Tensor`.
    y = model(tf.convert_to_tensor([[7.]]))
    self.assertIsInstance(y, Foo)
    # Confirm that (custom) loss sees `Foo` instance, not Tensor.
    obtained_prediction_box = [None]
    def custom_loss(y_obs, y_pred):
      del y_obs
      obtained_prediction_box[0] = y_pred
      return y_pred
    # Apparently `compile` calls the loss function enough to trigger the
    # side-effect.
    model.compile('SGD', loss=custom_loss)
    self.assertIsInstance(obtained_prediction_box[0], Foo)


class ConvertInnerNodeDataTest(tf.test.TestCase):

  def test_convert_inner_node_data(self):
    data = tf_utils.convert_inner_node_data((tf_utils.ListWrapper(['l', 2, 3]),
                                             tf_utils.ListWrapper(['l', 5, 6])))
    self.assertEqual(data, (['l', 2, 3], ['l', 5, 6]))

    data = tf_utils.convert_inner_node_data(((['l', 2, 3], ['l', 5, 6])),
                                            wrap=True)
    self.assertTrue(all(isinstance(ele, tf_utils.ListWrapper) for ele in data))


class AttrsTest(tf.test.TestCase):

  def test_map_structure_with_atomic_accept_attr(self):
    if attr is None:
      self.skipTest('attr module is unavailable.')

    @attr.s(frozen=True)
    class Foo(object):

      bar = attr.ib()

    self.assertEqual(
        Foo(2),
        tf_utils.map_structure_with_atomic(
            is_atomic_fn=lambda x: isinstance(x, int),
            map_fn=lambda x: x + 1,
            nested=Foo(1)))


class TestIsRagged(tf.test.TestCase):

  def test_is_ragged_return_true_for_ragged_tensor(self):
    tensor = tf.RaggedTensor.from_row_splits(
        values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
    self.assertTrue(tf_utils.is_ragged(tensor))

  def test_is_ragged_return_false_for_list(self):
    tensor = [1., 2., 3.]
    self.assertFalse(tf_utils.is_ragged(tensor))


class TestIsExtensionType(tf.test.TestCase):

  def test_is_extension_type_return_true_for_ragged_tensor(self):
    self.assertTrue(tf_utils.is_extension_type(
        tf.ragged.constant([[1, 2], [3]])))

  def test_is_extension_type_return_true_for_sparse_tensor(self):
    self.assertTrue(tf_utils.is_extension_type(
        tf.sparse.from_dense([[1, 2], [3, 4]])))

  def test_is_extension_type_return_false_for_dense_tensor(self):
    self.assertFalse(tf_utils.is_extension_type(
        tf.constant([[1, 2], [3, 4]])))

  def test_is_extension_type_return_false_for_list(self):
    tensor = [1., 2., 3.]
    self.assertFalse(tf_utils.is_extension_type(tensor))

if __name__ == '__main__':
  tf.test.main()
