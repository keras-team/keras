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
"""Tests for traceback_utils."""

from keras import layers
from keras.utils import traceback_utils
import tensorflow.compat.v2 as tf


class TracebackUtilsTest(tf.test.TestCase):

  def test_info_injection_basics(self):
    def error_fn(arg_1, arg_2, keyword_arg_1=None, keyword_arg_2=None):
      raise ValueError('Original message')

    with self.assertRaises(ValueError) as e:
      traceback_utils.inject_argument_info_in_traceback(
          error_fn, 'ObjName')(1, 2, keyword_arg_1=3, keyword_arg_2=4)
    self.assertIn('Original message', str(e.exception))
    self.assertIn('Exception encountered when calling ObjName',
                  str(e.exception))
    self.assertIn('Call arguments received:', str(e.exception))
    self.assertIn('arg_1=1', str(e.exception))
    self.assertIn('arg_2=2', str(e.exception))
    self.assertIn('keyword_arg_1=3', str(e.exception))
    self.assertIn('keyword_arg_2=4', str(e.exception))

    with self.assertRaises(ValueError) as e:
      traceback_utils.inject_argument_info_in_traceback(
          error_fn)(1, 2, keyword_arg_1=3, keyword_arg_2=4)
    self.assertIn('Exception encountered when calling error_fn',
                  str(e.exception))

  def test_info_injection_no_args(self):
    def error_fn():
      raise ValueError('Original message')

    with self.assertRaises(ValueError) as e:
      traceback_utils.inject_argument_info_in_traceback(error_fn)()
    self.assertEqual(str(e.exception).count('Call arguments received:'), 0)

  def test_info_injection_unbindable(self):
    def error_fn(arg_1, keyword_arg_1=1):
      return arg_1 + keyword_arg_1

    with self.assertRaises(TypeError) as e:
      traceback_utils.inject_argument_info_in_traceback(error_fn)()
    self.assertIn('missing 1 required positional argument', str(e.exception))

  def test_info_injection_nested(self):
    def inner_fn(arg_1):
      raise ValueError('Original message')

    def outer_fn(arg_1):
      return inner_fn(arg_1)

    with self.assertRaises(ValueError) as e:
      traceback_utils.inject_argument_info_in_traceback(
          outer_fn)(1)
    self.assertEqual(str(e.exception).count('Call arguments received:'), 1)

  def test_info_injection_tf_op_error(self):
    def error_fn(arg_1, keyword_arg_1=1):
      return arg_1 + keyword_arg_1 + tf.zeros((2, 3))

    with self.assertRaises(tf.errors.InvalidArgumentError) as e:
      traceback_utils.inject_argument_info_in_traceback(error_fn)(
          tf.zeros((3, 3)))
    self.assertIn('Incompatible shapes', str(e.exception))
    self.assertIn('Call arguments received', str(e.exception))


class LayerCallInfoInjectionTest(tf.test.TestCase):

  def assert_info_injected(self, fn):
    tf.debugging.enable_traceback_filtering()
    try:
      fn()
    except Exception as e:  # pylint: disable=broad-except
      # Info should be injected exactly once.
      self.assertEqual(str(e).count('Call arguments received:'), 1)  # pylint: disable=g-assert-in-except

  def test_custom_layer_call_nested(self):

    class InnerLayer(layers.Layer):

      def call(self, inputs, training=False, mask=None):
        return inputs + tf.zeros((3, 4))

    class OuterLayer(layers.Layer):

      def __init__(self):
        super().__init__()
        self.inner = InnerLayer()

      def call(self, inputs, training=True):
        return self.inner(inputs)

    def fn():
      layer = OuterLayer()
      layer(tf.zeros((3, 5)), training=False)

    self.assert_info_injected(fn)

  def test_custom_layer_call_eager_dense_input(self):

    class MyLayer(layers.Layer):

      def call(self, inputs, training=False, mask=None):
        return inputs + tf.zeros((3, 4))

    def fn():
      layer = MyLayer()
      layer(tf.zeros((3, 5)), training=False)

    self.assert_info_injected(fn)

  def test_custom_layer_call_eager_sparse_input(self):

    class MyLayer(layers.Layer):

      def call(self, inputs, training=False, mask=None):
        return inputs + tf.zeros((3, 4))

    def fn():
      layer = MyLayer()
      layer(
          tf.SparseTensor(indices=[[0, 0]], values=[1], dense_shape=[3, 5]),
          training=False)

    self.assert_info_injected(fn)

  def test_custom_layer_call_eager_ragged_input(self):

    class MyLayer(layers.Layer):

      def call(self, inputs, training=False, mask=None):
        return inputs + tf.zeros((3, 4))

    def fn():
      layer = MyLayer()
      layer(tf.ragged.constant([[0, 0, 0], [0, 0]]), training=False)

    self.assert_info_injected(fn)

  def test_custom_layer_call_symbolic(self):

    class MyLayer(layers.Layer):

      def call(self, inputs, training=False, mask=None):
        return inputs + tf.zeros((3, 4))

    def fn():
      layer = MyLayer()
      layer(layers.Input((3, 5)), training=False)

    self.assert_info_injected(fn)

  def test_custom_layer_call_unbindable(self):

    class MyLayer(layers.Layer):

      def __init__(self):
        super().__init__()
        self.input_spec = layers.InputSpec(shape=(3, 4))

      def call(self, inputs, training=False, mask=None):
        return inputs + tf.zeros((3, 4))

    def fn():
      layer = MyLayer()
      layer(bad=True, arguments=True)

    with self.assertRaisesRegex(
        ValueError, 'The first argument to `Layer.call` must always'):
      fn()


if __name__ == '__main__':
  if tf.__internal__.tf2.enabled():
    tf.test.main()
