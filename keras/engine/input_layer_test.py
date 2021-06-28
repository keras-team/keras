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
#,============================================================================
"""Tests for InputLayer construction."""

import tensorflow.compat.v2 as tf
from tensorflow.python.framework import type_spec
from keras import backend
from keras import combinations
from keras import keras_parameterized
from keras.engine import functional
from keras.engine import input_layer as input_layer_lib
from keras.layers import core
from keras.saving import model_config


class TwoTensors(tf.__internal__.CompositeTensor):
  """A simple value type to test TypeSpec.

  Contains two tensors (x, y) and a string (color).  The color value is a
  stand-in for any extra type metadata we might need to store.

  This value type contains no single dtype.
  """

  def __init__(self, x, y, color='red', assign_variant_dtype=False):
    assert isinstance(color, str)
    self.x = tf.convert_to_tensor(x)
    self.y = tf.convert_to_tensor(y)
    self.color = color
    self.shape = tf.TensorShape(None)
    self._shape = tf.TensorShape(None)
    if assign_variant_dtype:
      self.dtype = tf.variant
    self._assign_variant_dtype = assign_variant_dtype

  def _type_spec(self):
    return TwoTensorsSpecNoOneDtype(
        self.x.shape, self.x.dtype, self.y.shape,
        self.y.dtype, color=self.color,
        assign_variant_dtype=self._assign_variant_dtype)


def as_shape(shape):
  """Converts the given object to a TensorShape."""
  if isinstance(shape, tf.TensorShape):
    return shape
  else:
    return tf.TensorShape(shape)


@type_spec.register('tf.TwoTensorsSpec')
class TwoTensorsSpecNoOneDtype(tf.TypeSpec):
  """A TypeSpec for the TwoTensors value type."""

  def __init__(
      self, x_shape, x_dtype, y_shape, y_dtype, color='red',
      assign_variant_dtype=False):
    self.x_shape = as_shape(x_shape)
    self.x_dtype = tf.as_dtype(x_dtype)
    self.y_shape = as_shape(y_shape)
    self.y_dtype = tf.as_dtype(y_dtype)
    self.color = color
    self.shape = tf.TensorShape(None)
    self._shape = tf.TensorShape(None)
    if assign_variant_dtype:
      self.dtype = tf.variant
    self._assign_variant_dtype = assign_variant_dtype

  value_type = property(lambda self: TwoTensors)

  @property
  def _component_specs(self):
    return (tf.TensorSpec(self.x_shape, self.x_dtype),
            tf.TensorSpec(self.y_shape, self.y_dtype))

  def _to_components(self, value):
    return (value.x, value.y)

  def _from_components(self, components):
    x, y = components
    return TwoTensors(x, y, self.color)

  def _serialize(self):
    return (self.x_shape, self.x_dtype, self.y_shape, self.y_dtype, self.color)

  @classmethod
  def from_value(cls, value):
    return cls(value.x.shape, value.x.dtype, value.y.shape, value.y.dtype,
               value.color)


type_spec.register_type_spec_from_value_converter(
    TwoTensors, TwoTensorsSpecNoOneDtype.from_value)


class InputLayerTest(keras_parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testBasicOutputShapeNoBatchSize(self):
    # Create a Keras Input
    x = input_layer_lib.Input(shape=(32,), name='input_a')
    self.assertAllEqual(x.shape.as_list(), [None, 32])

    # Verify you can construct and use a model w/ this input
    model = functional.Functional(x, x * 2.0)
    self.assertAllEqual(model(tf.ones((3, 32))),
                        tf.ones((3, 32)) * 2.0)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testBasicOutputShapeWithBatchSize(self):
    # Create a Keras Input
    x = input_layer_lib.Input(batch_size=6, shape=(32,), name='input_b')
    self.assertAllEqual(x.shape.as_list(), [6, 32])

    # Verify you can construct and use a model w/ this input
    model = functional.Functional(x, x * 2.0)
    self.assertAllEqual(model(tf.ones(x.shape)),
                        tf.ones(x.shape) * 2.0)

  @combinations.generate(combinations.combine(mode=['eager']))
  def testBasicOutputShapeNoBatchSizeInTFFunction(self):
    model = None
    @tf.function
    def run_model(inp):
      nonlocal model
      if not model:
        # Create a Keras Input
        x = input_layer_lib.Input(shape=(8,), name='input_a')
        self.assertAllEqual(x.shape.as_list(), [None, 8])

        # Verify you can construct and use a model w/ this input
        model = functional.Functional(x, x * 2.0)
      return model(inp)

    self.assertAllEqual(run_model(tf.ones((10, 8))),
                        tf.ones((10, 8)) * 2.0)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def testInputTensorArg(self):
    # Create a Keras Input
    x = input_layer_lib.Input(tensor=tf.zeros((7, 32)))
    self.assertAllEqual(x.shape.as_list(), [7, 32])

    # Verify you can construct and use a model w/ this input
    model = functional.Functional(x, x * 2.0)
    self.assertAllEqual(model(tf.ones(x.shape)),
                        tf.ones(x.shape) * 2.0)

  @combinations.generate(combinations.combine(mode=['eager']))
  def testInputTensorArgInTFFunction(self):
    # We use a mutable model container instead of a model python variable,
    # because python 2.7 does not have `nonlocal`
    model_container = {}

    @tf.function
    def run_model(inp):
      if not model_container:
        # Create a Keras Input
        x = input_layer_lib.Input(tensor=tf.zeros((10, 16)))
        self.assertAllEqual(x.shape.as_list(), [10, 16])

        # Verify you can construct and use a model w/ this input
        model_container['model'] = functional.Functional(x, x * 3.0)
      return model_container['model'](inp)

    self.assertAllEqual(run_model(tf.ones((10, 16))),
                        tf.ones((10, 16)) * 3.0)

  @combinations.generate(combinations.combine(mode=['eager']))
  def testCompositeInputTensorArg(self):
    # Create a Keras Input
    rt = tf.RaggedTensor.from_row_splits(
        values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
    x = input_layer_lib.Input(tensor=rt)

    # Verify you can construct and use a model w/ this input
    model = functional.Functional(x, x * 2)

    # And that the model works
    rt = tf.RaggedTensor.from_row_splits(
        values=[3, 21, 4, 1, 53, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
    self.assertAllEqual(model(rt), rt * 2)

  @combinations.generate(combinations.combine(mode=['eager']))
  def testCompositeInputTensorArgInTFFunction(self):
    # We use a mutable model container instead of a model python variable,
    # because python 2.7 does not have `nonlocal`
    model_container = {}

    @tf.function
    def run_model(inp):
      if not model_container:
        # Create a Keras Input
        rt = tf.RaggedTensor.from_row_splits(
            values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
        x = input_layer_lib.Input(tensor=rt)

        # Verify you can construct and use a model w/ this input
        model_container['model'] = functional.Functional(x, x * 3)
      return model_container['model'](inp)

    # And verify the model works
    rt = tf.RaggedTensor.from_row_splits(
        values=[3, 21, 4, 1, 53, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
    self.assertAllEqual(run_model(rt), rt * 3)

  @combinations.generate(combinations.combine(mode=['eager']))
  def testNoMixingArgsWithTypeSpecArg(self):
    with self.assertRaisesRegexp(
        ValueError, 'all other args except `name` must be None'):
      input_layer_lib.Input(
          shape=(4, 7),
          type_spec=tf.TensorSpec((2, 7, 32), tf.float32))
    with self.assertRaisesRegexp(
        ValueError, 'all other args except `name` must be None'):
      input_layer_lib.Input(
          batch_size=4,
          type_spec=tf.TensorSpec((7, 32), tf.float32))
    with self.assertRaisesRegexp(
        ValueError, 'all other args except `name` must be None'):
      input_layer_lib.Input(
          dtype=tf.int64,
          type_spec=tf.TensorSpec((7, 32), tf.float32))
    with self.assertRaisesRegexp(
        ValueError, 'all other args except `name` must be None'):
      input_layer_lib.Input(
          sparse=True,
          type_spec=tf.TensorSpec((7, 32), tf.float32))
    with self.assertRaisesRegexp(
        ValueError, 'all other args except `name` must be None'):
      input_layer_lib.Input(
          ragged=True,
          type_spec=tf.TensorSpec((7, 32), tf.float32))

  @combinations.generate(combinations.combine(mode=['eager']))
  def testTypeSpecArg(self):
    # Create a Keras Input
    x = input_layer_lib.Input(
        type_spec=tf.TensorSpec((7, 32), tf.float32))
    self.assertAllEqual(x.shape.as_list(), [7, 32])

    # Verify you can construct and use a model w/ this input
    model = functional.Functional(x, x * 2.0)
    self.assertAllEqual(model(tf.ones(x.shape)),
                        tf.ones(x.shape) * 2.0)

    # Test serialization / deserialization
    model = functional.Functional.from_config(model.get_config())
    self.assertAllEqual(model(tf.ones(x.shape)),
                        tf.ones(x.shape) * 2.0)

    model = model_config.model_from_json(model.to_json())
    self.assertAllEqual(model(tf.ones(x.shape)),
                        tf.ones(x.shape) * 2.0)

  @combinations.generate(combinations.combine(mode=['eager']))
  def testTypeSpecArgInTFFunction(self):
    # We use a mutable model container instead of a model python variable,
    # because python 2.7 does not have `nonlocal`
    model_container = {}

    @tf.function
    def run_model(inp):
      if not model_container:
        # Create a Keras Input
        x = input_layer_lib.Input(
            type_spec=tf.TensorSpec((10, 16), tf.float32))
        self.assertAllEqual(x.shape.as_list(), [10, 16])

        # Verify you can construct and use a model w/ this input
        model_container['model'] = functional.Functional(x, x * 3.0)
      return model_container['model'](inp)

    self.assertAllEqual(run_model(tf.ones((10, 16))),
                        tf.ones((10, 16)) * 3.0)

  @combinations.generate(combinations.combine(mode=['eager']))
  def testCompositeTypeSpecArg(self):
    # Create a Keras Input
    rt = tf.RaggedTensor.from_row_splits(
        values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
    x = input_layer_lib.Input(type_spec=rt._type_spec)

    # Verify you can construct and use a model w/ this input
    model = functional.Functional(x, x * 2)

    # And that the model works
    rt = tf.RaggedTensor.from_row_splits(
        values=[3, 21, 4, 1, 53, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
    self.assertAllEqual(model(rt), rt * 2)

    # Test serialization / deserialization
    model = functional.Functional.from_config(model.get_config())
    self.assertAllEqual(model(rt), rt * 2)
    model = model_config.model_from_json(model.to_json())
    self.assertAllEqual(model(rt), rt * 2)

  @combinations.generate(combinations.combine(mode=['eager']))
  def testCompositeTypeSpecArgInTFFunction(self):
    # We use a mutable model container instead of a model pysthon variable,
    # because python 2.7 does not have `nonlocal`
    model_container = {}

    @tf.function
    def run_model(inp):
      if not model_container:
        # Create a Keras Input
        rt = tf.RaggedTensor.from_row_splits(
            values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
        x = input_layer_lib.Input(type_spec=rt._type_spec)

        # Verify you can construct and use a model w/ this input
        model_container['model'] = functional.Functional(x, x * 3)
      return model_container['model'](inp)

    # And verify the model works
    rt = tf.RaggedTensor.from_row_splits(
        values=[3, 21, 4, 1, 53, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
    self.assertAllEqual(run_model(rt), rt * 3)

  @combinations.generate(combinations.combine(mode=['eager']))
  def testCompositeTypeSpecArgWithoutDtype(self):
    for assign_variant_dtype in [False, True]:
      # Create a Keras Input
      spec = TwoTensorsSpecNoOneDtype(
          (1, 2, 3), tf.float32, (1, 2, 3), tf.int64,
          assign_variant_dtype=assign_variant_dtype)
      x = input_layer_lib.Input(type_spec=spec)

      def lambda_fn(tensors):
        return (tf.cast(tensors.x, tf.float64)
                + tf.cast(tensors.y, tf.float64))
      # Verify you can construct and use a model w/ this input
      model = functional.Functional(x, core.Lambda(lambda_fn)(x))

      # And that the model works
      two_tensors = TwoTensors(tf.ones((1, 2, 3)) * 2.0,
                               tf.ones(1, 2, 3))
      self.assertAllEqual(model(two_tensors), lambda_fn(two_tensors))

      # Test serialization / deserialization
      model = functional.Functional.from_config(model.get_config())
      self.assertAllEqual(model(two_tensors), lambda_fn(two_tensors))
      model = model_config.model_from_json(model.to_json())
      self.assertAllEqual(model(two_tensors), lambda_fn(two_tensors))

  def test_serialize_with_unknown_rank(self):
    inp = backend.placeholder(shape=None, dtype=tf.string)
    x = input_layer_lib.InputLayer(input_tensor=inp, dtype=tf.string)
    loaded = input_layer_lib.InputLayer.from_config(x.get_config())
    self.assertIsNone(loaded._batch_input_shape)


if __name__ == '__main__':
  tf.test.main()
