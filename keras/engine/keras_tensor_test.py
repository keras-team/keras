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
"""InputSpec tests."""
# pylint: disable=g-bad-import-order

import tensorflow.compat.v2 as tf

from absl.testing import parameterized
from keras import keras_parameterized
from keras import layers
from keras.engine import keras_tensor
from keras.engine import training


class CustomTypeSpec(tf.TypeSpec):
  """Stubbed-out custom type spec, for testing."""

  def __init__(self, shape, dtype):
    self.shape = tf.TensorShape(shape)
    self.dtype = tf.dtypes.as_dtype(dtype)

  # Stub implementations for all the TypeSpec methods:
  value_type = None
  _to_components = lambda self, value: None
  _from_components = lambda self, components: None
  _component_specs = property(lambda self: None)
  _serialize = lambda self: (self.shape, self.dtype)


class CustomTypeSpec2(CustomTypeSpec):
  """Adds a with_shape method to CustomTypeSpec."""

  def with_shape(self, new_shape):
    return CustomTypeSpec2(new_shape, self.dtype)


class KerasTensorTest(keras_parameterized.TestCase):

  def test_repr_and_string(self):
    kt = keras_tensor.KerasTensor(
        type_spec=tf.TensorSpec(shape=(1, 2, 3), dtype=tf.float32))
    expected_str = ("KerasTensor(type_spec=TensorSpec(shape=(1, 2, 3), "
                    "dtype=tf.float32, name=None))")
    expected_repr = "<KerasTensor: shape=(1, 2, 3) dtype=float32>"
    self.assertEqual(expected_str, str(kt))
    self.assertEqual(expected_repr, repr(kt))

    kt = keras_tensor.KerasTensor(
        type_spec=tf.TensorSpec(shape=(2,), dtype=tf.int32),
        inferred_value=[2, 3])
    expected_str = ("KerasTensor(type_spec=TensorSpec(shape=(2,), "
                    "dtype=tf.int32, name=None), inferred_value=[2, 3])")
    expected_repr = (
        "<KerasTensor: shape=(2,) dtype=int32 inferred_value=[2, 3]>")
    self.assertEqual(expected_str, str(kt))
    self.assertEqual(expected_repr, repr(kt))

    kt = keras_tensor.KerasTensor(
        type_spec=tf.SparseTensorSpec(
            shape=(1, 2, 3), dtype=tf.float32))
    expected_str = ("KerasTensor(type_spec=SparseTensorSpec("
                    "TensorShape([1, 2, 3]), tf.float32))")
    expected_repr = (
        "<KerasTensor: type_spec=SparseTensorSpec("
        "TensorShape([1, 2, 3]), tf.float32)>")
    self.assertEqual(expected_str, str(kt))
    self.assertEqual(expected_repr, repr(kt))

    inp = layers.Input(shape=(3, 5))
    kt = layers.Dense(10)(inp)
    expected_str = (
        "KerasTensor(type_spec=TensorSpec(shape=(None, 3, 10), "
        "dtype=tf.float32, name=None), name='dense/BiasAdd:0', "
        "description=\"created by layer 'dense'\")")
    expected_repr = (
        "<KerasTensor: shape=(None, 3, 10) dtype=float32 (created "
        "by layer 'dense')>")
    self.assertEqual(expected_str, str(kt))
    self.assertEqual(expected_repr, repr(kt))

    kt = tf.reshape(kt, shape=(3, 5, 2))
    expected_str = (
        "KerasTensor(type_spec=TensorSpec(shape=(3, 5, 2), dtype=tf.float32, "
        "name=None), name='tf.reshape/Reshape:0', description=\"created "
        "by layer 'tf.reshape'\")")
    expected_repr = ("<KerasTensor: shape=(3, 5, 2) dtype=float32 (created "
                     "by layer 'tf.reshape')>")
    self.assertEqual(expected_str, str(kt))
    self.assertEqual(expected_repr, repr(kt))

    kts = tf.unstack(kt)
    for i in range(3):
      expected_str = (
          "KerasTensor(type_spec=TensorSpec(shape=(5, 2), dtype=tf.float32, "
          "name=None), name='tf.unstack/unstack:%s', description=\"created "
          "by layer 'tf.unstack'\")" % (i,))
      expected_repr = ("<KerasTensor: shape=(5, 2) dtype=float32 "
                       "(created by layer 'tf.unstack')>")
      self.assertEqual(expected_str, str(kts[i]))
      self.assertEqual(expected_repr, repr(kts[i]))

  @parameterized.parameters(
      {"property_name": "values"},
      {"property_name": "indices"},
      {"property_name": "dense_shape"},
  )
  def test_sparse_instance_property(self, property_name):
    inp = layers.Input(shape=[3], sparse=True)
    out = getattr(inp, property_name)
    model = training.Model(inp, out)

    x = tf.SparseTensor([[0, 0], [0, 1], [1, 1], [1, 2]], [1, 2, 3, 4], [2, 3])
    expected_property = getattr(x, property_name)
    self.assertAllEqual(model(x), expected_property)

    # Test that it works with serialization and deserialization as well
    model_config = model.get_config()
    model2 = training.Model.from_config(model_config)
    self.assertAllEqual(model2(x), expected_property)

  @parameterized.parameters([
      (tf.TensorSpec([2, 3], tf.int32), [2, 3]),
      (tf.RaggedTensorSpec([2, None]), [2, None]),
      (tf.SparseTensorSpec([8]), [8]),
      (CustomTypeSpec([3, 8], tf.int32), [3, 8]),
  ])
  def test_shape(self, spec, expected_shape):
    kt = keras_tensor.KerasTensor(spec)
    self.assertEqual(kt.shape.as_list(), expected_shape)

  @parameterized.parameters([
      (tf.TensorSpec([8, 3], tf.int32), [8, 3]),
      (tf.TensorSpec([None, 3], tf.int32), [8, 3]),
      (tf.TensorSpec(None, tf.int32), [8, 3]),
      (tf.TensorSpec(None, tf.int32), [8, None]),
      (tf.TensorSpec(None, tf.int32), None),
      (tf.RaggedTensorSpec([2, None, None]), [2, None, 5]),
      (tf.SparseTensorSpec([8]), [8]),
      (CustomTypeSpec2([3, None], tf.int32), [3, 8]),
  ])
  def test_set_shape(self, spec, new_shape):
    kt = keras_tensor.KerasTensor(spec)
    kt.set_shape(new_shape)
    if new_shape is None:
      self.assertIsNone(kt.type_spec.shape.rank)
    else:
      self.assertEqual(kt.type_spec.shape.as_list(), new_shape)
    self.assertTrue(kt.type_spec.is_compatible_with(spec))

  def test_set_shape_error(self):
    spec = CustomTypeSpec([3, None], tf.int32)
    kt = keras_tensor.KerasTensor(spec)
    with self.assertRaisesRegex(
        ValueError, "Keras requires TypeSpec to have a `with_shape` method"):
      kt.set_shape([3, 3])

  def test_missing_shape_error(self):
    spec = CustomTypeSpec(None, tf.int32)
    del spec.shape
    with self.assertRaisesRegex(
        ValueError,
        "KerasTensor only supports TypeSpecs that have a shape field; .*"):
      keras_tensor.KerasTensor(spec)

  def test_wrong_shape_type_error(self):
    spec = CustomTypeSpec(None, tf.int32)
    spec.shape = "foo"
    with self.assertRaisesRegex(
        TypeError, "KerasTensor requires that wrapped TypeSpec's shape is a "
        "TensorShape; .*"):
      keras_tensor.KerasTensor(spec)

  def test_missing_dtype_error(self):
    spec = CustomTypeSpec(None, tf.int32)
    del spec.dtype
    kt = keras_tensor.KerasTensor(spec)
    with self.assertRaisesRegex(
        AttributeError,
        "KerasTensor wraps TypeSpec .* which does not have a dtype."):
      kt.dtype  # pylint: disable=pointless-statement

  def test_wrong_dtype_type_error(self):
    spec = CustomTypeSpec(None, tf.int32)
    spec.dtype = "foo"
    kt = keras_tensor.KerasTensor(spec)
    with self.assertRaisesRegex(
        TypeError,
        "KerasTensor requires that wrapped TypeSpec's dtype is a DType; .*"):
      kt.dtype  # pylint: disable=pointless-statement


if __name__ == "__main__":
  tf.compat.v1.enable_eager_execution()
  tf.compat.v1.enable_v2_tensorshape()
  tf.test.main()
