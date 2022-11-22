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


import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras import layers
from keras.engine import keras_tensor
from keras.engine import training
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils


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


@test_utils.run_v2_only
class KerasTensorTest(test_combinations.TestCase):
    def test_repr_and_string(self):
        kt = keras_tensor.KerasTensor(
            type_spec=tf.TensorSpec(shape=(1, 2, 3), dtype=tf.float32)
        )
        expected_str = (
            "KerasTensor(type_spec=TensorSpec(shape=(1, 2, 3), "
            "dtype=tf.float32, name=None))"
        )
        expected_repr = "<KerasTensor: shape=(1, 2, 3) dtype=float32>"
        self.assertEqual(expected_str, str(kt))
        self.assertEqual(expected_repr, repr(kt))

        kt = keras_tensor.KerasTensor(
            type_spec=tf.TensorSpec(shape=(2,), dtype=tf.int32),
            inferred_value=[2, 3],
        )
        expected_str = (
            "KerasTensor(type_spec=TensorSpec(shape=(2,), "
            "dtype=tf.int32, name=None), inferred_value=[2, 3])"
        )
        expected_repr = (
            "<KerasTensor: shape=(2,) dtype=int32 inferred_value=[2, 3]>"
        )
        self.assertEqual(expected_str, str(kt))
        self.assertEqual(expected_repr, repr(kt))

        kt = keras_tensor.KerasTensor(
            type_spec=tf.SparseTensorSpec(shape=(1, 2, 3), dtype=tf.float32)
        )
        expected_str = (
            "KerasTensor(type_spec=SparseTensorSpec("
            "TensorShape([1, 2, 3]), tf.float32))"
        )
        expected_repr = (
            "<KerasTensor: type_spec=SparseTensorSpec("
            "TensorShape([1, 2, 3]), tf.float32)>"
        )
        self.assertEqual(expected_str, str(kt))
        self.assertEqual(expected_repr, repr(kt))

        inp = layers.Input(shape=(3, 5))
        kt = layers.Dense(10)(inp)
        expected_str = (
            "KerasTensor(type_spec=TensorSpec(shape=(None, 3, 10), "
            "dtype=tf.float32, name=None), name='dense/BiasAdd:0', "
            "description=\"created by layer 'dense'\")"
        )
        expected_repr = (
            "<KerasTensor: shape=(None, 3, 10) dtype=float32 (created "
            "by layer 'dense')>"
        )
        self.assertEqual(expected_str, str(kt))
        self.assertEqual(expected_repr, repr(kt))

        kt = tf.reshape(kt, shape=(3, 5, 2))
        expected_str = (
            "KerasTensor(type_spec=TensorSpec(shape=(3, 5, 2), "
            "dtype=tf.float32, name=None), name='tf.reshape/Reshape:0', "
            "description=\"created by layer 'tf.reshape'\")"
        )
        expected_repr = (
            "<KerasTensor: shape=(3, 5, 2) dtype=float32 (created "
            "by layer 'tf.reshape')>"
        )
        self.assertEqual(expected_str, str(kt))
        self.assertEqual(expected_repr, repr(kt))

        kts = tf.unstack(kt)
        for i in range(3):
            expected_str = (
                "KerasTensor(type_spec=TensorSpec(shape=(5, 2), "
                "dtype=tf.float32, name=None), name='tf.unstack/unstack:%s', "
                "description=\"created by layer 'tf.unstack'\")" % (i,)
            )
            expected_repr = (
                "<KerasTensor: shape=(5, 2) dtype=float32 "
                "(created by layer 'tf.unstack')>"
            )
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

        x = tf.SparseTensor(
            [[0, 0], [0, 1], [1, 1], [1, 2]], [1, 2, 3, 4], [2, 3]
        )
        expected_property = getattr(x, property_name)
        self.assertAllEqual(model(x), expected_property)

        # Test that it works with serialization and deserialization as well
        model_config = model.get_config()
        model2 = training.Model.from_config(model_config)
        self.assertAllEqual(model2(x), expected_property)

    @parameterized.parameters(
        [
            (tf.TensorSpec([2, 3], tf.int32), [2, 3]),
            (tf.RaggedTensorSpec([2, None]), [2, None]),
            (tf.SparseTensorSpec([8]), [8]),
            (CustomTypeSpec([3, 8], tf.int32), [3, 8]),
        ]
    )
    def test_shape(self, spec, expected_shape):
        kt = keras_tensor.KerasTensor(spec)
        self.assertEqual(kt.shape.as_list(), expected_shape)

    @parameterized.parameters(
        [
            (tf.TensorSpec([8, 3], tf.int32), [8, 3], [8, 3]),
            (tf.TensorSpec([None, 3], tf.int32), [8, 3], [8, 3]),
            (tf.TensorSpec([8, 3], tf.int32), [None, 3], [8, 3]),
            (tf.TensorSpec(None, tf.int32), [8, 3], [8, 3]),
            (tf.TensorSpec(None, tf.int32), [8, None], [8, None]),
            (tf.TensorSpec(None, tf.int32), None, None),
            (tf.RaggedTensorSpec([2, None, None]), [2, None, 5], [2, None, 5]),
            (tf.SparseTensorSpec([8]), [8], [8]),
            (CustomTypeSpec2([3, None], tf.int32), [3, 8], [3, 8]),
        ]
    )
    def test_set_shape(self, spec, new_shape, expected_shape):
        kt = keras_tensor.KerasTensor(spec)
        kt.set_shape(new_shape)
        if expected_shape is None:
            self.assertIsNone(kt.type_spec.shape.rank)
        else:
            self.assertEqual(kt.type_spec.shape.as_list(), expected_shape)
        self.assertTrue(kt.type_spec.is_compatible_with(spec))

    @parameterized.parameters(
        [
            (layers.Input(shape=[3, 4], batch_size=7), tf.reshape),
            (layers.Input(shape=[3, 4], ragged=True, batch_size=7), tf.reshape),
            (
                layers.Input(shape=[3, 4], sparse=True, batch_size=7),
                tf.sparse.reshape,
            ),
        ]
    )
    def test_reshape(self, inp, reshape_op):
        out = reshape_op(inp, shape=[7, 4, 3])
        self.assertEqual(out.type_spec.shape.as_list(), [7, 4, 3])

    def test_set_shape_error(self):
        spec = CustomTypeSpec([3, None], tf.int32)
        kt = keras_tensor.KerasTensor(spec)
        with self.assertRaisesRegex(
            ValueError, "Keras requires TypeSpec to have a `with_shape` method"
        ):
            kt.set_shape([3, 3])

    def test_set_shape_equals_expected_shape(self):
        # Tests b/203201161: DenseSpec has both a _shape and a _shape_tuple
        # field, and we need to be sure both get updated.
        kt = keras_tensor.KerasTensor(tf.TensorSpec([8, None], tf.int32))
        kt.set_shape([8, 3])
        self.assertEqual(kt.type_spec, tf.TensorSpec([8, 3], tf.int32))

    def test_type_spec_with_shape_equals_expected_shape(self):
        # Tests b/203201161: DenseSpec has both a _shape and a _shape_tuple
        # field, and we need to be sure both get updated.
        spec1 = tf.TensorSpec([8, None], tf.int32)
        spec2 = keras_tensor.type_spec_with_shape(spec1, [8, 3])
        expected = tf.TensorSpec([8, 3], tf.int32)
        self.assertEqual(spec2, expected)

    def test_missing_shape_error(self):
        spec = CustomTypeSpec(None, tf.int32)
        del spec.shape
        with self.assertRaisesRegex(
            ValueError,
            "KerasTensor only supports TypeSpecs that have a shape field; .*",
        ):
            keras_tensor.KerasTensor(spec)

    def test_wrong_shape_type_error(self):
        spec = CustomTypeSpec(None, tf.int32)
        spec.shape = "foo"
        with self.assertRaisesRegex(
            TypeError,
            "KerasTensor requires that wrapped TypeSpec's shape is a "
            "TensorShape; .*",
        ):
            keras_tensor.KerasTensor(spec)

    def test_missing_dtype_error(self):
        spec = CustomTypeSpec(None, tf.int32)
        del spec.dtype
        kt = keras_tensor.KerasTensor(spec)
        with self.assertRaisesRegex(
            AttributeError,
            "KerasTensor wraps TypeSpec .* which does not have a dtype.",
        ):
            kt.dtype

    def test_wrong_dtype_type_error(self):
        spec = CustomTypeSpec(None, tf.int32)
        spec.dtype = "foo"
        kt = keras_tensor.KerasTensor(spec)
        with self.assertRaisesRegex(
            TypeError,
            "KerasTensor requires that wrapped TypeSpec's dtype is a DType; .*",
        ):
            kt.dtype

    def test_from_tensor_mask_tensor_is_none(self):
        tensor = tf.constant([1.0])
        kt = keras_tensor.keras_tensor_from_tensor(tensor)
        self.assertIsNone(getattr(kt, "_keras_mask", None))

    def test_from_tensor_mask_tensor_is_not_none(self):
        tensor = tf.constant([1.0])
        tensor._keras_mask = tf.constant([1.0])
        kt = keras_tensor.keras_tensor_from_tensor(tensor)
        self.assertIsInstance(kt._keras_mask, keras_tensor.KerasTensor)


if __name__ == "__main__":
    tf.test.main()
