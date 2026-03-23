"""Tests for keras.src.utils.tf_utils."""

import numpy as np

from keras.src import testing
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.tf_utils import ensure_tensor
from keras.src.utils.tf_utils import expand_dims
from keras.src.utils.tf_utils import get_tensor_spec
from keras.src.utils.tf_utils import is_ragged_tensor
from keras.src.utils.tf_utils import tf_encode_categorical_inputs


class GetTensorSpecTest(testing.TestCase):
    def test_from_tensor(self):
        t = tf.constant([1.0, 2.0])
        spec = get_tensor_spec(t)
        self.assertEqual(spec.shape, (2,))
        self.assertEqual(spec.dtype, tf.float32)

    def test_from_tensor_spec(self):
        ts = tf.TensorSpec(shape=(None, 5), dtype=tf.float32)
        spec = get_tensor_spec(ts)
        self.assertIs(spec, ts)

    def test_with_name(self):
        t = tf.constant([1.0])
        spec = get_tensor_spec(t, name="my_tensor")
        self.assertEqual(spec.name, "my_tensor")

    def test_dynamic_batch(self):
        t = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        spec = get_tensor_spec(t, dynamic_batch=True)
        self.assertIsNone(spec.shape[0])
        self.assertEqual(spec.shape[1], 2)

    def test_dynamic_batch_scalar(self):
        t = tf.constant(1.0)
        spec = get_tensor_spec(t, dynamic_batch=True)
        # Scalar has rank 0, so no batch dim to make dynamic
        self.assertEqual(spec.shape.rank, 0)

    def test_non_tensor_returns_none(self):
        self.assertIsNone(get_tensor_spec("not_a_tensor"))

    def test_numpy_like_object(self):
        arr = np.array([1.0, 2.0, 3.0])
        spec = get_tensor_spec(arr)
        self.assertEqual(spec.shape, (3,))


class EnsureTensorTest(testing.TestCase):
    def test_list_to_tensor(self):
        result = ensure_tensor([1, 2, 3])
        self.assertIsInstance(result, tf.Tensor)
        self.assertEqual(result.numpy().tolist(), [1, 2, 3])

    def test_numpy_to_tensor(self):
        arr = np.array([1.0, 2.0])
        result = ensure_tensor(arr)
        self.assertIsInstance(result, tf.Tensor)

    def test_tensor_passthrough(self):
        t = tf.constant([1.0])
        result = ensure_tensor(t)
        self.assertIsInstance(result, tf.Tensor)

    def test_dtype_cast(self):
        t = tf.constant([1, 2, 3], dtype=tf.int32)
        result = ensure_tensor(t, dtype=tf.float32)
        self.assertEqual(result.dtype, tf.float32)

    def test_sparse_passthrough(self):
        st = tf.SparseTensor(indices=[[0, 0]], values=[1.0], dense_shape=[2, 2])
        result = ensure_tensor(st)
        self.assertIsInstance(result, tf.SparseTensor)


class IsRaggedTensorTest(testing.TestCase):
    def test_ragged_tensor(self):
        rt = tf.ragged.constant([[1, 2], [3]])
        self.assertTrue(is_ragged_tensor(rt))

    def test_dense_tensor(self):
        t = tf.constant([1, 2, 3])
        self.assertFalse(is_ragged_tensor(t))

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        self.assertFalse(is_ragged_tensor(arr))

    def test_plain_list(self):
        self.assertFalse(is_ragged_tensor([1, 2, 3]))


class ExpandDimsTest(testing.TestCase):
    def test_dense_expand(self):
        t = tf.constant([1, 2, 3])
        result = expand_dims(t, axis=0)
        self.assertEqual(result.shape, (1, 3))

    def test_dense_expand_last_axis(self):
        t = tf.constant([1, 2, 3])
        result = expand_dims(t, axis=-1)
        self.assertEqual(result.shape, (3, 1))

    def test_sparse_expand(self):
        st = tf.SparseTensor(
            indices=[[0], [2]], values=[1.0, 3.0], dense_shape=[3]
        )
        result = expand_dims(st, axis=0)
        self.assertIsInstance(result, tf.SparseTensor)


class TfEncodeCategoricalInputsTest(testing.TestCase):
    def test_int_mode(self):
        inputs = tf.constant([0, 1, 2])
        result = tf_encode_categorical_inputs(
            inputs, output_mode="int", depth=3
        )
        np.testing.assert_array_equal(result.numpy(), [0.0, 1.0, 2.0])

    def test_multi_hot(self):
        inputs = tf.constant([1, 3])
        result = tf_encode_categorical_inputs(
            inputs, output_mode="multi_hot", depth=5
        )
        expected = [0, 1, 0, 1, 0]
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_one_hot(self):
        inputs = tf.constant([2])
        result = tf_encode_categorical_inputs(
            inputs, output_mode="one_hot", depth=4
        )
        # Input [2] shape (1,) → expand_dims → (1,1) → bincount → (1, 4)
        # But bincount for 1D returns (4,)
        expected = [0, 0, 1, 0]
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_scalar_input_upranked(self):
        inputs = tf.constant(2)
        result = tf_encode_categorical_inputs(
            inputs, output_mode="multi_hot", depth=4
        )
        np.testing.assert_array_equal(result.numpy(), [0, 0, 1, 0])

    def test_rank_too_high_raises(self):
        inputs = tf.constant([[[0, 1]]])
        with self.assertRaisesRegex(
            ValueError, "maximum supported output rank"
        ):
            tf_encode_categorical_inputs(
                inputs, output_mode="multi_hot", depth=3
            )

    def test_tf_idf_without_weights_raises(self):
        inputs = tf.constant([0, 1])
        with self.assertRaisesRegex(ValueError, "idf_weights must be provided"):
            tf_encode_categorical_inputs(
                inputs, output_mode="tf_idf", depth=3, idf_weights=None
            )

    def test_tf_idf_with_weights(self):
        inputs = tf.constant([0, 1])
        idf = tf.constant([0.5, 1.0, 0.0])
        result = tf_encode_categorical_inputs(
            inputs, output_mode="tf_idf", depth=3, idf_weights=idf
        )
        # bincounts = [1, 1, 0], idf = [0.5, 1.0, 0.0] → [0.5, 1.0, 0.0]
        np.testing.assert_array_almost_equal(result.numpy(), [0.5, 1.0, 0.0])

    def test_batched_multi_hot(self):
        inputs = tf.constant([[0, 1], [2, 3]])
        result = tf_encode_categorical_inputs(
            inputs, output_mode="multi_hot", depth=5
        )
        expected = [[1, 1, 0, 0, 0], [0, 0, 1, 1, 0]]
        np.testing.assert_array_equal(result.numpy(), expected)


if __name__ == "__main__":
    testing.run_tests()
