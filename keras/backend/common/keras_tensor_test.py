from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from keras import backend
from keras import ops
from keras import testing
from keras.backend.common import keras_tensor


class KerasTensorTest(testing.TestCase):
    def test_attributes(self):
        x = keras_tensor.KerasTensor(shape=(3,), dtype="float32", sparse=True)
        self.assertEqual(x.dtype, "float32")
        self.assertEqual(x.shape, (3,))
        self.assertEqual(x.sparse, True)

    def test_numpy_methods(self):
        x = keras_tensor.KerasTensor(shape=(3, 2), dtype="float32")

        # reshape
        x = x.reshape((6,))
        self.assertEqual(x.shape, (6,))

        # expand_dims, squeeze
        x = ops.expand_dims(x, -1)
        self.assertEqual(x.shape, (6, 1))
        x = x.squeeze()
        self.assertEqual(x.shape, (6,))
        x = ops.expand_dims(x, axis=0)
        self.assertEqual(x.shape, (1, 6))
        x = x.squeeze(axis=0)
        self.assertEqual(x.shape, (6,))

    def test_invalid_usage(self):
        x = keras_tensor.KerasTensor(shape=(3,), dtype="float32")
        with self.assertRaisesRegex(
            ValueError, "doesn't have any actual numerical value"
        ):
            np.array(x)

        if backend.backend() == "jax":
            from jax import numpy as jnp

            with self.assertRaisesRegex(
                ValueError, "cannot be used as input to a JAX function"
            ):
                jnp.array(x)

        with self.assertRaisesRegex(
            ValueError, "cannot be used as input to a TensorFlow function"
        ):
            tf.convert_to_tensor(x)

    def test_bool(self):
        tensor = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        with self.assertRaisesRegex(TypeError, "cannot be used as a boolean."):
            bool(tensor)

    def test_representation(self):
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        self.assertIn("<KerasTensor shape=(3, 4)", repr(x))

    def test_iterating(self):
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        with self.assertRaises(NotImplementedError):
            iter(x)

    def test_any_symbolic_tensors(self):
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        y = np.array([1, 2, 3])
        self.assertTrue(keras_tensor.any_symbolic_tensors(args=[x, y]))
        self.assertFalse(keras_tensor.any_symbolic_tensors(args=[y]))

    def test_is_keras_tensor(self):
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        self.assertTrue(keras_tensor.is_keras_tensor(x))
        y = np.array([1, 2, 3])
        self.assertFalse(keras_tensor.is_keras_tensor(y))

    @patch("keras.ops.Absolute.symbolic_call")
    def test_abs_method(self, mock_symbolic_call):
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        abs_x = abs(x)  # this will internally call x.__abs__()
        mock_symbolic_call.assert_called_once_with(x)
        self.assertEqual(abs_x, mock_tensor)

    @patch("keras.ops.Negative.symbolic_call")
    def test_neg_method(self, mock_method):
        self._test_unary_op_method(mock_method, lambda x: -x)

    @patch("keras.ops.Subtract.symbolic_call")
    def test_sub_method(self, mock_method):
        y = Mock()
        self._test_binary_op_method(mock_method, y, lambda x, y: x - y)

    @patch("keras.ops.Multiply.symbolic_call")
    def test_mul_method(self, mock_method):
        y = Mock()
        self._test_binary_op_method(mock_method, y, lambda x, y: x * y)

    @patch("keras.ops.Matmul.symbolic_call")
    def test_matmul_method(self, mock_method):
        y = Mock()
        self._test_binary_op_method(mock_method, y, lambda x, y: x @ y)

    @patch("keras.ops.Power.symbolic_call")
    def test_pow_method(self, mock_method):
        y = Mock()
        self._test_binary_op_method(mock_method, y, lambda x, y: x**y)

    @patch("keras.ops.Mod.symbolic_call")
    def test_mod_method(self, mock_method):
        y = Mock()
        self._test_binary_op_method(mock_method, y, lambda x, y: x % y)

    @patch("keras.ops.Less.symbolic_call")
    def test_lt_method(self, mock_method):
        y = Mock()
        self._test_binary_op_method(mock_method, y, lambda x, y: x < y)

    @patch("keras.ops.LogicalAnd.symbolic_call")
    def test_and_method(self, mock_method):
        y = Mock()
        self._test_binary_op_method(mock_method, y, lambda x, y: x & y)

    @patch("keras.ops.LogicalOr.symbolic_call")
    def test_or_method(self, mock_method):
        y = Mock()
        self._test_binary_op_method(mock_method, y, lambda x, y: x | y)

    @patch("keras.ops.GetItem.symbolic_call")
    def test_getitem_method(self, mock_method):
        y = Mock()
        self._test_binary_op_method(mock_method, y, lambda x, y: x[y])

    def _test_unary_op_method(self, mock_method, operator):
        mock_tensor = Mock()
        mock_method.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        result = operator(x)
        mock_method.assert_called_once_with(x)
        self.assertEqual(result, mock_tensor)

    def _test_binary_op_method(self, mock_method, other, operator):
        mock_tensor = Mock()
        mock_method.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        result = operator(x, other)
        mock_method.assert_called_once_with(x, other)
        self.assertEqual(result, mock_tensor)
