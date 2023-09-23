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

    @patch("keras.ops.Add.symbolic_call")
    def test_radd_method(self, mock_symbolic_call):
        """Test __radd__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        y = Mock()
        result = y + x
        mock_symbolic_call.assert_called_once_with(y, x)
        self.assertEqual(result, mock_tensor)

    @patch("keras.ops.Subtract.symbolic_call")
    def test_rsub_method(self, mock_symbolic_call):
        """Test __rsub__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        y = Mock()
        result = y - x
        mock_symbolic_call.assert_called_once_with(y, x)
        self.assertEqual(result, mock_tensor)

    @patch("keras.ops.Multiply.symbolic_call")
    def test_rmul_method(self, mock_symbolic_call):
        """Test __rmul__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        y = Mock()
        result = y * x
        mock_symbolic_call.assert_called_once_with(y, x)
        self.assertEqual(result, mock_tensor)

    @patch("keras.ops.Matmul.symbolic_call")
    def test_rmatmul_method(self, mock_symbolic_call):
        """Test __rmatmul__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        y = Mock()
        result = y @ x
        mock_symbolic_call.assert_called_once_with(y, x)
        self.assertEqual(result, mock_tensor)

    @patch("keras.ops.Power.symbolic_call")
    def test_rpow_method(self, mock_symbolic_call):
        """Test __rpow__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        y = Mock()
        result = y**x
        mock_symbolic_call.assert_called_once_with(y, x)
        self.assertEqual(result, mock_tensor)

    @patch("keras.ops.FloorDivide.symbolic_call")
    def test_floordiv_method(self, mock_symbolic_call):
        """Test __floordiv__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        y = Mock()
        result = x // y
        mock_symbolic_call.assert_called_once_with(x, y)
        self.assertEqual(result, mock_tensor)

    @patch("keras.ops.FloorDivide.symbolic_call")
    def test_rfloordiv_method(self, mock_symbolic_call):
        """Test __rfloordiv__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        y = Mock()
        result = y // x
        mock_symbolic_call.assert_called_once_with(y, x)
        self.assertEqual(result, mock_tensor)

    @patch("keras.ops.Mod.symbolic_call")
    def test_rmod_method(self, mock_symbolic_call):
        """Test __rmod__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        y = Mock()
        result = y % x
        mock_symbolic_call.assert_called_once_with(y, x)
        self.assertEqual(result, mock_tensor)

    @patch("keras.ops.LessEqual.symbolic_call")
    def test_le_method(self, mock_symbolic_call):
        """Test __le__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        y = Mock()
        result = x <= y
        mock_symbolic_call.assert_called_once_with(x, y)
        self.assertEqual(result, mock_tensor)

    @patch("keras.ops.Greater.symbolic_call")
    def test_gt_method(self, mock_symbolic_call):
        """Test __gt__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        y = Mock()
        result = x > y
        mock_symbolic_call.assert_called_once_with(x, y)
        self.assertEqual(result, mock_tensor)

    @patch("keras.ops.GreaterEqual.symbolic_call")
    def test_ge_method(self, mock_symbolic_call):
        """Test __ge__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        y = Mock()
        result = x >= y
        mock_symbolic_call.assert_called_once_with(x, y)
        self.assertEqual(result, mock_tensor)

    @patch("keras.ops.NotEqual.symbolic_call")
    def test_ne_method(self, mock_symbolic_call):
        """Test __ne__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        y = Mock()
        result = x != y
        mock_symbolic_call.assert_called_once_with(x, y)
        self.assertEqual(result, mock_tensor)

    @patch("keras.ops.LogicalAnd.symbolic_call")
    def test_rand_method(self, mock_symbolic_call):
        """Test __rand__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="bool")
        y = Mock()
        result = y & x
        mock_symbolic_call.assert_called_once_with(y, x)
        self.assertEqual(result, mock_tensor)

    @patch("keras.ops.LogicalOr.symbolic_call")
    def test_ror_method(self, mock_symbolic_call):
        """Test __ror__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="bool")
        y = Mock()
        result = y | x
        mock_symbolic_call.assert_called_once_with(y, x)
        self.assertEqual(result, mock_tensor)

    @patch("keras.ops.LogicalNot.symbolic_call")
    def test_invert_method(self, mock_symbolic_call):
        """Test __invert__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="bool")
        result = ~x
        mock_symbolic_call.assert_called_once_with(x)
        self.assertEqual(result, mock_tensor)

    @patch("keras.ops.LogicalXor.symbolic_call")
    def test_xor_method(self, mock_symbolic_call):
        """Test __xor__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="bool")
        y = Mock()
        result = x ^ y
        mock_symbolic_call.assert_called_once_with(x, y)
        self.assertEqual(result, mock_tensor)

    @patch("keras.ops.LogicalXor.symbolic_call")
    def test_rxor_method(self, mock_symbolic_call):
        """Test __rxor__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="bool")
        y = Mock()
        result = y ^ x
        mock_symbolic_call.assert_called_once_with(y, x)
        self.assertEqual(result, mock_tensor)

    @patch("keras.ops.TrueDivide.symbolic_call")
    def test_truediv_method(self, mock_symbolic_call):
        """Test __truediv__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        y = Mock()
        result = x / y
        mock_symbolic_call.assert_called_once_with(x, y)
        self.assertEqual(result, mock_tensor)

    @patch("keras.ops.TrueDivide.symbolic_call")
    def test_rtruediv_method(self, mock_symbolic_call):
        """Test __rtruediv__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        y = Mock()
        result = y / x
        mock_symbolic_call.assert_called_once_with(y, x)
        self.assertEqual(result, mock_tensor)

    @patch("keras.ops.Divide.symbolic_call")
    def test_div_method(self, mock_symbolic_call):
        """Test __div__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        y = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        # to ensure compatibility across Python versions
        result = x.__div__(y)
        mock_symbolic_call.assert_called_once_with(x, y)
        self.assertEqual(result, mock_tensor)

    @patch("keras.ops.Divide.symbolic_call")
    def test_rdiv_method(self, mock_symbolic_call):
        """Test __rdiv__ method"""
        mock_tensor = Mock()
        mock_symbolic_call.return_value = mock_tensor
        x = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        y = keras_tensor.KerasTensor(shape=(3, 4), dtype="float32")
        # to ensure compatibility across Python versions
        result = x.__rdiv__(y)
        mock_symbolic_call.assert_called_once_with(y, x)
        self.assertEqual(result, mock_tensor)
