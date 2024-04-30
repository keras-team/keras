import mlx.core as mx
import numpy as np
import pytest

from keras.src import backend
from keras.src import testing
from keras.src.backend.mlx import core


@pytest.mark.skipif(
    backend.backend() != "mlx",
    reason="Testing core MLX backend functionality",
)
class TestVariableMethods(testing.TestCase):
    def test_initialize(self):
        v = core.Variable(5, "int32")
        self.assertEqual(v._value, mx.array(5, dtype=mx.int32))

    def test_direct_assign(self):
        v = core.Variable(5, "int32")
        v._direct_assign(10)
        self.assertEqual(v._value, mx.array(10, dtype=mx.int32))

    def test_convert_to_tensor(self):
        v = core.Variable(5, "int32")
        tensor = v._convert_to_tensor(10)
        self.assertIsInstance(tensor, mx.array)
        self.assertEqual(tensor, mx.array(10, dtype=mx.int32))

    def test_array_conversion(self):
        v = core.Variable(mx.array([1, 2, 3]), "int32")
        arr = v.__array__()
        arr_mx = mx.array(arr)  # Convert arr to a mlx array
        self.assertTrue(mx.array_equal(arr_mx, mx.array([1, 2, 3])))

    def test_array_conversion_multidimensional(self):
        v = core.Variable(mx.array([[1, 2, 3], [4, 5, 6]]), "int32")
        arr = v.__array__()
        arr_mx = mx.array(arr)
        self.assertTrue(
            mx.array_equal(arr_mx, mx.array([[1, 2, 3], [4, 5, 6]]))
        )

    def test_null_initialization(self):
        with self.assertRaises(TypeError):
            core.Variable(None, "float32")

    def test_to_mlx_dtype(self):
        self.assertEqual(core.to_mlx_dtype("float32"), mx.float32)
        with self.assertRaises(ValueError):
            core.to_mlx_dtype("unsupported_dtype")

    def test_convert_to_tensor_exceptions(self):
        with self.assertRaises(ValueError):
            core.convert_to_tensor(10, sparse=True)

    def test_convert_to_numpy(self):
        arr = mx.array([1, 2, 3])
        np.testing.assert_array_equal(core.convert_to_numpy(arr), arr)

    def test_is_tensor(self):
        self.assertTrue(core.is_tensor(mx.array([1, 2, 3])))
        self.assertFalse(core.is_tensor([1, 2, 3]))

    def test_shape(self):
        arr = mx.array([1, 2, 3])
        self.assertEqual(core.shape(arr), (3,))

    def test_cast(self):
        tensor = core.cast([1, 2, 3], "float32")
        self.assertEqual(tensor.dtype, mx.float32)

    def test_tensor_to_numpy_and_back(self):
        tensor = core.cast(mx.array([1.5, 2.5, 3.5]), "float32")
        numpy_arr = core.convert_to_numpy(tensor)
        tensor_back = core.convert_to_tensor(numpy_arr, "float32")
        np.testing.assert_array_equal(tensor, tensor_back)

    def test_with_scalar_values(self):
        scalar = 5
        tensor = core.cast(scalar, "int32")
        self.assertEqual(tensor, mx.array(5, dtype=mx.int32))

    def test_with_zero_size_array(self):
        empty_arr = np.array([])
        tensor = core.convert_to_tensor(empty_arr, "float32")
        self.assertEqual(tensor.size, 0)

    def test_cond(self):
        result = core.cond(True, lambda: "true", lambda: "false")
        self.assertEqual(result, "true")

    def test_vectorized_map(self):
        result = core.vectorized_map(lambda x: x * 2, mx.array([1, 2, 3]))
        self.assertTrue(mx.array_equal(result, mx.array([2, 4, 6])))

    def test_scatter(self):
        zeros = mx.zeros((4,))
        result = core.scatter(mx.array([1]), mx.array([10]), zeros.shape)
        self.assertTrue(mx.array_equal(result, mx.array([0, 10, 0, 0])))

    def test_cond_complex_condition(self):
        result = core.cond(False, lambda: "true", lambda: "false")
        self.assertEqual(result, "false")

    def test_vectorized_map_complex_function(self):
        result = core.vectorized_map(lambda x: x * x + 2, mx.array([1, 2, 3]))
        self.assertTrue(mx.array_equal(result, mx.array([3, 6, 11])))

    def test_while_loop(self):
        result = core.while_loop(lambda x: x < 5, lambda x: x + 1, [0])
        self.assertEqual(result, (5,))

    def test_fori_loop(self):
        result = core.fori_loop(0, 5, lambda i, x: x + i, 0)
        self.assertEqual(result, 10)
