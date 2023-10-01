from unittest.mock import patch

import numpy as np
import pytest

from keras import backend
from keras import initializers
from keras.backend.common.variables import AutocastScope
from keras.backend.common.variables import KerasVariable
from keras.backend.common.variables import shape_equal
from keras.backend.common.variables import standardize_dtype
from keras.backend.common.variables import standardize_shape
from keras.testing import test_case


class VariableInitializationTest(test_case.TestCase):
    """Tests for KerasVariable.__init__()"""

    def test_deferred_initialization(self):
        """Tests deferred initialization of variables."""
        with backend.StatelessScope():
            v = backend.Variable(
                initializer=initializers.RandomNormal(), shape=(2, 2)
            )
            self.assertEqual(v._value, None)
            # Variables can nevertheless be accessed
            _ = v + 1
        self.assertEqual(v._value.shape, (2, 2))

        with self.assertRaisesRegex(ValueError, "while in a stateless scope"):
            with backend.StatelessScope():
                v = backend.Variable(initializer=0)

    def test_variable_initialization_with_non_callable(self):
        """Test variable init with non-callable initializer."""
        v = backend.Variable(initializer=np.ones((2, 2)))
        self.assertAllClose(v.value, np.ones((2, 2)))

    def test_variable_initialization_with_non_trainable(self):
        """Test variable initialization with non-trainable flag."""
        v = backend.Variable(initializer=np.ones((2, 2)), trainable=False)
        self.assertFalse(v.trainable)

    def test_variable_initialization_without_shape(self):
        """Test variable init without a shape."""
        with self.assertRaisesRegex(
            ValueError,
            "When creating a Variable from an initializer, the `shape` ",
        ):
            backend.Variable(initializer=initializers.RandomNormal())

    def test_deferred_initialize_already_initialized(self):
        """Test deferred init on an already initialized variable."""
        v = backend.Variable(initializer=np.ones((2, 2)))
        with self.assertRaisesRegex(
            ValueError, f"Variable {v.path} is already initialized."
        ):
            v._deferred_initialize()

    def test_variable_initialize(self):
        """Test initializing a variable."""
        v = backend.Variable(initializer=np.array([1, 2, 3]))
        init_value = np.array([4, 5, 6])
        v._initialize(value=init_value)
        self.assertAllClose(v.value, init_value)

    def test_variable_without_shape_from_callable_initializer(self):
        """Test that KerasVariable raises error
        if shape is not provided for callable initializer."""
        with self.assertRaisesRegex(
            ValueError, "When creating a Variable from an initializer"
        ):
            KerasVariable(initializer=lambda: np.ones((2, 2)))


class VariablePropertiesTest(test_case.TestCase):
    """Tests for KerasVariable._deferred_initialize
    KerasVariable._maybe_autocast"""

    def test_deferred_assignment(self):
        """Tests deferred assignment to variables."""
        with backend.StatelessScope() as scope:
            v = backend.Variable(
                initializer=initializers.RandomNormal(), shape=(2, 2)
            )
            self.assertEqual(v._value, None)
            v.assign(np.zeros((2, 2)))
            v.assign_add(2 * np.ones((2, 2)))
            v.assign_sub(np.ones((2, 2)))
        out = scope.get_current_value(v)
        self.assertAllClose(out, np.ones((2, 2)))

    def test_trainable_setter(self):
        v = backend.Variable(
            initializer=initializers.RandomNormal(),
            shape=(2, 2),
        )
        self.assertTrue(v.trainable)
        v.trainable = False
        self.assertFalse(v.trainable)

        if backend.backend() == "torch":
            v.trainable = True
            self.assertTrue(v._value.requires_grad)
            v.trainable = False
            self.assertFalse(v._value.requires_grad)

    def test_autocasting(self):
        """Tests autocasting of float variables."""
        v = backend.Variable(
            initializer=initializers.RandomNormal(),
            shape=(2, 2),
            dtype="float32",
        )
        self.assertEqual(v.dtype, "float32")
        self.assertEqual(backend.standardize_dtype(v.value.dtype), "float32")
        print("open scope")
        with AutocastScope("float16"):
            self.assertEqual(
                backend.standardize_dtype(v.value.dtype), "float16"
            )
        self.assertEqual(backend.standardize_dtype(v.value.dtype), "float32")

        # Test non-float variables are not affected
        v = backend.Variable(
            initializer=initializers.Ones(),
            shape=(2, 2),
            dtype="int32",
            trainable=False,
        )
        self.assertEqual(v.dtype, "int32")
        self.assertEqual(backend.standardize_dtype(v.value.dtype), "int32")

        with AutocastScope("float16"):
            self.assertEqual(backend.standardize_dtype(v.value.dtype), "int32")

    def test_standardize_dtype_with_torch_dtype(self):
        """Tests dtype standardization with PyTorch dtypes."""
        import torch

        x = torch.randn(4, 4)
        backend.standardize_dtype(x.dtype)

    def test_name_validation(self):
        """Tests validation of variable names."""
        with self.assertRaisesRegex(
            ValueError, "Argument `name` must be a string"
        ):
            KerasVariable(initializer=initializers.RandomNormal(), name=12345)

        with self.assertRaisesRegex(ValueError, "cannot contain character `/`"):
            KerasVariable(
                initializer=initializers.RandomNormal(), name="invalid/name"
            )

    def test_standardize_shape_with_none(self):
        """Tests standardizing shape with None."""
        with self.assertRaisesRegex(
            ValueError, "Undefined shapes are not supported."
        ):
            standardize_shape(None)

    def test_standardize_shape_with_non_iterable(self):
        """Tests shape standardization with non-iterables."""
        with self.assertRaisesRegex(
            ValueError, "Cannot convert '42' to a shape."
        ):
            standardize_shape(42)

    def test_standardize_shape_with_valid_input(self):
        """Tests standardizing shape with valid input."""
        shape = [3, 4, 5]
        standardized_shape = standardize_shape(shape)
        self.assertEqual(standardized_shape, (3, 4, 5))

    def test_standardize_shape_with_negative_entry(self):
        """Tests standardizing shape with negative entries."""
        with self.assertRaisesRegex(
            ValueError,
            "Cannot convert '\\(3, 4, -5\\)' to a shape. Negative dimensions",
        ):
            standardize_shape([3, 4, -5])

    def test_autocast_scope_with_non_float_dtype(self):
        """Tests autocast scope with non-float dtype."""
        with self.assertRaisesRegex(
            ValueError,
            "`AutocastScope` can only be used with a floating-point",
        ):
            _ = AutocastScope("int32")

    def test_variable_path_creation(self):
        """Test path creation for a variable."""
        v = backend.Variable(initializer=np.ones((2, 2)), name="test_var")
        self.assertEqual(v.path, "test_var")


class VariableNumpyValueAndAssignmentTest(test_case.TestCase):
    """tests for KerasVariable.numpy(), KerasVariable.value()
    and KerasVariable.assign()"""

    def test_variable_numpy(self):
        """Test retrieving the value of a variable as a numpy array."""
        v = backend.Variable(initializer=np.array([1, 2, 3]))
        self.assertIsInstance(v.numpy(), np.ndarray)
        self.assertAllClose(v.numpy(), np.array([1, 2, 3]))

    def test_variable_value(self):
        """Test retrieving the value of a variable."""
        v = backend.Variable(initializer=np.array([1, 2, 3]))
        self.assertAllClose(v.value, np.array([1, 2, 3]))

    def test_variable_assign(self):
        """Test assigning a new value to a variable."""
        v = backend.Variable(initializer=np.array([1, 2, 3]))
        v.assign(np.array([4, 5, 6]))
        self.assertAllClose(v.value, np.array([4, 5, 6]))

    def test_variable_assign_add(self):
        """Test the assign_add method on a variable."""
        v = backend.Variable(initializer=np.array([1, 2, 3]))
        v.assign_add(np.array([1, 1, 1]))
        self.assertAllClose(v.value, np.array([2, 3, 4]))

    def test_variable_assign_sub(self):
        """Test the assign_sub method on a variable."""
        v = backend.Variable(initializer=np.array([2, 3, 4]))
        v.assign_sub(np.array([1, 1, 1]))
        self.assertAllClose(v.value, np.array([1, 2, 3]))

    def test_deferred_initialize_within_stateless_scope(self):
        """Test deferred init within a stateless scope."""
        with backend.StatelessScope():
            v = backend.Variable(
                initializer=initializers.RandomNormal(), shape=(2, 2)
            )
            with self.assertRaisesRegex(
                ValueError,
                "You are attempting to initialize a variable "
                "while in a stateless scope. This is disallowed.",
            ):
                v._deferred_initialize()


class VariableDtypeShapeNdimRepr(test_case.TestCase):
    """tests for dtype, shape, ndim, __repr__"""

    def test_variable_dtype(self):
        """Test retrieving the dtype of a variable."""
        v = backend.Variable(initializer=np.array([1, 2, 3]))
        self.assertEqual(v.dtype, "float32")

    def test_variable_shape(self):
        """Test retrieving the shape of a variable."""
        v = backend.Variable(initializer=np.array([[1, 2], [3, 4]]))
        self.assertEqual(v.shape, (2, 2))

    def test_variable_ndim(self):
        """Test retrieving the number of dimensions of a variable."""
        v = backend.Variable(initializer=np.array([[1, 2], [3, 4]]))
        self.assertEqual(v.ndim, 2)

    def test_variable_repr(self):
        """Test the string representation of a variable."""
        v = backend.Variable(initializer=np.array([1, 2, 3]), name="test_var")
        expected_repr = (
            "<KerasVariable shape=(3,), dtype=float32, path=test_var>"
        )
        self.assertEqual(repr(v), expected_repr)

    def test_variable_getitem(self):
        """Test getting an item from a variable."""
        v = backend.Variable(initializer=np.array([1, 2, 3]))
        self.assertEqual(v[0], 1)

    def test_variable_initialize(self):
        """Test initializing a variable."""
        v = backend.Variable(initializer=np.array([1, 2, 3]))
        init_value = np.array([4, 5, 6])
        v._initialize(value=init_value)
        self.assertAllClose(v.value, init_value)

    def test_variable_convert_to_tensor(self):
        """Test converting a variable to a tensor."""
        v = backend.Variable(initializer=np.array([1, 2, 3]))
        self.assertAllClose(v._convert_to_tensor(v.value), np.array([1, 2, 3]))

    def test_variable_convert_to_tensor_with_dtype(self):
        """Test converting a variable to a tensor with a dtype."""
        v = backend.Variable(initializer=np.array([1, 2, 3]))
        self.assertAllClose(
            v._convert_to_tensor(v.value, dtype="float32"), np.array([1, 2, 3])
        )

    def test_variable_array(self):
        """Test converting a variable to an array."""
        v = backend.Variable(initializer=np.array([1, 2, 3]))
        self.assertAllClose(v.__array__(), np.array([1, 2, 3]))


class VariableOperationsTest(test_case.TestCase):
    def test_variable_as_boolean(self):
        """Test converting a variable to boolean."""
        v = backend.Variable(initializer=np.ones((2, 2)))
        with self.assertRaisesRegex(
            TypeError, "A Keras Variable cannot be used as a boolean."
        ):
            bool(v)

    def test_variable_negation(self):
        """Test negating a variable."""
        v = backend.Variable(initializer=np.array([-1, 2]))
        neg_v = -v
        self.assertAllClose(neg_v, np.array([1, -2]))

    def test_variable_div_numpy_array(self):
        """Test variable divided by numpy array."""
        v = backend.Variable(initializer=np.array([2, 4, 8]))
        arr = np.array([2, 8, 16])
        div_result = arr / v
        self.assertAllClose(div_result, np.array([1, 2, 2]))

    def test_variable_rdiv_numpy_array(self):
        """Test numpy array divided by variable."""
        v = backend.Variable(initializer=np.array([2, 4, 8]))
        arr = np.array([16, 32, 64])
        rdiv_result = arr / v
        self.assertAllClose(rdiv_result, np.array([8, 8, 8]))

    def test_variable_rsub_numpy_array(self):
        """Test numpy array minus variable."""
        v = backend.Variable(initializer=np.array([1, 2, 3]))
        arr = np.array([2, 2, 2])
        rsub_result = arr - v
        self.assertAllClose(rsub_result, np.array([1, 0, -1]))

    def test_variable_addition(self):
        """Test addition operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        result = v1 + v2
        self.assertAllClose(result, np.array([5, 7, 9]))

    def test_variable_subtraction(self):
        """Test subtraction operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        result = v1 - v2
        self.assertAllClose(result, np.array([-3, -3, -3]))

    def test_variable_multiplication(self):
        """Test multiplication operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        result = v1 * v2
        self.assertAllClose(result, np.array([4, 10, 18]))

    def test_variable_division(self):
        """Test division operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        result = v1 / v2
        self.assertAllClose(result, np.array([0.25, 0.4, 0.5]))

    def test_variable_floordiv(self):
        """Test floordiv operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        result = v1 // v2
        self.assertAllClose(result, np.array([0, 0, 0]))

    def test_variable_rfloordiv(self):
        """Test numpy array floordiv by variable."""
        v1 = backend.Variable(initializer=np.array([3, 4, 6]))
        v2 = backend.Variable(initializer=np.array([9, 12, 18]))
        result = v2 // v1
        self.assertAllClose(result, np.array([3, 3, 3]))

    def test_variable_divmod(self):
        """Test divmod operation on a variable."""
        x = backend.Variable(initializer=np.array([3, 4, 6]))
        y = backend.Variable(initializer=np.array([9, 12, 18]))
        result = (x // y, x % y)
        self.assertAllClose(result, (np.array([0, 0, 0]), np.array([3, 4, 6])))

    def test_variable_rdivmod(self):
        """Test reverse divmod operation on a variable."""
        x = backend.Variable(initializer=np.array([3, 4, 6]))
        y = backend.Variable(initializer=np.array([9, 12, 18]))
        result = (y // x, y % x)
        self.assertAllClose(result, (np.array([3, 3, 3]), np.array([0, 0, 0])))

    def test_variable_mod(self):
        """Test mod operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        result = v1 % v2
        self.assertAllClose(result, np.array([1, 2, 3]))

    def test_variable_rmod(self):
        """Test reverse mod operation on a variable."""
        v1 = backend.Variable(initializer=np.array([4, 5, 6]))
        v2 = backend.Variable(initializer=np.array([1, 2, 3]))
        result = v2 % v1
        self.assertAllClose(result, np.array([1, 2, 3]))

    def test_variable_pow(self):
        """Test pow operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        result = v1**v2
        self.assertAllClose(result, np.array([1, 32, 729]))

    def test_variable_rpow(self):
        """Test reverse power operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        result = v2**v1
        self.assertAllClose(result, np.array([4, 25, 216]))

    def test_variable_matmul(self):
        """Test matmul operation on a variable."""
        v1 = backend.Variable(initializer=np.array([[1, 2], [3, 4]]))
        v2 = backend.Variable(initializer=np.array([[5, 6], [7, 8]]))
        result = v1 @ v2
        self.assertAllClose(result, np.array([[19, 22], [43, 50]]))

    def test_variable_rmatmul(self):
        """Test reverse matmul operation on a variable."""
        v1 = backend.Variable(initializer=np.array([[1, 2], [3, 4]]))
        v2 = backend.Variable(initializer=np.array([[5, 6], [7, 8]]))
        result = v2 @ v1
        self.assertAllClose(result, np.array([[23, 34], [31, 46]]))

    def test_variable_ne(self):
        """Test ne operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([1, 2, 3]))
        result = v1 != v2
        self.assertAllClose(result, np.array([False, False, False]))

    def test_variable_lt(self):
        """Test lt operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([1, 2, 3]))
        result = v1 < v2
        self.assertAllClose(result, np.array([False, False, False]))

    def test_variable_le(self):
        """Test le operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([1, 2, 3]))
        result = v1 <= v2
        self.assertAllClose(result, np.array([True, True, True]))

    def test_variable_gt(self):
        """Test gt operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([1, 2, 3]))
        result = v1 > v2
        self.assertAllClose(result, np.array([False, False, False]))

    def test_variable_ge(self):
        """Test ge operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([1, 2, 3]))
        result = v1 >= v2
        self.assertAllClose(result, np.array([True, True, True]))

    def test_variable_dtype(self):
        """Test retrieving the dtype of a variable."""
        v = backend.Variable(initializer=np.array([1, 2, 3]))
        self.assertEqual(v.dtype, "float32")

    def test_variable_shape(self):
        """Test retrieving the shape of a variable."""
        v = backend.Variable(initializer=np.array([[1, 2], [3, 4]]))
        self.assertEqual(v.shape, (2, 2))

    def test_variable_ndim(self):
        """Test retrieving the number of dimensions of a variable."""
        v = backend.Variable(initializer=np.array([[1, 2], [3, 4]]))
        self.assertEqual(v.ndim, 2)

    def test_variable_repr(self):
        """Test the string representation of a variable."""
        v = backend.Variable(initializer=np.array([1, 2, 3]), name="test_var")
        expected_repr = (
            "<KerasVariable shape=(3,), dtype=float32, path=test_var>"
        )
        self.assertEqual(repr(v), expected_repr)

    def test_variable_getitem(self):
        """Test getting an item from a variable."""
        v = backend.Variable(initializer=np.array([1, 2, 3]))
        self.assertEqual(v[0], 1)

    def test_variable_bool(self):
        """Test converting a variable to boolean."""
        v = backend.Variable(initializer=np.array([1, 2, 3]))
        with self.assertRaises(TypeError):
            bool(v)

    def test_variable_neg(self):
        """Test negating a variable."""
        v = backend.Variable(initializer=np.array([-1, 2]))
        neg_v = -v
        self.assertAllClose(neg_v, np.array([1, -2]))

    def test_variable_abs(self):
        """Test absolute value of a variable."""
        v = backend.Variable(initializer=np.array([-1, 2]))
        abs_v = abs(v)
        self.assertAllClose(abs_v, np.array([1, 2]))

    def test_variable_eq(self):
        """Test eq operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([1, 2, 3]))
        result = v1 == v2
        self.assertAllClose(result, np.array([True, True, True]))

    def test_variable_add(self):
        """Test addition operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        result = v1 + v2
        self.assertAllClose(result, np.array([5, 7, 9]))

    def test_variable_radd(self):
        """Test addition operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        result = v2 + v1
        self.assertAllClose(result, np.array([5, 7, 9]))

    def test_variable_sub(self):
        """Test subtraction operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        result = v1 - v2
        self.assertAllClose(result, np.array([-3, -3, -3]))

    def test_variable_rsub(self):
        """Test subtraction operation on a variable."""
        v2 = backend.Variable(initializer=np.array([1, 2, 3]))
        v1 = backend.Variable(initializer=np.array([4, 5, 6]))
        result = v2 - v1
        self.assertAllClose(result, np.array([-3, -3, -3]))

    def test_variable_rmul(self):
        """Test multiplication operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        result = v1 * 2
        self.assertAllClose(result, np.array([2, 4, 6]))

    def test_variable_rdiv(self):
        """Test division operation on a variable."""
        v1 = backend.Variable(initializer=np.array([4, 8, 16]))
        result = v1 / 2
        self.assertAllClose(result, np.array([2, 4, 8]))

    def rtruediv(self):
        """Test rtruediv operation on a variable"""
        v1 = backend.Variable(initializer=np.array([4, 8, 16]))
        result = v1 / 2
        self.assertAllClose(result, np.array([2, 4, 8]))

    def test_variable_rtruediv(self):
        """Test rtruediv operation on a variable"""
        v1 = backend.Variable(initializer=np.array([4, 8, 16]))
        result = 2 / v1
        self.assertAllClose(result, np.array([0.5, 0.25, 0.125]))


class VariableBinaryOperationsTest(test_case.TestCase):
    """Tests for binary operations on KerasVariable."""

    def test_variable_bool(self):
        """Test converting a variable to boolean."""
        v = backend.Variable(initializer=np.array([1, 2, 3]))
        with self.assertRaises(TypeError):
            bool(v)

    def test_variable_neg(self):
        """Test negating a variable."""
        v = backend.Variable(initializer=np.array([-1, 2]))
        neg_v = -v
        self.assertAllClose(neg_v, np.array([1, -2]))

    def test_variable_abs(self):
        """Test absolute value of a variable."""
        v = backend.Variable(initializer=np.array([-1, 2]))
        abs_v = abs(v)
        self.assertAllClose(abs_v, np.array([1, 2]))

    def test_invalid_dtype(self):
        """Test invalid dtype standardization."""
        invalid_dtype = "invalid_dtype"
        with self.assertRaisesRegex(
            ValueError, f"Invalid dtype: {invalid_dtype}"
        ):
            standardize_dtype(invalid_dtype)

    @patch("keras.backend.config.backend", return_value="jax")
    def test_jax_backend_b_dimension(self, mock_backend):
        """Test 'b' dimension handling with JAX backend."""
        shape = (3, "b", 5)
        standardized_shape = standardize_shape(shape)
        self.assertEqual(standardized_shape, shape)

    def test_negative_shape_entry(self):
        """Test negative shape entry."""
        shape = (3, -1, 5)
        with self.assertRaisesRegex(
            ValueError,
            "Negative dimensions are not allowed",
        ):
            standardize_shape(shape)

    def test_shape_equal_length_mismatch(self):
        """Test mismatch in lengths of shapes."""
        self.assertFalse(shape_equal((3, 2), (3, 2, 4)))
        self.assertFalse(shape_equal((), (3,)))
        self.assertFalse(shape_equal((3, 2, 4, 5), (3, 2, 4)))

    # def __pos__(self):
    #     return self.value.__pos__()

    # def test_variable_rand(self):
    #     """Test reverse & operation on a variable."""
    #     # TODO

    # def test_variable_or(self):
    #     """Test | operation on a variable."""
    #     # TODO

    # def test_variable_rxor(self):
    #     """Test reverse ^ operation on a variable."""
    #     # TODO

    # def test_variable_le(self):
    #     """Test less than or equal operation on a variable."""
    #     # TODO

    # def test_variable_gt(self):
    #     """Test greater than operation on a variable."""
    #     # TODO

    # def test_variable_ge(self):
    #     """Test greater than or equal operation on a variable."""
    #     # TODO

    # def test_variable_radd(self):
    #     """Test reverse addition on a variable."""
    #     # TODO

    # def test_variable_rsub(self):
    #     """Test reverse subtraction on a variable."""
    #     # TODO

    # def test_variable_and(self):
    #     """Test & operation on a variable."""
    #     # TODO

    # def test_variable_ror(self):
    #     """Test reverse | operation on a variable."""
    #     # TODO

    # def test_variable_xor(self):
    #     """Test ^ operation on a variable."""
    #     # TODO

    # def test_variable_lshift(self):
    #     """Test left shift operation on a variable."""
    #     # TODO

    # def test_variable_rlshift(self):
    #     """Test reverse left shift operation on a variable."""
    #     # TODO

    # def test_variable_rshift(self):
    #     """Test right shift operation on a variable."""
    #     # TODO

    # def test_variable_rrshift(self):
    #     """Test reverse right shift operation on a variable."""
    #     # TODO

    # def test_variable_round(self):
    #     """Test round operation on a variable."""
    #     # TODO


@pytest.mark.skipif(
    backend.backend() == "torch",
    reason="tensorflow.python.framework.errors_impl.InvalidArgumentError",
)
# tensorflow.python.framework.errors_impl.InvalidArgumentError:
# Value for attr 'T' of float is not in the list of allowed values:
# int8, int16, int32, int64, uint8, uint16, uint32, uint64
class TestVariableInvertWithTorch(test_case.TestCase):
    def test_variable_invert(self):
        """Test inversion operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]), dtype="int32")
        result = ~v1
        self.assertAllClose(result, np.array([-2, -3, -4]))


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="torch RuntimeError",
)
class TestVariableInvertWithOutTorch(test_case.TestCase):
    def test_variable_invert(self):
        """Test inversion operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]), dtype="float32")
        result = ~v1
        self.assertAllClose(result, np.array([-2, -3, -4]))


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="Tests for standardize_shape with Torch backend",
)
class TestStandardizeShapeWithTorch(test_case.TestCase):
    def test_standardize_shape_with_torch_size_containing_negative_value(self):
        """Tests shape with a negative value."""
        shape_with_negative_value = (3, 4, -5)
        with self.assertRaisesRegex(
            ValueError,
            "Cannot convert '\\(3, 4, -5\\)' to a shape. Negative dimensions",
        ):
            _ = standardize_shape(shape_with_negative_value)

    # TODO FAILED keras/backend/common/variables_test.py::
    # TestStandardizeShapeWithTorch::
    # test_standardize_shape_with_torch_size_containing_string
    # - AssertionError: ValueError not raised
    # def test_standardize_shape_with_torch_size_containing_string(self):
    #     """Tests shape with a string value."""
    #     shape_with_string = (3, 4, "5")
    #     with self.assertRaisesRegex(
    #         ValueError,
    #         "Cannot convert .* to a shape. Found invalid entry '5'.",
    #     ):
    #         _ = standardize_shape(shape_with_string)

    # TODO FAILED keras/backend/common/variables_test.py::
    # TestStandardizeShapeWithTorch::
    # test_standardize_shape_with_torch_size_containing_string
    # - AssertionError: ValueError not raised
    # def test_standardize_shape_with_torch_size_containing_float(self):
    #     """Tests shape with a float value."""
    #     shape_with_float = (3, 4, 5.0)
    #     with self.assertRaisesRegex(
    #         ValueError,
    #         "Cannot convert .* to a shape. Found invalid entry '5.0'.",
    #     ):
    #         _ = standardize_shape(shape_with_float)

    def test_standardize_shape_with_torch_size_valid(self):
        """Tests a valid shape."""
        shape_valid = (3, 4, 5)
        standardized_shape = standardize_shape(shape_valid)
        self.assertEqual(standardized_shape, (3, 4, 5))

    def test_standardize_shape_with_torch_size_multidimensional(self):
        """Tests shape of a multi-dimensional tensor."""
        import torch

        tensor = torch.randn(3, 4, 5)
        shape = tensor.size()
        standardized_shape = standardize_shape(shape)
        self.assertEqual(standardized_shape, (3, 4, 5))

    def test_standardize_shape_with_torch_size_single_dimension(self):
        """Tests shape of a single-dimensional tensor."""
        import torch

        tensor = torch.randn(10)
        shape = tensor.size()
        standardized_shape = standardize_shape(shape)
        self.assertEqual(standardized_shape, (10,))

    def test_standardize_shape_with_torch_size_with_valid_1_dimension(self):
        """Tests a valid shape."""
        shape_valid = [3]
        standardized_shape = standardize_shape(shape_valid)
        self.assertEqual(standardized_shape, (3,))

    def test_standardize_shape_with_torch_size_with_valid_2_dimension(self):
        """Tests a valid shape."""
        shape_valid = [3, 4]
        standardized_shape = standardize_shape(shape_valid)
        self.assertEqual(standardized_shape, (3, 4))

    def test_standardize_shape_with_torch_size_with_valid_3_dimension(self):
        """Tests a valid shape."""
        shape_valid = [3, 4, 5]
        standardized_shape = standardize_shape(shape_valid)
        self.assertEqual(standardized_shape, (3, 4, 5))

    def test_standardize_shape_with_torch_size_with_negative_value(self):
        """Tests shape with a negative value appended."""
        import torch

        tensor = torch.randn(3, 4, 5)
        shape = tuple(tensor.size())
        shape_with_negative = shape + (-1,)
        with self.assertRaisesRegex(
            ValueError,
            "Cannot convert .* to a shape. Negative dimensions are not",
        ):
            _ = standardize_shape(shape_with_negative)

    def test_standardize_shape_with_non_integer_entry(self):
        with self.assertRaisesRegex(
            ValueError,
            # "Cannot convert '\\(3, 4, 'a'\\)' to a shape. Found invalid",
            # TODO ask is it ok to have different error message for torch
            "invalid literal for int() with base 10: 'a'",
        ):
            standardize_shape([3, 4, "a"])

    def test_standardize_shape_with_negative_entry(self):
        with self.assertRaisesRegex(
            ValueError,
            "Cannot convert '\\(3, 4, -5\\)' to a shape. Negative dimensions",
        ):
            standardize_shape([3, 4, -5])

    def test_standardize_shape_with_valid_not_tuple(self):
        """Tests a valid shape."""
        shape_valid = [3, 4, 5]
        standardized_shape = standardize_shape(shape_valid)
        self.assertEqual(standardized_shape, (3, 4, 5))


@pytest.mark.skipif(
    backend.backend() == "torch",
    reason="Tests for standardize_shape with others backend",
)
class TestStandardizeShapeWithOutTorch(test_case.TestCase):
    def test_standardize_shape_with_out_torch_negative_value(self):
        """Tests shape with a negative value."""
        shape_with_negative_value = (3, 4, -5)
        with self.assertRaisesRegex(
            ValueError,
            "Cannot convert '\\(3, 4, -5\\)' to a shape. Negative dimensions",
        ):
            _ = standardize_shape(shape_with_negative_value)

    def test_standardize_shape_with_out_torch_string(self):
        """Tests shape with a string value."""
        shape_with_string = (3, 4, "5")
        with self.assertRaisesRegex(
            ValueError,
            "Cannot convert .* to a shape. Found invalid entry '5'.",
        ):
            _ = standardize_shape(shape_with_string)

    def test_standardize_shape_with_out_torch_float(self):
        """Tests shape with a float value."""
        shape_with_float = (3, 4, 5.0)
        with self.assertRaisesRegex(
            ValueError,
            "Cannot convert .* to a shape. Found invalid entry '5.0'.",
        ):
            _ = standardize_shape(shape_with_float)

    def test_standardize_shape_with_out_torch_valid(self):
        """Tests a valid shape."""
        shape_valid = (3, 4, 5)
        standardized_shape = standardize_shape(shape_valid)
        self.assertEqual(standardized_shape, (3, 4, 5))

    def test_standardize_shape_with_out_torch_valid_not_tuple(self):
        """Tests a valid shape."""
        shape_valid = [3, 4, 5]
        standardized_shape = standardize_shape(shape_valid)
        self.assertEqual(standardized_shape, (3, 4, 5))
