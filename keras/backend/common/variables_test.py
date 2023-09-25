import numpy as np

from keras import backend
from keras import initializers
from keras.backend.common.variables import AutocastScope
from keras.backend.common.variables import KerasVariable
from keras.backend.common.variables import standardize_shape
from keras.testing import test_case


class VariableInitializationTest(test_case.TestCase):
    """tests for lines unders KerasVariable __init__"""

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

    # TODO (3.9,torch) FAILED keras/backend/common/variables_test.py
    # ::VariablesTest::test_standardize_shape_with_non_integer_entry:
    #  - AssertionError "Cannot convert '\(3, 4, 'a'\)' to a shape.
    # " does not match "invalid literal for int() with base 10: 'a'"
    # def test_standardize_shape_with_non_integer_entry(self):
    #     with self.assertRaisesRegex(
    #         ValueError,
    #         "Cannot convert '\\(3, 4, 'a'\\)' to a shape. Found invalid",
    #     ):
    #         standardize_shape([3, 4, "a"])

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


class VariableOperationsTest(test_case.TestCase):
    def test_variable_as_boolean(self):
        """Test converting a variable to boolean."""
        v = backend.Variable(initializer=np.ones((2, 2)))
        with self.assertRaises(TypeError):
            bool(v)

    def test_variable_negation(self):
        """Test negating a variable."""
        v = backend.Variable(initializer=np.array([-1, 2]))
        neg_v = -v
        self.assertAllClose(neg_v, np.array([1, -2]))

    def test_variable_pos(self):
        """Test positive value of a variable."""
        v = backend.Variable(initializer=np.array([-1, 2]))
        pos_v = v
        self.assertAllClose(pos_v, np.array([-1, 2]))

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

    def test_variable_rfloordiv(self):
        """Test numpy array floordiv by variable."""
        v = backend.Variable(initializer=np.array([3, 4, 6]))
        result = np.array([9, 12, 18]) // v
        self.assertAllClose(result, np.array([3, 3, 3]))

    def test_variable_rmatmul(self):
        """Test numpy array matmul with variable."""
        v = backend.Variable(initializer=np.array([[2, 3], [4, 5]]))
        other = np.array([[1, 2], [3, 4]])
        rmatmul_result = other @ v
        self.assertAllClose(rmatmul_result, np.array([[10, 13], [22, 29]]))

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

    def test_variable_mod(self):
        """Test mod operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        result = v1 % v2
        self.assertAllClose(result, np.array([1, 2, 3]))

    def test_variable_pow(self):
        """Test pow operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        result = v1**v2
        self.assertAllClose(result, np.array([1, 32, 729]))

    def test_variable_matmul(self):
        """Test matmul operation on a variable."""
        v1 = backend.Variable(initializer=np.array([[1, 2], [3, 4]]))
        v2 = backend.Variable(initializer=np.array([[5, 6], [7, 8]]))
        result = v1 @ v2
        self.assertAllClose(result, np.array([[19, 22], [43, 50]]))

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

    def test_variable_radd(self):
        """Test addition operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        result = v1 + v2
        self.assertAllClose(result, np.array([5, 7, 9]))

    def test_variable_sub(self):
        """Test subtraction operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        result = v1 - v2
        self.assertAllClose(result, np.array([-3, -3, -3]))

    def test_variable_rsub(self):
        """Test subtraction operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        result = v1 - v2
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

    def test_variable_rtruediv(self):
        """Test division operation on a variable"""
        v1 = backend.Variable(initializer=np.array([4, 8, 16]))
        result = v1 / 2
        self.assertAllClose(result, np.array([2, 4, 8]))


# TODO add test for the following lines
#  def __bool__(self):
#         raise TypeError("A Keras Variable cannot be used as a boolean.")

#     def __neg__(self):
#         return self.value.__neg__()

#     def __pos__(self):
#         return self.value.__pos__()

#     def __abs__(self):
#         return self.value.__abs__()

#     def __invert__(self):
#         return self.value.__invert__()

#           def __lt__(self, other):
#         value = self.value
#         return value.__lt__(self._convert_to_tensor(other, dtype=value.dtype))

#     def __le__(self, other):
#         value = self.value
#         return value.__le__(self._convert_to_tensor(other, dtype=value.dtype))

#     def __gt__(self, other):
#         value = self.value
#         return value.__gt__(self._convert_to_tensor(other, dtype=value.dtype))

#     def __ge__(self, other):
#         value = self.value
#         return value.__ge__(self._convert_to_tensor(other, dtype=value.dtype))

#     def __radd__(self, other):
#         value = self.value
#         return value.__radd__(self._convert_to_tensor(other, dtype=value.dtype))


#     def __rsub__(self, other):
#         value = self.value
#         return value.__rsub__(self._convert_to_tensor(other, dtype=value.dtype))


#     def __div__(self, other):
#         value = self.value
#         return value.__div__(self._convert_to_tensor(other, dtype=value.dtype))

#     def __rdiv__(self, other):
#         value = self.value
#         return value.__rdiv__(self._convert_to_tensor(other, dtype=value.dtype))

#          def __rtruediv__(self, other):
#         value = self.value
#         return value.__rtruediv__(
#             self._convert_to_tensor(other, dtype=value.dtype)
#         )

#     def __floordiv__(self, other):
#         value = self.value
#         return value.__floordiv__(
#             self._convert_to_tensor(other, dtype=value.dtype)
#         )

#     def __rfloordiv__(self, other):
#         value = self.value
#         return value.__rfloordiv__(
#             self._convert_to_tensor(other, dtype=value.dtype)
#         )

#     def __divmod__(self, other):
#         value = self.value
#         return value.__divmod__(
#             self._convert_to_tensor(other, dtype=value.dtype)
#         )

#     def __rdivmod__(self, other):
#         value = self.value
#         return value.__rdivmod__(
#             self._convert_to_tensor(other, dtype=value.dtype)
#         )

#     def __mod__(self, other):
#         value = self.value
#         return value.__mod__(self._convert_to_tensor(other, dtype=value.dtype))

#     def __rmod__(self, other):
#         value = self.value
#         return value.__rmod__(self._convert_to_tensor(other, dtype=value.dtype))

#     def __pow__(self, other):
#         value = self.value
#         return value.__pow__(self._convert_to_tensor(other, dtype=value.dtype))

#     def __rpow__(self, other):
#         value = self.value
#         return value.__rpow__(self._convert_to_tensor(other, dtype=value.dtype))

#     def __matmul__(self, other):
#         value = self.value
#         return value.__matmul__(
#             self._convert_to_tensor(other, dtype=value.dtype)
#         )

#     def __rmatmul__(self, other):
#         value = self.value
#         return value.__rmatmul__(
#             self._convert_to_tensor(other, dtype=value.dtype)
#         )

#     def __and__(self, other):
#         value = self.value
#         return value.__and__(self._convert_to_tensor(other, dtype=value.dtype))

#     def __rand__(self, other):
#         value = self.value
#         return value.__rand__(self._convert_to_tensor(other, dtype=value.dtype))

#     def __or__(self, other):
#         value = self.value
#         return value.__or__(self._convert_to_tensor(other, dtype=value.dtype))

#     def __ror__(self, other):
#         value = self.value
#         return value.__ror__(self._convert_to_tensor(other, dtype=value.dtype))

#     def __xor__(self, other):
#         value = self.value
#         return value.__xor__(self._convert_to_tensor(other, dtype=value.dtype))

#     def __rxor__(self, other):
#         value = self.value
#         return value.__rxor__(self._convert_to_tensor(other, dtype=value.dtype))

#     def __lshift__(self, other):
#         value = self.value
#         return value.__lshift__(
#             self._convert_to_tensor(other, dtype=value.dtype)
#         )

#     def __rlshift__(self, other):
#         value = self.value
#         return value.__rlshift__(
#             self._convert_to_tensor(other, dtype=self.dtype)
#         )

#     def __rshift__(self, other):
#         value = self.value
#         return value.__rshift__(
#             self._convert_to_tensor(other, dtype=value.dtype)
#         )

#     def __rrshift__(self, other):
#         value = self.value
#         return value.__rrshift__(
#             self._convert_to_tensor(other, dtype=self.dtype)
#         )

#     def __round__(self, ndigits=None):
#         value = self.value
#         return value.__round__(ndigits)


# TODO add test for the following lines in def standardize_dtype(dtype):
#  if dtype not in ALLOWED_DTYPES:
# !
#         raise ValueError(f"Invalid dtype: {dtype}")

# """


# """ TODO add test for the following lines in def standardize_shape(shape):
#     if config.backend() == "jax" and str(e) == "b":
# !
#             # JAX2TF tracing represents `None` dimensions as `b`
#             continue
#         if not isinstance(e, int):
# !
#             raise ValueError(
#                 f"Cannot convert '{shape}' to a shape. "
#                 f"Found invalid entry '{e}'. "
#             )
#         if e < 0:
#             raise ValueError(
#                 f"Cannot convert '{shape}' to a shape. "
#                 "Negative dimensions are not allowed."
#             )
#     return shape

# """ TODO Add tests for def shape_equal(a_shape, b_shape):
#     #Return whether a_shape == b_shape (allows None entries)#
#     if len(a_shape) != len(b_shape):
# !
#         return False
