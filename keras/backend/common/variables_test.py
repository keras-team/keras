from unittest.mock import patch

import numpy as np
import pytest
from absl.testing import parameterized

from keras import backend
from keras import initializers
from keras.backend.common.variables import ALLOWED_DTYPES
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

    def test_variable_initialization_with_strings(self):
        """Test variable init with non-callable initializer."""
        v = backend.Variable(initializer="ones", shape=(2, 2))
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


class VariablePropertiesTest(test_case.TestCase, parameterized.TestCase):
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
        """Tests the trainable setter."""
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

    @parameterized.parameters(
        *((dtype for dtype in ALLOWED_DTYPES if dtype != "string"))
    )
    def test_standardize_dtype(self, dtype):
        """Tests standardize_dtype for all ALLOWED_DTYPES except string."""
        if backend.backend() == "torch" and dtype in (
            "uint16",
            "uint32",
            "uint64",
        ):
            self.skipTest(f"torch backend does not support dtype {dtype}")

        if backend.backend() == "jax":
            import jax

            if not jax.config.x64_enabled and "64" in dtype:
                self.skipTest(
                    f"jax backend does not support {dtype} without x64 enabled"
                )

        x = backend.convert_to_tensor(np.zeros(()), dtype)
        actual = standardize_dtype(x.dtype)
        self.assertEqual(actual, dtype)

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

    @pytest.mark.skipif(
        backend.backend() != "tf",
        reason="Tests for MirroredVariable under tf backend",
    )
    def test_variable_numpy_scalar(self):
        from keras.utils.module_utils import tensorflow as tf

        strategy = tf.distribute.MirroredStrategy(["cpu:0", "cpu:1"])
        with strategy.scope():
            v = backend.Variable(initializer=0.0)

        np_value = backend.convert_to_numpy(v)
        self.assertIsInstance(np_value, np.ndarray)
        self.assertAllClose(np_value, 0.0)

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
    """Tests for operations on KerasVariable."""

    def test_variable_as_boolean(self):
        """Test converting a variable to boolean."""
        v = backend.Variable(initializer=np.ones((2, 2)))
        with self.assertRaisesRegex(
            TypeError, "A Keras Variable cannot be used as a boolean."
        ):
            bool(v)

    def test__neg__(self):
        """Test negating a variable."""
        v = backend.Variable(initializer=np.array([-1, 2]), trainable=False)
        self.assertAllClose(v.__neg__(), np.array([1, -2]))

    def test__abs__(self):
        """Test absolute value on a variable."""
        v = backend.Variable(initializer=np.array([-1, 2]), trainable=False)
        self.assertAllClose(v.__abs__(), np.array([1, 2]))

    def test__invert__(self):
        """Test bitwise not on a variable."""
        v = backend.Variable(
            initializer=np.array([True, False]), trainable=False, dtype="bool"
        )
        self.assertAllClose(v.__invert__(), np.array([False, True]))

    def test__eq__(self):
        """Test equality comparison on a variable."""
        v = backend.Variable(initializer=np.array([1, 2]), trainable=False)
        self.assertAllClose(v.__eq__(np.array([1, 2])), np.array([True, True]))

    def test__ne__(self):
        """Test inequality comparison on a variable."""
        v = backend.Variable(initializer=np.array([1, 2]), trainable=False)
        self.assertAllClose(
            v.__ne__(np.array([1, 2])), np.array([False, False])
        )

    def test__lt__(self):
        """Test less than comparison on a variable."""
        v = backend.Variable(initializer=np.array([1, 2]), trainable=False)
        self.assertAllClose(
            v.__lt__(np.array([1, 2])), np.array([False, False])
        )

    def test__le__(self):
        """Test less than or equal to comparison on a variable."""
        v = backend.Variable(initializer=np.array([1, 2]), trainable=False)
        self.assertAllClose(v.__le__(np.array([1, 2])), np.array([True, True]))

    def test__gt__(self):
        """Test greater than comparison on a variable."""
        v = backend.Variable(initializer=np.array([1, 2]), trainable=False)
        self.assertAllClose(
            v.__gt__(np.array([1, 2])), np.array([False, False])
        )

    def test__ge__(self):
        """Test greater than or equal to comparison on a variable."""
        v = backend.Variable(initializer=np.array([1, 2]), trainable=False)
        self.assertAllClose(v.__ge__(np.array([1, 2])), np.array([True, True]))

    def test__add__(self):
        """Test addition operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        self.assertAllClose(v1.__add__(v2), np.array([5, 7, 9]))

    def test__radd__(self):
        """Test reverse addition operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        self.assertAllClose(v1.__radd__(v2), np.array([5, 7, 9]))

    def test__sub__(self):
        """Test subtraction operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        self.assertAllClose(v1.__sub__(v2), np.array([-3, -3, -3]))

    def test__rsub__(self):
        """Test reverse subtraction operation on a variable."""
        v1 = backend.Variable(initializer=np.array([4, 5, 6]))
        v2 = backend.Variable(initializer=np.array([1, 2, 3]))
        self.assertAllClose(v1.__rsub__(v2), np.array([-3, -3, -3]))

    def test__mul__(self):
        """Test multiplication operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        self.assertAllClose(v1.__mul__(v2), np.array([4, 10, 18]))

    def test__rmul__(self):
        """Test reverse multiplication operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        self.assertAllClose(v1.__rmul__(v2), np.array([4, 10, 18]))

    def test__truediv__(self):
        """Test true division operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([4, 5, 6]))
        self.assertAllClose(v1.__truediv__(v2), np.array([0.25, 0.4, 0.5]))

    def test__rtruediv__(self):
        """Test reverse true division operation on a variable."""
        v1 = backend.Variable(initializer=np.array([4, 5, 6]))
        v2 = backend.Variable(initializer=np.array([1, 2, 3]))
        self.assertAllClose(v1.__rtruediv__(v2), np.array([0.25, 0.4, 0.5]))

    def test__floordiv__(self):
        """Test floordiv operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([-4, 5, 6]))
        self.assertAllClose(v1.__floordiv__(v2), np.array([-1, 0, 0]))

    def test__rfloordiv__(self):
        """Test reverse floordiv operation on a variable."""
        v1 = backend.Variable(initializer=np.array([-4, 5, 6]))
        v2 = backend.Variable(initializer=np.array([1, 2, 3]))
        self.assertAllClose(v1.__rfloordiv__(v2), np.array([-1, 0, 0]))

    def test__mod__(self):
        """Test mod operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([-4, 5, 6]))
        self.assertAllClose(v1.__mod__(v2), np.array([-3, 2, 3]))

    def test__rmod__(self):
        """Test reverse mod operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([1, 2, 3]))
        self.assertAllClose(v1.__rmod__(v2), np.array([0, 0, 0]))

    def test__pow__(self):
        """Test pow operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([-4, 5, 6]))
        self.assertAllClose(v1.__pow__(v2), np.array([1, 32, 729]))

    def test__rpow__(self):
        """Test reverse power operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([1, 2, 3]))
        self.assertAllClose(v1.__rpow__(v2), np.array([1, 4, 27]))

    def test__matmul__(self):
        """Test matmul operation on a variable."""
        v1 = backend.Variable(initializer=np.array([[1, 2], [3, 4]]))
        v2 = backend.Variable(initializer=np.array([[5, 6], [7, 8]]))
        self.assertAllClose(v1.__matmul__(v2), np.array([[19, 22], [43, 50]]))

    def test__rmatmul__(self):
        """Test reverse matmul operation on a variable."""
        v1 = backend.Variable(initializer=np.array([[1, 2], [3, 4]]))
        v2 = backend.Variable(initializer=np.array([[5, 6], [7, 8]]))
        self.assertAllClose(v1.__rmatmul__(v2), np.array([[23, 34], [31, 46]]))

    def test__and__(self):
        """Test bitwise and operation on a variable."""
        v1 = backend.Variable(
            initializer=np.array([True, False]), dtype="bool", trainable=False
        )
        v2 = backend.Variable(
            initializer=np.array([True, True]), dtype="bool", trainable=False
        )
        self.assertAllClose(v1.__and__(v2), np.array([True, False]))

    def test__rand__(self):
        """Test reverse bitwise and operation on a variable."""
        v1 = backend.Variable(
            initializer=np.array([True, False]), dtype="bool", trainable=False
        )
        v2 = backend.Variable(
            initializer=np.array([True, True]), dtype="bool", trainable=False
        )
        self.assertAllClose(v1.__rand__(v2), np.array([True, False]))

    def test__or__(self):
        """Test bitwise or operation on a variable."""
        v1 = backend.Variable(
            initializer=np.array([True, False]), dtype="bool", trainable=False
        )
        v2 = backend.Variable(
            initializer=np.array([True, True]), dtype="bool", trainable=False
        )
        self.assertAllClose(v1.__or__(v2), np.array([True, True]))

    def test__ror__(self):
        """Test reverse bitwise or operation on a variable."""
        v1 = backend.Variable(
            initializer=np.array([True, False]), dtype="bool", trainable=False
        )
        v2 = backend.Variable(
            initializer=np.array([True, True]), dtype="bool", trainable=False
        )
        self.assertAllClose(v1.__ror__(v2), np.array([True, True]))

    def test__xor__(self):
        """Test bitwise xor operation on a variable."""
        v1 = backend.Variable(
            initializer=np.array([True, False]), dtype="bool", trainable=False
        )
        v2 = backend.Variable(
            initializer=np.array([True, True]), dtype="bool", trainable=False
        )
        self.assertAllClose(v1.__xor__(v2), np.array([False, True]))

    def test__rxor__(self):
        """Test reverse bitwise xor operation on a variable."""
        v1 = backend.Variable(
            initializer=np.array([True, False]), dtype="bool", trainable=False
        )
        v2 = backend.Variable(
            initializer=np.array([True, True]), dtype="bool", trainable=False
        )
        self.assertAllClose(v1.__rxor__(v2), np.array([False, True]))

    def test__pos__(self):
        """Test unary plus on a variable."""
        v = backend.Variable(initializer=np.array([-1, 2]), trainable=False)
        self.assertAllClose(v.__pos__(), np.array([-1, 2]))

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


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="Tests for standardize_shape with Torch backend",
)
class TestStandardizeShapeWithTorch(test_case.TestCase):
    """Tests for standardize_shape with Torch backend."""

    def test_standardize_shape_with_torch_size_containing_negative_value(self):
        """Tests shape with a negative value."""
        shape_with_negative_value = (3, 4, -5)
        with self.assertRaisesRegex(
            ValueError,
            "Cannot convert '\\(3, 4, -5\\)' to a shape. Negative dimensions",
        ):
            _ = standardize_shape(shape_with_negative_value)

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
        """Tests shape with a non-integer value."""
        with self.assertRaisesRegex(
            # different error message for torch
            ValueError,
            r"invalid literal for int\(\) with base 10: 'a'",
        ):
            standardize_shape([3, 4, "a"])

    def test_standardize_shape_with_negative_entry(self):
        """Tests shape with a negative value."""
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
    """Tests for standardize_shape with others backend."""

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
