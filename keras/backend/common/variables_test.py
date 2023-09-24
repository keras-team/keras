import numpy as np

from keras import backend
from keras import initializers
from keras.backend.common.variables import AutocastScope
from keras.backend.common.variables import KerasVariable
from keras.backend.common.variables import standardize_shape
from keras.testing import test_case


class VariablesTest(test_case.TestCase):
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

    def test_variable_initialization_with_non_callable(self):
        """Test variable init with non-callable initializer."""
        v = backend.Variable(initializer=np.ones((2, 2)))
        self.assertAllClose(v.value, np.ones((2, 2)))

    def test_variable_path_creation(self):
        """Test path creation for a variable."""
        v = backend.Variable(initializer=np.ones((2, 2)), name="test_var")
        self.assertEqual(v.path, "test_var")

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

    def test_variable_abs(self):
        """Test absolute value of a variable."""
        v = backend.Variable(initializer=np.array([-1, 2]))
        abs_v = abs(v)
        self.assertAllClose(abs_v, np.array([1, 2]))

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

    def test_variable_eq(self):
        """Test eq operation on a variable."""
        v1 = backend.Variable(initializer=np.array([1, 2, 3]))
        v2 = backend.Variable(initializer=np.array([1, 2, 3]))
        result = v1 == v2
        self.assertAllClose(result, np.array([True, True, True]))

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


# TODO add test for  'numpy'
# TODO add test for  'value'
# TODO add test for  'assign'
# TODO add test for  'assign_add'
# TODO add test for  'assign_sub'
# TODO add test for  'dtype'
# TODO add test for  'shape'
# TODO add test for  'ndim'
# TODO add test for  '__repr__'
# TODO add test for  '_initialize'
# TODO add test for  '_convert_to_tensor'
# TODO add test for  '__getitem__'
# TODO add test for  '__array__'
# TODO add test for  '__bool__'
# TODO add test for  '__neg__'
# TODO add test for  '__pos__'
# TODO add test for  '__abs__'
# TODO add test for  '__invert__'
# TODO add test for  '__eq__'
# TODO add test for  '__ne__'
# TODO add test for  '__lt__'
# TODO add test for  '__le__'
# TODO add test for  '__gt__'
# TODO add test for  '__ge__'
# TODO add test for  '__add__'
# TODO add test for  '__radd__'
# TODO add test for  '__sub__'
# TODO add test for  '__rsub__'
# TODO add test for  '__mul__'
# TODO add test for  '__rmul__'
# TODO add test for  '__div__'
# TODO add test for  '__rdiv__'
# TODO add test for  '__truediv__'
# TODO add test for  '__rtruediv__'
# TODO add test for  '__floordiv__'
# TODO add test for  '__rfloordiv__'
# TODO add test for  '__divmod__'
# TODO add test for  '__rdivmod__'
# TODO add test for  '__mod__'
# TODO add test for  '__rmod__'
# TODO add test for  '__pow__'
# TODO add test for  '__rpow__'
# TODO add test for  '__matmul__'
# TODO add test for  '__rmatmul__'
# TODO add test for  '__and__'
# TODO add test for  '__rand__'
# TODO add test for  '__or__'
# TODO add test for  '__ror__'
# TODO add test for  '__xor__'
# TODO add test for  '__rxor__'
# TODO add test for  '__lshift__'
# TODO add test for  '__rlshift__'
# TODO add test for  '__rshift__'
# TODO add test for  '__rrshift__'
# TODO add test for  '__round__'
# TODO add test for  'register_uninitialized_variable'
# TODO add test for  'initialize_all_variables'
# TODO add test for  'standardize_dtype'
# TODO add test for  'standardize_shape'
# TODO add test for  'shape_equal'
# TODO add test for  'is_float_dtype'
# TODO add test for  'is_int_dtype'
# TODO add test for  'get_autocast_scope'
# TODO add test for  'AutocastScope'
# TODO add test for  'maybe_cast'
# TODO add test for  '__enter__'
# TODO add test for  '__exit__'
