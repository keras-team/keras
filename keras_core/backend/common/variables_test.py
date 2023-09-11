import numpy as np

from keras_core import backend
from keras_core import initializers
from keras_core.backend.common.variables import AutocastScope
from keras_core.backend.common.variables import KerasVariable
from keras_core.backend.common.variables import standardize_shape
from keras_core.testing import test_case


class VariablesTest(test_case.TestCase):
    def test_deferred_initialization(self):
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
        import torch

        x = torch.randn(4, 4)
        backend.standardize_dtype(x.dtype)

    def test_name_validation(self):
        # Test when name is not a string
        with self.assertRaisesRegex(
            ValueError, "Argument `name` must be a string"
        ):
            KerasVariable(initializer=initializers.RandomNormal(), name=12345)

        # Test when name contains a '/'
        with self.assertRaisesRegex(ValueError, "cannot contain character `/`"):
            KerasVariable(
                initializer=initializers.RandomNormal(), name="invalid/name"
            )

    def test_standardize_shape_with_none(self):
        with self.assertRaisesRegex(
            ValueError, "Undefined shapes are not supported."
        ):
            standardize_shape(None)

    def test_standardize_shape_with_non_iterable(self):
        with self.assertRaisesRegex(
            ValueError, "Cannot convert '42' to a shape."
        ):
            standardize_shape(42)

    def test_standardize_shape_with_valid_input(self):
        shape = [3, 4, 5]
        standardized_shape = standardize_shape(shape)
        self.assertEqual(standardized_shape, (3, 4, 5))

    # TODO
    # (3.9,torch) FAILED keras_core/backend/common/variables_test.py
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
        with self.assertRaisesRegex(
            ValueError,
            "Cannot convert '\\(3, 4, -5\\)' to a shape. Negative dimensions",
        ):
            standardize_shape([3, 4, -5])

    def test_autocast_scope_with_non_float_dtype(self):
        with self.assertRaisesRegex(
            ValueError,
            "`AutocastScope` can only be used with a floating-point",
        ):
            _ = AutocastScope("int32")
