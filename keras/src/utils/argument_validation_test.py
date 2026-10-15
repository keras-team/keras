import numpy as np

from keras.src import layers
from keras.src import testing
from keras.src.utils import argument_validation


class ArgumentValidationTest(testing.TestCase):
    def test_standardize_tuple_valid_inputs(self):
        # Python int scalar
        result = argument_validation.standardize_tuple(2, 2, "kernel_size")
        self.assertEqual(result, (2, 2))
        self.assertTrue(all(type(x) is int for x in result))

        # NumPy integer scalar
        result = argument_validation.standardize_tuple(
            np.int64(2), 2, "kernel_size"
        )
        self.assertEqual(result, (2, 2))
        self.assertTrue(all(type(x) is int for x in result))

        # Tuple and list of ints
        result = argument_validation.standardize_tuple((2, 3), 2, "kernel_size")
        self.assertEqual(result, (2, 3))
        self.assertTrue(all(type(x) is int for x in result))

        result = argument_validation.standardize_tuple([2, 3], 2, "kernel_size")
        self.assertEqual(result, (2, 3))
        self.assertTrue(all(type(x) is int for x in result))

        # Tuple of floats with integer values - must be converted to int
        result = argument_validation.standardize_tuple(
            (2.0, 3.0), 2, "kernel_size"
        )
        self.assertEqual(result, (2, 3))
        self.assertTrue(all(type(x) is int for x in result))

        # Tuple of NumPy integers - must be converted to Python int
        result = argument_validation.standardize_tuple(
            (np.int64(2), np.int32(3)), 2, "kernel_size"
        )
        self.assertEqual(result, (2, 3))
        self.assertTrue(all(type(x) is int for x in result))

        # NumPy 1D array
        result = argument_validation.standardize_tuple(
            np.array([2, 3]), 2, "kernel_size"
        )
        self.assertEqual(result, (2, 3))
        self.assertTrue(all(type(x) is int for x in result))

        # allow_zero=True
        result = argument_validation.standardize_tuple(
            (0, 1), 2, "kernel_size", allow_zero=True
        )
        self.assertEqual(result, (0, 1))
        self.assertTrue(all(type(x) is int for x in result))

        result = argument_validation.standardize_tuple(
            0, 2, "kernel_size", allow_zero=True
        )
        self.assertEqual(result, (0, 0))
        self.assertTrue(all(type(x) is int for x in result))

    def test_standardize_tuple_invalid_inputs(self):
        # Non-integer floats
        with self.assertRaises(ValueError):
            argument_validation.standardize_tuple((2.5, 3.0), 2, "kernel_size")

        # Booleans should be rejected even though bool subclasses int
        with self.assertRaises(ValueError):
            argument_validation.standardize_tuple(True, 2, "kernel_size")

        with self.assertRaises(ValueError):
            argument_validation.standardize_tuple(False, 2, "kernel_size")

        with self.assertRaises(ValueError):
            argument_validation.standardize_tuple(
                (True, False), 2, "kernel_size"
            )

        # Strings should not be accepted as an iterable of chars
        with self.assertRaises(ValueError):
            argument_validation.standardize_tuple("12", 2, "kernel_size")

        with self.assertRaises(ValueError):
            argument_validation.standardize_tuple("invalid", 2, "kernel_size")

        # Wrong length
        with self.assertRaises(ValueError):
            argument_validation.standardize_tuple((1, 2, 3), 2, "kernel_size")

        # None or non-iterable / non-int
        with self.assertRaises(ValueError):
            argument_validation.standardize_tuple(None, 2, "kernel_size")

        # Zero when allow_zero=False
        with self.assertRaises(ValueError):
            argument_validation.standardize_tuple(
                (0, 1), 2, "kernel_size", allow_zero=False
            )

        with self.assertRaises(ValueError):
            argument_validation.standardize_tuple(
                0, 2, "kernel_size", allow_zero=False
            )

        # Negative numbers
        with self.assertRaises(ValueError):
            argument_validation.standardize_tuple(
                (-1, 1), 2, "kernel_size", allow_zero=True
            )

    def test_standardize_padding(self):
        self.assertEqual(
            argument_validation.standardize_padding("valid"), "valid"
        )
        self.assertEqual(
            argument_validation.standardize_padding("SAME"), "same"
        )
        self.assertEqual(
            argument_validation.standardize_padding(
                "causal", allow_causal=True
            ),
            "causal",
        )
        self.assertEqual(
            argument_validation.standardize_padding((1, 2)), (1, 2)
        )
        with self.assertRaises(ValueError):
            argument_validation.standardize_padding(
                "causal", allow_causal=False
            )
        with self.assertRaises(ValueError):
            argument_validation.standardize_padding("unknown")

    def test_validate_string_arg(self):
        argument_validation.validate_string_arg(
            "valid", ["valid", "same"], "caller", "arg"
        )
        argument_validation.validate_string_arg(
            None, ["valid"], "caller", "arg", allow_none=True
        )
        argument_validation.validate_string_arg(
            lambda x: x, ["valid"], "caller", "arg", allow_callables=True
        )
        with self.assertRaises(ValueError):
            argument_validation.validate_string_arg(
                "invalid", ["valid"], "caller", "arg"
            )

    def test_standardize_tuple_conv_integration(self):
        conv = layers.Conv2D(filters=16, kernel_size=(2.0, 2.0))
        self.assertEqual(conv.kernel_size, (2, 2))
        self.assertTrue(all(type(x) is int for x in conv.kernel_size))
        config = conv.get_config()
        self.assertEqual(config["kernel_size"], (2, 2))
        self.assertTrue(all(type(x) is int for x in config["kernel_size"]))

        conv_np = layers.Conv2D(filters=16, kernel_size=np.int64(3))
        self.assertEqual(conv_np.kernel_size, (3, 3))
        self.assertTrue(all(type(x) is int for x in conv_np.kernel_size))
