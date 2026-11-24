import numpy as np

from keras.src import testing
from keras.src.utils import argument_validation


class ValidateIntTest(testing.TestCase):
    def test_python_and_numpy_integers(self):
        for value in (2, np.int8(2), np.int32(2), np.int64(2), np.uint8(2)):
            result = argument_validation.validate_int("units", value)
            self.assertIs(type(result), int)
            self.assertEqual(result, 2)

    def test_rejects_non_integers(self):
        for value in (2.0, np.float32(2.0), "2", True, np.bool_(True), None):
            with self.assertRaisesRegex(
                TypeError, "Expected `units` to be an integer"
            ):
                argument_validation.validate_int("units", value)

    def test_allow_none(self):
        self.assertIsNone(
            argument_validation.validate_int("units", None, allow_none=True)
        )
        with self.assertRaisesRegex(TypeError, "an integer or `None`"):
            argument_validation.validate_int("units", 2.0, allow_none=True)

    def test_min_value(self):
        self.assertEqual(
            argument_validation.validate_int("units", 1, min_value=1), 1
        )
        self.assertEqual(
            argument_validation.validate_int("padding", 0, min_value=0), 0
        )
        with self.assertRaisesRegex(
            ValueError, "Expected `units` to be an integer >= 1"
        ):
            argument_validation.validate_int("units", 0, min_value=1)
        with self.assertRaisesRegex(ValueError, ">= 1"):
            argument_validation.validate_int("units", np.int64(-3), min_value=1)


class ValidateIntSequenceTest(testing.TestCase):
    def test_list_and_tuple(self):
        result = argument_validation.validate_int_sequence(
            "axis", (np.int64(1), 2)
        )
        self.assertEqual(result, (1, 2))
        self.assertIs(type(result), tuple)
        self.assertIs(type(result[0]), int)

        result = argument_validation.validate_int_sequence(
            "axis", [np.int32(1), 2]
        )
        self.assertEqual(result, [1, 2])
        self.assertIs(type(result), list)
        self.assertIs(type(result[0]), int)

    def test_rejects_non_sequences(self):
        for value in (1, "12", np.array([1, 2]), None):
            with self.assertRaisesRegex(
                TypeError, "Expected `axis` to be a list or a tuple"
            ):
                argument_validation.validate_int_sequence("axis", value)

    def test_rejects_non_integer_elements(self):
        with self.assertRaisesRegex(
            TypeError, r"Expected `axis\[1\]` to be an integer"
        ):
            argument_validation.validate_int_sequence("axis", (1, 2.0))

    def test_length(self):
        with self.assertRaisesRegex(
            ValueError, "Expected `axes` to have 2 elements"
        ):
            argument_validation.validate_int_sequence("axes", (1,), length=2)

    def test_allow_none_and_min_value(self):
        result = argument_validation.validate_int_sequence(
            "noise_shape", (None, np.int64(1)), allow_none=True, min_value=1
        )
        self.assertEqual(result, (None, 1))
        with self.assertRaisesRegex(TypeError, r"`noise_shape\[0\]`"):
            argument_validation.validate_int_sequence("noise_shape", (None, 1))
        with self.assertRaisesRegex(
            ValueError, r"Expected `noise_shape\[1\]` to be an integer >= 1"
        ):
            argument_validation.validate_int_sequence(
                "noise_shape", (None, 0), allow_none=True, min_value=1
            )


class StandardizeTupleTest(testing.TestCase):
    def test_numpy_integer_scalar(self):
        result = argument_validation.standardize_tuple(
            np.int64(2), 2, "kernel_size"
        )
        self.assertEqual(result, (2, 2))
        for value in result:
            self.assertIs(type(value), int)

    def test_numpy_integer_scalar_validates_sign(self):
        with self.assertRaisesRegex(ValueError, "value > 0"):
            argument_validation.standardize_tuple(np.int64(0), 2, "kernel_size")
        self.assertEqual(
            argument_validation.standardize_tuple(
                np.int32(0), 2, "padding", allow_zero=True
            ),
            (0, 0),
        )
