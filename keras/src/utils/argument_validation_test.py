import numpy as np

from keras.src import testing
from keras.src.utils import argument_validation


class StandardizeIntTest(testing.TestCase):
    def test_numpy_integer(self):
        for value in (np.int8(2), np.int32(2), np.int64(2), np.uint8(2)):
            result = argument_validation.standardize_int(value)
            self.assertIs(type(result), int)
            self.assertEqual(result, 2)

    def test_list_and_tuple(self):
        result = argument_validation.standardize_int(
            (np.int64(1), 2, None, [np.int32(3), "a"])
        )
        self.assertEqual(result, (1, 2, None, [3, "a"]))
        self.assertIs(type(result), tuple)
        self.assertIs(type(result[0]), int)
        self.assertIs(type(result[3]), list)
        self.assertIs(type(result[3][0]), int)

    def test_other_values_are_unchanged(self):
        array = np.array([1, 2])
        for value in (2, True, 2.0, "2", None, np.float32(2.0), np.bool_(True)):
            self.assertIs(argument_validation.standardize_int(value), value)
        self.assertIs(argument_validation.standardize_int(array), array)


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
