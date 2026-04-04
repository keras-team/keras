"""Tests for keras.src.utils.grain_utils."""

import numpy as np

from keras.src import backend
from keras.src import testing
from keras.src.utils.grain_utils import make_batch
from keras.src.utils.grain_utils import make_string_batch


class MakeBatchTest(testing.TestCase):
    def test_basic_batch(self):
        vals = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = make_batch(vals)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertAllClose(result, expected)

    def test_single_element(self):
        vals = [np.array([1.0, 2.0])]
        result = make_batch(vals)
        self.assertEqual(backend.standardize_shape(result.shape), (1, 2))

    def test_empty_raises(self):
        with self.assertRaisesRegex(ValueError, "Cannot batch 0"):
            make_batch([])

    def test_nested_structure(self):
        vals = [
            {"x": np.array([1.0]), "y": np.array([2.0])},
            {"x": np.array([3.0]), "y": np.array([4.0])},
        ]
        result = make_batch(vals)
        self.assertIn("x", result)
        self.assertIn("y", result)


class MakeStringBatchTest(testing.TestCase):
    def test_numeric_batch(self):
        vals = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = make_string_batch(vals)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertAllClose(result, expected)

    def test_empty_raises(self):
        with self.assertRaisesRegex(ValueError, "Cannot batch 0"):
            make_string_batch([])


if __name__ == "__main__":
    testing.run_tests()
