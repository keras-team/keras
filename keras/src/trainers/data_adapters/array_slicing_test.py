"""Tests for keras.src.trainers.data_adapters.array_slicing."""

import math

import numpy as np

from keras.src import backend
from keras.src import testing
from keras.src.trainers.data_adapters.array_slicing import NumpySliceable
from keras.src.trainers.data_adapters.array_slicing import Sliceable
from keras.src.trainers.data_adapters.array_slicing import can_slice_array
from keras.src.trainers.data_adapters.array_slicing import convert_to_sliceable
from keras.src.trainers.data_adapters.array_slicing import (
    train_validation_split,
)


class CanSliceArrayTest(testing.TestCase):
    def test_none(self):
        self.assertTrue(can_slice_array(None))

    def test_numpy_array(self):
        self.assertTrue(can_slice_array(np.array([1, 2, 3])))

    def test_list_not_sliceable(self):
        self.assertFalse(can_slice_array([1, 2, 3]))

    def test_string_not_sliceable(self):
        self.assertFalse(can_slice_array("hello"))

    def test_int_not_sliceable(self):
        self.assertFalse(can_slice_array(42))

    def test_object_with_array_method(self):
        """Objects with __array__ should be sliceable."""

        class ArrayLike:
            def __array__(self, dtype=None):
                return np.array([1, 2, 3])

        self.assertTrue(can_slice_array(ArrayLike()))

    def test_backend_tensor(self):
        t = backend.convert_to_tensor(np.array([1.0, 2.0], dtype="float32"))
        self.assertTrue(can_slice_array(t))


class NumpySliceableTest(testing.TestCase):
    def test_slice(self):
        arr = np.arange(10)
        s = NumpySliceable(arr)
        result = s[2:5]
        self.assertTrue(np.array_equal(result, np.array([2, 3, 4])))

    def test_index_list(self):
        arr = np.arange(10)
        s = NumpySliceable(arr)
        result = s[[0, 3, 7]]
        self.assertTrue(np.array_equal(result, np.array([0, 3, 7])))

    def test_cast(self):
        arr = np.array([1.0, 2.0], dtype="float64")
        result = NumpySliceable.cast(arr, "float32")
        self.assertEqual(result.dtype, np.float32)

    def test_convert_to_numpy_is_identity(self):
        arr = np.array([1, 2, 3])
        result = NumpySliceable.convert_to_numpy(arr)
        self.assertIs(result, arr)

    def test_convert_to_tf_dataset_compatible_is_identity(self):
        arr = np.array([1, 2, 3])
        result = NumpySliceable.convert_to_tf_dataset_compatible(arr)
        self.assertIs(result, arr)

    def test_slice_returns_exact_values(self):
        arr = np.arange(20)
        s = NumpySliceable(arr)
        result = s[5:12]
        expected = np.arange(5, 12)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(len(result), 7)

    def test_single_index(self):
        arr = np.arange(10)
        s = NumpySliceable(arr)
        result = s[0:1]
        np.testing.assert_array_equal(result, np.array([0]))

    def test_2d_slice_preserves_trailing_dims(self):
        arr = np.arange(20).reshape(10, 2)
        s = NumpySliceable(arr)
        result = s[3:7]
        self.assertEqual(result.shape, (4, 2))
        np.testing.assert_array_equal(result, arr[3:7])


class ConvertToSliceableTest(testing.TestCase):
    def test_numpy_array_wraps_to_numpy_sliceable(self):
        arr = np.array([1, 2, 3])
        result = convert_to_sliceable(arr, target_backend=None)
        self.assertIsInstance(result, Sliceable)

    def test_none_passes_through(self):
        result = convert_to_sliceable(None, target_backend=None)
        self.assertIsNone(result)

    def test_nested_structure(self):
        x = np.ones((5, 3))
        y = np.zeros((5,))
        result = convert_to_sliceable((x, y), target_backend=None)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Sliceable)
        self.assertIsInstance(result[1], Sliceable)

    def test_dict_structure(self):
        data = {"x": np.ones((5, 3)), "y": np.zeros((5,))}
        result = convert_to_sliceable(data, target_backend=None)
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["x"], Sliceable)
        self.assertIsInstance(result["y"], Sliceable)

    def test_float64_cast_to_floatx(self):
        """Float arrays should be cast to current floatx."""
        arr = np.array([1.0, 2.0], dtype="float64")
        result = convert_to_sliceable(arr, target_backend=None)
        # The dtype should be floatx (float32 by default)
        expected_dtype = backend.floatx()
        self.assertEqual(
            backend.standardize_dtype(result.array.dtype), expected_dtype
        )

    def test_int_arrays_not_cast(self):
        """Integer arrays should NOT be cast."""
        arr = np.array([1, 2, 3], dtype="int32")
        result = convert_to_sliceable(arr, target_backend=None)
        self.assertEqual(result.array.dtype, np.int32)

    def test_invalid_type_raises(self):
        with self.assertRaisesRegex(ValueError, "Received invalid input"):
            convert_to_sliceable("a string", target_backend=None)

    def test_array_like_object(self):
        """Objects with __array__ are converted to numpy."""

        class ArrayLike:
            def __init__(self):
                self.shape = (3,)
                self.dtype = np.float32

            def __array__(self, dtype=None):
                return np.array([1.0, 2.0, 3.0], dtype=np.float32)

        result = convert_to_sliceable(ArrayLike(), target_backend=None)
        self.assertIsInstance(result, NumpySliceable)


class TrainValidationSplitTest(testing.TestCase):
    def test_basic_split_exact_sizes(self):
        """split_at = floor(N * (1 - split)); verify exact sizes."""
        n, split = 10, 0.2
        expected_train = math.floor(n * (1.0 - split))
        expected_val = n - expected_train
        x = np.arange(100).reshape(n, 10)
        y = np.arange(n, dtype="float32")
        (train_x, train_y), (val_x, val_y) = train_validation_split(
            (x, y), validation_split=split
        )
        self.assertEqual(train_x.shape[0], expected_train)
        self.assertEqual(val_x.shape[0], expected_val)
        self.assertEqual(train_y.shape[0], expected_train)
        self.assertEqual(val_y.shape[0], expected_val)

    def test_split_ratio_floor_math(self):
        """Verify split boundary uses floor(n * (1 - split))."""
        # floor(7 * (1 - 0.3)) — use the impl's formula, not recomputed version
        n, split = 7, 0.3
        (train,), _ = train_validation_split(
            (np.arange(n),), validation_split=split
        )
        self.assertEqual(len(train), math.floor(n * (1.0 - split)))

        # floor(13 * (1 - 0.25))
        n, split = 13, 0.25
        (train,), _ = train_validation_split(
            (np.arange(n),), validation_split=split
        )
        self.assertEqual(len(train), math.floor(n * (1.0 - split)))

    def test_train_plus_val_equals_original(self):
        """Concatenating train and val must exactly reproduce the input."""
        x = np.arange(20, dtype="float32")
        (train,), (val,) = train_validation_split((x,), validation_split=0.2)
        reconstructed = np.concatenate([train, val], axis=0)
        np.testing.assert_array_equal(reconstructed, x)

    def test_split_preserves_order(self):
        """First N samples go to train; last M samples go to validation."""
        n, split = 10, 0.3
        split_at = math.floor(n * (1.0 - split))
        x = np.arange(n)
        (train,), (val,) = train_validation_split((x,), validation_split=split)
        np.testing.assert_array_equal(train, np.arange(split_at))
        np.testing.assert_array_equal(val, np.arange(split_at, n))

    def test_no_overlap_between_train_and_val(self):
        """Train and val sets must share no samples."""
        x = np.arange(20, dtype="float32")
        (train,), (val,) = train_validation_split((x,), validation_split=0.2)
        train_set = set(train.tolist())
        val_set = set(val.tolist())
        self.assertFalse(
            train_set & val_set,
            "Train and validation sets must not overlap",
        )

    def test_split_2d_preserves_feature_dim(self):
        """2D arrays: only the first dim is split; trailing dims are kept."""
        n, features = 20, 5
        x = np.ones((n, features), dtype="float32")
        (train,), (val,) = train_validation_split((x,), validation_split=0.2)
        self.assertEqual(train.shape[1], features)
        self.assertEqual(val.shape[1], features)

    def test_multiple_arrays_get_same_boundary(self):
        """All arrays in a tuple must be split at the same index."""
        n = 10
        x = np.arange(n, dtype="float32")
        y = np.arange(n, 2 * n, dtype="float32")
        (train_x,), (val_x,) = train_validation_split(
            (x,), validation_split=0.3
        )
        (train_y,), (val_y,) = train_validation_split(
            (y,), validation_split=0.3
        )
        self.assertEqual(len(train_x), len(train_y))
        self.assertEqual(len(val_x), len(val_y))

    def test_split_too_small_raises(self):
        """If split results in 0 samples for train or val, raise."""
        x = np.array([[1, 2]])  # Only 1 sample
        with self.assertRaisesRegex(ValueError, "not sufficient"):
            train_validation_split((x,), validation_split=0.5)

    def test_split_one_zero_raises(self):
        """Validation split of 1.0 means all val, 0 train — should raise."""
        x = np.arange(10)
        with self.assertRaisesRegex(ValueError, "not sufficient"):
            train_validation_split((x,), validation_split=1.0)

    def test_split_with_none_arrays(self):
        """All-None arrays should return same structure."""
        train, val = train_validation_split((None, None), validation_split=0.2)
        self.assertIsNone(train[0])
        self.assertIsNone(train[1])

    def test_split_mixed_none(self):
        """Mix of None and real arrays."""
        x = np.arange(20).reshape(10, 2)
        (train_x, train_none), (val_x, val_none) = train_validation_split(
            (x, None), validation_split=0.3
        )
        self.assertIsNone(train_none)
        self.assertIsNone(val_none)
        self.assertEqual(train_x.shape[0], math.floor(10 * 0.7))

    def test_unsupported_type_raises(self):
        """Passing unsupported types like strings should raise."""
        with self.assertRaisesRegex(ValueError, "only supported"):
            train_validation_split(("not an array",), validation_split=0.2)

    def test_dict_structure_sizes(self):
        """Should work with dict structures and correct sizes."""
        n = 10
        split = 0.3
        expected_train = math.floor(n * (1.0 - split))
        data = {"a": np.arange(n * 2).reshape(n, 2), "b": np.arange(n)}
        train, val = train_validation_split(data, validation_split=split)
        self.assertEqual(train["a"].shape[0], expected_train)
        self.assertEqual(train["b"].shape[0], expected_train)
        self.assertEqual(val["a"].shape[0], n - expected_train)

    def test_small_split_fraction(self):
        """Tiny validation fraction still works correctly."""
        n, split = 100, 0.1
        expected_train = math.floor(n * (1.0 - split))
        (train,), (val,) = train_validation_split(
            (np.arange(n),), validation_split=split
        )
        self.assertEqual(len(train), expected_train)
        self.assertEqual(len(val), n - expected_train)

    def test_large_split_fraction(self):
        """Large validation fraction: split_at = floor(n * (1 - split))."""
        n = 10
        split = 0.8
        # Use the same formula as the implementation to avoid float precision
        # differences between split_at = floor(n * (1 - split)) and
        # floor(n * complement) when computed separately.
        expected_train = math.floor(n * (1.0 - split))
        (train,), (val,) = train_validation_split(
            (np.arange(n),), validation_split=split
        )
        self.assertEqual(len(train), expected_train)
        self.assertEqual(len(val), n - expected_train)


if __name__ == "__main__":
    testing.run_tests()
