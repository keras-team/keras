"""Tests for Dataset Utils"""

import os
import shutil

import numpy as np
import tensorflow.compat.v2 as tf

from keras.testing_infra import test_utils
from keras.utils import dataset_utils


@test_utils.run_v2_only
class SplitDatasetTest(tf.test.TestCase):
    def test_numpy_array(self):
        dataset = np.ones(shape=(200, 32))
        res = dataset_utils.split_dataset(
            dataset, left_size=0.8, right_size=0.2
        )

        self.assertLen(res, 2)
        left_split, right_split = res

        self.assertIsInstance(left_split, tf.data.Dataset)
        self.assertIsInstance(right_split, tf.data.Dataset)

        self.assertLen(left_split, 160)
        self.assertLen(right_split, 40)

        self.assertAllEqual(dataset[:160], list(left_split))
        self.assertAllEqual(dataset[-40:], list(right_split))

    def test_list_of_numpy_arrays(self):
        # test with list of np arrays with same shapes
        dataset = [np.ones(shape=(200, 32)), np.zeros(shape=(200, 32))]
        res = dataset_utils.split_dataset(dataset, left_size=4)

        self.assertLen(res, 2)
        left_split, right_split = res

        self.assertIsInstance(left_split, tf.data.Dataset)
        self.assertIsInstance(right_split, tf.data.Dataset)

        self.assertEqual(np.array(list(left_split)).shape, (4, 2, 32))
        self.assertEqual(np.array(list(right_split)).shape, (196, 2, 32))

        # test with different shapes
        dataset = [np.ones(shape=(5, 3)), np.ones(shape=(5,))]
        left_split, right_split = dataset_utils.split_dataset(
            dataset, left_size=0.3
        )

        self.assertEqual(np.array(list(left_split), dtype=object).shape, (2, 2))
        self.assertEqual(
            np.array(list(right_split), dtype=object).shape, (3, 2)
        )

        self.assertEqual(
            np.array(list(left_split)[0], dtype=object).shape, (2,)
        )
        self.assertEqual(np.array(list(left_split)[0][0]).shape, (3,))
        self.assertEqual(np.array(list(left_split)[0][1]).shape, ())

        self.assertEqual(
            np.array(list(right_split)[0], dtype=object).shape, (2,)
        )
        self.assertEqual(np.array(list(right_split)[0][0]).shape, (3,))
        self.assertEqual(np.array(list(right_split)[0][1]).shape, ())

    def test_dataset_with_invalid_shape(self):
        with self.assertRaisesRegex(
            ValueError,
            "Received a list of NumPy arrays with different lengths",
        ):
            dataset = [np.ones(shape=(200, 32)), np.zeros(shape=(100, 32))]
            dataset_utils.split_dataset(dataset, left_size=4)

        with self.assertRaisesRegex(
            ValueError,
            "Received a tuple of NumPy arrays with different lengths",
        ):
            dataset = (np.ones(shape=(200, 32)), np.zeros(shape=(201, 32)))
            dataset_utils.split_dataset(dataset, left_size=4)

    def test_tuple_of_numpy_arrays(self):
        dataset = (np.random.rand(4, 3), np.random.rand(4, 3))
        left_split, right_split = dataset_utils.split_dataset(
            dataset, left_size=2
        )

        self.assertIsInstance(left_split, tf.data.Dataset)
        self.assertIsInstance(right_split, tf.data.Dataset)

        self.assertEqual(len(left_split), 2)
        self.assertEqual(len(right_split), 2)

        self.assertEqual(np.array(list(left_split)[0]).shape, (2, 3))
        self.assertEqual(np.array(list(left_split)[1]).shape, (2, 3))

        # test with fractional size
        dataset = (np.random.rand(5, 32, 32), np.random.rand(5, 32, 32))
        left_split, right_split = dataset_utils.split_dataset(
            dataset, right_size=0.4
        )
        self.assertIsInstance(left_split, tf.data.Dataset)
        self.assertIsInstance(right_split, tf.data.Dataset)

        self.assertEqual(np.array(list(left_split)).shape, (3, 2, 32, 32))
        self.assertEqual(np.array(list(right_split)).shape, (2, 2, 32, 32))

        self.assertEqual(np.array(list(left_split))[0].shape, (2, 32, 32))
        self.assertEqual(np.array(list(left_split))[1].shape, (2, 32, 32))

        self.assertEqual(np.array(list(right_split))[0].shape, (2, 32, 32))
        self.assertEqual(np.array(list(right_split))[1].shape, (2, 32, 32))

        # test with tuple of np arrays with different shapes
        dataset = (
            np.random.rand(5, 32, 32),
            np.random.rand(
                5,
            ),
        )
        left_split, right_split = dataset_utils.split_dataset(
            dataset, left_size=2, right_size=3
        )
        self.assertIsInstance(left_split, tf.data.Dataset)
        self.assertIsInstance(right_split, tf.data.Dataset)

        self.assertEqual(np.array(list(left_split), dtype=object).shape, (2, 2))
        self.assertEqual(
            np.array(list(right_split), dtype=object).shape, (3, 2)
        )

        self.assertEqual(
            np.array(list(left_split)[0], dtype=object).shape, (2,)
        )
        self.assertEqual(np.array(list(left_split)[0][0]).shape, (32, 32))
        self.assertEqual(np.array(list(left_split)[0][1]).shape, ())

        self.assertEqual(
            np.array(list(right_split)[0], dtype=object).shape, (2,)
        )
        self.assertEqual(np.array(list(right_split)[0][0]).shape, (32, 32))
        self.assertEqual(np.array(list(right_split)[0][1]).shape, ())

    def test_batched_tf_dataset_of_vectors(self):
        vectors = np.ones(shape=(100, 32, 32, 1))
        dataset = tf.data.Dataset.from_tensor_slices(vectors)
        dataset = dataset.batch(10)
        left_split, right_split = dataset_utils.split_dataset(
            dataset, left_size=2
        )

        # Ensure that the splits are batched
        self.assertEqual(len(list(right_split)), 10)

        left_split, right_split = left_split.unbatch(), right_split.unbatch()
        self.assertAllEqual(np.array(list(left_split)).shape, (2, 32, 32, 1))
        self.assertAllEqual(np.array(list(right_split)).shape, (98, 32, 32, 1))
        dataset = dataset.unbatch()
        self.assertAllEqual(list(dataset), list(left_split) + list(right_split))

    def test_batched_tf_dataset_of_tuple_of_vectors(self):
        tuple_of_vectors = (
            np.random.rand(10, 32, 32),
            np.random.rand(10, 32, 32),
        )
        dataset = tf.data.Dataset.from_tensor_slices(tuple_of_vectors)
        dataset = dataset.batch(2)
        left_split, right_split = dataset_utils.split_dataset(
            dataset, left_size=4
        )

        # Ensure that the splits are batched
        self.assertEqual(np.array(list(right_split)).shape, (3, 2, 2, 32, 32))
        self.assertEqual(np.array(list(left_split)).shape, (2, 2, 2, 32, 32))

        left_split, right_split = left_split.unbatch(), right_split.unbatch()
        self.assertAllEqual(np.array(list(left_split)).shape, (4, 2, 32, 32))
        self.assertAllEqual(np.array(list(right_split)).shape, (6, 2, 32, 32))

        dataset = dataset.unbatch()
        self.assertAllEqual(list(dataset), list(left_split) + list(right_split))

    def test_batched_tf_dataset_of_dict_of_vectors(self):
        dict_samples = {"X": np.random.rand(10, 3), "Y": np.random.rand(10, 3)}
        dataset = tf.data.Dataset.from_tensor_slices(dict_samples)
        dataset = dataset.batch(2)
        left_split, right_split = dataset_utils.split_dataset(
            dataset, left_size=2
        )

        self.assertAllEqual(np.array(list(left_split)).shape, (1,))
        self.assertAllEqual(np.array(list(right_split)).shape, (4,))

        left_split, right_split = left_split.unbatch(), right_split.unbatch()
        self.assertEqual(len(list(left_split)), 2)
        self.assertEqual(len(list(right_split)), 8)
        for i in range(10):
            if i < 2:
                self.assertEqual(
                    list(left_split)[i], list(dataset.unbatch())[i]
                )
            else:
                self.assertEqual(
                    list(right_split)[i - 2], list(dataset.unbatch())[i]
                )

        # test with dict of np arrays with different shapes
        dict_samples = {
            "images": np.random.rand(10, 16, 16, 3),
            "labels": np.random.rand(
                10,
            ),
        }
        dataset = tf.data.Dataset.from_tensor_slices(dict_samples)
        dataset = dataset.batch(1)
        left_split, right_split = dataset_utils.split_dataset(
            dataset, right_size=0.3
        )

        self.assertAllEqual(np.array(list(left_split)).shape, (7,))
        self.assertAllEqual(np.array(list(right_split)).shape, (3,))

        dataset = dataset.unbatch()
        left_split, right_split = left_split.unbatch(), right_split.unbatch()
        self.assertEqual(len(list(left_split)), 7)
        self.assertEqual(len(list(right_split)), 3)
        for i in range(10):
            if i < 7:
                self.assertEqual(list(left_split)[i], list(dataset)[i])
            else:
                self.assertEqual(list(right_split)[i - 7], list(dataset)[i])

    def test_unbatched_tf_dataset_of_vectors(self):
        vectors = np.ones(shape=(100, 16, 16, 3))
        dataset = tf.data.Dataset.from_tensor_slices(vectors)

        left_split, right_split = dataset_utils.split_dataset(
            dataset, left_size=0.25
        )

        self.assertAllEqual(np.array(list(left_split)).shape, (25, 16, 16, 3))
        self.assertAllEqual(np.array(list(right_split)).shape, (75, 16, 16, 3))

        self.assertAllEqual(list(dataset), list(left_split) + list(right_split))

        dataset = [np.random.rand(10, 3, 3) for _ in range(5)]
        dataset = tf.data.Dataset.from_tensor_slices(dataset)

        left_split, right_split = dataset_utils.split_dataset(
            dataset, left_size=2
        )
        self.assertAllEqual(list(dataset), list(left_split) + list(right_split))

    def test_unbatched_tf_dataset_of_tuple_of_vectors(self):
        # test with tuple of np arrays with same shape
        X, Y = (np.random.rand(10, 32, 32, 1), np.random.rand(10, 32, 32, 1))
        dataset = tf.data.Dataset.from_tensor_slices((X, Y))

        left_split, right_split = dataset_utils.split_dataset(
            dataset, left_size=5
        )

        self.assertEqual(len(list(left_split)), 5)
        self.assertEqual(len(list(right_split)), 5)
        self.assertAllEqual(list(dataset), list(left_split) + list(right_split))

        # test with tuple of np arrays with different shapes
        X, Y = (
            np.random.rand(5, 3, 3),
            np.random.rand(
                5,
            ),
        )
        dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        left_split, right_split = dataset_utils.split_dataset(
            dataset, left_size=0.5
        )

        self.assertEqual(len(list(left_split)), 2)
        self.assertEqual(len(list(right_split)), 3)
        self.assertEqual(np.array(list(left_split)[0][0]).shape, (3, 3))
        self.assertEqual(np.array(list(left_split)[0][1]).shape, ())

    def test_unbatched_tf_dataset_of_dict_of_vectors(self):
        # test with dict of np arrays of same shape
        dict_samples = {"X": np.random.rand(10, 2), "Y": np.random.rand(10, 2)}
        dataset = tf.data.Dataset.from_tensor_slices(dict_samples)
        left_split, right_split = dataset_utils.split_dataset(
            dataset, left_size=2
        )
        self.assertEqual(len(list(left_split)), 2)
        self.assertEqual(len(list(right_split)), 8)
        for i in range(10):
            if i < 2:
                self.assertEqual(list(left_split)[i], list(dataset)[i])
            else:
                self.assertEqual(list(right_split)[i - 2], list(dataset)[i])

        # test with dict of np arrays with different shapes
        dict_samples = {
            "images": np.random.rand(10, 16, 16, 3),
            "labels": np.random.rand(
                10,
            ),
        }
        dataset = tf.data.Dataset.from_tensor_slices(dict_samples)
        left_split, right_split = dataset_utils.split_dataset(
            dataset, left_size=0.3
        )
        self.assertEqual(len(list(left_split)), 3)
        self.assertEqual(len(list(right_split)), 7)
        for i in range(10):
            if i < 3:
                self.assertEqual(list(left_split)[i], list(dataset)[i])
            else:
                self.assertEqual(list(right_split)[i - 3], list(dataset)[i])

        # test with dict of text arrays
        txt_feature = ["abb", "bb", "cc", "d", "e", "f", "g", "h", "i", "j"]
        dict_samples = {
            "txt_feature": txt_feature,
            "label": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
        dataset = tf.data.Dataset.from_tensor_slices(dict_samples)
        left_split, right_split = dataset_utils.split_dataset(
            dataset, left_size=0.45, right_size=0.55
        )
        self.assertEqual(len(list(left_split)), 4)
        self.assertEqual(len(list(right_split)), 6)
        for i in range(10):
            if i < 4:
                self.assertEqual(list(left_split)[i], list(dataset)[i])
            else:
                self.assertEqual(list(right_split)[i - 4], list(dataset)[i])

    def test_list_dataset(self):
        dataset = [np.ones(shape=(10, 10, 10)) for _ in range(10)]
        left_split, right_split = dataset_utils.split_dataset(
            dataset, left_size=5, right_size=5
        )
        self.assertEqual(len(left_split), len(right_split))
        self.assertIsInstance(left_split, tf.data.Dataset)
        self.assertIsInstance(left_split, tf.data.Dataset)

        dataset = [np.ones(shape=(10, 10, 10)) for _ in range(10)]
        left_split, right_split = dataset_utils.split_dataset(
            dataset, left_size=0.6, right_size=0.4
        )
        self.assertEqual(len(left_split), 6)
        self.assertEqual(len(right_split), 4)

    def test_invalid_dataset(self):
        with self.assertRaisesRegex(
            TypeError,
            "The `dataset` argument must be either a `tf.data.Dataset` "
            "object or a list/tuple of arrays.",
        ):
            dataset_utils.split_dataset(dataset=None, left_size=5)
        with self.assertRaisesRegex(
            TypeError,
            "The `dataset` argument must be either a `tf.data.Dataset` "
            "object or a list/tuple of arrays.",
        ):
            dataset_utils.split_dataset(dataset=1, left_size=5)
        with self.assertRaisesRegex(
            TypeError,
            "The `dataset` argument must be either a `tf.data.Dataset` "
            "object or a list/tuple of arrays.",
        ):
            dataset_utils.split_dataset(dataset=float(1.2), left_size=5)
        with self.assertRaisesRegex(
            TypeError,
            "The `dataset` argument must be either a `tf.data.Dataset` "
            "object or a list/tuple of arrays.",
        ):
            dataset_utils.split_dataset(dataset=dict({}), left_size=5)
        with self.assertRaisesRegex(
            TypeError,
            "The `dataset` argument must be either a `tf.data.Dataset` "
            "object or a list/tuple of arrays.",
        ):
            dataset_utils.split_dataset(dataset=float("INF"), left_size=5)

    def test_valid_left_and_right_sizes(self):
        dataset = np.array([1, 2, 3])
        splitted_dataset = dataset_utils.split_dataset(dataset, 1, 2)
        self.assertLen(splitted_dataset, 2)
        left_split, right_split = splitted_dataset
        self.assertEqual(len(left_split), 1)
        self.assertEqual(len(right_split), 2)
        self.assertEqual(list(left_split), [1])
        self.assertEqual(list(right_split), [2, 3])

        dataset = np.ones(shape=(200, 32))
        res = dataset_utils.split_dataset(dataset, left_size=150, right_size=50)
        self.assertLen(res, 2)
        self.assertIsInstance(res[0], tf.data.Dataset)
        self.assertIsInstance(res[1], tf.data.Dataset)

        self.assertLen(res[0], 150)
        self.assertLen(res[1], 50)

        dataset = np.ones(shape=(200, 32))
        res = dataset_utils.split_dataset(dataset, left_size=120)
        self.assertLen(res, 2)
        self.assertIsInstance(res[0], tf.data.Dataset)
        self.assertIsInstance(res[1], tf.data.Dataset)

        self.assertLen(res[0], 120)
        self.assertLen(res[1], 80)

        dataset = np.ones(shape=(10000, 16))
        res = dataset_utils.split_dataset(dataset, right_size=20)
        self.assertLen(res, 2)
        self.assertIsInstance(res[0], tf.data.Dataset)
        self.assertIsInstance(res[1], tf.data.Dataset)

        self.assertLen(res[0], 9980)
        self.assertLen(res[1], 20)

        dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        splitted_dataset = dataset_utils.split_dataset(
            dataset, left_size=0.1, right_size=0.9
        )
        self.assertLen(splitted_dataset, 2)
        left_split, right_split = splitted_dataset
        self.assertEqual(len(left_split), 1)
        self.assertEqual(len(right_split), 9)
        self.assertEqual(list(left_split), [1])
        self.assertEqual(list(right_split), [2, 3, 4, 5, 6, 7, 8, 9, 10])

        dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        splitted_dataset = dataset_utils.split_dataset(
            dataset, left_size=2, right_size=5
        )
        self.assertLen(splitted_dataset, 2)
        left_split, right_split = splitted_dataset
        self.assertEqual(len(left_split), 2)
        self.assertEqual(len(right_split), 5)
        self.assertEqual(list(left_split), [1, 2])
        self.assertEqual(list(right_split), [6, 7, 8, 9, 10])

    def test_float_left_and_right_sizes(self):
        X = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        dataset = tf.data.Dataset.from_tensor_slices(X)
        left_split, right_split = dataset_utils.split_dataset(
            dataset, left_size=0.8, right_size=0.2
        )
        self.assertEqual(len(left_split), 2)
        self.assertEqual(len(right_split), 1)

    def test_invalid_float_left_and_right_sizes(self):
        expected_regex = (
            r"^(.*?(\bleft_size\b).*?(\bshould be\b)"
            r".*?(\bwithin the range\b).*?(\b0\b).*?(\b1\b))"
        )
        with self.assertRaisesRegexp(ValueError, expected_regex):
            dataset = [
                np.ones(shape=(200, 32, 32)),
                np.zeros(shape=(200, 32, 32)),
            ]
            dataset_utils.split_dataset(dataset, left_size=1.5, right_size=0.2)

        expected_regex = (
            r"^(.*?(\bright_size\b).*?(\bshould be\b)"
            r".*?(\bwithin the range\b).*?(\b0\b).*?(\b1\b))"
        )
        with self.assertRaisesRegex(ValueError, expected_regex):
            dataset = [np.ones(shape=(200, 32)), np.zeros(shape=(200, 32))]
            dataset_utils.split_dataset(dataset, left_size=0.8, right_size=-0.8)

    def test_None_and_zero_left_and_right_size(self):
        expected_regex = (
            r"^.*?(\bleft_size\b).*?(\bright_size\b).*?(\bmust "
            r"be specified\b).*?(\bReceived: left_size=None and"
            r" right_size=None\b)"
        )

        with self.assertRaisesRegex(ValueError, expected_regex):
            dataset_utils.split_dataset(
                dataset=np.array([1, 2, 3]), left_size=None
            )
        with self.assertRaisesRegex(ValueError, expected_regex):
            dataset_utils.split_dataset(
                np.array([1, 2, 3]), left_size=None, right_size=None
            )

        expected_regex = (
            r"^.*?(\bleft_size\b).*?(\bshould be\b)"
            r".*?(\bpositive\b).*?(\bsmaller than 3\b)"
        )
        with self.assertRaisesRegex(ValueError, expected_regex):
            dataset_utils.split_dataset(np.array([1, 2, 3]), left_size=3)

        expected_regex = (
            "Both `left_size` and `right_size` are zero. "
            "At least one of the split sizes must be non-zero."
        )
        with self.assertRaisesRegex(ValueError, expected_regex):
            dataset_utils.split_dataset(
                np.array([1, 2, 3]), left_size=0, right_size=0
            )

    def test_invalid_left_and_right_size_types(self):
        expected_regex = (
            r"^.*?(\bInvalid `left_size` and `right_size` Types"
            r"\b).*?(\bExpected: integer or float or None\b)"
        )
        with self.assertRaisesRegex(TypeError, expected_regex):
            dataset_utils.split_dataset(
                np.array([1, 2, 3]), left_size="1", right_size="1"
            )

        expected_regex = r"^.*?(\bInvalid `right_size` Type\b)"
        with self.assertRaisesRegex(TypeError, expected_regex):
            dataset_utils.split_dataset(
                np.array([1, 2, 3]), left_size=0, right_size="1"
            )

        expected_regex = r"^.*?(\bInvalid `left_size` Type\b)"
        with self.assertRaisesRegex(TypeError, expected_regex):
            dataset_utils.split_dataset(
                np.array([1, 2, 3]), left_size="100", right_size=None
            )

        expected_regex = r"^.*?(\bInvalid `right_size` Type\b)"
        with self.assertRaisesRegex(TypeError, expected_regex):
            dataset_utils.split_dataset(np.array([1, 2, 3]), right_size="1")

        expected_regex = r"^.*?(\bInvalid `right_size` Type\b)"
        with self.assertRaisesRegex(TypeError, expected_regex):
            dataset_utils.split_dataset(
                np.array([1, 2, 3]), left_size=0.5, right_size="1"
            )

    def test_end_to_end(self):
        x_train = np.random.random((10000, 28, 28))
        y_train = np.random.randint(0, 10, size=(10000,))

        left_split, right_split = dataset_utils.split_dataset(
            (x_train, y_train), left_size=0.8
        )

        self.assertIsInstance(left_split, tf.data.Dataset)
        self.assertIsInstance(right_split, tf.data.Dataset)

        self.assertEqual(len(left_split), 8000)
        self.assertEqual(len(right_split), 2000)


@test_utils.run_v2_only
class IndexDirectoryStructureTest(tf.test.TestCase):
    def test_explicit_labels_and_unnested_files(self):

        # Get a unique temp directory
        temp_dir = os.path.join(
            self.get_temp_dir(), str(np.random.randint(1e6))
        )
        os.mkdir(temp_dir)
        self.addCleanup(shutil.rmtree, temp_dir)

        # Number of temp files, each of which
        # will have its own explicit label
        num_files = 10

        explicit_labels = np.random.randint(0, 10, size=num_files).tolist()

        # Save empty text files to root of temp directory
        # (content is not important, only location)
        for i in range(len(explicit_labels)):
            with open(os.path.join(temp_dir, f"file{i}.txt"), "w"):
                pass

        file_paths, labels, class_names = dataset_utils.index_directory(
            temp_dir, labels=explicit_labels, formats=".txt"
        )

        # Files are found at the root of the temp directory, when
        # `labels` are passed explicitly to `index_directory` and
        # the number of returned and passed labels match
        self.assertLen(file_paths, num_files)
        self.assertLen(labels, num_files)

        # Class names are returned as a sorted list
        expected_class_names = sorted(set(explicit_labels))
        self.assertEqual(expected_class_names, class_names)


if __name__ == "__main__":
    tf.test.main()
