import numpy as np

from keras.testing import test_case
from keras.utils.dataset_utils import split_dataset
from keras.utils.module_utils import tensorflow as tf


class DatasetUtilsTest(test_case.TestCase):
    def test_split_dataset_list(self):
        n_sample, n_cols, n_pred, left_size, right_size = 100, 2, 1, 0.2, 0.8
        dataset = [
            np.random.sample((n_sample, n_cols)),
            np.random.sample((n_sample, n_pred)),
        ]
        dataset_left, dataset_right = split_dataset(
            dataset, left_size=left_size, right_size=right_size
        )
        self.assertEqual(
            int(dataset_left.cardinality()), int(n_sample * left_size)
        )
        self.assertEqual(
            int(dataset_right.cardinality()), int(n_sample * right_size)
        )
        self.assertEqual(
            [sample for sample in dataset_right][0][0].shape, (n_cols)
        )

        n_sample, n_cols, n_pred, left_size, right_size = 100, 2, 1, 0.2, 0.8
        dataset = [
            np.random.sample((n_sample, 100, n_cols)),
            np.random.sample((n_sample, n_pred)),
        ]
        dataset_left, dataset_right = split_dataset(
            dataset, left_size=left_size, right_size=right_size
        )
        self.assertEqual(
            int(dataset_left.cardinality()), int(n_sample * left_size)
        )
        self.assertEqual(
            int(dataset_right.cardinality()), int(n_sample * right_size)
        )
        self.assertEqual(
            [sample for sample in dataset_right][0][0].shape, (100, n_cols)
        )

        n_sample, n_cols, n_pred, left_size, right_size = 100, 2, 1, 0.2, 0.8
        dataset = [
            np.random.sample((n_sample, 10, 10, n_cols)),
            np.random.sample((n_sample, n_pred)),
        ]
        dataset_left, dataset_right = split_dataset(
            dataset, left_size=left_size, right_size=right_size
        )
        self.assertEqual(
            int(dataset_left.cardinality()), int(n_sample * left_size)
        )
        self.assertEqual(
            int(dataset_right.cardinality()), int(n_sample * right_size)
        )
        self.assertEqual(
            [sample for sample in dataset_right][0][0].shape, (10, 10, n_cols)
        )

        n_sample, n_cols, n_pred, left_size, right_size = 100, 2, 1, 0.2, 0.8
        dataset = [
            np.random.sample((n_sample, 100, 10, 30, n_cols)),
            np.random.sample((n_sample, n_pred)),
        ]
        dataset_left, dataset_right = split_dataset(
            dataset, left_size=left_size, right_size=right_size
        )
        self.assertEqual(
            int(dataset_left.cardinality()), int(n_sample * left_size)
        )
        self.assertEqual(
            int(dataset_right.cardinality()), int(n_sample * right_size)
        )
        self.assertEqual(
            [sample for sample in dataset_right][0][0].shape,
            (100, 10, 30, n_cols),
        )

    def test_split_dataset_tuple(self):
        n_sample, n_cols, n_pred, left_size, right_size = 100, 2, 1, 0.2, 0.8
        dataset = (
            np.random.sample((n_sample, n_cols)),
            np.random.sample((n_sample, n_pred)),
        )
        dataset_left, dataset_right = split_dataset(
            dataset, left_size=left_size, right_size=right_size
        )
        self.assertEqual(
            int(dataset_left.cardinality()), int(n_sample * left_size)
        )
        self.assertEqual(
            int(dataset_right.cardinality()), int(n_sample * right_size)
        )
        self.assertEqual(
            [sample for sample in dataset_right][0][0].shape, (n_cols)
        )

        n_sample, n_cols, n_pred, left_size, right_size = 100, 2, 1, 0.2, 0.8
        dataset = (
            np.random.sample((n_sample, 100, n_cols)),
            np.random.sample((n_sample, n_pred)),
        )
        dataset_left, dataset_right = split_dataset(
            dataset, left_size=left_size, right_size=right_size
        )
        self.assertEqual(
            int(dataset_left.cardinality()), int(n_sample * left_size)
        )
        self.assertEqual(
            int(dataset_right.cardinality()), int(n_sample * right_size)
        )
        self.assertEqual(
            [sample for sample in dataset_right][0][0].shape, (100, n_cols)
        )

        n_sample, n_cols, n_pred, left_size, right_size = 100, 2, 1, 0.2, 0.8
        dataset = (
            np.random.sample((n_sample, 10, 10, n_cols)),
            np.random.sample((n_sample, n_pred)),
        )
        dataset_left, dataset_right = split_dataset(
            dataset, left_size=left_size, right_size=right_size
        )
        self.assertEqual(
            int(dataset_left.cardinality()), int(n_sample * left_size)
        )
        self.assertEqual(
            int(dataset_right.cardinality()), int(n_sample * right_size)
        )
        self.assertEqual(
            [sample for sample in dataset_right][0][0].shape, (10, 10, n_cols)
        )

        n_sample, n_cols, n_pred, left_size, right_size = 100, 2, 1, 0.2, 0.8
        dataset = (
            np.random.sample((n_sample, 100, 10, 30, n_cols)),
            np.random.sample((n_sample, n_pred)),
        )
        dataset_left, dataset_right = split_dataset(
            dataset, left_size=left_size, right_size=right_size
        )
        self.assertEqual(
            int(dataset_left.cardinality()), int(n_sample * left_size)
        )
        self.assertEqual(
            int(dataset_right.cardinality()), int(n_sample * right_size)
        )
        self.assertEqual(
            [sample for sample in dataset_right][0][0].shape,
            (100, 10, 30, n_cols),
        )

    def test_split_dataset_tensorflow(self):
        n_sample, n_cols, n_pred, left_size, right_size = 100, 2, 1, 0.2, 0.8
        features, labels = (
            np.random.sample((n_sample, n_cols)),
            np.random.sample((n_sample, n_pred)),
        )
        tf_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset_left, dataset_right = split_dataset(
            tf_dataset, left_size=left_size, right_size=right_size
        )
        self.assertEqual(
            int(dataset_left.cardinality()), int(n_sample * left_size)
        )
        self.assertEqual(
            int(dataset_right.cardinality()), int(n_sample * right_size)
        )
        self.assertEqual(
            [sample for sample in dataset_right][0][0].shape, (n_cols)
        )

        n_sample, n_cols, n_pred, left_size, right_size = 100, 2, 1, 0.2, 0.8
        features, labels = (
            np.random.sample((n_sample, 100, n_cols)),
            np.random.sample((n_sample, n_pred)),
        )
        tf_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset_left, dataset_right = split_dataset(
            tf_dataset, left_size=left_size, right_size=right_size
        )
        self.assertEqual(
            int(dataset_left.cardinality()), int(n_sample * left_size)
        )
        self.assertEqual(
            int(dataset_right.cardinality()), int(n_sample * right_size)
        )
        self.assertEqual(
            [sample for sample in dataset_right][0][0].shape, (100, n_cols)
        )

        n_sample, n_cols, n_pred, left_size, right_size = 100, 2, 1, 0.2, 0.8
        features, labels = (
            np.random.sample((n_sample, 10, 10, n_cols)),
            np.random.sample((n_sample, n_pred)),
        )
        tf_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset_left, dataset_right = split_dataset(
            tf_dataset, left_size=left_size, right_size=right_size
        )
        self.assertEqual(
            int(dataset_left.cardinality()), int(n_sample * left_size)
        )
        self.assertEqual(
            int(dataset_right.cardinality()), int(n_sample * right_size)
        )
        self.assertEqual(
            [sample for sample in dataset_right][0][0].shape, (10, 10, n_cols)
        )

        n_sample, n_cols, n_pred, left_size, right_size = 100, 2, 1, 0.2, 0.8
        features, labels = (
            np.random.sample((n_sample, 100, 10, 30, n_cols)),
            np.random.sample((n_sample, n_pred)),
        )
        tf_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset_left, dataset_right = split_dataset(
            tf_dataset, left_size=left_size, right_size=right_size
        )
        self.assertEqual(
            int(dataset_left.cardinality()), int(n_sample * left_size)
        )
        self.assertEqual(
            int(dataset_right.cardinality()), int(n_sample * right_size)
        )
        self.assertEqual(
            [sample for sample in dataset_right][0][0].shape,
            (100, 10, 30, n_cols),
        )

    def test_split_dataset_torch(self):
        # sample torch dataset class
        from torch.utils.data import Dataset as torchDataset

        class Dataset(torchDataset):
            "Characterizes a dataset for PyTorch"

            def __init__(self, x, y):
                "Initialization"
                self.x = x
                self.y = y

            def __len__(self):
                "Denotes the total number of samples"
                return len(self.x)

            def __getitem__(self, index):
                "Generates one sample of data"
                return self.x[index], self.y[index]

        n_sample, n_cols, n_pred, left_size, right_size = 100, 2, 1, 0.2, 0.8
        features, labels = (
            np.random.sample((n_sample, n_cols)),
            np.random.sample((n_sample, n_pred)),
        )
        torch_dataset = Dataset(features, labels)
        dataset_left, dataset_right = split_dataset(
            torch_dataset, left_size=left_size, right_size=right_size
        )
        self.assertEqual(
            len([sample for sample in dataset_left]), int(n_sample * left_size)
        )
        self.assertEqual(
            len([sample for sample in dataset_right]),
            int(n_sample * right_size),
        )
        self.assertEqual(
            [sample for sample in dataset_right][0][0].shape, (n_cols,)
        )

        n_sample, n_cols, n_pred, left_size, right_size = 100, 2, 1, 0.2, 0.8
        features, labels = (
            np.random.sample((n_sample, 100, n_cols)),
            np.random.sample((n_sample, n_pred)),
        )
        torch_dataset = Dataset(features, labels)
        dataset_left, dataset_right = split_dataset(
            torch_dataset, left_size=left_size, right_size=right_size
        )
        self.assertEqual(
            len([sample for sample in dataset_left]), int(n_sample * left_size)
        )
        self.assertEqual(
            len([sample for sample in dataset_right]),
            int(n_sample * right_size),
        )
        self.assertEqual(
            [sample for sample in dataset_right][0][0].shape, (100, n_cols)
        )

        n_sample, n_cols, n_pred, left_size, right_size = 100, 2, 1, 0.2, 0.8
        features, labels = (
            np.random.sample((n_sample, 10, 10, n_cols)),
            np.random.sample((n_sample, n_pred)),
        )
        torch_dataset = Dataset(features, labels)
        dataset_left, dataset_right = split_dataset(
            torch_dataset, left_size=left_size, right_size=right_size
        )
        self.assertEqual(
            len([sample for sample in dataset_left]), int(n_sample * left_size)
        )
        self.assertEqual(
            len([sample for sample in dataset_right]),
            int(n_sample * right_size),
        )
        self.assertEqual(
            [sample for sample in dataset_right][0][0].shape, (10, 10, n_cols)
        )

        n_sample, n_cols, n_pred, left_size, right_size = 100, 2, 1, 0.2, 0.8
        features, labels = (
            np.random.sample((n_sample, 100, 10, 30, n_cols)),
            np.random.sample((n_sample, n_pred)),
        )
        torch_dataset = Dataset(features, labels)
        dataset_left, dataset_right = split_dataset(
            torch_dataset, left_size=left_size, right_size=right_size
        )
        self.assertEqual(
            len([sample for sample in dataset_left]), int(n_sample * left_size)
        )
        self.assertEqual(
            len([sample for sample in dataset_right]),
            int(n_sample * right_size),
        )
        self.assertEqual(
            [sample for sample in dataset_right][0][0].shape,
            (100, 10, 30, n_cols),
        )
