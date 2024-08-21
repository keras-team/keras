import itertools

import numpy as np
from absl.testing import parameterized
from torch.utils.data import Dataset as TorchDataset

from keras.src.testing import test_case
from keras.src.testing.test_utils import named_product
from keras.src.utils.dataset_utils import split_dataset
from keras.src.utils.module_utils import tensorflow as tf


class MyTorchDataset(TorchDataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class DatasetUtilsTest(test_case.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        named_product(
            dataset_type=["list", "tuple", "tensorflow", "torch"],
            features_shape=[(2,), (100, 2), (10, 10, 2)],
        )
    )
    def test_split_dataset(self, dataset_type, features_shape):
        n_sample, left_size, right_size = 100, 0.2, 0.8
        features = np.random.sample((n_sample,) + features_shape)
        labels = np.random.sample((n_sample, 1))

        if dataset_type == "list":
            dataset = [features, labels]
        elif dataset_type == "tuple":
            dataset = (features, labels)
        elif dataset_type == "tensorflow":
            dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        elif dataset_type == "torch":
            dataset = MyTorchDataset(features, labels)

        dataset_left, dataset_right = split_dataset(
            dataset, left_size=left_size, right_size=right_size
        )
        self.assertEqual(
            int(dataset_left.cardinality()), int(n_sample * left_size)
        )
        self.assertEqual(
            int(dataset_right.cardinality()), int(n_sample * right_size)
        )
        for sample in itertools.chain(dataset_left, dataset_right):
            self.assertEqual(sample[0].shape, features_shape)
            self.assertEqual(sample[1].shape, (1,))

    @parameterized.named_parameters(
        named_product(structure_type=["dict", "tuple"])
    )
    def test_split_dataset_nested_structures(self, structure_type):
        n_sample, left_size, right_size = 100, 0.2, 0.8
        features1 = np.random.sample((n_sample, 2))
        features2 = np.random.sample((n_sample, 10, 2))
        labels = np.random.sample((n_sample, 1))

        if structure_type == "dict":
            dataset = tf.data.Dataset.from_tensor_slices(
                {"x1": features1, "x2": features2, "labels": labels}
            )
        elif structure_type == "tuple":
            dataset = tf.data.Dataset.from_tensor_slices(
                ((features1, features2), labels)
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
        for sample in itertools.chain(dataset_left, dataset_right):
            if structure_type == "dict":
                x1, x2, labels = sample["x1"], sample["x2"], sample["labels"]
            elif structure_type == "tuple":
                (x1, x2), labels = sample
            self.assertEqual(x1.shape, (2,))
            self.assertEqual(x2.shape, (10, 2))
            self.assertEqual(labels.shape, (1,))
