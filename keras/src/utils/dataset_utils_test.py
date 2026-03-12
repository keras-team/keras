import collections
import itertools

import numpy as np
import torch
from absl.testing import parameterized
from torch.utils.data import Dataset as TorchDataset

from keras.src import backend
from keras.src.testing import test_case
from keras.src.testing.test_utils import named_product
from keras.src.utils.dataset_utils import split_dataset
from keras.src.utils.module_utils import tensorflow as tf


class MyTorchDataset(TorchDataset):
    def __init__(self, x, y=None):
        # Convert NumPy → Torch tensors if needed
        def to_tensor(v):
            if isinstance(v, torch.Tensor):
                return v
            if hasattr(v, "shape"):
                return torch.as_tensor(v, dtype=torch.float32)
            return v

        # Convert structured input recursively
        def map_structure(obj):
            if isinstance(obj, (dict, collections.OrderedDict)):
                return {k: map_structure(v) for k, v in obj.items()}
            if isinstance(obj, (tuple, list)):
                typ = type(obj)
                return typ(map_structure(v) for v in obj)
            return to_tensor(obj)

        self.x = map_structure(x)
        self.y = None if y is None else map_structure(y)

        # Infer dataset length from the first tensor in x
        def first_tensor(obj):
            if isinstance(obj, (dict, collections.OrderedDict)):
                return first_tensor(next(iter(obj.values())))
            if isinstance(obj, (tuple, list)):
                return first_tensor(obj[0])
            return obj

        self.length = len(first_tensor(self.x))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        def index_structure(obj):
            if isinstance(obj, (dict, collections.OrderedDict)):
                return obj.__class__(
                    (k, index_structure(v)) for k, v in obj.items()
                )
            if isinstance(obj, (tuple, list)):
                typ = type(obj)
                return typ(index_structure(v) for v in obj)
            return obj[idx]

        if self.y is None:
            return index_structure(self.x)
        return index_structure(self.x), index_structure(self.y)


class DatasetUtilsTest(test_case.TestCase):
    @parameterized.named_parameters(
        named_product(
            dataset_type=["list", "tuple", "tensorflow", "torch"],
            features_shape=[(2,), (100, 2), (10, 10, 2)],
            preferred_backend=[None, "tensorflow", "torch"],
        )
    )
    def test_split_dataset(
        self, dataset_type, features_shape, preferred_backend
    ):
        n_sample, left_size, right_size = 100, 0.2, 0.8
        features = np.random.sample((n_sample,) + features_shape)
        labels = np.random.sample((n_sample, 1))
        cardinality_function = (
            tf.data.Dataset.cardinality
            if (backend.backend() != "torch" and preferred_backend != "torch")
            else len
        )

        if dataset_type == "list":
            dataset = [features, labels]
        elif dataset_type == "tuple":
            dataset = (features, labels)
        elif dataset_type == "tensorflow":
            dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        elif dataset_type == "torch":
            dataset = MyTorchDataset(features, labels)
            cardinality_function = len
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        dataset_left, dataset_right = split_dataset(
            dataset,
            left_size=left_size,
            right_size=right_size,
            preferred_backend=preferred_backend,
        )
        self.assertEqual(
            int(cardinality_function(dataset_left)), int(n_sample * left_size)
        )
        self.assertEqual(
            int(cardinality_function(dataset_right)), int(n_sample * right_size)
        )
        for sample in itertools.chain(dataset_left, dataset_right):
            self.assertEqual(sample[0].shape, features_shape)
            self.assertEqual(sample[1].shape, (1,))

    @parameterized.named_parameters(
        named_product(structure_type=["tuple", "dict", "OrderedDict"])
    )
    def test_split_dataset_nested_structures(self, structure_type):
        n_sample, left_size, right_size = 100, 0.2, 0.8
        features1 = np.random.sample((n_sample, 2))
        features2 = np.random.sample((n_sample, 10, 2))
        labels = np.random.sample((n_sample, 1))

        if backend.backend() != "torch":
            create_dataset_function = tf.data.Dataset.from_tensor_slices
            cardinality_function = tf.data.Dataset.cardinality
        else:
            create_dataset_function = MyTorchDataset
            cardinality_function = len

        if structure_type == "tuple":
            dataset = create_dataset_function(((features1, features2), labels))
        if structure_type == "dict":
            dataset = create_dataset_function(
                {"y": features2, "x": features1, "labels": labels}
            )
        if structure_type == "OrderedDict":
            dataset = create_dataset_function(
                collections.OrderedDict(
                    [("y", features2), ("x", features1), ("labels", labels)]
                )
            )

        dataset_left, dataset_right = split_dataset(
            dataset, left_size=left_size, right_size=right_size
        )
        self.assertEqual(
            int(cardinality_function(dataset_left)), int(n_sample * left_size)
        )
        self.assertEqual(
            int(cardinality_function(dataset_right)), int(n_sample * right_size)
        )
        for sample in itertools.chain(dataset_left, dataset_right):
            if structure_type in ("dict", "OrderedDict"):
                x, y, labels = sample["x"], sample["y"], sample["labels"]
            elif structure_type == "tuple":
                (x, y), labels = sample
            self.assertEqual(x.shape, (2,))
            self.assertEqual(y.shape, (10, 2))
            self.assertEqual(labels.shape, (1,))
