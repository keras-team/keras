import math

import numpy as np
import tensorflow as tf
import torch
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.testing.test_utils import named_product
from keras.src.trainers.data_adapters.torch_data_loader_adapter import (
    TorchDataLoaderAdapter,
)


class TestTorchDataLoaderAdapter(testing.TestCase):
    def test_basic_dataloader(self):
        x = torch.normal(2, 3, size=(34, 4))
        y = torch.normal(1, 3, size=(34, 2))
        ds = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=16)
        adapter = TorchDataLoaderAdapter(dataloader)

        self.assertEqual(adapter.num_batches, 3)
        self.assertEqual(adapter.batch_size, 16)
        self.assertEqual(adapter.has_partial_batch, True)
        self.assertEqual(adapter.partial_batch_size, 2)

        if backend.backend() == "numpy":
            it = adapter.get_numpy_iterator()
            expected_class = np.ndarray
        elif backend.backend() == "tensorflow":
            it = adapter.get_tf_dataset()
            expected_class = tf.Tensor
        elif backend.backend() == "jax":
            it = adapter.get_jax_iterator()
            expected_class = np.ndarray
        elif backend.backend() == "torch":
            it = adapter.get_torch_dataloader()
            expected_class = torch.Tensor

        for i, batch in enumerate(it):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, expected_class)
            self.assertIsInstance(by, expected_class)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertContainsExactSubsequence(str(bx.dtype), "float32")
            if i < 2:
                self.assertEqual(bx.shape, (16, 4))
                self.assertEqual(by.shape, (16, 2))
            else:
                self.assertEqual(bx.shape, (2, 4))
                self.assertEqual(by.shape, (2, 2))

    @parameterized.named_parameters(
        named_product(batch_size=[None, 3], implements_len=[True, False])
    )
    def test_dataloader_iterable_dataset(self, batch_size, implements_len):
        class TestIterableDataset(torch.utils.data.IterableDataset):
            def __init__(self):
                self.x = torch.normal(2, 3, size=(16, 4))
                self.y = torch.normal(1, 3, size=(16, 2))

            def __iter__(self):
                for _ in range(10):
                    yield (self.x, self.y)

        class TestIterableDatasetWithLen(TestIterableDataset):
            def __len__(self):
                return 10

        ds = (
            TestIterableDatasetWithLen()
            if implements_len
            else TestIterableDataset()
        )
        dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
        adapter = TorchDataLoaderAdapter(dataloader)

        if implements_len and batch_size:
            self.assertEqual(adapter.num_batches, math.ceil(10 / batch_size))
            self.assertEqual(adapter.batch_size, batch_size)
            self.assertEqual(adapter.has_partial_batch, True)
            self.assertEqual(adapter.partial_batch_size, 10 % batch_size)
        elif implements_len:
            self.assertEqual(adapter.num_batches, 10)
            self.assertEqual(adapter.batch_size, None)
            self.assertEqual(adapter.has_partial_batch, None)
            self.assertEqual(adapter.partial_batch_size, None)
        else:
            self.assertIsNone(adapter.num_batches)
            self.assertEqual(adapter.batch_size, batch_size)
            self.assertIsNone(adapter.has_partial_batch)
            self.assertIsNone(adapter.partial_batch_size)

        if backend.backend() == "numpy":
            it = adapter.get_numpy_iterator()
            expected_class = np.ndarray
        elif backend.backend() == "tensorflow":
            it = adapter.get_tf_dataset()
            expected_class = tf.Tensor
        elif backend.backend() == "jax":
            it = adapter.get_jax_iterator()
            expected_class = np.ndarray
        elif backend.backend() == "torch":
            it = adapter.get_torch_dataloader()
            expected_class = torch.Tensor

        batch_count = 0
        for i, batch in enumerate(it):
            batch_count += 1
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, expected_class)
            self.assertIsInstance(by, expected_class)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertContainsExactSubsequence(str(bx.dtype), "float32")
            if batch_size:
                if i < 3:
                    self.assertEqual(bx.shape, (batch_size, 16, 4))
                    self.assertEqual(by.shape, (batch_size, 16, 2))
                else:
                    self.assertEqual(bx.shape, (10 % batch_size, 16, 4))
                    self.assertEqual(by.shape, (10 % batch_size, 16, 2))
            else:
                self.assertEqual(bx.shape, (16, 4))
                self.assertEqual(by.shape, (16, 2))

        if batch_size:
            self.assertEqual(batch_count, math.ceil(10 / batch_size))
        else:
            self.assertEqual(batch_count, 10)

    def test_with_different_shapes(self):
        x = (
            [np.ones([4], "float32")] * 16
            + [np.ones([5], "float32")] * 16
            + [np.ones([6], "float32")] * 2
        )
        y = np.ones((34, 2), "float32")
        ds = torch.utils.data.StackDataset(x, y)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=16)
        adapter = TorchDataLoaderAdapter(dataloader)

        self.assertEqual(adapter.num_batches, 3)
        self.assertEqual(adapter.batch_size, 16)
        self.assertEqual(adapter.has_partial_batch, True)
        self.assertEqual(adapter.partial_batch_size, 2)

        if backend.backend() == "numpy":
            it = adapter.get_numpy_iterator()
        elif backend.backend() == "tensorflow":
            it = adapter.get_tf_dataset()
        elif backend.backend() == "jax":
            it = adapter.get_jax_iterator()
        elif backend.backend() == "torch":
            it = adapter.get_torch_dataloader()

        for i, batch in enumerate(it):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertEqual(bx.dtype, by.dtype)
            self.assertContainsExactSubsequence(str(bx.dtype), "float32")
            if i == 0:
                self.assertEqual(bx.shape, (16, 4))
                self.assertEqual(by.shape, (16, 2))
            elif i == 1:
                self.assertEqual(bx.shape, (16, 5))
                self.assertEqual(by.shape, (16, 2))
            else:
                self.assertEqual(bx.shape, (2, 6))
                self.assertEqual(by.shape, (2, 2))

    @parameterized.named_parameters(named_product(num_workers=[0, 2]))
    def test_builtin_prefetch(self, num_workers):
        x = torch.normal(2, 3, size=(34, 4))
        y = torch.normal(1, 3, size=(34, 2))
        ds = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(
            ds, batch_size=16, num_workers=num_workers
        )
        adapter = TorchDataLoaderAdapter(dataloader)
        if num_workers > 0:
            self.assertTrue(adapter.builtin_prefetch)
        else:
            self.assertFalse(adapter.builtin_prefetch)
