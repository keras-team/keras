import numpy as np
import pytest
import tensorflow as tf

from keras import backend
from keras import testing
from keras.trainers.data_adapters.torch_data_loader_adapter import (
    TorchDataLoaderAdapter,
)


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="Backend does not support TorchDataLoaderAdapter.",
)
class TestTorchDataLoaderAdapter(testing.TestCase):
    def test_basic_dataloader(self):
        import torch
        from torch.utils.data import DataLoader
        from torch.utils.data import TensorDataset

        x = torch.normal(2, 3, size=(34, 4))
        y = torch.normal(1, 3, size=(34, 2))
        base_ds = TensorDataset(x, y)
        base_dataloader = DataLoader(base_ds, batch_size=16)
        adapter = TorchDataLoaderAdapter(base_dataloader)

        self.assertEqual(adapter.num_batches, 3)
        self.assertEqual(adapter.batch_size, 16)
        self.assertEqual(adapter.has_partial_batch, True)
        self.assertEqual(adapter.partial_batch_size, 2)

        gen = adapter.get_numpy_iterator()
        for i, batch in enumerate(gen):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, np.ndarray)
            self.assertIsInstance(by, np.ndarray)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.dtype, "float32")
            if i < 2:
                self.assertEqual(bx.shape, (16, 4))
                self.assertEqual(by.shape, (16, 2))
            else:
                self.assertEqual(bx.shape, (2, 4))
                self.assertEqual(by.shape, (2, 2))

        ds = adapter.get_torch_dataloader()
        for i, batch in enumerate(ds):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, torch.Tensor)
            self.assertIsInstance(by, torch.Tensor)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.dtype, torch.float32)
            if i < 2:
                self.assertEqual(tuple(bx.shape), (16, 4))
                self.assertEqual(tuple(by.shape), (16, 2))
            else:
                self.assertEqual(tuple(bx.shape), (2, 4))
                self.assertEqual(tuple(by.shape), (2, 2))

        ds = adapter.get_tf_dataset()
        for i, batch in enumerate(ds):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, tf.Tensor)
            self.assertIsInstance(by, tf.Tensor)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.dtype, tf.float32)
            if i < 2:
                self.assertEqual(tuple(bx.shape), (16, 4))
                self.assertEqual(tuple(by.shape), (16, 2))
            else:
                self.assertEqual(tuple(bx.shape), (2, 4))
                self.assertEqual(tuple(by.shape), (2, 2))
