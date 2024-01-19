import jax
import numpy as np
import tensorflow as tf
import torch
from absl.testing import parameterized

from keras import testing
from keras.testing.test_utils import named_product
from keras.trainers.data_adapters.torch_data_loader_adapter import (
    TorchDataLoaderAdapter,
)


class TestTorchDataLoaderAdapter(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        named_product(iterator_type=["np", "tf", "jax", "torch"])
    )
    def test_basic_dataloader(self, iterator_type):
        x = torch.normal(2, 3, size=(34, 4))
        y = torch.normal(1, 3, size=(34, 2))
        base_ds = torch.utils.data.TensorDataset(x, y)
        base_dataloader = torch.utils.data.DataLoader(base_ds, batch_size=16)
        adapter = TorchDataLoaderAdapter(base_dataloader)

        self.assertEqual(adapter.num_batches, 3)
        self.assertEqual(adapter.batch_size, 16)
        self.assertEqual(adapter.has_partial_batch, True)
        self.assertEqual(adapter.partial_batch_size, 2)

        if iterator_type == "np":
            it = adapter.get_numpy_iterator()
            expected_class = np.ndarray
        elif iterator_type == "tf":
            it = adapter.get_tf_dataset()
            expected_class = tf.Tensor
        elif iterator_type == "jax":
            it = adapter.get_jax_iterator()
            expected_class = jax.Array
        elif iterator_type == "torch":
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
