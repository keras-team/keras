import logging
import unittest

import numpy as np
import pytest
import torch

from keras.src import backend
from keras.src.backend.torch.distributed_backend import TorchDistributedBackend

logging.disable(logging.WARNING)


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="PyTorch-specific distributed backend tests",
)
class TestTorchDistributedBackend(unittest.TestCase):
    """Unit tests for the TorchDistributedBackend class."""

    def setUp(self):
        """Set up the test case by instantiating the backend."""
        self.backend = TorchDistributedBackend()

    def tearDown(self):
        """Re-enable logging after tests are done."""
        logging.disable(logging.NOTSET)

    def test_get_tensor_lib(self):
        """Test if the correct tensor library (torch) is returned."""
        self.assertIs(self.backend.get_tensor_lib(), torch)

    def test_convert_to_backend_tensor(self):
        """Test tensor conversion to torch.Tensor."""
        np_array = np.array([1.0, 2.0, 3.0])
        torch_tensor = self.backend.convert_to_backend_tensor(np_array)
        self.assertIsInstance(torch_tensor, torch.Tensor)
        expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch_tensor.dtype)
        torch.testing.assert_close(torch_tensor, expected)

    def test_compute_gradients_returns_zeros(self):
        """
        Test that compute_gradients returns zero gradients as a fallback.
        """
        var1 = torch.randn(3, 4, requires_grad=True)
        var2 = torch.randn(5, requires_grad=True)
        trainable_vars = [var1, var2]

        gradients = self.backend.compute_gradients(None, trainable_vars)

        self.assertEqual(len(gradients), 2)
        torch.testing.assert_close(gradients[0], torch.zeros_like(var1))
        torch.testing.assert_close(gradients[1], torch.zeros_like(var2))

    def test_apply_gradients(self):
        """Test applying gradients to torch.Tensor objects."""
        var = torch.tensor([10.0, 20.0])
        grad = torch.tensor([0.5, 1.5])
        trainable_vars = [var]
        gradients = [grad]

        self.backend.apply_gradients(
            gradients, trainable_vars, learning_rate=0.1
        )

        expected = torch.tensor([10.0 - 0.1 * 0.5, 20.0 - 0.1 * 1.5])
        torch.testing.assert_close(var, expected)

    def test_create_optimizer(self):
        """Test the creation of torch.optim optimizers."""
        adam = self.backend.create_optimizer(
            "adam", params=[torch.tensor(1.0)], lr=0.1
        )
        self.assertIsInstance(adam, torch.optim.Adam)

        sgd = self.backend.create_optimizer(
            "sgd", params=[torch.tensor(1.0)], lr=0.1
        )
        self.assertIsInstance(sgd, torch.optim.SGD)

        default = self.backend.create_optimizer(
            "unknown", params=[torch.tensor(1.0)]
        )
        self.assertIsInstance(default, torch.optim.Adam)

    def test_get_device_info_on_cpu(self):
        """Test retrieving device information in a CPU-only environment."""
        info = self.backend.get_device_info()
        self.assertEqual(info["backend"], "pytorch")
        self.assertEqual(info["devices"], ["cpu"])
        self.assertEqual(info["device_count"], 1)

    def test_is_multi_device_capable(self):
        """Test the multi-device capability check."""
        self.assertIsInstance(self.backend.is_multi_device_capable(), bool)

    def test_get_communication_ops_simulated(self):
        """
        Test the simulated communication ops for a non-distributed context.
        """
        ops = self.backend.get_communication_ops()

        device_info = self.backend.get_device_info()
        world_size = device_info.get("device_count", 1)
        if world_size == 0:
            world_size = 1

        x_reduce = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        reduced = ops["all_reduce"](x_reduce, op="sum")
        expected_reduce = x_reduce * world_size
        self.assertEqual(reduced.shape, x_reduce.shape)
        torch.testing.assert_close(reduced, expected_reduce)

        x_gather = torch.tensor([[1.0, 2.0]])
        gathered = ops["all_gather"](x_gather, axis=0)
        expected_gather = torch.cat([x_gather] * world_size, dim=0)
        self.assertEqual(gathered.shape, (world_size, 2))
        torch.testing.assert_close(gathered, expected_gather)

        scatter_data = list(range(world_size * 2))
        x_scatter = torch.tensor(scatter_data, dtype=torch.float32)
        scattered = ops["scatter"](x_scatter)
        expected_scatter = torch.tensor(scatter_data[:2], dtype=torch.float32)
        self.assertEqual(scattered.shape, (2,))
        torch.testing.assert_close(scattered, expected_scatter)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
