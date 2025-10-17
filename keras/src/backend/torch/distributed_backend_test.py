import pytest
import torch

from keras.src import backend
from keras.src.backend import distributed_backend


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="Torch Backend specific test",
)
class TestPytorchDistributedFunctions:
    """Unit tests for the PyTorch distributed backend standalone functions."""

    def test_get_device_info(self):
        """Test retrieving device information from the PyTorch backend."""
        info = distributed_backend.get_device_info()
        assert info["backend"] == "torch (CPU)"
        assert isinstance(info["devices"], list)
        assert isinstance(info["device_count"], int)
        assert info["device_count"] > 0
        assert len(info["devices"]) == info["device_count"]
        if torch.cuda.is_available():
            assert info["device_count"] == torch.cuda.device_count()
        else:
            assert info["device_count"] == 1
            assert info["devices"] == ["cpu"]

    def test_is_multi_device_capable(self):
        """Test the boolean check for multi-device capability."""
        assert isinstance(distributed_backend.is_multi_device_capable(), bool)

    def test_communication_ops_simulation_logic(self):
        """Test the simulated communication ops in a single-device context."""
        comm_ops = distributed_backend.get_communication_ops()
        device_info = distributed_backend.get_device_info()
        world_size = device_info.get("device_count", 1)

        # Test all_reduce
        x_reduce = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        reduced = comm_ops["all_reduce"](x_reduce, op="sum")
        expected_reduce = (
            x_reduce * float(world_size) if world_size > 1 else x_reduce
        )
        torch.testing.assert_close(reduced, expected_reduce)

        # Test all_gather
        x_gather = torch.tensor([[1.0, 2.0]])
        gathered = comm_ops["all_gather"](x_gather, axis=0)
        expected_gather = torch.cat([x_gather] * world_size, dim=0)
        torch.testing.assert_close(gathered, expected_gather)
