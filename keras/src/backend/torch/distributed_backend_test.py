import pytest
import torch

from keras.src import backend
from keras.src.backend import distributed_backend


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="Jax Backend specific test",
)
class TestPytorchDistributedFunctions:
    """Unit tests for the PyTorch distributed backend standalone functions."""

    def test_compute_gradients_computes_correctly(self):
        """Test that compute_gradients returns correct gradients."""
        w = torch.tensor([2.0, 3.0], requires_grad=True)
        b = torch.tensor(1.0, requires_grad=True)
        x = torch.tensor([4.0, 5.0])
        y_true = torch.tensor(25.0)

        # loss = (w.x + b - y_true)^2 = ((2*4 + 3*5 + 1) - 25)^2 = (24-25)^2 = 1
        y_pred = torch.dot(w, x) + b
        loss = (y_pred - y_true) ** 2

        trainable_vars = [w, b]
        gradients = distributed_backend.compute_gradients(loss, trainable_vars)

        # d_loss/d_w = 2*(y_pred - y_true)*x = 2*(-1)*[4, 5] = [-8, -10]
        # d_loss/d_b = 2*(y_pred - y_true)*1 = 2*(-1)*1 = -2
        expected_grad_w = torch.tensor([-8.0, -10.0])
        expected_grad_b = torch.tensor(-2.0)

        assert len(gradients) == 2
        torch.testing.assert_close(gradients[0], expected_grad_w)
        torch.testing.assert_close(gradients[1], expected_grad_b)

    def test_apply_gradients(self):
        """Test the application of gradients to PyTorch tensors."""
        var1 = torch.tensor([1.0, 2.0], requires_grad=True)
        var2 = torch.tensor(5.0, requires_grad=True)
        trainable_vars = [var1, var2]
        grad1 = torch.tensor([0.1, 0.2])
        grad2 = torch.tensor(0.5)
        gradients = [grad1, grad2]
        learning_rate = 0.1

        original_var1 = var1.clone()
        original_var2 = var2.clone()

        updated_vars = distributed_backend.apply_gradients(
            gradients, trainable_vars, learning_rate
        )

        assert updated_vars[0] is var1
        assert updated_vars[1] is var2

        expected_var1 = original_var1 - (grad1 * learning_rate)
        expected_var2 = original_var2 - (grad2 * learning_rate)
        torch.testing.assert_close(updated_vars[0], expected_var1)
        torch.testing.assert_close(updated_vars[1], expected_var2)

    def test_create_optimizer(self):
        """Test optimizer configuration creation."""
        adam_config = distributed_backend.create_optimizer(
            "adam", learning_rate=0.01
        )
        assert isinstance(adam_config, dict)
        assert adam_config["name"] == "adam"
        assert adam_config["learning_rate"] == 0.01

        sgd_config = distributed_backend.create_optimizer(
            "sgd", learning_rate=0.1, momentum=0.9
        )
        assert isinstance(sgd_config, dict)
        assert sgd_config["name"] == "sgd"
        assert sgd_config["learning_rate"] == 0.1
        assert sgd_config["momentum"] == 0.9

    def test_get_device_info(self):
        """Test retrieving device information from the PyTorch backend."""
        info = distributed_backend.get_device_info()
        assert info["backend"] == "pytorch"
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

        # Test broadcast
        x_broadcast = torch.tensor([5.0, 6.0])
        broadcasted = comm_ops["broadcast"](x_broadcast)
        torch.testing.assert_close(broadcasted, x_broadcast)

        # Test scatter
        if world_size > 0:
            scatter_data = torch.arange(world_size * 4, dtype=torch.float32)
            x_scatter = scatter_data.reshape(world_size * 2, 2)
            scattered = comm_ops["scatter"](x_scatter, axis=0)
            expected_scatter = torch.chunk(x_scatter, world_size, dim=0)[0]
            torch.testing.assert_close(scattered, expected_scatter)
