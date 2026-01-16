import os

import distribution_lib
import pytest
import torch
import torch.distributed as dist


@pytest.fixture(scope="session", autouse=True)
def setup_torch_distributed():
    """Sets up a mock distributed environment for testing."""
    if not torch.cuda.is_available() and not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    yield

    if dist.is_initialized():
        dist.destroy_process_group()


class TestTorchDistributionLib:
    """Tests for the specific distribution_lib implementation provided."""

    def test_list_devices(self):
        """Tests device discovery logic."""
        cpu_devices = distribution_lib.list_devices("cpu")
        assert cpu_devices == ["cpu:0"]

        if torch.cuda.is_available():
            gpu_devices = distribution_lib.list_devices("cuda")
            assert len(gpu_devices) == torch.cuda.device_count()
            assert "cuda:0" in gpu_devices
        else:
            assert distribution_lib.list_devices("gpu") == []

    def test_process_info(self):
        """Tests num_processes and process_id integration with torch.dist."""
        assert distribution_lib.num_processes() == dist.get_world_size()
        assert distribution_lib.process_id() == dist.get_rank()

    def test_distribute_variable(self):
        """Tests if distribute_variable currently acts as identity."""
        sample_tensor = torch.tensor([1, 2, 3])
        result = distribution_lib.distribute_variable(
            sample_tensor, layout=None
        )
        assert torch.equal(sample_tensor, result)

    def test_all_reduce(self):
        """Tests the all_reduce wrapper logic."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        tensor = torch.tensor([1.0, 2.0], device="cpu")
        reduced = distribution_lib.all_reduce(tensor, op="sum")
        assert torch.equal(tensor, reduced)

        assert reduced is not tensor

    def test_all_gather(self):
        """Tests the all_gather wrapper logic."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        tensor = torch.tensor([1.0], device="cpu")
        gathered = distribution_lib.all_gather(tensor, axis=0)

        assert gathered.shape == (1,)
        assert gathered[0] == 1.0

    def test_initialize_rng(self):
        """Tests that RNG initialization runs without error."""
        try:
            distribution_lib.initialize_rng()
        except Exception as e:
            pytest.fail(f"initialize_rng failed with {e}")

    def test_initialize_full(self):
        """
        Tests the environment variable setup in initialize.
        """
        distribution_lib.initialize("localhost:29506", 1, 0)
        assert os.environ["RANK"] == "0"
        assert os.environ["WORLD_SIZE"] == "1"
        assert os.environ["MASTER_ADDR"] == "localhost"
        assert os.environ["MASTER_PORT"] == "29506"
