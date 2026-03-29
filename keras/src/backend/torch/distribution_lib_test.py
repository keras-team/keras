import os

import pytest
import torch
import torch.distributed as dist

from keras.src.backend.torch import distribution_lib


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
            with pytest.raises(RuntimeError):
                distribution_lib.list_devices("gpu")

    def test_get_device_count(self):
        """Tests device count logic."""
        cpu_count = distribution_lib.get_device_count("cpu")
        assert cpu_count == 1

        if torch.cuda.is_available():
            gpu_count = distribution_lib.get_device_count("cuda")
            assert gpu_count == torch.cuda.device_count()
        else:
            gpu_count = distribution_lib.get_device_count("gpu")
            assert gpu_count == 0

        mps_count = distribution_lib.get_device_count("mps")
        if hasattr(torch, "mps") and torch.mps.is_available():
            assert mps_count == 1
        else:
            assert mps_count == 0

    def test_process_info(self):
        """Tests num_processes and process_id integration with torch.dist."""
        assert distribution_lib.num_processes() == dist.get_world_size()
        assert distribution_lib.process_id() == dist.get_rank()

    def test_get_current_rank(self):
        """Tests _get_current_rank function."""
        rank = distribution_lib._get_current_rank()
        assert rank == 0

        if dist.is_initialized():
            assert rank == dist.get_rank()

    def test_get_current_device(self):
        """Tests _get_current_device function."""
        device = distribution_lib._get_current_device()
        if torch.cuda.is_available():
            assert device.type == "cuda"
        elif hasattr(torch, "mps") and torch.mps.is_available():
            assert device.type == "mps"
        else:
            assert device.type == "cpu"

    def test_distribute_variable_no_layout(self):
        """Tests distribute_variable with no layout (identity-like behavior)."""
        sample_tensor = torch.tensor([1, 2, 3])
        result = distribution_lib.distribute_variable(
            sample_tensor, layout=None
        )
        assert torch.equal(sample_tensor, result)

    def test_distribute_variable_with_layout(self):
        """Tests distribute_variable with a TensorLayout."""
        from keras.src.distribution import TensorLayout, DeviceMesh

        # Create a simple device mesh and layout
        devices = ["cpu:0"]
        device_mesh = DeviceMesh(
            devices=devices,
            mesh=[1],
            axis_names=("batch",),
        )

        # Test with no sharding (all axes are None)
        layout = TensorLayout(axes=[None, None], device_mesh=device_mesh)
        sample_tensor = torch.tensor([[1, 2], [3, 4]])
        result = distribution_lib.distribute_variable(sample_tensor, layout=layout)
        assert torch.equal(sample_tensor, result)

    def test_distribute_tensor_no_layout(self):
        """Tests distribute_tensor with no layout."""
        sample_tensor = torch.tensor([1, 2, 3])
        result = distribution_lib.distribute_tensor(sample_tensor, layout=None)
        assert torch.equal(sample_tensor, result)

    def test_distribute_tensor_with_layout(self):
        """Tests distribute_tensor with a TensorLayout."""
        from keras.src.distribution import TensorLayout, DeviceMesh

        devices = ["cpu:0"]
        device_mesh = DeviceMesh(
            devices=devices,
            mesh=[1],
            axis_names=("batch",),
        )

        # Test with no sharding (all axes are None)
        layout = TensorLayout(axes=[None, None], device_mesh=device_mesh)
        sample_tensor = torch.tensor([[1, 2], [3, 4]])
        result = distribution_lib.distribute_tensor(sample_tensor, layout=layout)
        assert torch.equal(sample_tensor, result)

    def test_all_reduce(self):
        """Tests the all_reduce wrapper logic."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        tensor = torch.tensor([1.0, 2.0], device="cpu")
        reduced = distribution_lib.all_reduce(tensor, op="sum")
        assert torch.equal(tensor, reduced)

        assert reduced is not tensor

    def test_all_reduce_ops(self):
        """Tests all_reduce with different reduction operations."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        tensor = torch.tensor([1.0, 2.0, 3.0], device="cpu")

        # Test sum
        result_sum = distribution_lib.all_reduce(tensor.clone(), op="sum")
        assert result_sum is not tensor

        # Test product
        result_prod = distribution_lib.all_reduce(tensor.clone(), op="product")
        assert result_prod is not tensor

        # Test min
        result_min = distribution_lib.all_reduce(tensor.clone(), op="min")
        assert result_min is not tensor

        # Test max
        result_max = distribution_lib.all_reduce(tensor.clone(), op="max")
        assert result_max is not tensor

    def test_all_gather(self):
        """Tests the all_gather wrapper logic."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        tensor = torch.tensor([1.0], device="cpu")
        gathered = distribution_lib.all_gather(tensor, axis=0)

        assert gathered.shape == (1,)
        assert gathered[0] == 1.0

    def test_all_gather_variable(self):
        """Tests all_gather_variable function."""
        # Test with non-sharded variable
        sample_tensor = torch.tensor([1, 2, 3])
        result = distribution_lib.all_gather_variable(sample_tensor)
        assert torch.equal(sample_tensor, result)

    def test_broadcast(self):
        """Tests the broadcast function."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        tensor = torch.tensor([1.0, 2.0], device="cpu")
        result = distribution_lib.broadcast(tensor, src=0)

        assert result is not tensor
        assert result.shape == tensor.shape

    def test_broadcast_different_src(self):
        """Tests broadcast with different source rank."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        tensor = torch.tensor([1.0, 2.0], device="cpu")
        result = distribution_lib.broadcast(tensor, src=0)

        assert result.shape == tensor.shape

    def test_initialize_single_process(self):
        """Tests initialize with single process (no-op case)."""
        # Should not raise any errors
        distribution_lib.initialize("localhost:29506", 1, 0)

    def test_initialize_env_vars(self):
        """Tests that environment variables are set correctly in initialize."""
        distribution_lib.initialize("localhost:29507", 1, 0)
        assert os.environ["RANK"] == "0"
        assert os.environ["WORLD_SIZE"] == "1"
        assert os.environ["MASTER_ADDR"] == "localhost"
        assert os.environ["MASTER_PORT"] == "29507"

    def test_distribute_data_input(self):
        """Tests distribute_data_input function."""
        # Test with single tensor
        sample_batch = torch.tensor([1, 2, 3])
        result = distribution_lib.distribute_data_input(
            sample_batch, layout=None, batch_dim_name="batch"
        )
        assert torch.equal(sample_batch, result)

        # Test with tuple of batches
        batch_tuple = (torch.tensor([1, 2]), torch.tensor([3, 4]))
        result = distribution_lib.distribute_data_input(
            batch_tuple, layout=None, batch_dim_name="batch"
        )
        assert isinstance(result, tuple)
        assert torch.equal(result[0], batch_tuple[0])
        assert torch.equal(result[1], batch_tuple[1])

        # Test with list of batches
        batch_list = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        result = distribution_lib.distribute_data_input(
            batch_list, layout=None, batch_dim_name="batch"
        )
        assert isinstance(result, list)
        assert torch.equal(result[0], batch_list[0])
        assert torch.equal(result[1], batch_list[1])

    def test_to_backend_mesh(self):
        """Tests _to_backend_mesh function."""
        from keras.src.distribution import DeviceMesh

        devices = ["cpu:0"]
        device_mesh = DeviceMesh(
            devices=devices,
            mesh=[1],
            axis_names=("batch",),
        )

        backend_mesh = distribution_lib._to_backend_mesh(device_mesh)

        assert "devices" in backend_mesh
        assert "axis_names" in backend_mesh
        assert "shape" in backend_mesh
        assert backend_mesh["devices"] == devices
        assert backend_mesh["axis_names"] == ("batch",)
        assert backend_mesh["shape"] == [1]

    def test_to_backend_layout(self):
        """Tests _to_backend_layout function."""
        from keras.src.distribution import TensorLayout, DeviceMesh

        devices = ["cpu:0"]
        device_mesh = DeviceMesh(
            devices=devices,
            mesh=[1],
            axis_names=("batch",),
        )

        layout = TensorLayout(axes=[None], device_mesh=device_mesh)

        backend_layout = distribution_lib._to_backend_layout(layout)

        assert "axes" in backend_layout
        assert "mesh" in backend_layout
        assert backend_layout["axes"] == [None]

    def test_to_backend_layout_no_mesh(self):
        """Tests _to_backend_layout raises error when no device mesh."""
        from keras.src.distribution import TensorLayout

        layout = TensorLayout(axes=[None], device_mesh=None)

        with pytest.raises(ValueError):
            distribution_lib._to_backend_layout(layout)

