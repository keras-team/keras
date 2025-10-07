import os

import numpy as np
import pytest
import torch
import torch.distributed as dist

from keras.src import backend
from keras.src.backend import distribution_lib
from keras.src.distribution import DeviceMesh
from keras.src.distribution import TensorLayout


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="Backend specific test",
)
def setup_torch_distributed():
    """
    A fixture to initialize the distributed process group if not already done.
    This allows test file to be run directly with `pytest` for single-process
    checks, while also working correctly when launched with `torchrun`.
    """
    if not dist.is_available() or dist.is_initialized():
        return

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(backend="gloo")


@pytest.mark.skipif(
    not torch.distributed.is_available(),
    reason="PyTorch distributed components are not available.",
)
class TestTorchDistributionLibLive:
    """
    Tests for the Torch distribution library without using mocks.
    These tests will reflect the capabilities of environment they are run in.
    """

    def test_device_listing_and_info(self):
        """Tests device discovery functions against the runtime environment."""
        if torch.cuda.is_available():
            gpu_devices = distribution_lib.list_devices("gpu")
            assert len(gpu_devices) == torch.cuda.device_count()
            assert gpu_devices[0] == "cuda:0"
        else:
            assert distribution_lib.list_devices("gpu") == []

        cpu_devices = distribution_lib.list_devices("cpu")
        assert cpu_devices == ["cpu:0"]

        with pytest.raises(ValueError, match="Unknown device type"):
            distribution_lib.list_devices("unsupported_device")

    def test_device_helpers(self):
        """Tests validation, backend, and memory info functions."""
        device_str = "cpu:0"
        if torch.cuda.is_available():
            device_str = "cuda:0"

        assert distribution_lib.validate_device_placement(device_str) is True
        assert distribution_lib.validate_device_placement("invalid:0") is False

        assert distribution_lib.get_device_backend("cpu") == "torch"
        assert distribution_lib.get_device_backend("gpu") == "torch"

        mem_info = distribution_lib.get_device_memory_info(device_str)
        assert mem_info is not None
        assert "type" in mem_info
        assert mem_info["index"] == 0

    def test_process_discovery(self):
        """Tests process_id and num_processes in the live environment."""
        rank = distribution_lib.process_id()
        world_size = distribution_lib.num_processes()

        if dist.is_initialized():
            assert rank == dist.get_rank()
            assert world_size == dist.get_world_size()
        else:
            assert rank == 0
            assert world_size == 1

    def test_backend_conversions(self):
        """Tests the conversion of Keras objects to Torch backend objects."""
        world_size = distribution_lib.num_processes()
        if world_size < 2:
            pytest.skip(
                "Skipping conversion tests in a single-process environment."
            )

        devices = [f"cpu:{i}" for i in range(world_size)]
        shape = (world_size,)
        axis_names = ("data",)
        keras_mesh = DeviceMesh(shape, axis_names, devices)

        torch_mesh = distribution_lib._to_backend_mesh(keras_mesh)
        assert isinstance(torch_mesh, dist.DeviceMesh)
        assert torch_mesh.mesh.shape == shape

        keras_layout = TensorLayout(axes=("data",), device_mesh=keras_mesh)
        placements = distribution_lib._to_backend_layout(keras_layout)
        assert isinstance(placements[0], dist.Shard)

        keras_layout_replicated = TensorLayout(
            axes=(None,), device_mesh=keras_mesh
        )
        placements_replicated = distribution_lib._to_backend_layout(
            keras_layout_replicated
        )
        assert isinstance(placements_replicated[0], dist.Replicate)

    def test_tensor_distribution(self):
        """Tests the distribution of a tensor into a DTensor."""
        if not dist.is_initialized() or distribution_lib.num_processes() < 2:
            pytest.skip(
                "Tensor distribution test requires a multi-process environment."
            )

        world_size = distribution_lib.num_processes()
        devices = np.arange(world_size)
        keras_mesh = DeviceMesh((world_size,), ("batch",), devices)
        keras_layout = TensorLayout(("batch", None), keras_mesh)

        local_tensor = torch.randn((10, 20))

        dtensor = distribution_lib.distribute_tensor(local_tensor, keras_layout)
        assert isinstance(dtensor, torch.distributed.dtensor.DTensor)
        assert dtensor.device_mesh.mesh.shape == (world_size,)
        assert isinstance(dtensor.placements[0], dist.Shard)

        dvariable = distribution_lib.distribute_variable(
            local_tensor, keras_layout
        )
        assert isinstance(dvariable, torch.distributed.dtensor.DTensor)

    def test_distribute_data_input(self):
        """Tests the `from_local` logic for distributing input data."""
        if not dist.is_initialized() or distribution_lib.num_processes() < 2:
            pytest.skip(
                "Input distribution test requires a multi-process environment."
            )

        world_size = distribution_lib.num_processes()
        devices = np.arange(world_size)
        keras_mesh = DeviceMesh((world_size,), ("batch",), devices)
        keras_layout = TensorLayout(("batch", None), keras_mesh)

        per_process_batch = torch.ones((8, 16))

        global_batch = distribution_lib.distribute_data_input(
            per_process_batch, keras_layout, batch_dim_name="batch"
        )

        assert isinstance(global_batch, torch.distributed.dtensor.DTensor)
        assert global_batch.shape == (world_size * 8, 16)
