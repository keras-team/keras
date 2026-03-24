import contextlib
import os

import torch

from keras.src.backend.common import global_state


def list_devices(device_type=None):
    """Return all the available devices based on the device type."""
    device_type = device_type or "gpu"
    if torch.distributed.is_initialized():
        count = torch.distributed.get_world_size()
    elif "WORLD_SIZE" in os.environ:
        count = int(os.environ["WORLD_SIZE"])
    else:
        count = torch.cuda.device_count() or 1
    return [f"{device_type.lower()}:{i}" for i in range(count)]


def get_device_count(device_type=None):
    """Returns the number of available devices based on the device type."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    elif "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    else:
        return torch.cuda.device_count() or 1


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the distribution system."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=backend, rank=rank, world_size=world_size
        )


def num_processes():
    """Returns the number of processes in the distribution."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def process_id():
    """Returns the process ID of the current process."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _to_backend_device(device_name):
    """Returns the local device for the current process."""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _to_backend_mesh(keras_mesh):
    """Converts Keras DeviceMesh to PyTorch DeviceMesh."""
    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh(
        "cuda" if torch.cuda.is_available() else "cpu",
        mesh_shape=tuple(keras_mesh.shape),
        mesh_dim_names=tuple(keras_mesh.axis_names),
    )


def _to_backend_layout(tensor_layout):
    """Convert Keras TensorLayout to PyTorch DTensor placement spec."""
    if tensor_layout is None:
        return None

    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor import Shard

    keras_mesh = tensor_layout.device_mesh
    torch_mesh = _to_backend_mesh(keras_mesh)    
    placements = []
    for mesh_dim_name in keras_mesh.axis_names:
        shard_dim = None
        if tensor_layout.axes is not None:
            for tensor_dim, axis_name in enumerate(tensor_layout.axes):
                if axis_name == mesh_dim_name:
                    shard_dim = tensor_dim
                    break
        if shard_dim is not None:
            placements.append(Shard(shard_dim))
        else:
            placements.append(Replicate())
    
    return torch_mesh, tuple(placements)


class DTensorLayout:
    """Wraps a torch DeviceMesh + placements for use as a backend layout."""

    def __init__(self, device_mesh, placements):
        self.device_mesh = device_mesh
        self.placements = placements


def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout."""
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor import (
        distribute_tensor as torch_distribute_tensor,
    )

    torch_mesh, placements = _to_backend_layout(layout)
    return torch_distribute_tensor(
        tensor, device_mesh=torch_mesh, placements=placements
    )


def distribute_variable(value, layout):
    """Distribute the variable based on the layout."""
    dtensor = distribute_tensor(value, layout)
    if isinstance(value, torch.nn.Parameter):
        return torch.nn.Parameter(dtensor, requires_grad=value.requires_grad)
    return dtensor


def distribute_data_input(per_process_batch, layout, batch_dim_name=None):
    """Distribute the input data with the corresponding layout."""
    if layout is None:
        return per_process_batch

    from torch.distributed.tensor import DTensor

    torch_mesh, placements = _to_backend_layout(layout)
    return DTensor.from_local(
        per_process_batch, device_mesh=torch_mesh, placements=placements
    )


def distribution():
    """Returns the current distribution strategy."""
    from keras.src.distribution import distribution_lib

    return distribution_lib.distribution()


@contextlib.contextmanager
def sharding_scope():
    """Context manager to enable automatic sharding in convert_to_tensor."""
    previous_value = global_state.get_global_attribute(
        "enable_torch_sharding", False
    )
    global_state.set_global_attribute("enable_torch_sharding", True)
    try:
        yield
    finally:
        global_state.set_global_attribute(
            "enable_torch_sharding", previous_value
        )


def all_reduce(tensor, op="sum"):
    """Perform all-reduce on the tensor."""
    if not torch.distributed.is_initialized():
        return tensor

    if op.lower() in ("sum", "mean"):
        reduce_op = torch.distributed.ReduceOp.SUM
    else:
        raise ValueError(f"Unsupported op: {op}")

    from torch.distributed.tensor import DTensor

    if isinstance(tensor, DTensor):
        local_tensor = tensor.to_local()
        torch.distributed.all_reduce(local_tensor, op=reduce_op)
        if op.lower() == "mean":
            local_tensor /= torch.distributed.get_world_size()
    else:
        torch.distributed.all_reduce(tensor, op=reduce_op)
        if op.lower() == "mean":
            tensor /= torch.distributed.get_world_size()
    return tensor
