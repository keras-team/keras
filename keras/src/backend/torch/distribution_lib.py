"""Utilities for distribution strategy with Torch backend."""

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import Shard
from torch.distributed.tensor import (
    distribute_tensor as distribute_tensor_torch,
)

from keras.src.backend.torch.core import get_device


def list_devices(device_type=None):
    """Return all the available devices based on the device type.

    Note that this should return the global devices in a distributed setting.

    Args:
        device_type: string of `"cpu"`, `"gpu"` or `"tpu"`. Defaults to `"gpu"`
            or `"tpu"` if available when device_type is not provided. Otherwise
            will return the `"cpu"` devices.

    Returns:
        List of devices that are available for distribute computation.
    """
    if device_type is None:
        if torch.cuda.is_available():
            device_type = "cuda"
        elif "xla" in get_device():
            device_type = "xla"
        elif torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"

    device_type = device_type.lower()
    if "gpu" in device_type:
        device_type = "cuda"
    if "tpu" in device_type:
        device_type = "xla"

    if device_type == "cuda":
        count = torch.cuda.device_count()
    elif device_type == "xla":
        import torch_xla.core.xla_model as xm

        count = len(xm.get_xla_supported_devices())
    elif device_type == "mps":
        count = 1 if torch.backends.mps.is_available() else 0
    else:
        count = 1

    if dist.is_initialized():
        world_size = dist.get_world_size()
        return [f"{device_type}:{i}" for i in range(world_size)]

    return [f"{device_type}:{i}" for i in range(count)]


def get_device_count(device_type=None):
    """Returns the number of available devices.

    Args:
        device_type: Optional device type to count (e.g., "cpu", "gpu", "tpu").
            If `None`, it defaults to counting "gpu" or "tpu" devices if
            available, otherwise it counts "cpu" devices.

    Returns:
        int: The total number of devices for the specified type.
    """
    return len(list_devices(device_type))


def distribute_value(value, layout):
    """Distribute the value based on the layout.

    Args:
        value: The value to distribute.
        layout: The layout to use for distribution.

    Returns:
        The distributed value.
    """
    return distribute_tensor(value, layout)


def distribute_variable(value, layout):
    """Create a distributed variable for Torch.

    Args:
        value: The initial value of the variable.
        layout: The layout to use for distribution.

    Returns:
        The distributed variable value.
    """
    return distribute_tensor(value, layout)


def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout.

    Args:
        tensor: The tensor to distribute.
        layout: The layout to use for distribution.

    Returns:
        The distributed tensor.
    """
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        torch_mesh = layout.device_mesh.backend_mesh
        placements = _get_placements(layout)

        if isinstance(tensor, DTensor):
            if tensor.device_mesh == torch_mesh and tensor.placements == tuple(
                placements
            ):
                return tensor
            return tensor.redistribute(torch_mesh, placements)

        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, device=get_device())

        if get_device() == "meta":
            return tensor

        if (
            not isinstance(tensor, DTensor)
            and tensor.device.type == torch_mesh.device_type
        ):
            local_tensor = tensor
            should_shard_locally = True
            for i, placement in enumerate(placements):
                if isinstance(placement, Shard):
                    shard_dim = placement.dim
                    num_chunks = torch_mesh.shape[i]
                    if local_tensor.shape[shard_dim] % num_chunks != 0:
                        should_shard_locally = False
                        break

                    chunk_idx = torch_mesh.get_local_rank(mesh_dim=i)
                    local_tensor = torch.chunk(
                        local_tensor, num_chunks, dim=shard_dim
                    )[chunk_idx]

            if should_shard_locally:
                return DTensor.from_local(local_tensor, torch_mesh, placements)

        if tensor.device.type != torch_mesh.device_type:
            if tensor.is_meta:
                tensor = torch.empty_like(tensor, device=torch_mesh.device_type)
            else:
                tensor = tensor.to(torch_mesh.device_type)

        return distribute_tensor_torch(tensor, torch_mesh, placements)

    return tensor


def _sync_tensors(*args):
    """Ensure all tensors are DTensors if any of them is a DTensor.

    Args:
        *args: The structures (tensors, lists, dicts) to synchronize.

    Returns:
        The synchronized structures.
    """
    from keras.src import tree
    from keras.src.backend.torch.core import Variable

    flat_args = tree.flatten(args)
    flat_values = [t.value if isinstance(t, Variable) else t for t in flat_args]

    has_dtensor = any(isinstance(t, DTensor) for t in flat_values)
    if not has_dtensor:
        return args

    if any(isinstance(t, torch.Tensor) and t.is_meta for t in flat_values):
        return args

    ref_dtensor = next(t for t in flat_values if isinstance(t, DTensor))
    mesh = ref_dtensor.device_mesh

    from torch.distributed.tensor import Partial

    def _sync_op(t):
        if isinstance(t, Variable):
            t = t.value

        if isinstance(t, DTensor):
            if any(isinstance(p, Partial) for p in t.placements):
                return t.redistribute(mesh, [Replicate()] * mesh.ndim)
            return t
        elif isinstance(t, torch.Tensor):
            if t.is_meta:
                t = torch.empty_like(t, device=mesh.device_type)
            elif t.device.type != mesh.device_type:
                t = t.to(mesh.device_type)

            return distribute_tensor_torch(t, mesh, [Replicate()] * mesh.ndim)
        return t

    return tree.map_structure(_sync_op, args)


def distribute_data_input(per_process_batch, layout, batch_dim_name):
    """Distribute the input data with the corresponding layout.

    Args:
        per_process_batch: The batch of data for the current process.
        layout: The layout to use for distribution.
        batch_dim_name: The name of the batch dimension.

    Returns:
        The distributed input data.
    """
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        torch_mesh = layout.device_mesh.backend_mesh
        placements = _get_placements(layout)

        if not isinstance(per_process_batch, torch.Tensor):
            per_process_batch = torch.as_tensor(
                per_process_batch, device=get_device()
            )

        if per_process_batch.device.type != torch_mesh.device_type:
            per_process_batch = per_process_batch.to(torch_mesh.device_type)

        if isinstance(per_process_batch, DTensor):
            return per_process_batch.redistribute(torch_mesh, placements)

        return distribute_tensor_torch(
            per_process_batch, torch_mesh, placements
        )

    return per_process_batch


def initialize_rng():
    """Initializes the global random number generator across processes."""
    from keras.src.utils import rng_utils

    global_seed = rng_utils.get_random_seed()
    if global_seed is None:
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank == 0:
                seed = np.random.randint(0, 2**31)
            else:
                seed = 0

            backend_name = dist.get_backend()
            if backend_name == "nccl":
                device = "cuda"
            elif backend_name == "xla":
                device = "xla"
            else:
                device = "cpu"

            if device == "cuda":
                torch.cuda.set_device(rank % torch.cuda.device_count())
            elif device == "xla":
                import torch_xla.core.xla_model as xm

                device = xm.xla_device()

            seed_tensor = torch.tensor([seed], dtype=torch.int64, device=device)
            dist.broadcast(seed_tensor, src=0)
            global_seed = int(seed_tensor.item())
            rng_utils.set_random_seed(global_seed)


def initialize(job_addresses, num_processes, process_id):
    """Initialize the distribution system.

    Args:
        job_addresses: Comma separated IP addresses for all the jobs.
        num_processes: The number of worker/processes.
        process_id: The ID number of the current worker/process.
    """
    if dist.is_initialized():
        return

    import os

    if job_addresses:
        coordinator_address = job_addresses.split(",")[0]
        if ":" in coordinator_address:
            addr, _, port = coordinator_address.rpartition(":")
        else:
            addr, port = coordinator_address, "12345"
        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = port

    if num_processes is not None:
        os.environ["WORLD_SIZE"] = str(num_processes)
    if process_id is not None:
        os.environ["RANK"] = str(process_id)

    if torch.cuda.is_available():
        backend = "nccl"
    elif os.environ.get("TPU_NAME"):
        import torch_xla.core.xla_model as xm

        backend = "xla"
    else:
        backend = "gloo"

    if backend == "nccl":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    elif backend == "xla":
        import torch_xla.core.xla_model as xm

        xm.xla_device()

    dist.init_process_group(backend=backend)

    initialize_rng()


def num_processes():
    """Return the number of processes for the current distribution setting.

    Returns:
        int: The number of processes.
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def process_id():
    """Return the current process ID for the distribution setting.

    Returns:
        int: The current process ID.
    """
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def _to_backend_mesh(device_mesh):
    """Convert the DeviceMesh to Torch backend specific Mesh.

    Args:
        device_mesh: The DeviceMesh to convert.

    Returns:
        The Torch-specific DeviceMesh.
    """
    from keras.src.backend.torch import core

    mesh_shape = device_mesh.devices.shape
    mesh_dim_names = device_mesh.axis_names

    device = core.get_device()
    if ":" in device:
        device_type = device.split(":")[0]
    else:
        device_type = device

    if device_type == "mps":
        device_type = "cpu"

    return init_device_mesh(
        device_type, mesh_shape, mesh_dim_names=mesh_dim_names
    )


def _to_backend_layout(tensor_layout):
    """Convert the TensorLayout to Torch backend specific Placements.

    Args:
        tensor_layout: The TensorLayout to convert.

    Returns:
        A list of Shard/Replicate objects.
    """
    return _get_placements(tensor_layout)


def _maybe_distribute_input(x, distribution):
    """Distribute the input data if it's not already a DTensor.

    Args:
        x: The input data.
        distribution: The distribution strategy to use.

    Returns:
        The distributed input data.
    """
    from keras.src import tree

    if isinstance(x, torch.Tensor) and not isinstance(x, DTensor):
        layout = _get_data_layout(x.shape, distribution)
        return distribute_tensor(x, layout)

    def _distribute_if_tensor(t):
        if isinstance(t, torch.Tensor) and not isinstance(t, DTensor):
            layout = _get_data_layout(t.shape, distribution)
            return distribute_tensor(t, layout)
        return t

    return tree.map_structure(_distribute_if_tensor, x)


def _get_data_layout(shape, distribution):
    """Default data layout if not provided.

    Args:
        shape: The shape of the data.
        distribution: The distribution strategy.

    Returns:
        The TensorLayout for the data.
    """
    from keras.src.distribution import TensorLayout

    spec = [None] * len(shape)
    if distribution.batch_dim_name:
        spec[0] = distribution.batch_dim_name
    return TensorLayout(spec, distribution.device_mesh)


def _get_placements(layout):
    """Helper to get PyTorch placements from TensorLayout.

    Args:
        layout: The TensorLayout.

    Returns:
        A list of Shard/Replicate objects.
    """
    mesh = layout.device_mesh
    axes = layout.axes
    placements = []
    for mesh_axis_name in mesh.axis_names:
        found = False
        for i, axis_name in enumerate(axes):
            if axis_name == mesh_axis_name:
                placements.append(Shard(i))
                found = True
                break
        if not found:
            placements.append(Replicate())
    return placements


def distribute_dataset(dataset, distribution):
    """Create a distributed dataset for Torch.

    Args:
        dataset: The dataset to distribute.
        distribution: The distribution strategy.

    Returns:
        The distributed dataset.
    """
    if not dist.is_initialized():
        return dataset

    if isinstance(dataset, torch.utils.data.DataLoader):
        return dataset

    from keras.src.utils.module_utils import tensorflow as tf

    if tf.available and isinstance(dataset, tf.data.Dataset):
        return distribution.distribute_dataset(dataset)

    return dataset
