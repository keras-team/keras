"""Utilities for distribution strategy with Torch backend.

This file contains the core Torch distribution primitives from Keras,
along with higher-level device management and auto-configuration utilities.
This version does not use try-except blocks for error handling.
"""

import logging
import os
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

from keras.src.backend.common import global_state
from keras.src.random import seed_generator
from keras.src.utils import rng_utils

logger = logging.getLogger(__name__)


def list_devices(device_type=None):
    """Return all the available devices based on the device type.

    Note that this should return the global devices in a distributed setting.

    Args:
        device_type: string of `"cpu"`, `"gpu"`. Defaults to `"gpu"` if
            available when device_type is not provided. Otherwise will return
            the `"cpu"` devices. `"tpu"` is not supported by the default
            torch backend.

    Return:
        List of devices that are available for distribute computation.
    """
    if device_type:
        device_type = device_type.lower()
    else:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    if device_type in ("gpu", "cuda"):
        if not torch.cuda.is_available():
            return []
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    elif device_type == "cpu":
        return ["cpu:0"]
    elif device_type == "tpu":
        logger.warning(
            "TPU device type is not supported by the default "
            "PyTorch backend. Use the `torch_xla` package."
        )
        return []
    raise ValueError(f"Unknown device type: {device_type}")


def get_device_info(device_id: str) -> Dict[str, any]:
    """
    Get detailed information about a specific device.

    Args:
        device_id: Device identifier (e.g., 'cuda:0', 'cpu:0')

    Returns:
        Dictionary containing device information
    """
    device_info = {
        "id": device_id,
        "type": None,
        "index": None,
        "memory": None,
        "capabilities": None,
    }

    device_type, device_index = device_id.split(":")
    device_type_map = {"cuda": "GPU", "cpu": "CPU"}
    device_info["type"] = device_type_map.get(device_type, device_type.upper())
    device_info["index"] = int(device_index)

    return device_info


def get_best_devices(count: int = 1) -> List[str]:
    """
    Get the best available devices for tensor parallelism.

    Args:
        count: Number of devices needed

    Returns:
        List of best device identifiers
    """
    all_devices = list_devices("cuda")
    if not all_devices:
        all_devices = list_devices("cpu")

    if count <= 0:
        return []

    if count > len(all_devices):
        logger.warning(
            f"Requested {count} devices but only {len(all_devices)} available"
        )
        count = len(all_devices)

    return all_devices[:count]


def get_device_backend(device_type: str) -> str:
    """
    Get the recommended backend for a device type.

    Args:
        device_type: Device type ('tpu', 'gpu', 'cpu')

    Returns:
        Recommended backend name
    """
    backend_mapping = {"gpu": "torch", "cuda": "torch", "cpu": "torch"}

    return backend_mapping.get(device_type.lower(), "torch")


def validate_device_placement(device_id: str) -> bool:
    """
    Validate if a device can be used for tensor operations.

    Args:
        device_id: Device identifier

    Returns:
        True if device is valid and available
    """
    if ":" not in device_id:
        return False

    device_type = device_id.split(":")[0]
    known_device_types = ("cpu", "gpu", "cuda", "tpu")
    if device_type not in known_device_types:
        return False

    all_devices = list_devices(device_type)
    return device_id in all_devices


def get_device_memory_info(device_id: str) -> Optional[Dict[str, any]]:
    """
    Get memory information for a device (if available).

    Args:
        device_id: Device identifier

    Returns:
        Memory information dictionary or None if not available
    """
    if device_id.startswith("cuda:"):
        return {
            "type": "GPU",
            "index": int(device_id.split(":")[1]),
            "memory": "Available",
        }
    elif device_id.startswith("cpu:"):
        return {
            "type": "CPU",
            "index": int(device_id.split(":")[1]),
            "memory": "System RAM",
        }

    return None


def auto_configure_tensor_parallel(
    world_size: int = None, backend: str = None
) -> Dict[str, any]:
    """
    Automatically configure tensor parallelism with the best available devices.

    Args:
        world_size: Number of devices to use (if None, uses all available GPUs)
        backend: Backend to use (if None, will be set to 'torch')

    Returns:
        Configuration dictionary with devices, backend, and other settings
    """
    all_devices = list_devices()

    if not all_devices:
        raise RuntimeError("No devices available for tensor parallelism")

    if world_size is None:
        world_size = len(all_devices)
    else:
        world_size = min(world_size, len(all_devices))

    selected_devices = all_devices[:world_size]

    recommended_backend = "torch"

    config = {
        "devices": selected_devices,
        "world_size": world_size,
        "backend": recommended_backend,
    }

    logger.info(f"Auto-configured tensor parallelism: {config}")
    return config


def distribute_variable(value, layout):
    """Create a distributed variable for PyTorch.

    This function creates a `torch.Tensor` distributed according to the given
    layout. In PyTorch, variables and tensors are unified in the `Tensor` class.

    Args:
        value: The initial value of the variable as a `torch.Tensor`.
        layout: `TensorLayout` for the created variable, or a PyTorch-supported
            layout instance (e.g., a list of `Placement` types).

    Returns:
        `torch.Tensor` which is the distributed variable.
    """
    return distribute_tensor(value, layout)


def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout.

    Args:
        tensor: `torch.Tensor` that needs to be distributed.
        layout: `TensorLayout` for the created variable, or a PyTorch-supported
            layout instance (e.g., a list of `Placement` types).

    Returns:
        Distributed `torch.Tensor`.
    """
    # Avoid circular imports.
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        placements = layout.backend_layout
        device_mesh = layout.device_mesh.backend_mesh
    else:
        raise ValueError(
            "Directly passing backend layout is not yet supported for torch. "
            "Please provide a `keras.distribution.TensorLayout` instance."
        )

    return dist.dtensor.distribute_tensor(
        tensor.to("cpu"), device_mesh, placements
    )


def distribute_data_input(per_process_batch, layout, batch_dim_name):
    """Distribute the input data with the corresponding layout.

    Note that the input here is a local worker batch. PyTorch's `from_local`
    is used to construct a global DTensor from these local shards.

    Args:
        per_process_batch: `torch.Tensor` that is local shard for this process.
        layout: `TensorLayout` for the distribution information.

    Returns:
        A global batch distributed according to `layout`.
    """
    from keras.src.distribution import TensorLayout

    if not isinstance(layout, TensorLayout):
        raise ValueError(
            "A `keras.distribution.TensorLayout` instance is required."
        )

    placements = layout.backend_layout
    device_mesh = layout.device_mesh.backend_mesh
    return dist.dtensor.from_local(
        per_process_batch, device_mesh, placements, run_check=True
    )


def initialize_rng():
    """Initializes the global random number generator across processes.

    This is required for consistent initialization in multi-host settings.
    It works by generating a seed on rank 0 and broadcasting it to all other
    processes.
    """
    global_seed = rng_utils.get_random_seed()
    if global_seed is None:
        if not dist.is_initialized():
            seed = seed_generator.make_default_seed()
        else:
            if process_id() == 0:
                seed = seed_generator.make_default_seed()
                seed_tensor = torch.tensor(
                    seed, dtype=torch.int64, device="cpu"
                )
            else:
                seed_tensor = torch.empty(1, dtype=torch.int64, device="cpu")
            dist.broadcast(seed_tensor, src=0)
            seed = seed_tensor.item()
        global_seed = seed
        rng_utils.set_random_seed(global_seed)

    global_seed_generator = global_state.get_global_attribute(
        "global_seed_generator"
    )
    if global_seed_generator is not None and global_seed_generator.seed is None:
        global_state.set_global_attribute(
            "global_seed_generator",
            seed_generator.SeedGenerator(
                seed=global_seed,
                name=global_seed_generator.name,
                backend=global_seed_generator.backend,
            ),
        )


def initialize(job_addresses, num_processes, process_id):
    """Initializes the distributed process group in PyTorch."""
    os.environ["RANK"] = str(process_id)
    os.environ["WORLD_SIZE"] = str(num_processes)

    if "," in job_addresses:
        master_addr = job_addresses.split(",")[0]
    else:
        master_addr = job_addresses

    if ":" not in master_addr:
        raise ValueError(
            "Invalid `job_addresses`. Expected format `hostname:port`, "
            f"but got {master_addr}"
        )

    master_host, master_port = master_addr.split(":")
    os.environ["MASTER_ADDR"] = master_host
    os.environ["MASTER_PORT"] = master_port

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    initialize_rng()


def num_processes():
    """Return the number of processes for the current distribution setting."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def process_id():
    """Return the current process ID for the distribution setting."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def _to_backend_device(device_name):
    if isinstance(device_name, torch.device):
        return device_name
    return torch.device(device_name)


def _to_backend_mesh(device_mesh):
    """Convert the DeviceMesh to Torch backend specific Mesh.

    Args:
        device_mesh: DeviceMesh instance to convert.

    Returns:
        A `torch.distributed.DeviceMesh` instance.
    """
    mesh_shape = device_mesh.devices.shape
    mesh_devices = np.array(device_mesh.devices.flatten()).reshape(mesh_shape)
    return dist.DeviceMesh(
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        mesh=mesh_devices,
    )


def _to_backend_layout(tensor_layout):
    """Convert the TensorLayout to Torch backend specific placement.

    Args:
        tensor_layout: TensorLayout instance to convert.

    Returns:
        A list of `torch.distributed.placement_types.Placement` instances.
    """
    if tensor_layout.device_mesh is None:
        raise ValueError(
            "Cannot create sharding when device mesh is not set "
            "for TensorLayout."
        )

    mesh_axes = tensor_layout.device_mesh.axis_names
    placements = []
    for axis in tensor_layout.axes:
        if axis is None:
            placements.append(dist.Replicate())
        else:
            try:
                mesh_dim = mesh_axes.index(axis)
                placements.append(dist.Shard(mesh_dim))
            except ValueError:
                raise ValueError(
                    f"Tensor axis `{axis}` is not found in the "
                    f"device mesh axes `{mesh_axes}`."
                ) from None
    return placements
