import os

import numpy as np
import torch
import torch.distributed

from keras.src.backend.torch.core import _parse_device_input
from keras.src.backend.torch.core import get_device


def list_devices(device_type=None):
    """Return all available devices as a list of strings.

    Args:
        device_type: Optional string, one of "cpu", "gpu", "cuda", or "tpu".
            Defaults to the primary available device type.

    Returns:
        A list of strings like ["gpu:0", "gpu:1"] or ["cpu:0"].
    """
    resolved_device_type = _parse_device_input(
        device_type or get_device()
    ).split(":")[0]
    count = get_device_count(device_type)

    display_type = (
        "gpu" if resolved_device_type == "cuda" else resolved_device_type
    )

    return [f"{display_type}:{i}" for i in range(count)]


def get_device_count(device_type=None):
    """Return the total number of devices for a given type.

    In a distributed setting, this returns the total number of processes
    managing that device type across the cluster.

    Args:
        device_type: Optional string, one of "cpu", "gpu", "cuda", or "tpu".

    Returns:
        An integer representing the device count.
    """
    device_type = device_type.lower() if device_type else None

    if torch.distributed.is_initialized() or "WORLD_SIZE" in os.environ:
        actual_device_type = _parse_device_input(get_device()).split(":")[0]

        if device_type in (None, "cpu", actual_device_type) or (
            device_type == "gpu" and actual_device_type == "cuda"
        ):
            return num_processes()

        return 0

    resolved_device_type = _parse_device_input(
        device_type or get_device()
    ).split(":")[0]

    if resolved_device_type == "cuda":
        return torch.cuda.device_count()

    if resolved_device_type == "mps":
        return 1

    if resolved_device_type == "xpu":
        return torch.xpu.device_count()

    if resolved_device_type == "tpu":
        from keras.src.utils.module_utils import torch_xla

        if torch_xla.available:
            import torch_xla.core.xla_model as xm

            return xm.xla_device_count()

    return 1 if resolved_device_type == "cpu" else 0


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the distributed process group.

    Args:
        job_addresses: Optional string, comma-separated list of host:port
            addresses. The first address is used as the MASTER_ADDR/MASTER_PORT.
        num_processes: Optional integer, the total number of processes
            (WORLD_SIZE).
        process_id: Optional integer, the rank of the current process.
    """
    if job_addresses:
        address = job_addresses.split(",")[0]

        if ":" in address:
            master_addr, master_port = address.split(":")
            os.environ.setdefault("MASTER_ADDR", master_addr)
            os.environ.setdefault("MASTER_PORT", master_port)
        else:
            os.environ.setdefault("MASTER_ADDR", address)

    if num_processes is not None:
        os.environ.setdefault("WORLD_SIZE", str(num_processes))

    if process_id is not None:
        os.environ.setdefault("RANK", str(process_id))

    if not torch.distributed.is_initialized():
        world_size = int(os.environ.get("WORLD_SIZE", -1))
        rank = int(os.environ.get("RANK", -1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        resolved_device_type = _parse_device_input(get_device()).split(":")[0]
        if resolved_device_type == "cuda":
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        elif resolved_device_type == "xpu":
            torch.xpu.set_device(local_rank)
            backend = "ccl"
        elif resolved_device_type == "tpu":
            backend = "xla"
        else:
            backend = "gloo"

        torch.distributed.init_process_group(
            backend=backend, rank=rank, world_size=world_size
        )


def num_processes():
    """Return the total number of processes in the distributed group."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()

    return int(os.environ.get("WORLD_SIZE", 1))


def process_id():
    """Return the rank of the current process."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    return int(os.environ.get("RANK", 0))


def _to_backend_mesh(device_mesh):
    """Convert the DeviceMesh to PyTorch backend specific Mesh.

    Args:
        device_mesh: DeviceMesh instance to convert.

    Returns:
        A `torch.distributed.DeviceMesh` instance.
    """
    devices = device_mesh.devices

    ranks = np.array(
        [int(d.split(":")[-1]) for d in devices.flatten()]
    ).reshape(devices.shape)

    first_device = (
        devices.flatten()[0].split(":")[0] if devices.size > 0 else "cpu"
    )

    resolved_device_type = _parse_device_input(
        first_device or get_device()
    ).split(":")[0]

    return torch.distributed.device_mesh.DeviceMesh(
        resolved_device_type,
        ranks,
        mesh_dim_names=tuple(device_mesh.axis_names),
    )


def _to_backend_device(device_name):
    """Convert a device name string to a torch.device object."""
    if isinstance(device_name, torch.device):
        return device_name

    name = str(device_name).lower()
    parts = name.split(":")

    device_type_str = parts[0]
    device_index = (
        parts[1] if len(parts) > 1 else os.environ.get("LOCAL_RANK", "0")
    )

    resolved_device_type = _parse_device_input(
        device_type_str or get_device()
    ).split(":")[0]

    if resolved_device_type == "cpu":
        return torch.device("cpu")

    return torch.device(f"{resolved_device_type}:{device_index}")
