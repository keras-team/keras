import os

import torch


def _is_available(device_type):
    """Check if a device type is available."""
    if device_type in ("gpu", "cuda"):
        return torch.cuda.is_available()
    if device_type == "mps":
        return torch.backends.mps.is_available()
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "tpu":
        from keras.src.utils.module_utils import torch_xla

        return torch_xla.available
    return device_type == "cpu"


def _get_default_device_type():
    """Get the default device type for the current environment."""
    for dt in ["gpu", "mps", "xpu", "tpu"]:
        if _is_available(dt):
            return dt
    return "cpu"


def get_device_count(device_type=None):
    """Returns total device count across all hosts.

    Args:
        device_type: String of `"cpu"`, `"gpu"`, `"mps"`, `"xpu"`, or `"tpu"`.
            Defaults to the best available device type.

    Returns:
        Int, total number of devices across all hosts.
    """
    device_type = (device_type or _get_default_device_type()).lower()
    if device_type == "cuda":
        device_type = "gpu"
    if not _is_available(device_type):
        return 0

    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])

    if device_type == "gpu":
        return torch.cuda.device_count()
    if device_type == "xpu":
        return torch.xpu.device_count()
    if device_type == "tpu":
        from keras.src.utils.module_utils import torch_xla

        return torch_xla.runtime.global_device_count()
    return 1


def list_devices(device_type=None):
    """Returns Keras device strings representing global indices.

    Args:
        device_type: String of `"cpu"`, `"gpu"`, `"mps"`, `"xpu"`, or `"tpu"`.
            Defaults to the best available device type.

    Returns:
        List of strings like `["gpu:0", "gpu:1", ...]`.
    """
    device_type = (device_type or _get_default_device_type()).lower()
    if device_type == "cuda":
        device_type = "gpu"
    return [f"{device_type}:{i}" for i in range(get_device_count(device_type))]


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the current process for distributed training.

    Args:
        job_addresses: Address of the coordinator process (process 0).
        num_processes: Total number of processes.
        process_id: Rank of the current process.
    """
    if torch.distributed.is_initialized():
        return

    if num_processes is not None:
        os.environ["WORLD_SIZE"] = str(num_processes)
    if process_id is not None:
        os.environ["RANK"] = str(process_id)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        # RANK is the global rank of the process across all nodes.
        rank = int(os.environ.get("RANK", 0))
        # LOCAL_RANK is the rank of the process on the current node.
        # This is used to set the device (e.g. which GPU to use).
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        init_method = None
        if job_addresses:
            if "," in job_addresses:
                raise ValueError(
                    "For the torch backend, `job_addresses` should only "
                    "contain the coordinator address (the address of "
                    f"process 0). Received: job_addresses={job_addresses}"
                )
            init_method = (
                job_addresses
                if "://" in job_addresses
                else f"tcp://{job_addresses}"
            )

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            init_method=init_method,
        )


def num_processes():
    """Return the number of processes for the current distribution setting."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def process_id():
    """Return the current process ID for the distribution setting."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def to_backend_device(device_name):
    """Returns the local device for the current process.

    Args:
        device_name: Keras device string (e.g. `"gpu:0"`).

    Returns:
        A `torch.device` instance.
    """
    if device_name:
        device_name = device_name.lower()
        if "meta" in device_name:
            return torch.device("meta")
        if "cpu" in device_name:
            return torch.device("cpu")

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        idx = (
            device_name.split(":")[1]
            if device_name and ":" in device_name
            else local_rank
        )
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")
