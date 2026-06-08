import os

import torch


def get_device_count(device_type=None):
    """Returns total device count across all hosts."""
    if device_type is None:
        if torch.cuda.is_available():
            device_type = "gpu"
        elif torch.backends.mps.is_available():
            device_type = "mps"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device_type = "xpu"
        else:
            from keras.src.utils.module_utils import torch_xla

            if torch_xla.available:
                device_type = "tpu"
            else:
                device_type = "cpu"

    device_type = device_type.lower()
    if device_type in ("gpu", "cuda"):
        if not torch.cuda.is_available():
            return 0
    elif device_type == "mps":
        if not torch.backends.mps.is_available():
            return 0
    elif device_type == "xpu":
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return 0
    elif device_type == "tpu":
        from keras.src.utils.module_utils import torch_xla

        if not torch_xla.available:
            return 0
    elif device_type == "cpu":
        pass
    else:
        return 0

    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])

    if device_type in ("gpu", "cuda"):
        return torch.cuda.device_count()
    if device_type == "xpu":
        return torch.xpu.device_count()
    if device_type == "tpu":
        from keras.src.utils.module_utils import torch_xla

        return torch_xla.runtime.global_device_count()
    return 1


def list_devices(device_type=None):
    """Returns Keras device strings representing global indices."""
    if device_type is None:
        if torch.cuda.is_available():
            device_type = "gpu"
        elif torch.backends.mps.is_available():
            device_type = "mps"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device_type = "xpu"
        else:
            from keras.src.utils.module_utils import torch_xla

            if torch_xla.available:
                device_type = "tpu"
            else:
                device_type = "cpu"

    device_type = device_type.lower()
    if device_type == "cuda":
        device_type = "gpu"

    count = get_device_count(device_type)
    return [f"{device_type}:{i}" for i in range(count)]


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the current process for distributed training."""
    if not torch.distributed.is_initialized():
        if num_processes is not None:
            world_size = int(num_processes)
        elif "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            world_size = 1

        if process_id is not None:
            rank = int(process_id)
        elif "RANK" in os.environ:
            rank = int(os.environ["RANK"])
        else:
            rank = 0

        if world_size > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            torch.distributed.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
            )


def num_processes():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def process_id():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def to_backend_device(device_name):
    """Returns the local device for the current process."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device_name is not None:
        device_name_lower = device_name.lower()
        if "meta" in device_name_lower:
            return torch.device("meta")
        if "cpu" in device_name_lower:
            return torch.device("cpu")
        if "gpu" in device_name_lower or "cuda" in device_name_lower:
            if ":" in device_name_lower:
                device_idx = int(device_name_lower.split(":")[1])
                return torch.device(f"cuda:{device_idx}")
            if torch.cuda.is_available():
                return torch.device(f"cuda:{local_rank}")
            return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")
