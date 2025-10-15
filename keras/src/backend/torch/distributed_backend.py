import os

import torch
import torch.distributed as dist


def get_device_info():
    """Retrieves information about the available PyTorch devices.

    This function queries PyTorch to identify the type and number of
    available computational devices (e.g., CPU, GPU).

    Returns:
        dict: A dictionary containing the backend name ('torch'), a list of
        device string representations, and the total count of devices.
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        devices = [
            f"cuda:{i} ({torch.cuda.get_device_name(i)})"
            for i in range(device_count)
        ]
        backend = "torch (CUDA)"
    else:
        device_count = 1
        devices = ["cpu"]
        backend = "torch (CPU)"

    return {
        "backend": backend,
        "devices": devices,
        "device_count": device_count,
    }


def is_multi_device_capable():
    """Checks if more than one device is available for distributed computation.

    Returns:
        bool: True if the PyTorch distributed environment is initialized and
        has a world size greater than one, False otherwise.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size() > 1
    elif torch.cuda.is_available():
        return torch.cuda.device_count() > 1
    return False


def setup_distributed_environment():
    """
    A helper function to initialize the distributed process group.

    This is a prerequisite for using the communication operations.
    In a real application, this would be called at the start of the script.
    It uses environment variables commonly set by launchers like torchrun.
    """
    if dist.is_available() and not dist.is_initialized():
        required_env_vars = ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"]
        if not all(v in os.environ for v in required_env_vars):
            return False

        dist.init_process_group(backend="nccl")
        return True
    elif dist.is_initialized():
        return True
    else:
        return False


def get_communication_ops():
    """Provides a dictionary of PyTorch collective communication operations.

    Note: The torch.distributed process group must be initialized before
    calling these functions.

    Returns:
        dict: A dictionary mapping operation names (e.g., 'all_reduce') to their
        corresponding PyTorch implementation functions.
    """

    def all_reduce(x, op="sum"):
        """Reduces a tensor across all devices in the process group.

        This function performs a collective reduction operation
        across all devices in the distributed group.

        Args:
            x (torch.Tensor): The input tensor on the local device.
            op (str, optional): The reduction operation to perform. Supported
                values are 'sum' and 'mean'. Defaults to 'sum'.

        Returns:
            torch.Tensor: The reduced tensor, which is identical across all
            devices participating in the reduction.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return x

        if op == "sum":
            reduce_op = dist.ReduceOp.SUM
        elif op == "mean":
            reduce_op = dist.ReduceOp.AVG
        else:
            raise ValueError(
                f"Unsupported reduction operation: {op}. "
                "Supported options are 'sum' and 'mean'."
            )

        result = x.clone()
        dist.all_reduce(result, op=reduce_op)
        return result

    def all_gather(x, axis):
        """Gathers and concatenates tensors from all devices.

        This function takes the local tensor `x` from each device and
        concatenates them along the specified tensor `axis` to form a single,
        larger tensor that is then replicated on all participating devices.

        Args:
            x (torch.Tensor): The input tensor shard on the local device.
            axis (int): The tensor axis along which to concatenate the gathered
                shards.

        Returns:
            torch.Tensor: The full, gathered tensor, which is identical across
            all devices participating in the gather.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return x

        world_size = dist.get_world_size()
        tensor_list = [torch.empty_like(x) for _ in range(world_size)]

        dist.all_gather(tensor_list, x)
        return torch.cat(tensor_list, dim=axis)

    return {
        "all_reduce": all_reduce,
        "all_gather": all_gather,
    }
