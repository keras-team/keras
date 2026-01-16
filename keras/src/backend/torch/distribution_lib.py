import logging
import os

import torch
import torch.distributed as dist

from keras.src.random import seed_generator
from keras.src.utils import rng_utils

logger = logging.getLogger(__name__)


def list_devices(device_type=None):
    """Returns a list of available device identifiers.

    Args:
        device_type (str, optional): The type of device to list
        (e.g., "cpu", "cuda", "gpu"). Defaults to "cuda" if available,
        otherwise "cpu".

    Returns:
        list: A list of strings representing device addresses
        (e.g., ["cuda:0", "cuda:1"]).
    """
    if device_type:
        device_type = device_type.lower()
    else:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    if device_type in ("gpu", "cuda"):
        return (
            [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        )
    return ["cpu:0"] if device_type == "cpu" else []


def distribute_variable(value, layout):
    """Placeholder for distributing a variable across a specific layout.

    Currently returns the value as-is.

    Args:
        value: The variable or tensor to distribute.
        layout: The distribution layout configuration.

    Returns:
        The undistributed value.
    """
    return value


def initialize(job_addresses, num_processes, process_id):
    """Initializes the distributed process group and environment variables.

    Sets up the RANK, WORLD_SIZE, and MASTER_ADDR/PORT for PyTorch distributed.
    It defaults to the NCCL backend for GPUs and Gloo for CPUs.

    Args:
        job_addresses (str): A comma-separated list of host addresses.
            The first address is used as the master.
        num_processes (int): Total number of processes in the job (world size).
        process_id (int): The rank of the current process.
    """
    os.environ["RANK"] = str(process_id)
    os.environ["WORLD_SIZE"] = str(num_processes)
    if job_addresses:
        master_addr = job_addresses.split(",")[0]
        if ":" in master_addr:
            host, port = master_addr.split(":")
            os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = host, port

    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo"
    )
    initialize_rng()


def initialize_rng():
    """Initializes the random number generator across all distributed processes.

    If distributed training is active, rank 0 generates a seed and broadcasts
    it to all other ranks to ensure synchronized stochastic operations.
    """
    global_seed = rng_utils.get_random_seed()
    if global_seed is None:
        if not dist.is_initialized():
            seed = seed_generator.make_default_seed()
        else:
            seed_tensor = (
                torch.tensor(
                    seed_generator.make_default_seed(),
                    dtype=torch.int64,
                    device="cpu",
                )
                if process_id() == 0
                else torch.empty(1, dtype=torch.int64, device="cpu")
            )
            dist.broadcast(seed_tensor, src=0)
            seed = seed_tensor.item()
        rng_utils.set_random_seed(seed)


def num_processes():
    """Returns the total number of processes in the distributed group.

    Returns:
        int: The world size, or 1 if distributed training is not initialized.
    """
    return dist.get_world_size() if dist.is_initialized() else 1


def process_id():
    """Returns the rank of the current process.

    Returns:
        int: The process rank, or 0 if distributed training is not initialized.
    """
    return dist.get_rank() if dist.is_initialized() else 0


def all_reduce(x, op="sum", axis_name="model"):
    """Reduces the tensor data across all processes using a specific operation.

    Args:
        x (torch.Tensor): The input tensor to be reduced.
        op (str, optional): The reduction operation. Supports "sum" or "avg".
            Defaults to "sum".
        axis_name (str, optional): The named axis for distribution.
        Defaults to "model".

    Returns:
        torch.Tensor: The reduced tensor, identical on all processes.
    """
    if not dist.is_initialized():
        return x
    reduce_op = dist.ReduceOp.SUM if op == "sum" else dist.ReduceOp.AVG
    x_out = x.clone()
    dist.all_reduce(x_out, op=reduce_op)
    return x_out


def all_gather(x, axis, axis_name="model"):
    """Gathers tensors from all processes and concatenates them along a
    specified axis.

    Args:
        x (torch.Tensor): The tensor from the local process to be gathered.
        axis (int): The dimension along which to concatenate the gathered
        tensors.
        axis_name (str, optional): The named axis for distribution.
        Defaults to "model".

    Returns:
        torch.Tensor: A single concatenated tensor containing data from all
        ranks.
    """
    if not dist.is_initialized():
        return x
    gather_list = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, x)
    return torch.cat(gather_list, dim=axis)
