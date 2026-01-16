import logging
import os

import torch
import torch.distributed as dist

from keras.src.random import seed_generator
from keras.src.utils import rng_utils

logger = logging.getLogger(__name__)


def list_devices(device_type=None):
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
    return value


def initialize(job_addresses, num_processes, process_id):
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
    return dist.get_world_size() if dist.is_initialized() else 1


def process_id():
    return dist.get_rank() if dist.is_initialized() else 0


def all_reduce(x, op="sum", axis_name="model"):
    if not dist.is_initialized():
        return x
    reduce_op = dist.ReduceOp.SUM if op == "sum" else dist.ReduceOp.AVG
    x_out = x.clone()
    dist.all_reduce(x_out, op=reduce_op)
    return x_out


def all_gather(x, axis, axis_name="model"):
    if not dist.is_initialized():
        return x
    gather_list = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, x)
    return torch.cat(gather_list, dim=axis)
