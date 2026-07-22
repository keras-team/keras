#!/usr/bin/env python3
"""Two-GPU model-parallel experiment launcher.

This script runs a model-parallel experiment on exactly two GPU devices for
both Torch and JAX backends.

Examples:
    python model_parallel_experiment_2gpu.py torch
    python model_parallel_experiment_2gpu.py jax
"""

import os
import sys

from model_parallel_experiment import find_free_port
from model_parallel_experiment import get_layout_map
from model_parallel_experiment import run_training


def run_backend(backend="torch", world_size=2):
    backend = backend.lower()
    if backend not in {"torch", "jax"}:
        raise ValueError("Backend must be 'torch' or 'jax'.")

    os.environ["KERAS_BACKEND"] = backend

    if backend == "jax":
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
        import jax

        if len(jax.devices()) < world_size:
            raise RuntimeError(
                f"Expected at least {world_size} visible JAX devices,"
                f"{len(jax.devices())}."
            )
        _run_jax(world_size)
    else:
        import torch

        if torch.cuda.device_count() < world_size:
            raise RuntimeError(
                f"Expected at least {world_size} visible GPUs, but found "
                f"{torch.cuda.device_count()}."
            )

        port = str(find_free_port())
        torch.multiprocessing.spawn(
            _run_torch,
            args=(world_size, port),
            nprocs=world_size,
            join=True,
        )


def _run_jax(world_size):
    import keras

    keras.utils.set_random_seed(42)
    devices = keras.distribution.list_devices()
    if len(devices) > world_size:
        devices = devices[:world_size]
    print(f"Using JAX devices: {devices}")

    mesh = keras.distribution.DeviceMesh(
        shape=(1, 2),
        axis_names=("data", "model"),
        devices=devices,
    )
    run_training(0, world_size, get_layout_map(mesh), "jax")


def _run_torch(rank, world_size, port):
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(rank)

    os.environ.update(
        {
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": port,
        }
    )
    os.environ["KERAS_TORCH_DEVICE"] = "cuda"

    import keras

    keras.utils.set_random_seed(42)
    keras.distribution.initialize()

    devices = keras.distribution.list_devices("cuda")[:world_size]
    mesh = keras.distribution.DeviceMesh(
        shape=(1, 2),
        axis_names=("data", "model"),
        devices=devices,
    )
    run_training(rank, world_size, get_layout_map(mesh), "torch")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else "torch"
    run_backend(backend)
