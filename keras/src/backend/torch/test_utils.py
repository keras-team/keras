"""Shared test utilities for the PyTorch backend."""

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from keras.src.backend.torch.ops.core import get_device


def host_data_ops(fn):
    """Runs `fn` and returns the torch ops that use host data.

    An op uses host data when it makes a tensor from host data
    (`aten.lift_fresh`, for example a NumPy array that becomes a tensor),
    or, on a GPU or MPS device, when it reads or returns a CPU tensor.
    Torch copies such data to the device on each call, and a CUDA graph
    capture refuses the copy. The `lift_fresh` check applies on the CPU
    device too.

    Args:
        fn: A function with no arguments.

    Returns:
        The names of the ops, in call order.
    """
    on_cpu = torch.device(get_device()).type == "cpu"
    found = []

    class HostData(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            outputs = func(*args, **(kwargs or {}))
            tensors = tree_flatten((args, kwargs, outputs))[0]
            reads_cpu = not on_cpu and any(
                isinstance(t, torch.Tensor) and t.device.type == "cpu"
                for t in tensors
            )
            if reads_cpu or func is torch.ops.aten.lift_fresh.default:
                found.append(str(func))
            return outputs

    with HostData():
        fn()
    return found
