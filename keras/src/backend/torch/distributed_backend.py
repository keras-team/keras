from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal

import torch
import torch.distributed as dist


def compute_gradients(
    loss: torch.Tensor, trainable_vars: List[torch.Tensor]
) -> List[torch.Tensor]:
    """Computes gradients of the loss with respect to trainable variables.

    This function leverages PyTorch's `autograd.grad` for a stateless,
    functional approach similar to `jax.grad`.

    Args:
        loss (torch.Tensor): The loss value for which to compute gradients.
        trainable_vars (List[torch.Tensor]): A list of variables (tensors with
            `requires_grad=True`) to compute gradients with respect to.

    Returns:
        List[torch.Tensor]: A list of gradients corresponding to the
        trainable variables.
    """
    return list(torch.autograd.grad(loss, trainable_vars))


def apply_gradients(
    gradients: List[torch.Tensor],
    trainable_vars: List[torch.Tensor],
    learning_rate: float = 0.001,
) -> List[torch.Tensor]:
    """Applies gradients and returns the updated variables.

    Updates are performed in-place within a `torch.no_grad()` context
    to prevent the update operation from being part of the computation graph.
    """
    with torch.no_grad():
        updated_vars = []
        for grad, var in zip(gradients, trainable_vars):
            if grad is not None:
                var.sub_(learning_rate * grad)
            updated_vars.append(var)
    return updated_vars


def create_optimizer(optimizer_class: str, **kwargs) -> Dict[str, Any]:
    """Creates a configuration dictionary for a PyTorch optimizer.

    This function returns a dictionary containing the optimizer's configuration,
    maintaining a consistent interface with the JAX backend. The user is
    expected to instantiate the optimizer from this config.

    Args:
        optimizer_class (str): The name of the optimizer to create (e.g.,
            `"adam"`, `"sgd"`).
        **kwargs: Keyword arguments for the optimizer (e.g., `learning_rate`).

    Returns:
        Dict[str, Any]: A dictionary representing the optimizer configuration.
    """
    config = kwargs.copy()
    config["name"] = optimizer_class.lower()
    config.setdefault("learning_rate", 0.001)
    return config


def get_device_info() -> Dict[str, Any]:
    """Retrieves information about the available PyTorch devices.

    Returns:
        Dict[str, Any]: A dictionary containing the backend name, a list of
        available device strings, and the total device count.
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        devices = [torch.cuda.get_device_name(i) for i in range(device_count)]
    else:
        device_count = 1
        devices = ["cpu"]
    return {
        "backend": "pytorch",
        "devices": devices,
        "device_count": device_count,
    }


def is_multi_device_capable() -> bool:
    """Checks if more than one CUDA device is available.

    Returns:
        bool: `True` if PyTorch reports more than one CUDA device, `False`
        otherwise.
    """
    return torch.cuda.device_count() > 1


def get_communication_ops() -> Dict[str, Callable]:
    """Provides a dictionary of PyTorch collective communication operations.

    These operations rely on the `torch.distributed` package. They are
    designed to work in a multi-process, multi-device environment. If the
    distributed package is not initialized, they provide a sensible fallback
    for single-device execution.

    Returns:
        Dict[str, Callable]: A dictionary mapping operation names to their
        PyTorch implementations.
    """

    def _is_distributed() -> bool:
        """Checks if the default process group is initialized."""
        return dist.is_available() and dist.is_initialized()

    def all_reduce(
        x: torch.Tensor,
        op: Literal["sum", "mean"] = "sum",
        axis_name: str = None,
    ) -> torch.Tensor:
        """Reduces a tensor across all devices."""
        if not _is_distributed():
            world_size = (
                torch.cuda.device_count() if torch.cuda.is_available() else 1
            )
            if world_size <= 1:
                return x
            if op == "sum":
                return x * float(world_size)
            elif op == "mean":
                return x
            else:
                raise ValueError(f"Unsupported all_reduce op: {op}")

        reduce_op = {"sum": dist.ReduceOp.SUM, "mean": dist.ReduceOp.AVG}.get(
            op
        )
        if reduce_op is None:
            raise ValueError(f"Unsupported all_reduce op: {op}")

        result = x.clone()
        dist.all_reduce(result, op=reduce_op)
        return result

    def all_gather(
        x: torch.Tensor, axis: int = 0, axis_name: str = None
    ) -> torch.Tensor:
        """Gathers tensors from all devices and concatenates them."""
        if not _is_distributed():
            world_size = (
                torch.cuda.device_count() if torch.cuda.is_available() else 1
            )
            if world_size <= 1:
                return x
            return torch.cat([x] * world_size, dim=axis)

        world_size = dist.get_world_size()
        tensor_list = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(tensor_list, x)
        return torch.cat(tensor_list, dim=axis)

    def broadcast(
        x: torch.Tensor, root: int = 0, axis_name: str = None
    ) -> torch.Tensor:
        """Broadcasts a tensor from a root device to all other devices."""
        if not _is_distributed():
            return x

        dist.broadcast(x, src=root)
        return x

    def scatter(
        x: torch.Tensor,
        root: int = 0,
        axis: int = 0,
        axis_name: str = None,
    ) -> torch.Tensor:
        """Scatters a tensor from a root device to all devices."""
        if not _is_distributed():
            world_size = (
                torch.cuda.device_count() if torch.cuda.is_available() else 1
            )
            if world_size <= 1:
                return x
            if x.shape[axis] % world_size != 0:
                raise ValueError(
                    f"Tensor with shape {x.shape} cannot be scattered along "
                    f"axis {axis} across {world_size} devices."
                )
            return torch.chunk(x, world_size, dim=axis)[0]

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if x.shape[axis] % world_size != 0:
            raise ValueError(
                f"Tensor with shape {x.shape} cannot be scattered along "
                f"axis {axis} across {world_size} devices."
            )

        if rank == root:
            scatter_list = list(torch.chunk(x, world_size, dim=axis))
        else:
            scatter_list = None

        chunk_shape = list(x.shape)
        chunk_shape[axis] //= world_size
        local_chunk = torch.empty(chunk_shape, dtype=x.dtype, device=x.device)

        dist.scatter(local_chunk, scatter_list, src=root)
        return local_chunk

    return {
        "all_reduce": all_reduce,
        "all_gather": all_gather,
        "broadcast": broadcast,
        "scatter": scatter,
    }
