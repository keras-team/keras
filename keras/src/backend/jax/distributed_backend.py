from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal

import jax
import jax.lax as lax
import jax.numpy as jnp
import optax

import keras
from keras.src.backend.distributed.base import DistributedBackend


class JaxDistributedBackend(DistributedBackend):
    """JAX-specific implementation of distributed operations.

    This class provides the JAX-based logic for distributed training,
    including device management, optimizer creation, and collective

    communication operations like all-reduce and all-gather.
    """

    def compute_gradients(
        self, loss: Any, trainable_vars: List[Any]
    ) -> List[Any]:
        """Computes gradients of the loss with respect to trainable variables.

        Note: The standard JAX paradigm for gradient computation involves using
        `jax.grad` on a function that computes the loss from the parameters.
        This method's signature, which takes a pre-computed loss, is not
        directly compatible with JAX's gradient transformation. As a fallback,
        this implementation returns zero gradients. For actual gradient
        computation in a JAX workflow, the training step logic should be
        encapsulated in a function and differentiated with `jax.grad`.

        Args:
            loss: The loss tensor. In the JAX backend, this is unused.
            trainable_vars: A list of trainable variables.

        Returns:
            A list of zero tensors, each with the same shape as the
            corresponding trainable variable.
        """
        return [jnp.zeros_like(var) for var in trainable_vars]

    def apply_gradients(
        self,
        gradients: List[Any],
        trainable_vars: List[Any],
        learning_rate: float = 0.001,
    ) -> None:
        """Applies gradients to trainable variables.

        This method performs a basic gradient descent update. It is a simplified
        implementation and does not use a stateful optimizer.

        Args:
            gradients: A list of gradient tensors.
            trainable_vars: A list of variables to be updated.
            learning_rate: The learning rate for the gradient descent update.
        """
        for grad, var in zip(gradients, trainable_vars):
            if grad is not None:
                new_value = var - (learning_rate * grad)
                if hasattr(var, "assign"):
                    var.assign(new_value)

    def create_optimizer(
        self, optimizer_class: str, **kwargs
    ) -> optax.GradientTransformation:
        """Creates an Optax optimizer instance from a string identifier.

        Args:
            optimizer_class: The name of the optimizer (e.g., 'adam', 'sgd').
            **kwargs: Keyword arguments to be passed to the optimizer's
                constructor (e.g., `learning_rate`).

        Returns:
            An instance of an `optax` optimizer. Defaults to `optax.adam` if
            the specified class is not found.
        """
        if optimizer_class.lower() == "adam":
            return optax.adam(**kwargs)
        elif optimizer_class.lower() == "sgd":
            return optax.sgd(**kwargs)
        else:
            kwargs.setdefault("learning_rate", 0.001)
            return optax.adam(**kwargs)

    def get_device_info(self) -> Dict[str, Any]:
        """Retrieves information about the available JAX devices.

        Returns:
            A dictionary containing the backend name ('jax'), a list of
            device strings, and the total count of local devices.
        """
        available_devices = jax.devices()
        if available_devices:
            return {
                "backend": "jax",
                "devices": [str(d) for d in available_devices],
                "device_count": len(available_devices),
            }
        else:
            return {"backend": "jax", "devices": ["cpu"], "device_count": 1}

    def is_multi_device_capable(self) -> bool:
        """Checks if more than one JAX device is available.

        Returns:
            `True` if the local device count is greater than 1, `False`
            otherwise.
        """
        return self.get_device_info()["device_count"] > 1

    def get_communication_ops(self) -> Dict[str, Callable]:
        """Provides a dictionary of JAX collective communication operations.

        These operations are designed to be robust, working correctly both
        inside and outside a `jax.pmap` context by dynamically checking the
        execution environment.

        Returns:
            A dictionary mapping operation names (e.g., 'all_reduce') to their
            JAX-based implementation functions.
        """

        def _is_in_pmap(axis_name: str = "data") -> bool:
            """Checks if currently executing inside a `pmap` transformation.

            This is the standard JAX idiom for context detection. It works by
            attempting to resolve an axis name, which only succeeds inside a
            `pmap` context.

            Args:
                axis_name: The `pmap` axis name to check for.

            Returns:
                `True` if inside a `pmap` context, `False` otherwise.
            """
            try:
                lax.axis_index(axis_name)
                return True
            except NameError:
                return False

        def all_reduce(
            x: jnp.ndarray,
            op: Literal["sum", "mean"] = "sum",
            axis_name: str = "data",
        ) -> jnp.ndarray:
            """Reduces a tensor across all devices.

            If inside a `pmap`, it uses JAX's collective operations (`psum` or
            `pmean`). Outside `pmap`, it simulates the reduction on a single
            device based on the total device count.

            Args:
                x: The tensor to reduce.
                op: The reduction operation, either 'sum' or 'mean'.
                axis_name: The `pmap` axis name for the reduction.

            Returns:
                The reduced tensor.
            """
            if _is_in_pmap(axis_name):
                if op == "sum":
                    return lax.psum(x, axis_name=axis_name)
                elif op == "mean":
                    return lax.pmean(x, axis_name=axis_name)
                raise ValueError(f"Unsupported all_reduce op: {op}")
            else:
                world_size = self.get_device_info()["device_count"]
                if world_size <= 1:
                    return x
                if op == "sum":
                    return keras.ops.multiply(x, world_size)
                elif op == "mean":
                    return x
                raise ValueError(f"Unsupported all_reduce op: {op}")

        def all_gather(
            x: jnp.ndarray, axis: int = 0, axis_name: str = "data"
        ) -> jnp.ndarray:
            """Gathers tensors from all devices and concatenates them.

            If inside a `pmap`, it uses `lax.all_gather`. Outside `pmap`, it
            simulates the operation by concatenating the input tensor `N` times,
            where `N` is the number of devices.

            Args:
                x: The tensor to gather from each device.
                axis: The axis along which to concatenate the gathered tensors.
                axis_name: The `pmap` axis name.

            Returns:
                The concatenated tensor containing data from all devices.
            """
            if _is_in_pmap(axis_name):
                return lax.all_gather(x, axis_name=axis_name, axis=axis)
            else:
                world_size = self.get_device_info()["device_count"]
                if world_size <= 1:
                    return x
                return keras.ops.concatenate([x] * world_size, axis=axis)

        def broadcast(
            x: jnp.ndarray, root: int = 0, axis_name: str = "data"
        ) -> jnp.ndarray:
            """Broadcasts a tensor from a root device to all other devices.

            If inside a `pmap`, it gathers the tensor from all devices and then
            selects the tensor from the `root` device. Outside `pmap`, this is
            a no-op and returns the tensor as-is.

            Args:
                x: The tensor to broadcast.
                root: The device index of the root (source) device.
                axis_name: The `pmap` axis name.

            Returns:
                The broadcasted tensor.
            """
            if _is_in_pmap(axis_name):
                return lax.all_gather(x, axis_name=axis_name, axis=0)[root]
            else:
                return x

        def scatter(
            x: jnp.ndarray,
            root: int = 0,
            axis: int = 0,
            axis_name: str = "data",
        ) -> jnp.ndarray:
            """Scatters a tensor from a root device to all devices.

            The tensor on the `root` device is split into chunks, and each
            device receives one chunk. If inside a `pmap`, it uses `all_gather`
            to get the full tensor and `dynamic_slice_in_dim` to extract the
            local chunk. Outside `pmap`, it simulates by splitting the tensor
            and returning the chunk corresponding to the `root` index.

            Args:
                x: The full tensor on the root device to be scattered.
                root: The device index of the root (source) device.
                axis: The axis along which to split the tensor.
                axis_name: The `pmap` axis name.

            Returns:
                A chunk of the original tensor specific to the local device.
            """
            if _is_in_pmap(axis_name):
                full_tensor = lax.all_gather(x, axis_name=axis_name, axis=0)[
                    root
                ]

                device_id = lax.axis_index(axis_name=axis_name)
                num_devices = lax.psum(1, axis_name=axis_name)

                chunk_size = full_tensor.shape[axis] // num_devices
                start_index = device_id * chunk_size
                return lax.dynamic_slice_in_dim(
                    operand=full_tensor,
                    start_index=start_index,
                    slice_size=chunk_size,
                    axis=axis,
                )
            else:
                world_size = self.get_device_info()["device_count"]
                if world_size <= 1:
                    return x
                chunks = keras.ops.split(x, world_size, axis=axis)
                return chunks[root]

        return {
            "all_reduce": all_reduce,
            "all_gather": all_gather,
            "broadcast": broadcast,
            "scatter": scatter,
        }