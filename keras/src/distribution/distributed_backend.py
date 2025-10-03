from typing import Any
from typing import List

from keras.src.api_export import keras_export
from keras.src.backend import distributed_backend


@keras_export("keras.distribution.apply_gradients")
def apply_gradients(
    gradients: List[Any],
    trainable_vars: List[Any],
    learning_rate: float = 0.001,
) -> None:
    """Applies gradients to trainable variables.

    This function is a distribution-aware wrapper that delegates the gradient
    application to the current backend's implementation.

    Args:
        gradients (List[Any]): A list of gradients to be applied.
        trainable_vars (List[Any]): A list of trainable variables to be updated.
        learning_rate (float, optional): The learning rate to use for the
            update. Defaults to 0.001.
    """
    return distributed_backend.apply_gradients(
        gradients, trainable_vars, learning_rate
    )


@keras_export("keras.distribution.create_optimizer")
def create_optimizer(optimizer_class: str, **kwargs):
    """Creates a backend-specific optimizer instance.

    This function instantiates an optimizer suitable for the current distributed
    backend, forwarding all keyword arguments to the optimizer's constructor.

    Args:
        optimizer_class (str): The class name of the optimizer to create (e.g.,
            `"Adam"`).
        **kwargs: Additional keyword arguments to be passed to the optimizer's
            constructor.

    Returns:
        An instance of the requested optimizer.
    """
    return distributed_backend.create_optimizer(optimizer_class, **kwargs)


@keras_export("keras.distribution.get_device_info")
def get_device_info() -> dict:
    """Gets information about available computational devices.

    Retrieves details about the devices (e.g., CPU, GPU) that are visible
    to the current backend.

    Returns:
        dict: A dictionary containing information about the available devices.
    """
    return distributed_backend.get_device_info()


@keras_export("keras.distribution.is_multi_device_capable")
def is_multi_device_capable() -> bool:
    """Checks if the backend supports multi-device operations.

    This function determines if the underlying backend is configured and
    capable of running computations across multiple devices.

    Returns:
        bool: `True` if the backend supports multi-device training,
        `False` otherwise.
    """
    return distributed_backend.is_multi_device_capable()


@keras_export("keras.distribution.get_communication_ops")
def get_communication_ops() -> dict:
    """Gets collective communication operations for the backend.

    This function returns a dictionary of collective ops (e.g., `all_reduce`,
    `all_gather`) that can be used for distributed communication.

    Returns:
        dict: A dictionary mapping the names of communication operations
        (str) to their callable implementations.
    """
    return distributed_backend.get_communication_ops()
