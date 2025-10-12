from keras.src.backend import distributed_backend


def get_device_info() -> dict:
    """Gets information about available computational devices.

    Retrieves details about the devices (e.g., CPU, GPU) that are visible
    to the current backend.

    Returns:
        dict: A dictionary containing information about the available devices.
    """
    return distributed_backend.get_device_info()


def is_multi_device_capable() -> bool:
    """Checks if the backend supports multi-device operations.

    This function determines if the underlying backend is configured and
    capable of running computations across multiple devices.

    Returns:
        bool: `True` if the backend supports multi-device training,
        `False` otherwise.
    """
    return distributed_backend.is_multi_device_capable()


def get_communication_ops() -> dict:
    """Gets collective communication operations for the backend.

    This function returns a dictionary of collective ops (e.g., `all_reduce`,
    `all_gather`) that can be used for distributed communication.

    Returns:
        dict: A dictionary mapping the names of communication operations
        (str) to their callable implementations.
    """
    return distributed_backend.get_communication_ops()
