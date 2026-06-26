"""Backend-agnostic device utility functions for Keras."""

from keras.src.api_export import keras_export


@keras_export("keras.utils.get_memory_info")
def get_memory_info(device):
    """Returns memory usage info for the given device.

    Args:
        device: A string device identifier (e.g. "GPU:0", "cuda:0")
            or backend-native device object.

    Returns:
        A dict with keys:
            - "allocated": currently allocated memory in bytes (int)
            - "peak": peak allocated memory in bytes (int)

    Raises:
        NotImplementedError: If the active backend does not support
            memory introspection (e.g. NumPy, OpenVINO).
    """
    from keras.src.backend import get_memory_info as _backend_get_memory_info

    return _backend_get_memory_info(device)
