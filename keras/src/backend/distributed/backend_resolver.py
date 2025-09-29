import logging

from keras.src.backend.distributed.base import DistributedBackend

logger = logging.getLogger(__name__)


def get_distributed_backend(
    backend_name: str = "auto",
) -> DistributedBackend:
    """
    Backend resolver to get a specific distributed backend.

    Note: Currently, only the JAX backend is implemented.

    Args:
        backend_name: Name of the backend to use. Currently accepts "auto"
            or "jax". Other backends are reserved for future implementation.

    Returns:
        An instance of a class that inherits from `BaseDistributedBackend`.

    Raises:
        ValueError: If an unknown backend name is provided.
        NotImplementedError: If a backend other than JAX is requested.
        RuntimeError: If `backend_name` is "auto" and JAX is not installed.
    """
    if backend_name == "auto":
        try:
            from keras.src.backend.jax.distributed_backend import (
                JaxDistributedBackend,
            )

            logger.info("Auto-detected JAX for distributed backend.")
            return JaxDistributedBackend()
        except ImportError:
            raise RuntimeError(
                "Could not automatically detect a distributed backend. "
                "Currently, only the JAX backend is supported, so please "
                "ensure JAX is installed."
            )

    elif backend_name == "jax":
        from keras.src.backend.jax.distributed_backend import (
            JaxDistributedBackend,
        )

        return JaxDistributedBackend()
    elif backend_name == "tensorflow":
        raise NotImplementedError(
            "The TensorFlow distributed backend is not yet implemented."
        )
    elif backend_name == "torch":
        raise NotImplementedError(
            "The PyTorch distributed backend is not yet implemented."
        )
    elif backend_name == "numpy":
        raise NotImplementedError(
            "The NumPy distributed backend is not yet implemented."
        )
    else:
        raise ValueError(
            f"Unknown distributed backend: {backend_name}. "
            "Currently, the only available option is 'jax' or 'auto'."
        )
