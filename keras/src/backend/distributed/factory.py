import logging

from keras.src.backend.distributed.base import BaseDistributedBackend

logger = logging.getLogger(__name__)


def get_distributed_backend(
    backend_name: str = "auto",
) -> BaseDistributedBackend:
    """
    Factory to get the best available or a specific distributed backend.
    """
    if backend_name == "auto":
        try:
            from keras.src.backend.jax.distributed_backend import (
                JaxDistributedBackend,
            )

            logger.info("Auto-detected JAX for distributed backend.")
            return JaxDistributedBackend()
        except ImportError:
            pass

    elif backend_name == "jax":
        from keras.src.backend.jax.distributed_backend import (
            JaxDistributedBackend,
        )

        return JaxDistributedBackend()
    elif backend_name == "tensorflow":
        pass
    elif backend_name == "torch":
        pass
    elif backend_name == "numpy":
        pass
    else:
        raise ValueError(f"Unknown distributed backend: {backend_name}")
