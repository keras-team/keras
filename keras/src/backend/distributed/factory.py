import logging

from keras.src.backend.distributed.base import BaseDistributedBackend

from keras.src.backend.jax.distributed_backend import JaxDistributedBackend
from keras.src.backend.numpy.distributed_backend import NumpyDistributedBackend
from keras.src.backend.tensorflow.distributed_backend import (
    TensorflowDistributedBackend,
)
from keras.src.backend.torch.distributed_backend import (
    PytorchDistributedBackend,
)

logger = logging.getLogger(__name__)


def get_distributed_backend(
    backend_name: str = "auto",
) -> BaseDistributedBackend:
    """
    Factory to get the best available or a specific distributed backend.
    """
    if backend_name == "auto":
        try:
            logger.info("Auto-detected JAX for distributed backend.")
            return JaxDistributedBackend()
        except ImportError:
            try:
                logger.info("Auto-detected TensorFlow for distributed backend.")
                return TensorflowDistributedBackend()
            except ImportError:
                try:
                    logger.info(
                        "Auto-detected PyTorch for distributed backend."
                    )
                    return PytorchDistributedBackend()
                except ImportError:
                    logger.warning("Using NumPy distributed backend.")
                    return NumpyDistributedBackend()

    elif backend_name == "jax":
        return JaxDistributedBackend()
    elif backend_name == "tensorflow":
        return TensorflowDistributedBackend()
    elif backend_name == "pytorch":
        return PytorchDistributedBackend()
    elif backend_name == "numpy":
        return NumpyDistributedBackend()
    else:
        raise ValueError(f"Unknown distributed backend: {backend_name}")
