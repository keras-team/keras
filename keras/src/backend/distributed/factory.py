import logging

from keras.src.backend.distributed.base import BaseDistributedBackend
import traceback  # <-- Add this import

logger = logging.getLogger(__name__)


def get_distributed_backend(
    backend_name: str = "auto",
) -> BaseDistributedBackend:
    """
    Factory to get the best available or a specific distributed backend.
    """
    print("!!! Keras Distributed Backend Factory was called !!!")
    traceback.print_stack()
    if backend_name == "auto":
        try:
            from keras.src.backend.jax.distributed_backend import (
                JaxDistributedBackend,
            )

            logger.info("Auto-detected JAX for distributed backend.")
            return JaxDistributedBackend()
        except ImportError:
            try:
                from keras.src.backend.tensorflow.distributed_backend import (
                    TensorflowDistributedBackend,
                )

                logger.info("Auto-detected TensorFlow for distributed backend.")
                return TensorflowDistributedBackend()
            except ImportError:
                try:
                    from keras.src.backend.torch.distributed_backend import (
                        TorchDistributedBackend,
                    )

                    logger.info(
                        "Auto-detected PyTorch for distributed backend."
                    )
                    return TorchDistributedBackend()
                except ImportError:
                    error_msg = (
                        "Could not automatically detect a distributed backend "
                        "(JAX, TensorFlow, or PyTorch). Please install them "
                        "or explicitly specify a backend."
                    )
                    logger.error(error_msg)
                    raise ImportError(error_msg)

    elif backend_name == "jax":
        from keras.src.backend.jax.distributed_backend import (
            JaxDistributedBackend,
        )

        return JaxDistributedBackend()
    elif backend_name == "tensorflow":
        from keras.src.backend.tensorflow.distributed_backend import (
            TensorflowDistributedBackend,
        )

        return TensorflowDistributedBackend()
    elif backend_name == "torch":
        from keras.src.backend.torch.distributed_backend import (
            TorchDistributedBackend,
        )

        return TorchDistributedBackend()
    elif backend_name == "numpy":
        from keras.src.backend.numpy.distributed_backend import (
            NumpyDistributedBackend,
        )

        logger.warning(
            "Using explicitly requested NumPy distributed backend. "
            "This backend is for simulation and does not support "
            "multi-device computation."
        )
        return NumpyDistributedBackend()
    else:
        raise ValueError(f"Unknown distributed backend: {backend_name}")
