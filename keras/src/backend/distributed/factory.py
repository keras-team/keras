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
                    from keras.src.backend.numpy.distributed_backend import (
                        NumpyDistributedBackend,
                    )

                    logger.warning("Using NumPy distributed backend.")
                    return NumpyDistributedBackend()

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

        return NumpyDistributedBackend()
    else:
        raise ValueError(f"Unknown distributed backend: {backend_name}")
