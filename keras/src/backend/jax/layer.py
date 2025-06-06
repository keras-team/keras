from keras.src.backend import config


class JaxLayer:
    pass


if config.is_nnx_backend_enabled():
    try:
        from flax import nnx

        class NnxLayer(nnx.Module):
            def __init_subclass__(cls):
                super().__init_subclass__()
    except ImportError:
        raise ImportError(
            "To use the NNX backend, you must install `flax`."
            "Try: `pip install flax`"
        )
