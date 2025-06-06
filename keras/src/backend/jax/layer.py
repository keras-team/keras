from keras.src.backend import config

if config.is_nnx_backend_enabled():
    try:
        from flax import nnx
    except ImportError:
        raise ImportError(
            "To use the NNX backend, you must install `flax`."
            "Try: `pip install flax`"
        )


class JaxLayer:
    pass


class NnxLayer(nnx.Module):
    def __init_subclass__(cls):
        super().__init_subclass__()
