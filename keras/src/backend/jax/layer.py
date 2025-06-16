from keras.src.backend.config import is_nnx_enabled

if is_nnx_enabled():
    from flax import nnx

    class NnxLayer(nnx.Module):
        pass


class JaxLayer:
    pass
