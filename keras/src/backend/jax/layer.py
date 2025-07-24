from keras.src.backend.config import is_nnx_enabled

if is_nnx_enabled():
    from flax import nnx

    BaseLayer = nnx.Module
else:
    BaseLayer = object


class JaxLayer(BaseLayer):
    pass
