from keras.src.backend.config import is_nnx_enabled

if is_nnx_enabled():
    from flax import nnx

    class BaseLayer(nnx.Module):
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(pytree=False, **kwargs)
else:
    BaseLayer = object


class JaxLayer(BaseLayer):
    pass
