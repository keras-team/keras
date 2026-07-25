from keras.src.backend.common.symbolic_scope import in_symbolic_scope
from keras.src.backend.config import is_nnx_enabled

if is_nnx_enabled():
    from flax import nnx

    class BaseLayer(nnx.Module):
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(pytree=False, **kwargs)

        def __init__(self, *args, **kwargs):
            object.__setattr__(
                self, "_created_in_symbolic_scope", in_symbolic_scope()
            )
            super().__init__(*args, **kwargs)

        def _check_valid_context(self, error_msg):
            if in_symbolic_scope() or getattr(
                self, "_created_in_symbolic_scope", False
            ):
                return
            super()._check_valid_context(error_msg)
else:
    BaseLayer = object


class JaxLayer(BaseLayer):
    pass
