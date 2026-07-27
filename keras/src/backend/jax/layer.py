from keras.src.backend.common.symbolic_scope import in_symbolic_scope
from keras.src.backend.config import is_nnx_enabled

if is_nnx_enabled():
    from flax import nnx

    class BaseLayer(nnx.Module):
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(pytree=False, **kwargs)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            from keras.src.backend.jax.core import stamp_with_enclosing_trace

            stamp_with_enclosing_trace(self)

        def _check_valid_context(self, error_msg):
            # NNX forbids mutating an object from a trace level other than
            # the one it was created at. These mutations are safe, since
            # initializers only see concrete shapes and only shapes escape
            # the trace.
            if in_symbolic_scope():
                return
            super()._check_valid_context(error_msg)
else:
    BaseLayer = object


class JaxLayer(BaseLayer):
    pass
