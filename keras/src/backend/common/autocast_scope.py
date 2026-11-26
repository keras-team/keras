from keras.src.backend.common import global_state
from keras.src.backend.common.dtypes import is_float_dtype
from keras.src.backend.common.dtypes import standardize_dtype


def get_autocast_scope():
    return global_state.get_global_attribute("autocast_scope")


class AutocastScope:
    """Context manager that enables the autocasting of float variables.

    Under this context manager, float `Variables`s will be cast to `dtype`
    (note that `dtype` must also be float).
    """

    def __init__(self, dtype):
        if dtype is not None:
            dtype = standardize_dtype(dtype)
            if not is_float_dtype(dtype):
                raise ValueError(
                    "`AutocastScope` can only be used with "
                    "a floating-point target dtype, such as 'float16'. "
                    f"Received: dtype={dtype}"
                )
        self.dtype = dtype
        self.original_scope = None

    def maybe_cast(self, value):
        from keras.src import backend

        if self.dtype is not None and is_float_dtype(value.dtype):
            return backend.ops.cast(value, dtype=self.dtype)
        return value

    def __enter__(self):
        self.original_scope = get_autocast_scope()
        global_state.set_global_attribute("autocast_scope", self)

    def __exit__(self, *args, **kwargs):
        global_state.set_global_attribute("autocast_scope", self.original_scope)
