import threading

from tensorflow import nest

from keras_core.backend.config import floatx


class KerasVariable:
    def __init__(self, value, dtype, trainable=True, name=None):
        raise NotImplementedError

    @property
    def value(self):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def ndim(self):
        raise NotImplementedError

    def numpy(self):
        raise NotImplementedError

    def assign(self, value):
        raise NotImplementedError

    def __repr__(self):
        return f"<KerasVariable shape={self.shape}, dtype={self.dtype}, name={self.name}>"


ALLOWED_DTYPES = {
    "float16",
    "float32",
    "float64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "bfloat16",
    "bool",
}


def standardize_dtype(dtype):
    if dtype is None:
        return floatx()
    if hasattr(dtype, "name"):
        dtype = dtype.name
    if dtype not in ALLOWED_DTYPES:
        raise ValueError(f"Invalid dtype: {dtype}")
    return dtype


def standardize_shape(shape, fully_defined=False):
    if not isinstance(shape, tuple):
        if shape is None:
            raise ValueError("Undefined shapes are not supported.")
        if not hasattr(shape, "__iter__"):
            raise ValueError(f"Cannot convert '{shape}' to a shape.")
        shape = tuple(shape)
    for e in shape:
        if not fully_defined and e is None:
            continue
        if not isinstance(e, int):
            raise ValueError(
                f"Cannot convert '{shape}' to a shape. Found invalid entry '{e}'"
            )
        if e < 0:
            raise ValueError(
                f"Cannot convert '{shape}' to a shape. Negative dimensions are not allowed."
            )
    return shape


### Stateless context manager

GLOBAL_SCOPE_TRACKER = threading.local()


class StatelessScope:
    def __init__(self, state_mapping=None, collect_losses=False):
        from keras_core import backend

        self.collect_losses = collect_losses
        self.losses = []
        self.state_mapping = {}
        state_mapping = state_mapping or {}
        for k, v in state_mapping:
            if not isinstance(k, KerasVariable):
                raise ValueError(
                    "Invalid reference variable in VariableSwapScope: "
                    "all keys in argument `mapping` must be KerasVariable "
                    f"instances. Received instead: {k}"
                )
            v = backend.convert_to_tensor(v, dtype=k.dtype)
            if k.shape != v.shape:
                raise ValueError(
                    "Invalid variable value in VariableSwapScope: "
                    "all values in argument `mapping` must be tensors with "
                    "a shape that matches the corresponding variable shape. "
                    f"For variable {k}, received invalid value {v} with shape {v.shape}."
                )
            self.state_mapping[id(k)] = v

    def __enter__(self):
        self.original_scope = get_stateless_scope()
        GLOBAL_SCOPE_TRACKER.stateless_scope = self
        return self

    def add_loss(self, loss):
        self.losses.append(loss)

    def add_update(self, update):
        variable, value = update
        self.state_mapping[id(variable)] = value

    def get_current_value(self, variable):
        return self.state_mapping.get(id(variable), None)

    def __exit__(self, *args, **kwargs):
        GLOBAL_SCOPE_TRACKER.stateless_scope = self.original_scope


def in_stateless_scope():
    return getattr(GLOBAL_SCOPE_TRACKER, "stateless_scope", None) is not None


def get_stateless_scope():
    return getattr(GLOBAL_SCOPE_TRACKER, "stateless_scope", None)
