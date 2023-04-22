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
