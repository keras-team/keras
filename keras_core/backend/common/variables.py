import threading

from keras_core.backend.config import floatx
from keras_core.utils.naming import auto_name


class KerasVariable:
    def __init__(
        self, initializer, shape=None, dtype=None, trainable=True, name=None
    ):
        self.name = name or auto_name(self.__class__.__name__)
        dtype = standardize_dtype(dtype)
        self._dtype = dtype
        self._initializer = None
        self.trainable = trainable
        if callable(initializer):
            if shape is None:
                raise ValueError(
                    "When creating a Variable from an initializer, "
                    "the `shape` argument should be specified. "
                    f"Received: initializer={initializer} "
                    f"and shape={shape}"
                )
        from keras_core.backend.stateless_scope import in_stateless_scope

        if in_stateless_scope():
            if callable(initializer):
                self._value = None
                self._initializer = initializer
                self._shape = standardize_shape(shape)
                register_uninitialized_variable(self)
            else:
                raise ValueError(
                    "You are attempting to create a variable "
                    "while in a stateless scope. This is disallowed. "
                    "Make sure that all variables are created "
                    "before you start using your layer/model objects. "
                    "Most of this time, this means you need to "
                    "implement a `def build(self, input_shape)` method "
                    "on your layer/model, which will "
                    "create its variables."
                )
        else:
            if callable(initializer):
                value = initializer(shape, dtype=dtype)
            else:
                value = initializer
            self._initialize(value)
            self._shape = tuple(self._value.shape)
        self._ndim = len(self._shape)

    def _deferred_initialize(self):
        if self._value is not None:
            raise ValueError(f"Variable {self.name} is already initialized.")
        from keras_core.backend.stateless_scope import in_stateless_scope

        if in_stateless_scope():
            raise ValueError(
                "You are attempting to initialize a variable "
                "while in a stateless scope. This is disallowed. "
                "Make sure that all variables are initialized "
                "before you start using your layer/model objects."
            )
        value = self._initializer(self._shape, dtype=self._dtype)
        self._initialize(value)

    def _initialize(self, value):
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
        return (
            f"<KerasVariable shape={self.shape}, dtype={self.dtype}, "
            f"name={self.name}>"
        )


GLOBAL_VARIABLE_TRACKER = threading.local()


def register_uninitialized_variable(variable):
    global GLOBAL_VARIABLE_TRACKER
    if not hasattr(GLOBAL_VARIABLE_TRACKER, "uninitialized_variables"):
        GLOBAL_VARIABLE_TRACKER.uninitialized_variables = []
    GLOBAL_VARIABLE_TRACKER.uninitialized_variables.append(variable)


def initialize_all_variables():
    global GLOBAL_VARIABLE_TRACKER
    collection = getattr(GLOBAL_VARIABLE_TRACKER, "uninitialized_variables", [])
    for v in collection:
        v._deferred_initialize()
    GLOBAL_VARIABLE_TRACKER.uninitialized_variables = []


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
                f"Cannot convert '{shape}' to a shape. "
                f"Found invalid entry '{e}'"
            )
        if e < 0:
            raise ValueError(
                f"Cannot convert '{shape}' to a shape. "
                "Negative dimensions are not allowed."
            )
    return shape
