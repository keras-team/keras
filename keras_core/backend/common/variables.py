import numpy as np

from keras_core.backend import config
from keras_core.backend.common import global_state
from keras_core.backend.common.stateless_scope import get_stateless_scope
from keras_core.backend.common.stateless_scope import in_stateless_scope
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
                    "before you start using your layer/model objects.\n\n"
                    "In some cases, you might be seeing this error "
                    "because you need to "
                    "implement a `def build(self, input_shape)` method "
                    "on your layer/model, which will "
                    "create its variables.\n\n"
                    "In some other cases, you might be seeing this error "
                    "because you are instantiating a `Variable` and "
                    "assigning it to a layer without going through "
                    "self.add_variable()/self.add_weight(). Always prefer "
                    "using these methods "
                    "(with a `shape` and `initializer` argument)."
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
        from keras_core.backend.common.stateless_scope import in_stateless_scope

        if in_stateless_scope():
            raise ValueError(
                "You are attempting to initialize a variable "
                "while in a stateless scope. This is disallowed. "
                "Make sure that all variables are initialized "
                "before you start using your layer/model objects."
            )
        value = self._initializer(self._shape, dtype=self._dtype)
        self._initialize(value)

    def _maybe_autocast(self, value):
        autocast_scope = get_autocast_scope()
        if autocast_scope is not None:
            return autocast_scope.maybe_cast(value)
        return value

    def numpy(self):
        return np.array(self.value)

    @property
    def value(self):
        if in_stateless_scope():
            scope = get_stateless_scope()
            value = scope.get_current_value(self)
            if value is not None:
                return self._maybe_autocast(value)
        if self._value is None:
            # Unitialized variable. Return a placeholder.
            # This is fine because it's only ever used
            # in during shape inference / graph tracing
            # (anything else would be a bug, to be fixed.)
            return self._maybe_autocast(
                self._initializer(self._shape, dtype=self._dtype)
            )
        return self._maybe_autocast(self._value)

    def assign(self, value):
        value = self._convert_to_tensor(value, dtype=self.dtype)
        if value.shape != self.value.shape:
            raise ValueError(
                "The shape of the target variable and "
                "the shape of the target value in "
                "`variable.assign(value)` must match. "
                f"Received: value.shape={value.shape}; "
                f"variable.shape={self.value.shape}"
            )
        if in_stateless_scope():
            scope = get_stateless_scope()
            scope.add_update((self, value))
        else:
            self._direct_assign(value)

    @property
    def dtype(self):
        autocast_scope = get_autocast_scope()
        if autocast_scope is not None and is_float_dtype(self._dtype):
            return autocast_scope.dtype
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return self._ndim

    def __repr__(self):
        return (
            f"<KerasVariable shape={self.shape}, dtype={self.dtype}, "
            f"name={self.name}>"
        )

    def _initialize(self, value):
        raise NotImplementedError

    def _convert_to_tensor(self, value, dtype=None):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.value.__getitem__(idx)

    def __array__(self, dtype=None):
        return self.value.__array__(dtype)

    def __bool__(self):
        raise TypeError("A Keras Variable cannot be used as a boolean.")

    def __neg__(self):
        return self.value.__neg__()

    def __pos__(self):
        return self.value.__pos__()

    def __abs__(self):
        return self.value.__abs__()

    def __invert__(self):
        return self.value.__invert__()

    def __eq__(self, other):
        value = self.value
        return value.__eq__(self._convert_to_tensor(other, dtype=value.dtype))

    def __ne__(self, other):
        value = self.value
        return value.__ne__(self._convert_to_tensor(other, dtype=value.dtype))

    def __lt__(self, other):
        value = self.value
        return value.__lt__(self._convert_to_tensor(other, dtype=value.dtype))

    def __le__(self, other):
        value = self.value
        return value.__le__(self._convert_to_tensor(other, dtype=value.dtype))

    def __gt__(self, other):
        value = self.value
        return value.__gt__(self._convert_to_tensor(other, dtype=value.dtype))

    def __ge__(self, other):
        value = self.value
        return value.__ge__(self._convert_to_tensor(other, dtype=value.dtype))

    def __add__(self, other):
        value = self.value
        return value.__add__(self._convert_to_tensor(other, dtype=value.dtype))

    def __radd__(self, other):
        value = self.value
        return value.__radd__(self._convert_to_tensor(other, dtype=value.dtype))

    def __sub__(self, other):
        value = self.value
        return value.__sub__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rsub__(self, other):
        value = self.value
        return value.__rsub__(self._convert_to_tensor(other, dtype=value.dtype))

    def __mul__(self, other):
        value = self.value
        return value.__mul__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rmul__(self, other):
        value = self.value
        return value.__rmul__(self._convert_to_tensor(other, dtype=value.dtype))

    def __div__(self, other):
        value = self.value
        return value.__div__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rdiv__(self, other):
        value = self.value
        return value.__rdiv__(self._convert_to_tensor(other, dtype=value.dtype))

    def __truediv__(self, other):
        value = self.value
        return value.__truediv__(
            self._convert_to_tensor(other, dtype=value.dtype)
        )

    def __rtruediv__(self, other):
        value = self.value
        return value.__rtruediv__(
            self._convert_to_tensor(other, dtype=value.dtype)
        )

    def __floordiv__(self, other):
        value = self.value
        return value.__floordiv__(
            self._convert_to_tensor(other, dtype=value.dtype)
        )

    def __rfloordiv__(self, other):
        value = self.value
        return value.__rfloordiv__(
            self._convert_to_tensor(other, dtype=value.dtype)
        )

    def __divmod__(self, other):
        value = self.value
        return value.__divmod__(
            self._convert_to_tensor(other, dtype=value.dtype)
        )

    def __rdivmod__(self, other):
        value = self.value
        return value.__rdivmod__(
            self._convert_to_tensor(other, dtype=value.dtype)
        )

    def __mod__(self, other):
        value = self.value
        return value.__mod__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rmod__(self, other):
        value = self.value
        return value.__rmod__(self._convert_to_tensor(other, dtype=value.dtype))

    def __pow__(self, other):
        value = self.value
        return value.__pow__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rpow__(self, other):
        value = self.value
        return value.__rpow__(self._convert_to_tensor(other, dtype=value.dtype))

    def __matmul__(self, other):
        value = self.value
        return value.__matmul__(
            self._convert_to_tensor(other, dtype=value.dtype)
        )

    def __rmatmul__(self, other):
        value = self.value
        return value.__rmatmul__(
            self._convert_to_tensor(other, dtype=value.dtype)
        )

    def __and__(self, other):
        value = self.value
        return value.__and__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rand__(self, other):
        value = self.value
        return value.__rand__(self._convert_to_tensor(other, dtype=value.dtype))

    def __or__(self, other):
        value = self.value
        return value.__or__(self._convert_to_tensor(other, dtype=value.dtype))

    def __ror__(self, other):
        value = self.value
        return value.__ror__(self._convert_to_tensor(other, dtype=value.dtype))

    def __xor__(self, other):
        value = self.value
        return value.__xor__(self._convert_to_tensor(other, dtype=value.dtype))

    def __rxor__(self, other):
        value = self.value
        return value.__rxor__(self._convert_to_tensor(other, dtype=value.dtype))

    def __lshift__(self, other):
        value = self.value
        return value.__lshift__(
            self._convert_to_tensor(other, dtype=value.dtype)
        )

    def __rlshift__(self, other):
        value = self.value
        return value.__rlshift__(
            self._convert_to_tensor(other, dtype=self.dtype)
        )

    def __rshift__(self, other):
        value = self.value
        return value.__rshift__(
            self._convert_to_tensor(other, dtype=value.dtype)
        )

    def __rrshift__(self, other):
        value = self.value
        return value.__rrshift__(
            self._convert_to_tensor(other, dtype=self.dtype)
        )

    def __round__(self, ndigits=None):
        value = self.value
        return value.__round__(ndigits)


def register_uninitialized_variable(variable):
    uninitialized_variables = global_state.get_global_attribute(
        "uninitialized_variables", [], set_to_default=True
    )
    uninitialized_variables.append(variable)


def initialize_all_variables():
    collection = global_state.get_global_attribute("uninitialized_variables")
    if collection:
        for v in collection:
            v._deferred_initialize()
    global_state.set_global_attribute("uninitialized_variables", [])


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
    "string",
}

PYTHON_DTYPES_MAP = {
    bool: "bool",
    int: "int",  # TBD by backend
    float: "float32",
}


def standardize_dtype(dtype):
    if dtype is None:
        return config.floatx()
    if dtype in PYTHON_DTYPES_MAP:
        dtype = PYTHON_DTYPES_MAP.get(dtype)
    if dtype == "int":
        if config.backend() == "tensorflow":
            dtype = "int64"
        else:
            dtype = "int32"
    if hasattr(dtype, "name"):
        dtype = dtype.name

    if dtype not in ALLOWED_DTYPES:
        raise ValueError(f"Invalid dtype: {dtype}")
    return dtype


def standardize_shape(
    shape, allow_dynamic_batch_size=True, allow_all_dynamic=True
):
    if not isinstance(shape, tuple):
        if shape is None:
            raise ValueError("Undefined shapes are not supported.")
        if not hasattr(shape, "__iter__"):
            raise ValueError(f"Cannot convert '{shape}' to a shape.")
        shape = tuple(shape)

    for i, e in enumerate(shape):
        if i == 0 and allow_dynamic_batch_size and e is None:
            continue
        if allow_all_dynamic and e is None:
            continue
        if not isinstance(e, int):
            msg = (
                f"Cannot convert '{shape}' to a shape. "
                f"Found invalid entry '{e}'. "
            )
            if allow_dynamic_batch_size:
                msg += (
                    "Dynamic shapes (shapes with `None` entries) "
                    f"are not allowed with the {config.backend()}, "
                    "except for the batch size (axis 0)."
                )
            else:
                msg += (
                    "Dynamic shapes (shapes with `None` entries) "
                    f"are not allowed with the {config.backend()}. "
                    "All dimensions should be positive integers, "
                    "including the batch size (axis 0)."
                )
            raise ValueError(msg)
        if e < 0:
            raise ValueError(
                f"Cannot convert '{shape}' to a shape. "
                "Negative dimensions are not allowed."
            )
    return shape


def is_float_dtype(dtype):
    if hasattr(dtype, "name"):
        dtype = dtype.name
    return dtype.startswith("float") or dtype.startswith("bfloat")


def get_autocast_scope():
    return global_state.get_global_attribute("autocast_scope")


class AutocastScope:
    """Context manager that enables the autocasting of float variables.

    Under this context manager, float `KerasVariables`s will be cast to `dtype`
    (note that `dtype` must also be float).
    """

    def __init__(self, dtype):
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
        from keras_core import backend

        if is_float_dtype(value.dtype):
            return backend.cast(value, dtype=self.dtype)
        return value

    def __enter__(self):
        self.original_scope = get_autocast_scope()
        global_state.set_global_attribute("autocast_scope", self)

    def __exit__(self, *args, **kwargs):
        global_state.set_global_attribute("autocast_scope", self.original_scope)
