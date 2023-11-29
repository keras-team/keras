import numpy as np

from keras.api_export import keras_export
from keras.backend import config
from keras.backend.common import global_state
from keras.backend.common.name_scope import current_path
from keras.backend.common.stateless_scope import get_stateless_scope
from keras.backend.common.stateless_scope import in_stateless_scope
from keras.utils.naming import auto_name


class KerasVariable:
    def __init__(
        self, initializer, shape=None, dtype=None, trainable=True, name=None
    ):
        name = name or auto_name(self.__class__.__name__)
        if not isinstance(name, str) or "/" in name:
            raise ValueError(
                "Argument `name` must be a string and "
                "cannot contain character `/`. "
                f"Received: name={name}"
            )
        self.name = name
        parent_path = current_path()
        if parent_path:
            self.path = current_path() + "/" + self.name
        else:
            self.path = self.name
        dtype = standardize_dtype(dtype)
        self._dtype = dtype
        self._shape = None
        self._initializer = None
        self._trainable = trainable
        if isinstance(initializer, str):
            from keras import initializers

            initializer = initializers.get(initializer)
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
                self._shape = self._validate_shape(shape)
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
                shape = self._validate_shape(shape)
                value = initializer(shape, dtype=dtype)
            else:
                value = initializer
            self._initialize(value)
            self._shape = tuple(self._value.shape)
        self._ndim = len(self._shape)

    def _deferred_initialize(self):
        if self._value is not None:
            raise ValueError(f"Variable {self.path} is already initialized.")

        if in_stateless_scope():
            raise ValueError(
                "You are attempting to initialize a variable "
                "while in a stateless scope. This is disallowed. "
                "Make sure that all variables are initialized "
                "before you start using your layer/model objects."
            )
        value = self._initializer(self._shape, dtype=self._dtype)
        self._initialize(value)

    def _validate_shape(self, shape):
        shape = standardize_shape(shape)
        if None in shape:
            raise ValueError(
                "Shapes used to initialize variables must be "
                "fully-defined (no `None` dimensions). Received: "
                f"shape={shape} for variable path='{self.path}'"
            )
        return shape

    def _maybe_autocast(self, value):
        autocast_scope = get_autocast_scope()
        if autocast_scope is not None:
            return autocast_scope.maybe_cast(value)
        return value

    def numpy(self):
        return np.array(self)

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
        if not shape_equal(value.shape, self.shape):
            raise ValueError(
                "The shape of the target variable and "
                "the shape of the target value in "
                "`variable.assign(value)` must match. "
                f"variable.shape={self.value.shape}, "
                f"Received: value.shape={value.shape}. "
                f"Target variable: {self}"
            )
        if in_stateless_scope():
            scope = get_stateless_scope()
            scope.add_update((self, value))
        else:
            self._direct_assign(value)

    def assign_add(self, value):
        self.assign(self + value)

    def assign_sub(self, value):
        self.assign(self - value)

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

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value

    def __repr__(self):
        return (
            f"<KerasVariable shape={self.shape}, dtype={self.dtype}, "
            f"path={self.path}>"
        )

    def _initialize(self, value):
        raise NotImplementedError

    def _convert_to_tensor(self, value, dtype=None):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.value.__getitem__(idx)

    def __array__(self, dtype=None):
        # We can't directly use self.value.__array__ here because of scalar.
        # Numpy require this method to return as array like object. In the case
        # of scalar, it will fail the type checking from numpy. We need to
        # return a 0d array via numpy.
        return np.asarray(self.value.__array__(dtype))

    def __bool__(self):
        raise TypeError("A Keras Variable cannot be used as a boolean.")

    def __neg__(self):
        return self.value.__neg__()

    def __pos__(self):
        return self.value

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
    int: "int64" if config.backend() == "tensorflow" else "int32",
    float: "float32",
    str: "string",
    # special case for string value
    "int": "int64" if config.backend() == "tensorflow" else "int32",
}


@keras_export("keras.backend.standardize_dtype")
def standardize_dtype(dtype):
    if dtype is None:
        return config.floatx()
    dtype = PYTHON_DTYPES_MAP.get(dtype, dtype)
    if hasattr(dtype, "name"):
        dtype = dtype.name
    elif hasattr(dtype, "__str__") and (
        "torch" in str(dtype) or "jax.numpy" in str(dtype)
    ):
        dtype = str(dtype).split(".")[-1]
    elif hasattr(dtype, "__name__"):
        dtype = dtype.__name__

    if dtype not in ALLOWED_DTYPES:
        raise ValueError(f"Invalid dtype: {dtype}")
    return dtype


def standardize_shape(shape):
    if not isinstance(shape, tuple):
        if shape is None:
            raise ValueError("Undefined shapes are not supported.")
        if not hasattr(shape, "__iter__"):
            raise ValueError(f"Cannot convert '{shape}' to a shape.")
        shape = tuple(shape)

    if config.backend() == "torch":
        # `shape` might be `torch.Size`. We need to convert the items in it to
        # either int or `None`
        shape = tuple(map(lambda x: int(x) if x is not None else None, shape))

    for e in shape:
        if e is None:
            continue
        if config.backend() == "jax" and str(e) == "b":
            # JAX2TF tracing represents `None` dimensions as `b`
            continue
        if not is_int_dtype(type(e)):
            raise ValueError(
                f"Cannot convert '{shape}' to a shape. "
                f"Found invalid entry '{e}' of type '{type(e)}'. "
            )
        if e < 0:
            raise ValueError(
                f"Cannot convert '{shape}' to a shape. "
                "Negative dimensions are not allowed."
            )
    return shape


def shape_equal(a_shape, b_shape):
    """Return whether a_shape == b_shape (allows None entries)."""
    if len(a_shape) != len(b_shape):
        return False
    for e1, e2 in zip(a_shape, b_shape):
        if e1 is not None and e2 is not None and e1 != e2:
            return False
    return True


@keras_export("keras.backend.is_float_dtype")
def is_float_dtype(dtype):
    dtype = standardize_dtype(dtype)
    return dtype.startswith("float") or dtype.startswith("bfloat")


@keras_export("keras.backend.is_int_dtype")
def is_int_dtype(dtype):
    dtype = standardize_dtype(dtype)
    return dtype.startswith("int") or dtype.startswith("uint")


def get_autocast_scope():
    return global_state.get_global_attribute("autocast_scope")


class AutocastScope:
    """Context manager that enables the autocasting of float variables.

    Under this context manager, float `KerasVariables`s will be cast to `dtype`
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
        from keras import backend

        if self.dtype is not None and is_float_dtype(value.dtype):
            return backend.cast(value, dtype=self.dtype)
        return value

    def __enter__(self):
        self.original_scope = get_autocast_scope()
        global_state.set_global_attribute("autocast_scope", self)

    def __exit__(self, *args, **kwargs):
        global_state.set_global_attribute("autocast_scope", self.original_scope)
