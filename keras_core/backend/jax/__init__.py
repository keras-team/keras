import jax
import numpy as np
from jax import numpy as jnp
from tensorflow import nest

from keras_core.backend.common import KerasVariable
from keras_core.backend.common import standardize_dtype
from keras_core.backend.jax import nn
from keras_core.backend.jax import numpy
from keras_core.backend.jax import random
from keras_core.backend.keras_tensor import KerasTensor
from keras_core.backend.stateless_scope import StatelessScope
from keras_core.backend.stateless_scope import get_stateless_scope
from keras_core.backend.stateless_scope import in_stateless_scope
from keras_core.utils.naming import auto_name

DYNAMIC_SHAPES_OK = False  # Dynamic shapes NG


def convert_to_tensor(x, dtype=None):
    dtype = standardize_dtype(dtype)
    if isinstance(x, Variable):
        if dtype and dtype != x.dtype:
            return x.value.astype(dtype)
        return x.value
    return jnp.array(x, dtype=dtype)


def is_tensor(x):
    if isinstance(x, jnp.ndarray):
        return True
    return False


def shape(x):
    # This will work as long as we disallow
    # dynamic shapes in JAX.
    return x.shape


def cast(x, dtype):
    return convert_to_tensor(x, dtype=dtype)


def cond(pred, true_fn, false_fn):
    return jax.lax.cond(pred, true_fn=true_fn, false_fun=false_fn)


def name_scope(name):
    return jax.named_scope(name)


class Variable(KerasVariable):
    def __init__(self, value, dtype=None, trainable=True, name=None):
        self.name = name or auto_name(self.__class__.__name__)
        dtype = standardize_dtype(dtype)
        self._value = jnp.array(value, dtype=dtype)
        self._dtype = dtype
        self._shape = tuple(self._value.shape)
        self._ndim = len(self._shape)
        self.trainable = trainable

    def assign(self, value):
        value = convert_to_tensor(value, dtype=self.dtype)
        if value.shape != self.shape:
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
            if isinstance(value, jnp.ndarray) and value.dtype == self.dtype:
                # Avoid a memory copy
                self._value = value
            else:
                self._value = jnp.array(value, dtype=self.dtype)

    @property
    def value(self):
        if in_stateless_scope():
            scope = get_stateless_scope()
            value = scope.get_current_value(self)
            if value is not None:
                return value
        return self._value

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return self._ndim

    def numpy(self):
        return np.array(self.value)

    # Overload native accessor.
    def __jax_array__(self):
        return self.value

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
        return self.value.__eq__(convert_to_tensor(other))

    def __ne__(self, other):
        return self.value.__ne__(convert_to_tensor(other))

    def __lt__(self, other):
        return self.value.__lt__(convert_to_tensor(other))

    def __le__(self, other):
        return self.value.__le__(convert_to_tensor(other))

    def __gt__(self, other):
        return self.value.__gt__(convert_to_tensor(other))

    def __ge__(self, other):
        return self.value.__ge__(convert_to_tensor(other))

    def __add__(self, other):
        return self.value.__add__(convert_to_tensor(other))

    def __radd__(self, other):
        return self.value.__radd__(convert_to_tensor(other))

    def __sub__(self, other):
        return self.value.__sub__(convert_to_tensor(other))

    def __rsub__(self, other):
        return self.value.__rsub__(convert_to_tensor(other))

    def __mul__(self, other):
        return self.value.__mul__(convert_to_tensor(other))

    def __rmul__(self, other):
        return self.value.__rmul__(convert_to_tensor(other))

    def __div__(self, other):
        return self.value.__div__(convert_to_tensor(other))

    def __rdiv__(self, other):
        return self.value.__rdiv__(convert_to_tensor(other))

    def __truediv__(self, other):
        return self.value.__truediv__(convert_to_tensor(other))

    def __rtruediv__(self, other):
        return self.value.__rtruediv__(convert_to_tensor(other))

    def __floordiv__(self, other):
        return self.value.__floordiv__(convert_to_tensor(other))

    def __rfloordiv__(self, other):
        return self.value.__rfloordiv__(convert_to_tensor(other))

    def __divmod__(self, other):
        return self.value.__divmod__(convert_to_tensor(other))

    def __rdivmod__(self, other):
        return self.value.__rdivmod__(convert_to_tensor(other))

    def __mod__(self, other):
        return self.value.__mod__(convert_to_tensor(other))

    def __rmod__(self, other):
        return self.value.__rmod__(convert_to_tensor(other))

    def __pow__(self, other):
        return self.value.__pow__(convert_to_tensor(other))

    def __rpow__(self, other):
        return self.value.__rpow__(convert_to_tensor(other))

    def __matmul__(self, other):
        return self.value.__matmul__(convert_to_tensor(other))

    def __rmatmul__(self, other):
        return self.value.__rmatmul__(convert_to_tensor(other))

    def __and__(self, other):
        return self.value.__and__(convert_to_tensor(other))

    def __rand__(self, other):
        return self.value.__rand__(convert_to_tensor(other))

    def __or__(self, other):
        return self.value.__or__(convert_to_tensor(other))

    def __ror__(self, other):
        return self.value.__ror__(convert_to_tensor(other))

    def __xor__(self, other):
        return self.value.__xor__(convert_to_tensor(other))

    def __rxor__(self, other):
        return self.value.__rxor__(convert_to_tensor(other))

    def __lshift__(self, other):
        return self.value.__lshift__(convert_to_tensor(other))

    def __rlshift__(self, other):
        return self.value.__rlshift__(convert_to_tensor(other))

    def __rshift__(self, other):
        return self.value.__rshift__(convert_to_tensor(other))

    def __rrshift__(self, other):
        return self.value.__rrshift__(convert_to_tensor(other))

    def __round__(self, ndigits=None):
        return self.value.__round__(ndigits)


# Shape / dtype inference util
def compute_output_spec(fn, *args, **kwargs):
    with StatelessScope():

        def convert_keras_tensor_to_jax(x):
            if isinstance(x, KerasTensor):
                return jax.ShapeDtypeStruct(x.shape, dtype=x.dtype)
            return x

        built_in_types = (type(None), int, float, str, bool, complex, bytes)
        static_args = []
        maybe_symbolic_args = []
        for arg in args:
            if isinstance(arg, built_in_types):
                static_args.append(arg)
            else:
                maybe_symbolic_args.append(arg)
        static_kwargs = {}
        maybe_symbolic_kwargs = {}
        for (
            k,
            arg,
        ) in kwargs.items():
            if isinstance(arg, built_in_types):
                static_kwargs[k] = arg
            else:
                maybe_symbolic_kwargs[k] = arg

        def wrapped_fn(*args, **kwargs):
            return fn(*args, *static_args, **kwargs, **static_kwargs)

        maybe_symbolic_args, maybe_symbolic_kwargs = nest.map_structure(
            convert_keras_tensor_to_jax,
            (maybe_symbolic_args, maybe_symbolic_kwargs),
        )
        _, jax_out = jax.make_jaxpr(wrapped_fn, return_shape=True)(
            *maybe_symbolic_args, **maybe_symbolic_kwargs
        )

        def convert_jax_spec_to_keras_tensor(x):
            if isinstance(x, jax.ShapeDtypeStruct):
                return KerasTensor(x.shape, x.dtype)
            return x

        return nest.map_structure(convert_jax_spec_to_keras_tensor, jax_out)


# NumPy op delegation
def execute(op_name, *args, **kwargs):
    if hasattr(jnp, op_name):
        op = getattr(jnp, op_name)
        return op(*args, **kwargs)
    raise AttributeError(f"The JAX backend does not support op '{op_name}'")
