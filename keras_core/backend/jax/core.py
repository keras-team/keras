import jax
import jax.numpy as jnp
import numpy as np
from tensorflow import nest

from keras_core.backend.common import KerasVariable
from keras_core.backend.common import standardize_dtype
from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.backend.common.stateless_scope import StatelessScope
from keras_core.backend.common.stateless_scope import get_stateless_scope
from keras_core.backend.common.stateless_scope import in_stateless_scope

DYNAMIC_SHAPES_OK = True  # Dynamic shapes NG


class Variable(KerasVariable):
    def _initialize(self, value):
        self._value = jnp.array(value, dtype=self._dtype)

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
                return self._maybe_autocast(value)
        if self._value is None:
            # Unitialized variable. Return a placeholder.
            # This is fine because it's only ever used
            # in during shape inference with JAX tracer objects
            # (anything else would be a bug, to be fixed.)
            return self._maybe_autocast(
                self._initializer(self._shape, dtype=self._dtype)
            )
        return self._maybe_autocast(self._value)

    def numpy(self):
        return np.array(self.value)

    # Overload native accessor.
    def __jax_array__(self):
        return self.value

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)


def convert_to_tensor(x, dtype=None):
    if dtype is not None:
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


def name_scope(name):
    return jax.named_scope(name)


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

        output_shape = nest.map_structure(
            convert_jax_spec_to_keras_tensor, jax_out
        )
    return output_shape


def cond(pred, true_fn, false_fn):
    return jax.lax.cond(pred, true_fun=true_fn, false_fun=false_fn)


def vectorized_map(function, elements):
    return jax.vmap(function)(elements)


def scatter(indices, values, shape):
    zeros = jnp.zeros(shape, values.dtype)
    key = tuple(jnp.moveaxis(indices, -1, 0))
    return zeros.at[key].set(values)
