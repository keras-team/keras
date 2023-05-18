import jax
import jax.numpy as jnp
from tensorflow import nest

from keras_core.backend.common import KerasVariable
from keras_core.backend.common import standardize_dtype
from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.backend.common.stateless_scope import StatelessScope

DYNAMIC_SHAPES_OK = False  # Dynamic shapes NG
DYNAMIC_BATCH_SIZE_OK = True


class Variable(KerasVariable):
    def _initialize(self, value):
        self._value = jnp.array(value, dtype=self._dtype)

    def _direct_assign(self, value):
        self._value = value

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)

    # Overload native accessor.
    def __jax_array__(self):
        return self.value


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
        dynamic_batch_map = {}
        magic_number = 3

        def convert_keras_tensor_to_jax(x):
            if isinstance(x, KerasTensor):
                shape = x.shape
                if shape and x.shape[0] is None:
                    shape = list(shape)
                    shape[0] = magic_number
                    dynamic_batch = True
                else:
                    dynamic_batch = False

                jax_tensor = jax.ShapeDtypeStruct(shape, dtype=x.dtype)
                dynamic_batch_map[jax_tensor] = dynamic_batch
                return jax_tensor
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
                if dynamic_batch_map.get(x, False):
                    shape = list(x.shape)
                    if shape[0] != magic_number:
                        raise ValueError(
                            f"Function {fn} appears to change the "
                            "batch size of its input. This is not "
                            "allowed when used in conjunction with "
                            "dynamic batch sizes. Consider using "
                            "a static batch size here."
                        )
                    shape[0] = None
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
    return zeros.at[key].add(values)
