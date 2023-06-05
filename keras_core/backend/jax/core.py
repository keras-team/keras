import jax
import jax.numpy as jnp
from tensorflow import nest

from keras_core.backend.common import KerasVariable
from keras_core.backend.common import standardize_dtype
from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.backend.common.stateless_scope import StatelessScope

DYNAMIC_SHAPES_OK = True


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
        all_input_ktensors = []
        built_in_types = (type(None), int, float, str, bool, complex, bytes)

        # First, separate symbolic args from other args
        static_args = []
        maybe_symbolic_args = []
        static_kwargs = {}
        maybe_symbolic_kwargs = {}
        for arg in args:
            if isinstance(arg, built_in_types):
                static_args.append(arg)
            else:
                maybe_symbolic_args.append(arg)
        for k, v in kwargs.items():
            if isinstance(v, built_in_types):
                static_kwargs[k] = v
            else:
                maybe_symbolic_kwargs[k] = v

        # Second, identify all ktensors
        def index_all_ktensors(x):
            if isinstance(x, KerasTensor):
                all_input_ktensors.append(x)
            return x

        # Third, find out if there are dynamic shapes
        maybe_symbolic_args, maybe_symbolic_kwargs = nest.map_structure(
            index_all_ktensors, (maybe_symbolic_args, maybe_symbolic_kwargs)
        )
        none_count = 0
        for x in all_input_ktensors:
            for d in x.shape:
                if d is None:
                    none_count += 1

        def convert_keras_tensor_to_jax(x, fill_value=None):
            if isinstance(x, KerasTensor):
                shape = list(x.shape)
                if fill_value:
                    for i, e in enumerate(shape):
                        if e is None:
                            shape[i] = fill_value
                jax_tensor = jax.ShapeDtypeStruct(shape, dtype=x.dtype)
                return jax_tensor
            return x

        def wrapped_fn(*args, **kwargs):
            return fn(*args, *static_args, **kwargs, **static_kwargs)

        jax_out = None
        if none_count:
            try:
                ms_args_1, ms_kwargs_1 = nest.map_structure(
                    lambda x: convert_keras_tensor_to_jax(x, fill_value=83),
                    (maybe_symbolic_args, maybe_symbolic_kwargs),
                )
                _, jax_out_1 = jax.make_jaxpr(wrapped_fn, return_shape=True)(
                    *ms_args_1, **ms_kwargs_1
                )

                ms_args_2, ms_kwargs_2 = nest.map_structure(
                    lambda x: convert_keras_tensor_to_jax(x, fill_value=89),
                    (maybe_symbolic_args, maybe_symbolic_kwargs),
                )
                _, jax_out_2 = jax.make_jaxpr(wrapped_fn, return_shape=True)(
                    *ms_args_2, **ms_kwargs_2
                )

                flat_out_1 = nest.flatten(jax_out_1)
                flat_out_2 = nest.flatten(jax_out_2)

                flat_out = []
                for x1, x2 in zip(flat_out_1, flat_out_2):
                    if isinstance(x1, jax.ShapeDtypeStruct):
                        if not isinstance(x2, jax.ShapeDtypeStruct):
                            raise ValueError("Indeterministic output ordering.")
                        shape = list(x1.shape)
                        for i, e in enumerate(x2.shape):
                            if e != shape[i]:
                                shape[i] = None
                        flat_out.append(
                            jax.ShapeDtypeStruct(shape, dtype=x1.dtype)
                        )
                    else:
                        flat_out.append(x1)
                jax_out = nest.pack_sequence_as(jax_out_1, flat_out)
            except:
                # Errors can happen when the filled dimensions
                # are not compatible with the function
                # (or when the function contains a bug).
                # In such cases we don't want to confuse users
                # with random filled dimensions and the like,
                # so we rerun a pass on the dynamic shapes,
                # which will likely error out when JAX tries to
                # validate shapes as fully static.
                # The error message will be much easier to understand.
                pass

        if jax_out is None:
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
    return zeros.at[key].add(values)


def scatter_update(inputs, indices, updates):
    indices = jnp.array(indices)
    indices = jnp.transpose(indices)
    inputs[tuple(indices)] = updates
    return inputs


def block_update(inputs, start_indices, updates):
    update_shape = updates.shape
    slices = [
        slice(start_index, start_index + update_length)
        for start_index, update_length in zip(start_indices, update_shape)
    ]
    inputs[tuple(slices)] = updates
    return inputs


def while_loop(
    cond,
    body,
    loop_vars,
    maximum_iterations=None,
):
    loop_vars = tuple(loop_vars)
    if maximum_iterations is not None:
        current_iter = 0
        loop_vars = loop_vars + (current_iter,)

        # Unpack list/tuple args. The last argument is `current_iter`.
        def _cond(args):
            return cond(*args[:-1]) & (args[-1] < maximum_iterations)

        def _body(args):
            return tuple(body(*args[:-1])) + (args[-1] + 1,)

    else:

        def _cond(args):
            return cond(*args)

        def _body(args):
            return tuple(body(*args))

    outputs = jax.lax.while_loop(_cond, _body, loop_vars)
    if maximum_iterations is not None:
        outputs = outputs[:-1]
    return outputs
