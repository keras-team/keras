import types

import jax
import jax.numpy as jnp
import numpy as np
import tree
from jax.tree_util import Partial

from keras.backend.common import KerasVariable
from keras.backend.common import global_state
from keras.backend.common import standardize_dtype
from keras.backend.common.keras_tensor import KerasTensor
from keras.backend.common.stateless_scope import StatelessScope
from keras.backend.jax import distribution_lib
from keras.utils.nest import pack_sequence_as

SUPPORTS_SPARSE_TENSORS = False


class Variable(KerasVariable):
    def _initialize(self, value):
        value = jnp.array(value, dtype=self._dtype)
        # Note that variable.shape is needed by distribution_lib
        self._shape = tuple(value.shape)
        # We can't import the keras/distribution/distribution_lib
        # due to circular dependency.
        distribution = global_state.get_global_attribute("distribution")
        if distribution is not None:
            self._layout = distribution_lib._to_jax_layout(
                distribution.get_variable_layout(self)
            )
        else:
            self._layout = None
        self._direct_assign(value)

    def _direct_assign(self, value):
        if getattr(self, "_layout", None) is not None:
            value = distribution_lib.distribute_variable(value, self._layout)
        self._value = value

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)

    # Overload native accessor.
    def __jax_array__(self):
        return self.value


def convert_to_tensor(x, dtype=None, sparse=None):
    if sparse:
        raise ValueError("`sparse=True` is not supported with jax backend")
    if dtype is not None:
        dtype = standardize_dtype(dtype)
    if isinstance(x, (jnp.ndarray, jax.Array)) and dtype == x.dtype:
        # Skip the conversion early if the instance is already a JAX array.
        # This is important in the multi-process context since jax.array(x) for
        # an existing distributed jax array will raise error.
        return x

    if isinstance(x, Variable):
        if dtype and dtype != x.dtype:
            return x.value.astype(dtype)
        return x.value
    return jnp.array(x, dtype=dtype)


def convert_to_numpy(x):
    return np.array(x)


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


# Shape / dtype inference util
def compute_output_spec(fn, *args, **kwargs):
    with StatelessScope():
        all_input_ktensors = []
        built_in_types = (type(None), int, float, str, bool, complex, bytes)

        # First, separate symbolic args from other args
        static_args_idx = []
        static_args = []
        maybe_symbolic_args = []
        static_kwargs = {}
        maybe_symbolic_kwargs = {}
        for idx, arg in enumerate(args):
            if isinstance(arg, built_in_types):
                static_args_idx.append(idx)
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
        maybe_symbolic_args, maybe_symbolic_kwargs = tree.map_structure(
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
            if isinstance(x, types.FunctionType):

                def _fn(*args, **kwargs):
                    out = x(*args, **kwargs)
                    out = convert_keras_tensor_to_jax(
                        out, fill_value=fill_value
                    )
                    return out

                return Partial(_fn)
            if isinstance(x, dict):
                return {
                    k: convert_keras_tensor_to_jax(v, fill_value=fill_value)
                    for k, v in x.items()
                }
            if isinstance(x, list):
                return [
                    convert_keras_tensor_to_jax(xi, fill_value=fill_value)
                    for xi in x
                ]
            return x

        def wrapped_fn(*args, **kwargs):
            rec_args = []
            idx_static = 0
            idx_sym = 0
            i = 0
            while idx_static < len(static_args) or idx_sym < len(args):
                if i in static_args_idx:
                    rec_args.append(static_args[idx_static])
                    idx_static += 1
                else:
                    rec_args.append(args[idx_sym])
                    idx_sym += 1

                i += 1
            with StatelessScope():
                return fn(*rec_args, **kwargs, **static_kwargs)

        jax_out = None
        if none_count:
            try:
                ms_args_1, ms_kwargs_1 = tree.map_structure(
                    lambda x: convert_keras_tensor_to_jax(x, fill_value=83),
                    (maybe_symbolic_args, maybe_symbolic_kwargs),
                )
                _, jax_out_1 = jax.make_jaxpr(wrapped_fn, return_shape=True)(
                    *ms_args_1, **ms_kwargs_1
                )

                ms_args_2, ms_kwargs_2 = tree.map_structure(
                    lambda x: convert_keras_tensor_to_jax(x, fill_value=89),
                    (maybe_symbolic_args, maybe_symbolic_kwargs),
                )
                _, jax_out_2 = jax.make_jaxpr(wrapped_fn, return_shape=True)(
                    *ms_args_2, **ms_kwargs_2
                )

                flat_out_1 = tree.flatten(jax_out_1)
                flat_out_2 = tree.flatten(jax_out_2)

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
                jax_out = pack_sequence_as(jax_out_1, flat_out)
            except Exception as e:
                if "[JAX RNG]" in str(e):
                    raise e
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
            maybe_symbolic_args, maybe_symbolic_kwargs = tree.map_structure(
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

        output_shape = tree.map_structure(
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


def slice(inputs, start_indices, shape):
    return jax.lax.dynamic_slice(inputs, start_indices, shape)


def slice_update(inputs, start_indices, updates):
    return jax.lax.dynamic_update_slice(inputs, updates, start_indices)


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


def fori_loop(lower, upper, body_fun, init_val):
    return jax.lax.fori_loop(lower, upper, body_fun, init_val)


def stop_gradient(variable):
    return jax.lax.stop_gradient(variable)


def unstack(x, num=None, axis=0):
    return [
        jax.lax.index_in_dim(x, i, axis, keepdims=False)
        for i in range(x.shape[axis])
    ]


def device_scope(device_name):
    if isinstance(device_name, str):
        # We support string value like "cpu:0", "gpu:1", etc.
        device_name = device_name.lower()
        jax_device = distribution_lib._to_jax_device(device_name)
    elif not isinstance(device_name, jax.Device):
        raise ValueError(
            "Invalid value for argument `device_name`. "
            "Expected a string like 'gpu:0' or a `jax.Device` instance. "
            f"Received: device_name='{device_name}'"
        )
    else:
        jax_device = device_name
    return jax.default_device(jax_device)
