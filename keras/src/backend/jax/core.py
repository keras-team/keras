import jax
import jax.experimental.sparse as jax_sparse
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.jax import distribution_lib

SUPPORTS_SPARSE_TENSORS = True


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
        return convert_to_tensor(value, dtype=dtype, sparse=False)

    # Overload native accessor.
    def __jax_array__(self):
        return self.value


def convert_to_tensor(x, dtype=None, sparse=True):
    if dtype is not None:
        dtype = standardize_dtype(dtype)
    if isinstance(x, (jnp.ndarray, jax.Array)) and (
        dtype is None or x.dtype == dtype
    ):
        # Skip the conversion early if the instance is already a JAX array.
        # This is important in the multi-process context since jax.array(x) for
        # an existing distributed jax array will raise error.
        return x

    if isinstance(x, Variable):
        if dtype is not None and x.dtype != dtype:
            return x.value.astype(dtype)
        return x.value

    if isinstance(x, jax_sparse.JAXSparse):
        if sparse is not None and not sparse:
            x = x.todense()
        elif dtype is not None and x.dtype != dtype:
            return x.astype(dtype)
        else:
            return x

    if not is_tensor(x) and standardize_dtype(dtype) == "bfloat16":
        # Can't create bfloat16 arrays on the fly (e.g. from a h5 Dataset).
        # Instead we convert "as is" (to stored dtype) and cast.
        return jnp.asarray(x).astype(dtype)
    return jnp.asarray(x, dtype=dtype)


def convert_to_numpy(x):
    if isinstance(x, jax_sparse.JAXSparse):
        x = x.todense()
    if is_tensor(x) and x.dtype == "bfloat16":
        return np.asarray(x, ml_dtypes.bfloat16)
    return np.asarray(x)


def is_tensor(x):
    if isinstance(x, (jnp.ndarray, jax_sparse.JAXSparse)):
        return True
    return False


def shape(x):
    # This will work as long as we disallow
    # dynamic shapes in JAX.
    return x.shape


def cast(x, dtype):
    return convert_to_tensor(x, dtype=dtype)


# Shape / dtype / sparseness inference util
def compute_output_spec(fn, *args, **kwargs):
    with StatelessScope():
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
        maybe_symbolic_args = tuple(maybe_symbolic_args)
        for k, v in kwargs.items():
            if isinstance(v, built_in_types):
                static_kwargs[k] = v
            else:
                maybe_symbolic_kwargs[k] = v

        # Second, find out if there are dynamic shapes
        has_none = False
        for x in tree.flatten((maybe_symbolic_args, maybe_symbolic_kwargs)):
            if isinstance(x, KerasTensor) and any(d is None for d in x.shape):
                has_none = True

        def convert_keras_tensor_to_jax(x, fill_value=None):
            if isinstance(x, KerasTensor):
                shape = list(x.shape)
                if fill_value:
                    for i, e in enumerate(shape):
                        if e is None:
                            shape[i] = fill_value
                jax_tensor = jax.ShapeDtypeStruct(shape, dtype=x.dtype)
                return jax_tensor
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

            # Turn inputs that are sparse to BCOO tensors
            def to_bcoo_if_sparse(x, maybe_symbolic_x):
                if (
                    isinstance(maybe_symbolic_x, KerasTensor)
                    and maybe_symbolic_x.sparse
                ):
                    return jax_sparse.BCOO.fromdense(x, nse=1)
                return x

            args, kwargs = tree.map_structure(
                to_bcoo_if_sparse,
                (args, kwargs),
                (maybe_symbolic_args, maybe_symbolic_kwargs),
            )

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

        output_spec = None
        if has_none:
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

                def merge_shapes(shape1, shape2):
                    return tuple(
                        [
                            d1 if d1 == d2 else None
                            for d1, d2 in zip(shape1, shape2)
                        ]
                    )

                def convert_jax_specs_to_keras_tensor(x1, x2):
                    if isinstance(x1, jax.ShapeDtypeStruct):
                        if not isinstance(x2, jax.ShapeDtypeStruct):
                            raise ValueError("Indeterministic output ordering.")
                        return KerasTensor(
                            merge_shapes(x1.shape, x2.shape), dtype=x1.dtype
                        )
                    elif isinstance(x1, jax_sparse.BCOO):
                        if not isinstance(x2, jax_sparse.BCOO):
                            raise ValueError("Indeterministic output ordering.")
                        return KerasTensor(
                            merge_shapes(x1.shape, x2.shape),
                            dtype=x1.dtype,
                            sparse=True,
                        )
                    else:
                        return x1

                output_spec = tree.map_structure(
                    convert_jax_specs_to_keras_tensor, jax_out_1, jax_out_2
                )
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

        if output_spec is None:
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
                elif isinstance(x, jax_sparse.BCOO):
                    return KerasTensor(x.shape, x.dtype, sparse=True)
                return x

            output_spec = tree.map_structure(
                convert_jax_spec_to_keras_tensor, jax_out
            )

    return output_spec


def cond(pred, true_fn, false_fn):
    return jax.lax.cond(pred, true_fun=true_fn, false_fun=false_fn)


def vectorized_map(function, elements):
    return jax.vmap(function)(elements)


def scatter(indices, values, shape):
    zeros = jnp.zeros(shape, values.dtype)
    key = tuple(jnp.moveaxis(indices, -1, 0))
    return zeros.at[key].add(values)


def scatter_update(inputs, indices, updates):
    inputs = convert_to_tensor(inputs)
    indices = jnp.array(indices)
    indices = jnp.transpose(indices)
    inputs = inputs.at[tuple(indices)].set(updates)
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
    is_tuple = isinstance(loop_vars, (tuple, list))
    loop_vars = tuple(loop_vars) if is_tuple else (loop_vars,)
    if maximum_iterations is not None:
        current_iter = 0
        loop_vars = loop_vars + (current_iter,)

        # Unpack list/tuple args. The last argument is `current_iter`.
        def _cond(args):
            return cond(*args[:-1]) & (args[-1] < maximum_iterations)

        def _body(args):
            outputs = body(*args[:-1])
            outputs = tuple(outputs) if is_tuple else (outputs,)
            return outputs + (args[-1] + 1,)

    else:

        def _cond(args):
            return cond(*args)

        def _body(args):
            outputs = body(*args)
            return tuple(outputs) if is_tuple else (outputs,)

    outputs = jax.lax.while_loop(_cond, _body, loop_vars)
    if maximum_iterations is not None:
        outputs = outputs[:-1]
    return outputs if is_tuple else outputs[0]


def fori_loop(lower, upper, body_fun, init_val):
    return jax.lax.fori_loop(lower, upper, body_fun, init_val)


def stop_gradient(variable):
    return jax.lax.stop_gradient(variable)


def unstack(x, num=None, axis=0):
    return [
        jax.lax.index_in_dim(x, i, axis, keepdims=False)
        for i in range(x.shape[axis])
    ]


def custom_gradient(fun):
    return jax.custom_gradient(fun=fun)


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
