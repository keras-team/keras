import jax
import jax.experimental.sparse as jax_sparse
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax import nnx

from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.name_scope import name_scope as base_name_scope
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.symbolic_scope import SymbolicScope
from keras.src.backend.jax import distribution_lib

SUPPORTS_SPARSE_TENSORS = True
SUPPORTS_RAGGED_TENSORS = False
IS_THREAD_SAFE = True


def in_stateless_scope():
    return global_state.get_global_attribute("stateless_scope") is not None


def get_stateless_scope():
    return global_state.get_global_attribute("stateless_scope")


def shape_equal(a_shape, b_shape):
    """Return whether a_shape == b_shape (allows None entries)."""
    if len(a_shape) != len(b_shape):
        return False
    for e1, e2 in zip(a_shape, b_shape):
        if e1 is not None and e2 is not None and e1 != e2:
            return False
    return True


# existing implementation
class JaxVariable(KerasVariable):
    def __init__(self, *args, layout=None, **kwargs):
        self._layout = layout
        super().__init__(*args, **kwargs)

    def _initialize(self, value):
        self._shape = self._validate_shape(value.shape)
        distribution = global_state.get_global_attribute("distribution")
        if self._layout is None and distribution is not None:
            tensor_layout = distribution.get_variable_layout(self)
            from keras.src.distribution import TensorLayout

            if isinstance(tensor_layout, TensorLayout):
                self._layout = tensor_layout.backend_layout
            else:
                self._layout = tensor_layout
        self._direct_assign(value)

    def _direct_assign(self, value):
        if self._layout is not None:
            value = distribution_lib.distribute_variable(value, self._layout)
        self._value = value

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype, sparse=False)

    def __jax_array__(self):
        return self.value


class Variable(JaxVariable, nnx.Variable):
    def __init__(
        self,
        initializer,
        shape=None,
        dtype=None,
        trainable=True,
        # Keras specific args from KerasVariable
        autocast=True,
        aggregation="none",
        synchronization="auto",
        name=None,
        # Keras JAX backend specific
        layout=None,
        # NNX specific args
        nnx_mutable=None,  # NNX's own mutable flag
        *args,
        **nnx_metadata,  # For nnx.Variable's **metadata
    ):
        # We need to call KerasJaxVariableImpl.__init__ first
        # KerasJaxVariableImpl's __init__ takes `layout` specifically
        # and forwards other Keras common args to CommonKerasVariable.__init__
        super(
            JaxVariable, self
        ).__init__(  # Explicitly call KerasJaxVariableImpl's __init__
            initializer=initializer,
            shape=shape,
            dtype=dtype,
            trainable=trainable,
            autocast=autocast,
            aggregation=aggregation,
            synchronization=synchronization,
            name=name,
            layout=layout,
            *args,
        )

        # Store NNX args for potential deferred initialization
        self._nnx_mutable_arg = nnx_mutable
        self._nnx_metadata_arg = nnx_metadata.copy()
        self._nnx_init_pending = True

        # If Keras initialization was not deferred, self._value is now set.
        # So we can proceed to initialize the nnx.Variable part.
        if self._initializer is None:
            self._complete_nnx_init()

    def _complete_nnx_init(self):
        """Initializes the nnx.Variable part of this instance."""
        if not self._nnx_init_pending:
            return  # Already done

        if self._value is None:
            # This can happen if _deferred_initialize was called but _value somehow didn't get set
            # Or if this is called too early. Keras's _initializer should not be None here.
            raise ValueError(
                "Cannot initialize NNX part: Keras self._value is None, "
                "but Keras initializer is also None (should not be deferred)."
            )

        # Determine nnx_mutable for nnx.Variable.__init__
        # If user didn't specify nnx_mutable, default to Keras's trainable status.
        current_nnx_mutable = self._nnx_mutable_arg
        if current_nnx_mutable is None:
            current_nnx_mutable = self.trainable  # A sensible default link

        # initialize the nnx.Variable
        nnx.Variable.__init__(
            self,
            value=self._value,
            mutable=current_nnx_mutable,
            **self._nnx_metadata_arg,
        )
        self._nnx_init_pending = False

    def _deferred_initialize(self):
        # This is called by Keras when it's time to actually create the variable's value
        super()._deferred_initialize()
        self._complete_nnx_init()

    def _direct_assign(self, value):
        super()._direct_assign(value)  # This sets self._value

        # After self._value is updated by Keras, sync nnx.Variable.raw_value
        # Only if NNX part is already initialized.
        if not self._nnx_init_pending:
            nnx_stores_mutable = False
            if (
                self._nnx_mutable_arg is None
            ):  # Check how nnx_mutable was resolved
                nnx_stores_mutable = self.trainable
            else:
                nnx_stores_mutable = self._nnx_mutable_arg

            if nnx_stores_mutable and nnx.utils.is_mutable_array(
                self.raw_value
            ):
                # If raw_value is a mutable_array, update its content
                self.raw_value[...] = self._value
            else:
                object.__setattr__(self, "raw_value", self._value)

    @property
    def value(self):
        # This will be KerasVariable.value:
        return super().value

    @value.setter
    def value(self, new_value):
        self.assign(
            new_value
        )  # assign will call _direct_assign, which syncs raw_value

    # Overriding NNX methods that modify `raw_value` or `_var_metadata` directly
    # to ensure Keras's `_value` and other Keras states are in sync.

    def copy_from(self, other: nnx.Variable):  # type: ignore
        if not isinstance(other, nnx.Variable):  # Basic check from nnx
            raise TypeError(
                f"Expected nnx.Variable, got {type(other).__name__}"
            )
        if not isinstance(other, Variable):
            pass

        # Let nnx.Variable handle its part (updates self.raw_value and self._var_metadata)
        # Need to call nnx.Variable.copy_from specifically.
        nnx.Variable.copy_from(self, other)

        # Now, self.raw_value is updated. Sync Keras's self._value.
        # Extract the JAX array if raw_value is a nnx.mutable_array
        keras_value_to_assign = self.raw_value
        if nnx.utils.is_mutable_array(keras_value_to_assign):
            keras_value_to_assign = keras_value_to_assign.__array__()

        self.assign(keras_value_to_assign)

        # Sync Keras-specific attributes if `other` is also a JaxNnxVariable
        if isinstance(other, Variable):
            self.trainable = other.trainable
            self._autocast = other._autocast
            self._aggregation = other._aggregation
            if hasattr(other, "_layout"):
                self._layout = other._layout

    def update_from_state(self, variable_state: nnx.graph.VariableState):
        # Let nnx.Variable handle its part (updates self.raw_value and self._var_metadata)
        nnx.Variable.update_from_state(self, variable_state)

        # Sync Keras's self._value
        keras_value_to_assign = self.raw_value
        if nnx.utils.is_mutable_array(keras_value_to_assign):
            keras_value_to_assign = keras_value_to_assign.__array__()

        self.assign(keras_value_to_assign)

        # Sync Keras attributes if they were part of variable_state.metadata
        if "trainable" in variable_state._var_metadata:  # type: ignore
            self.trainable = variable_state._var_metadata["trainable"]
            self._autocast = variable_state._var_metadata["autocast"]

    def __getstate__(self):
        keras_state = {
            # Keras common attributes (from CommonKerasVariable)
            "_name": self._name,
            "_path": self._path,
            "_trainable": self._trainable,
            "_dtype": self._dtype,
            "_shape": self._shape,
            "_autocast": self._autocast,
            "_aggregation": self._aggregation,
            "_synchronization": self._synchronization,
            "_regularizer": self._regularizer,
            "_constraint": self._constraint,
            # Keras JAX backend specific
            "_layout": self._layout,
            # Value itself (will be part of nnx_state's raw_value too)
            "_value": self._value,  # Keras's value (JAX array)
            "_initializer": self._initializer,  # In case it's not initialized
            # NNX specific args that were stored at init
            "_nnx_mutable_arg": self._nnx_mutable_arg,
            "_nnx_metadata_arg": self._nnx_metadata_arg,
            "_nnx_init_pending": self._nnx_init_pending,
        }
        nnx_state = nnx.Variable.__getstate__(self)
        return {"keras_state": keras_state, "nnx_state": nnx_state}

    def __setstate__(self, state):
        keras_state = state["keras_state"]
        nnx_state = state["nnx_state"]

        # Restore Keras attributes
        for k, v in keras_state.items():
            object.__setattr__(self, k, v)

        # Restore NNX attributes using its __setstate__
        nnx.Variable.__setstate__(self, nnx_state)

        if (
            self._initializer is not None and self._value is None
        ):  # Was deferred pre-pickle
            if (
                not self._nnx_init_pending
                and hasattr(self, "raw_value")
                and self.raw_value is not None
            ):
                pass  # self._value is already set from keras_state.

        # If self._value exists (from Keras state), ensure nnx.raw_value matches
        if self._value is not None:
            if self._nnx_init_pending:
                self._complete_nnx_init()
            else:
                # This is similar to _direct_assign's sync logic.
                current_nnx_mutable = self._nnx_mutable_arg
                if current_nnx_mutable is None:
                    current_nnx_mutable = self.trainable

                if current_nnx_mutable and nnx.utils.is_mutable_array(
                    self.raw_value
                ):
                    self.raw_value[...] = self._value
                else:
                    object.__setattr__(self, "raw_value", self._value)
        elif (
            not self._nnx_init_pending
            and hasattr(self, "raw_value")
            and self.raw_value is not None
        ):
            object.__setattr__(self, "_value", self.raw_value)


def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    if ragged:
        raise ValueError("`ragged=True` is not supported with jax backend")
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
        return np.array(x, dtype=ml_dtypes.bfloat16)
    return np.array(x)


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
    with StatelessScope(), SymbolicScope():
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

        if has_none:
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
                    [d1 if d1 == d2 else None for d1, d2 in zip(shape1, shape2)]
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

            return tree.map_structure(
                convert_jax_specs_to_keras_tensor, jax_out_1, jax_out_2
            )

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

        return tree.map_structure(convert_jax_spec_to_keras_tensor, jax_out)


def cond(pred, true_fn, false_fn):
    return jax.lax.cond(pred, true_fun=true_fn, false_fun=false_fn)


def vectorized_map(function, elements):
    return jax.vmap(function)(elements)


def map(f, xs):
    return jax.lax.map(f, xs)


def scan(f, init, xs=None, length=None, reverse=False, unroll=1):
    if not isinstance(unroll, bool):
        if not isinstance(unroll, int) or unroll < 1:
            raise ValueError(
                "`unroll` must be an positive integer or boolean. "
                f"Received: unroll={unroll}"
            )
    return jax.lax.scan(
        f, init=init, xs=xs, length=length, reverse=reverse, unroll=unroll
    )


def associative_scan(f, elems, reverse=False, axis=0):
    return jax.lax.associative_scan(f, elems, reverse, axis)


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


def switch(index, branches, *operands):
    return jax.lax.switch(index, branches, *operands)


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
    if isinstance(variable, Variable):
        variable = variable.value
    return jax.lax.stop_gradient(variable)


def unstack(x, num=None, axis=0):
    return [
        jax.lax.index_in_dim(x, i, axis, keepdims=False)
        for i in range(x.shape[axis])
    ]


def random_seed_dtype():
    # jax random seed uses uint32.
    return "uint32"


def custom_gradient(fun):
    return jax.custom_gradient(fun=fun)


def remat(f):
    """Implementation of rematerialization.

    Args:
        f: The function or operation to rematerialize.
    Returns:
        A function wrapping f that defines a custom gradient, which
        recomputes f on the backwards pass of a gradient call.
    """
    return jax.checkpoint(f)


class name_scope(base_name_scope):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._jax_name_scope = jax.named_scope(name)

    def __enter__(self):
        name_scope_stack = global_state.get_global_attribute(
            "name_scope_stack", default=[], set_to_default=True
        )
        if self.deduplicate and name_scope_stack:
            parent_caller = name_scope_stack[-1].caller
            parent_name = name_scope_stack[-1].name
            if (
                self.caller is not None
                and self.caller is parent_caller
                and self.name == parent_name
            ):
                return self
        name_scope_stack.append(self)
        self._pop_on_exit = True
        self._jax_name_scope.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        super().__exit__(*args, **kwargs)
        if self._pop_on_exit:
            self._jax_name_scope.__exit__(*args, **kwargs)


def device_scope(device_name):
    if isinstance(device_name, str):
        # We support string value like "cpu:0", "gpu:1", etc.
        device_name = device_name.lower()
        jax_device = distribution_lib._to_backend_device(device_name)
    elif not isinstance(device_name, jax.Device):
        raise ValueError(
            "Invalid value for argument `device_name`. "
            "Expected a string like 'gpu:0' or a `jax.Device` instance. "
            f"Received: device_name='{device_name}'"
        )
    else:
        jax_device = device_name
    return jax.default_device(jax_device)
