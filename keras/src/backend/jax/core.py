import jax
import jax.experimental.sparse as jax_sparse
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from absl import logging
from jax import export as jax_export

from keras.src import tree
from keras.src.backend import config
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.name_scope import name_scope as base_name_scope
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.stateless_scope import get_stateless_scope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.backend.common.symbolic_scope import SymbolicScope
from keras.src.backend.jax import distribution_lib
from keras.src.utils import jax_utils

SUPPORTS_SPARSE_TENSORS = True
SUPPORTS_RAGGED_TENSORS = False
IS_THREAD_SAFE = True


def _safe_has_addressable_shards(x):
    """Safely check if x has addressable_shards without tracer errors."""
    return (
        isinstance(x, jax.Array)
        and not jax_utils.is_in_jax_tracing_scope(x)
        and hasattr(x, "addressable_shards")
    )


class _ProtectedShardedArray:
    """Wrapper that prevents deletion of sharded JAX arrays.

    This wrapper intercepts delete() calls from jax_memory_cleanup
    and prevents deletion of sharded arrays that are needed for inference.
    """

    def __init__(self, array):
        self._array = array
        self._is_sharded = _safe_has_addressable_shards(array)

    def __getattr__(self, name):
        # Delegate all attribute access to the wrapped array
        return getattr(self._array, name)

    def delete(self):
        """Intercept delete() calls and prevent deletion of sharded arrays."""
        if self._is_sharded:
            # Don't actually delete sharded arrays
            return
        else:
            # Allow deletion of non-sharded arrays
            self._array.delete()

    def __repr__(self):
        return f"_ProtectedShardedArray({self._array})"


def _initialize_variable_with_sharding(
    variable, value, log_prefix="_initialize"
):
    """Shared helper for variable initialization with sharding support.

    This function handles the common logic for both JaxVariable and NnxVariable
    initialization, including layout detection, logging, and tensor
    distribution.

    Args:
        variable: The variable instance being initialized
        value: The initial value
        log_prefix: Prefix for logging messages

    Returns:
        The processed value ready for assignment
    """
    import numpy as np

    # Validate shape first
    variable._shape = variable._validate_shape(value.shape)

    # Detect layout from distribution if needed
    distribution = global_state.get_global_attribute("distribution")
    if variable._layout is None and distribution is not None:
        logging.debug(
            f"{log_prefix}: Getting layout for variable "
            f"'{variable.path}' from distribution"
        )
        tensor_layout = distribution.get_variable_layout(variable)
        logging.debug(
            f"{log_prefix}: Distribution returned layout: {tensor_layout}"
        )
        from keras.src.distribution import TensorLayout

        if isinstance(tensor_layout, TensorLayout):
            variable._layout = tensor_layout.backend_layout
            logging.debug(
                f"{log_prefix}: Using backend_layout: {variable._layout}"
            )
        else:
            variable._layout = tensor_layout
            logging.debug(
                f"{log_prefix}: Using layout directly: {variable._layout}"
            )

    # Log initialization details
    total_elements = np.prod(variable._shape)
    element_size = np.dtype(variable.dtype).itemsize
    total_size_mb = (total_elements * element_size) / (1024 * 1024)

    logging.info(f"{log_prefix}: Creating variable '{variable.path}'")
    logging.debug(
        f"{log_prefix}: Shape: {variable._shape}, Size: {total_size_mb:.2f} MB"
    )
    logging.debug(f"{log_prefix}: Has layout: {variable._layout is not None}")

    # If we have a layout, distribute the tensor to avoid OOM
    if variable._layout is not None:
        logging.info(
            f"{log_prefix}: Sharded initialization (layout: {variable._layout})"
        )

        if isinstance(value, (jnp.ndarray, jax.Array)):
            if hasattr(value, "device") and value.device.platform == "cpu":
                logging.debug(
                    f"{log_prefix}: JAX array already on CPU (host memory)"
                )
            else:
                value = jax.device_put(value, jax.devices("cpu")[0])
                logging.debug(
                    f"{log_prefix}: Moved JAX array to CPU (host memory)"
                )
        elif not isinstance(value, np.ndarray):
            value = np.array(value)
            logging.debug(
                f"{log_prefix}: Converted to numpy array (host memory)"
            )
        else:
            logging.debug(
                f"{log_prefix}: Value already numpy array (host memory)"
            )

        # Distribute to devices - this shards the tensor
        value = distribution_lib.distribute_tensor(value, variable._layout)
        logging.debug(f"{log_prefix}: Tensor distributed across devices")

        # Log sharding info
        if hasattr(value, "sharding") and _safe_has_addressable_shards(value):
            shards = value.addressable_shards
            num_devices = len(shards)
            shard_0_elements = np.prod(shards[0].data.shape)
            shard_0_size_mb = (shard_0_elements * element_size) / (1024 * 1024)

            logging.debug(f"{log_prefix}: Sharded across {num_devices} devices")
            logging.debug(
                f"{log_prefix}: Device 0 shard: {shards[0].data.shape}, "
                f"{shard_0_size_mb:.2f} MB"
            )
            # Calculate memory reduction percentage
            mem_reduction = (
                (total_size_mb - shard_0_size_mb) / total_size_mb * 100
            )
            logging.debug(
                f"{log_prefix}: Memory reduction: {mem_reduction:.1f}%"
            )
    else:
        logging.debug(f"{log_prefix}: NORMAL (non-sharded) initialization")
        # Convert to tensor using normal path
        value = variable._convert_to_tensor(value)

    variable._maybe_create_strong_reference(value)

    return value


class JaxVariable(KerasVariable):
    def __init__(self, *args, layout=None, **kwargs):
        # Intercept layout parameter so that it is available
        # during initialization.
        self._layout = layout
        super().__init__(*args, **kwargs)

    def _maybe_create_strong_reference(self, value):
        """Create a strong ref to a JAX array to prevent GC."""
        # Skip creating references for NNX variables during symbolic computation
        # as NNX doesn't allow mutation during tracing
        if hasattr(self, "_trace_state") and SymbolicScope():
            return

        if isinstance(value, jax.Array):
            try:
                # Check if this is a JAX tracer (during compilation/tracing)
                if jax_utils.is_in_jax_tracing_scope(value):
                    # During tracing, we can't access addressable_shards
                    # Just hold a reference to the tracer itself
                    self._strong_reference = value
                elif hasattr(value, "addressable_shards"):
                    # For sharded arrays, hold references to the shards' data.
                    shard_data = [
                        shard.data for shard in value.addressable_shards
                    ]
                    self._shard_references = [shard_data]
                else:
                    # For non-sharded arrays, hold a ref to the array itself.
                    self._strong_reference = value
            except (AttributeError, TypeError):
                # If we can't set attributes (e.g., during tracing), skip
                pass

    @property
    def value(self):
        var_name = (
            getattr(self, "path", None)
            or getattr(self, "name", None)
            or str(self)
        )
        logging.debug(f" JaxVariable.value for {var_name}")
        current_value = super().value
        # Unwrap protected arrays
        if isinstance(current_value, _ProtectedShardedArray):
            current_value = current_value._array
        self._maybe_create_strong_reference(current_value)
        return current_value

    def _initialize(self, value):
        """Initialize variable with sharding support.

        This method handles both regular and sharded variable initialization.
        When a layout is present, it distributes the tensor across devices
        during initialization to avoid OOM on device 0.
        """
        value = _initialize_variable_with_sharding(self, value)

        # Set the value (this is the critical part!)
        if hasattr(self, "raw_value"):
            # NNX variable
            object.__setattr__(self, "raw_value", value)
        else:
            # Regular JAX variable - protect sharded arrays from deletion
            if _safe_has_addressable_shards(value):
                self._value = _ProtectedShardedArray(value)
            else:
                self._value = value

        logging.info(
            f"_initialize: Variable '{self.path}' initialized successfully"
        )

    def _initialize_with_initializer(self, initializer):
        """Initialize variable with initializer, running on CPU if sharding
        is needed."""
        if self._layout is not None:
            # For sharded variables, run initializer on CPU to avoid device
            # placement issues
            with jax.default_device(jax.devices("cpu")[0]):
                value = self._convert_to_tensor(
                    initializer(self._shape, dtype=self._dtype)
                )
        else:
            # For non-sharded variables, use the default behavior
            value = self._convert_to_tensor(
                initializer(self._shape, dtype=self._dtype)
            )
        self._initialize(value)

    def _direct_assign(self, value):
        """Assign value to variable with sharding support.

        This is used during weight loading. For sharded variables,
        it distributes the weight data across devices to avoid OOM.
        """

        if self._layout is not None:
            logging.debug(
                f"_direct_assign: Distributing variable '{self.path}' "
                f"with layout"
            )
            logging.debug(
                f"_direct_assign: Original value shape: {value.shape}"
            )
            # Distribute the value (this shards it)
            value = distribution_lib.distribute_variable(value, self._layout)
            logging.debug("_direct_assign: Value distributed successfully")

            # Log sharding details
            if hasattr(value, "sharding") and _safe_has_addressable_shards(
                value
            ):
                shards = value.addressable_shards
                num_devices = len(shards)
                logging.debug(
                    f"_direct_assign: Sharded across {num_devices} devices"
                )

        self._maybe_create_strong_reference(value)

        # Assign the value - protect sharded arrays from deletion
        if _safe_has_addressable_shards(value):
            self._value = _ProtectedShardedArray(value)
        else:
            self._value = value

        logging.info(
            f"_direct_assign: Variable '{self.path}' assigned successfully"
        )

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype, sparse=False)

    # Overload native accessor.
    def __jax_array__(self):
        return self.value


Variable = JaxVariable
if config.is_nnx_enabled():
    from flax import nnx

    class NnxVariable(JaxVariable, nnx.Variable):
        def __init__(
            self,
            initializer,
            shape=None,
            dtype=None,
            trainable=True,
            autocast=True,
            aggregation="none",
            synchronization="auto",
            name=None,
            layout=None,
            mutable=None,
            **nnx_metadata,
        ):
            # Ensure 'mutable' is in nnx_metadata, but explicit 'mutable'
            # param takes precedence.
            nnx_metadata["mutable"] = trainable if mutable is None else mutable

            # First, initialize a basic nnx.Variable with a dummy value
            # This sets up the NNX variable structure
            if shape is None:
                dummy_value = jnp.array(0.0)
            else:
                dummy_value = jnp.zeros(shape, dtype=standardize_dtype(dtype))

            # Initialize nnx.Variable first
            nnx.Variable.__init__(self, value=dummy_value, **nnx_metadata)

            # Now we can safely set layout
            self._layout = layout

            # Initialize JaxVariable (which will call KerasVariable.__init__
            # and set up the real value).
            JaxVariable.__init__(
                self,
                initializer=initializer,
                shape=shape,
                dtype=dtype,
                trainable=trainable,
                autocast=autocast,
                aggregation=aggregation,
                synchronization=synchronization,
                name=name,
            )

            # The real value is now set in self._value, sync it to raw_value
            object.__setattr__(self, "raw_value", self._value)

        @property
        def _value(self):
            if hasattr(self, "raw_value"):
                return self.raw_value
            return None

        @_value.setter
        def _value(self, new_keras_value):
            self._direct_assign(new_keras_value)

        def __getstate__(self):
            # Get the state from KerasVariable (attributes in __dict__)
            # KerasVariable does not have a custom __getstate__, so we mimic
            # default behavior.
            try:
                keras_state = KerasVariable.__getstate__(self)
            except AttributeError:
                keras_state = object.__getstate__(self)

            # Get the state from nnx.Variable
            nnx_specific_state = nnx.Variable.__getstate__(self)

            # Merge them. Keras state is primary. NNX specific state adds
            # to it.
            if "raw_value" in nnx_specific_state:
                keras_state["_value"] = nnx_specific_state["raw_value"]

            # Add NNX attributes that are not in Keras's __dict__
            if "_trace_state" in nnx_specific_state:
                keras_state["_trace_state"] = nnx_specific_state["_trace_state"]
            if "_var_metadata" in nnx_specific_state:
                keras_state["_var_metadata"] = nnx_specific_state[
                    "_var_metadata"
                ]

            # Remove elements that might be problematic or redundant if
            # nnx.Variable's __getstate__
            keras_state.pop("raw_value", None)

            return keras_state

        def __setstate__(self, state):
            # Separate nnx specific keys that we added if they are not part
            # of Keras __dict__ this __getstate__ puts them into the main
            # state dictionary.
            nnx_raw_value = state["_value"]  # This was raw_value
            nnx_trace_state = state.pop("_trace_state", None)
            nnx_var_metadata = state.pop("_var_metadata", None)

            # Populate the instance's __dict__ with the Keras attributes.
            self.__dict__.update(state)

            # restore the nnx.Variable specific slotted attributes.
            object.__setattr__(self, "raw_value", nnx_raw_value)

            if nnx_trace_state is not None:
                object.__setattr__(self, "_trace_state", nnx_trace_state)
            else:
                pass

            if nnx_var_metadata is not None:
                object.__setattr__(self, "_var_metadata", nnx_var_metadata)
            else:
                pass

            # Ensure Keras's self._value is also consistent with the
            # restored raw_value
            self._value = nnx_raw_value

            if hasattr(self, "_shape") and self._shape is not None:
                self._ndim = len(self._shape)
            else:
                # Fallback if shape isn't immediately available.
                self._ndim = len(self.raw_value.shape)

        def _initialize(self, value):
            """Initialize NNX variable with sharding support."""
            value = _initialize_variable_with_sharding(
                self, value, "_initialize (NNX)"
            )

            # Set value for NNX
            object.__setattr__(self, "raw_value", value)

            logging.info(
                f"_initialize (NNX): Variable '{self.path}' initialized"
            )

        def _direct_assign(self, value):
            """Assign value to NNX variable with sharding support.

            Used during weight loading for sharded variables.
            Accepts both NumPy arrays and JAX arrays.
            """
            import numpy as np

            if self._layout is not None:
                logging.debug(
                    f"_direct_assign (NNX): Distributing '{self.path}'"
                )

                if isinstance(value, np.ndarray):
                    logging.debug("_direct_assign (NNX): Value is numpy array")
                elif isinstance(value, (jnp.ndarray, jax.Array)):
                    logging.debug("_direct_assign (NNX): Value is JAX array")

                # Distribute
                value = distribution_lib.distribute_variable(
                    value, self._layout
                )
                logging.debug("_direct_assign (NNX): Distributed successfully")

            # Apply on_set_value hook if exists
            if (
                hasattr(self, "_var_metadata")
                and "on_set_value" in self._var_metadata
            ):
                value = self._var_metadata["on_set_value"](self, value)

            # JAX automatically blocks when array properties are accessed
            self._maybe_create_strong_reference(value)
            # Set value for NNX
            object.__setattr__(self, "raw_value", value)

            logging.info(
                f"_direct_assign (NNX): Variable '{self.path}' assigned"
            )

        @property
        def value(self):
            if in_stateless_scope():
                scope = get_stateless_scope()
                stateless_value = scope.get_current_value(self)
                if stateless_value is not None:
                    return self._maybe_autocast(stateless_value)
            if not hasattr(self, "raw_value"):
                if self._initializer is not None:
                    self._initialize(
                        self._initializer(self.shape, dtype=self.dtype)
                    )
                else:
                    raise AttributeError(
                        "Variable is not properly initialized (raw_value "
                        "missing) and has no initializer."
                    )
            current_value = self.raw_value
            self._maybe_create_strong_reference(current_value)

            if (
                hasattr(self, "_var_metadata")
                and "on_get_value" in self._var_metadata
            ):
                current_value = self._var_metadata["on_get_value"](
                    self, current_value
                )
            return self._maybe_autocast(current_value)

    Variable = NnxVariable


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

        # We create a single dynamic dimension and reuse it instead of creating
        # N dynamic dimensions. This is for backwards compatibility. Previously
        # we would fill all dynamic dimensions with the same concrete value.
        # This can handle the case where there is an implicit assumption that
        # two dimensions are the same (e.g. square images).
        #
        # We add the constraint "dynamic_dimension>=2" to prevent JAX from
        # assuming that the dimension can be broadcastable or squeezable. It
        # removes this ambiguity.
        dynamic_dimension = jax_export.symbolic_shape(
            "(dynamic_dimension)",
            constraints=["dynamic_dimension>=2"],
        )[0]

        def convert_keras_tensor_to_jax(x):
            if isinstance(x, KerasTensor):
                shape = tuple(
                    [d if d is not None else dynamic_dimension for d in x.shape]
                )
                return jax.ShapeDtypeStruct(shape, dtype=x.dtype)
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

        maybe_symbolic_args_jax, maybe_symbolic_kwargs_jax = tree.map_structure(
            convert_keras_tensor_to_jax,
            (maybe_symbolic_args, maybe_symbolic_kwargs),
        )
        jax_out = jax.eval_shape(
            wrapped_fn, *maybe_symbolic_args_jax, **maybe_symbolic_kwargs_jax
        )

        def convert_jax_spec_to_keras_tensor(x):
            if isinstance(x, jax.ShapeDtypeStruct):
                shape = tuple(
                    d if isinstance(d, int) else None for d in x.shape
                )
                return KerasTensor(shape, x.dtype)
            elif isinstance(x, jax_sparse.BCOO):
                shape = tuple(
                    d if isinstance(d, int) else None for d in x.shape
                )
                return KerasTensor(shape, x.dtype, sparse=True)
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
    # If shape[i] is -1, all remaining elements in dimension i are included in
    # the slice.
    final_shape = tuple(
        inputs.shape[i] - start_indices[i] if s == -1 else s
        for i, s in enumerate(shape)
    )
    return jax.lax.dynamic_slice(inputs, start_indices, final_shape)


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
