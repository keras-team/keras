import builtins
import contextlib
import weakref

import ml_dtypes
import numpy as np
import paddle
import paddle.nn

from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.stateless_scope import get_stateless_scope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.backend.common.symbolic_scope import SymbolicScope
from keras.src.backend.config import floatx

SUPPORTS_SPARSE_TENSORS = False
SUPPORTS_RAGGED_TENSORS = False
SUPPORTS_COMPLEX_DTYPES = True
IS_THREAD_SAFE = True

DEFAULT_DEVICE = "cpu"

PADDLE_DTYPES = {
    "float16": paddle.float16,
    "float32": paddle.float32,
    "float64": paddle.float64,
    "uint8": paddle.uint8,
    "uint16": paddle.int32,  # Paddle doesn't have uint16
    "uint32": paddle.int64,  # Paddle doesn't have uint32
    "int8": paddle.int8,
    "int16": paddle.int16,
    "int32": paddle.int32,
    "int64": paddle.int64,
    "bfloat16": paddle.bfloat16,
    "bool": paddle.bool,
    "float8_e4m3fn": paddle.float8_e4m3fn,
    "float8_e5m2": paddle.float8_e5m2,
    "complex64": paddle.complex64,
    "complex128": paddle.complex128,
}

# Track logical dtypes for uint16/uint32 which Paddle maps to int32/int64
_logical_dtypes = {}
# Dtypes that Paddle maps to different physical dtypes
_MAPPED_DTYPES = {"uint16", "uint32"}
# Track tensors created from Python scalars (weak types in JAX sense)
_weak_tensors = weakref.WeakSet()


def _maybe_track_dtype(tensor, logical_dtype):
    """Store the logical dtype if it differs from the physical paddle dtype."""
    if logical_dtype is None:
        return
    std = standardize_dtype(logical_dtype)
    if std in _MAPPED_DTYPES:
        _logical_dtypes[id(tensor)] = std


def paddle_standardize_dtype(dtype):
    """standardize_dtype that checks paddle's logical dtype tracking."""
    # Check if this dtype belongs to a tensor with a tracked logical dtype
    tid = getattr(dtype, "_paddle_tensor_id", None)
    if tid is not None and tid in _logical_dtypes:
        return _logical_dtypes[tid]
    return standardize_dtype(dtype)


@contextlib.contextmanager
def device_scope(device_name):
    previous_device = paddle.get_device()
    current_device = _parse_device_input(device_name)
    paddle.set_device(current_device)
    global_state.set_global_attribute("paddle_device", current_device)
    try:
        yield current_device
    finally:
        paddle.set_device(previous_device)
        global_state.set_global_attribute("paddle_device", previous_device)


def get_device():
    device = global_state.get_global_attribute("paddle_device", None)
    if device is None:
        return DEFAULT_DEVICE
    return device


def _parse_device_input(device_name):
    if isinstance(device_name, str):
        device_name = device_name.lower()
        # Paddle uses "gpu:0", "cpu" format
        return device_name
    raise ValueError(
        "Invalid value for argument `device_name`. "
        "Expected a string like 'gpu:0' or 'cpu'. "
        f"Received: device_name='{device_name}'"
    )


def to_paddle_dtype(dtype):
    standardized_dtype = PADDLE_DTYPES.get(standardize_dtype(dtype), None)
    if standardized_dtype is None:
        raise ValueError(f"Unsupported dtype for Paddle: {dtype}")
    return standardized_dtype


class Variable(KerasVariable):
    def _initialize(self, value):
        if isinstance(value, paddle.Tensor) and not value.stop_gradient:
            self._value = value
        else:
            self._value = convert_to_tensor(value, dtype=self._dtype)
            self._value.stop_gradient = not self.trainable

    def _direct_assign(self, value):
        self._value.set_value(value)

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)

    def __array__(self, dtype=None):
        value = convert_to_numpy(self.value)
        if dtype:
            return value.astype(dtype)
        return value

    @property
    def value(self):
        def maybe_use_symbolic_tensor(value):
            return value

        if in_stateless_scope():
            scope = get_stateless_scope()
            value = scope.get_current_value(self)
            if value is not None:
                value = self._maybe_autocast(value)
                return maybe_use_symbolic_tensor(value)
        if self._value is None:
            value = self._maybe_autocast(
                self._initializer(self._shape, dtype=self._dtype)
            )
        else:
            value = self._maybe_autocast(self._value)
        return maybe_use_symbolic_tensor(value)

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value
        if self._value is not None:
            self._value.stop_gradient = not value


def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    if sparse:
        raise ValueError("`sparse=True` is not supported with paddle backend")
    if ragged:
        raise ValueError("`ragged=True` is not supported with paddle backend")
    if isinstance(x, Variable) or is_tensor(x):
        if isinstance(x, Variable):
            x = x.value
        if dtype is not None:
            x = x.cast(to_paddle_dtype(dtype))
        _maybe_track_dtype(x, dtype)
        return x
    if isinstance(x, (bool, int, float, complex)):
        if dtype is not None:
            dt = to_paddle_dtype(dtype)
        elif isinstance(x, bool):
            dt = paddle.bool
        elif isinstance(x, int):
            dt = paddle.int64 if x < -(2**31) or x >= 2**31 else paddle.int32
        elif isinstance(x, float):
            dt = to_paddle_dtype(floatx())
        else:
            dt = paddle.complex64
        t = paddle.to_tensor(x, dtype=dt)
        # Only mark as weak when no explicit dtype was provided
        if dtype is None:
            _weak_tensors.add(t)
        return t

    # Convert to np in case of any array-like that is not list or tuple.
    if isinstance(x, (list, tuple)):
        if len(x) > 0 and any(
            is_tensor(item) or isinstance(item, Variable)
            for item in tree.flatten(x)
        ):
            return paddle.stack(
                [convert_to_tensor(x1, dtype=dtype) for x1 in x]
            )
    elif not isinstance(x, (bool, int, float)):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        if x.dtype == np.uint32:
            x = x.astype(np.int64)
        if standardize_dtype(x.dtype) == "bfloat16":
            x = x.astype(np.float32)
            dtype = "bfloat16"
        dtype = dtype or x.dtype
    if dtype is None:
        dtype = result_type(
            *[getattr(item, "dtype", type(item)) for item in tree.flatten(x)]
        )
    dtype = to_paddle_dtype(dtype)
    return paddle.to_tensor(x, dtype=dtype)


def convert_to_numpy(x):
    def transform(x):
        if is_tensor(x):
            if not x.stop_gradient:
                x = x.detach()
            if x.dtype == paddle.bfloat16:
                return np.array(x.cast(paddle.float32)).astype(
                    ml_dtypes.bfloat16
                )
        return np.array(x)

    if isinstance(x, (list, tuple)):
        if tree.is_nested(x):
            return tree.map_structure(transform, x)
        return np.array([transform(e) for e in x])
    return transform(x)


def is_tensor(x):
    return isinstance(x, paddle.Tensor)


def shape(x):
    return tuple(d if d >= 0 else None for d in x.shape)


def cast(x, dtype):
    dtype = to_paddle_dtype(dtype)
    if isinstance(x, Variable):
        x = x.value
    if is_tensor(x):
        if x.dtype == dtype:
            return x
        return x.cast(dtype)
    return convert_to_tensor(x, dtype)


def compute_output_spec(fn, *args, **kwargs):
    def has_none_shape(x):
        if isinstance(x, KerasTensor):
            return None in x.shape
        return False

    def convert_keras_tensor_to_paddle(x, fill_value=None):
        if isinstance(x, KerasTensor):
            if x.shape is None:
                out_shape = [fill_value or 1]
            else:
                out_shape = list(x.shape)
                if fill_value:
                    for i, e in enumerate(out_shape):
                        if e is None:
                            out_shape[i] = fill_value
            return paddle.ones(
                shape=out_shape,
                dtype=to_paddle_dtype(x.dtype),
            )
        return x

    def convert_paddle_to_keras_tensor(x):
        if is_tensor(x):
            return KerasTensor(x.shape, standardize_dtype(x.dtype))
        return x

    def symbolic_call(fn, args, kwargs, fill_value):
        eager_args, eager_kwargs = tree.map_structure(
            lambda x: convert_keras_tensor_to_paddle(x, fill_value),
            (args, kwargs),
        )
        return fn(*eager_args, **eager_kwargs)

    with StatelessScope(), SymbolicScope(), paddle.no_grad():
        outputs = symbolic_call(fn, args, kwargs, fill_value=83)

        none_in_shape = any(
            builtins.map(has_none_shape, tree.flatten((args, kwargs)))
        )
        if none_in_shape:
            outputs_1 = outputs
            outputs_2 = symbolic_call(fn, args, kwargs, fill_value=89)

            flat_out_1 = tree.flatten(outputs_1)
            flat_out_2 = tree.flatten(outputs_2)

            flat_out = []
            for x1, x2 in zip(flat_out_1, flat_out_2):
                if not is_tensor(x1):
                    flat_out.append(x1)
                    continue
                out_shape = list(x1.shape)
                for i, e in enumerate(x2.shape):
                    if e != out_shape[i]:
                        out_shape[i] = None
                flat_out.append(
                    KerasTensor(out_shape, standardize_dtype(x1.dtype))
                )
            outputs = tree.pack_sequence_as(outputs_1, flat_out)

        output_spec = tree.map_structure(
            convert_paddle_to_keras_tensor, outputs
        )
    return output_spec


def cond(pred, true_fn, false_fn):
    if is_tensor(pred):
        pred = bool(pred.item())
    if pred:
        return true_fn()
    return false_fn()


def vectorized_map(function, elements):
    # Simple fallback for paddle: map over the first (batch) dimension
    if isinstance(elements, (list, tuple)):
        batch_size = elements[0].shape[0]
        results = [
            function(tuple(e[i] for e in elements)) for i in range(batch_size)
        ]
    else:
        batch_size = elements.shape[0]
        results = [function(elements[i]) for i in range(batch_size)]
    return paddle.stack(results)


def map(f, xs):
    def g(_, x):
        return (), f(x)

    _, ys = scan(g, (), xs)
    return ys


def scan(f, init, xs=None, length=None, reverse=False, unroll=1):
    if not callable(f):
        raise TypeError(f"`f` should be a callable. Received: f={f}")
    if not isinstance(unroll, bool):
        if not isinstance(unroll, int) or unroll < 1:
            raise ValueError(
                "`unroll` must be an positive integer or boolean. "
                f"Received: unroll={unroll}"
            )
    if xs is None and length is None:
        raise ValueError("Got no `xs` to scan over and `length` not provided.")

    input_is_sequence = tree.is_nested(xs)
    output_is_sequence = tree.is_nested(init)

    def pack_input(x):
        return tree.pack_sequence_as(xs, x) if input_is_sequence else x[0]

    def pack_output(x):
        return tree.pack_sequence_as(init, x) if output_is_sequence else x[0]

    if xs is None:
        xs_flat = []
        n = int(length)
    else:
        xs_flat = tree.flatten(xs)
        xs_flat = [convert_to_tensor(elem) for elem in xs_flat]
        n = int(length) if length is not None else shape(xs_flat[0])[0]

    init_flat = tree.flatten(init)
    init_flat = [convert_to_tensor(i) for i in init_flat]
    init = pack_output(init_flat)
    dummy_y = [paddle.zeros_like(i) for i in init_flat]

    carry = init
    ys = []
    maybe_reversed = reversed if reverse else lambda x: x
    for i in maybe_reversed(range(n)):
        xs_slice = [x[i] for x in xs_flat]
        packed_xs = pack_input(xs_slice) if len(xs_slice) > 0 else None
        carry, y = f(carry, packed_xs)
        ys.append(y if y is not None else dummy_y)
    ordered_ys = list(maybe_reversed(ys))
    flat_ys = [tree.flatten(y) for y in ordered_ys]
    flat_stacked = [paddle.stack(tensors) for tensors in zip(*flat_ys)]
    stacked_y = tree.pack_sequence_as(ordered_ys[0], flat_stacked)
    return carry, stacked_y


def associative_scan(f, elems, reverse=False, axis=0):
    raise NotImplementedError(
        "`associative_scan` is not supported with paddle backend"
    )


def scatter(indices, values, shape):
    indices = convert_to_tensor(indices)
    values = convert_to_tensor(values)
    zeros = paddle.zeros(shape, dtype=values.dtype)

    index_length = indices.shape[-1]
    value_shape = shape[index_length:]
    indices = paddle.reshape(indices, [-1, index_length])
    values = paddle.reshape(values, [-1] + list(value_shape))

    return paddle.scatter_nd_add(zeros, indices, values)


def scatter_update(inputs, indices, updates, reduction=None):
    inputs = convert_to_tensor(inputs)
    indices = convert_to_tensor(indices, dtype="int64")
    updates = convert_to_tensor(updates, dtype=inputs.dtype)

    if reduction is None:
        current_values = paddle.gather_nd(inputs, indices)
        return paddle.scatter_nd_add(inputs, indices, updates - current_values)
    elif reduction == "add":
        return paddle.scatter_nd_add(inputs, indices, updates)

    indices_t = paddle.transpose(indices, [1, 0])
    outputs = inputs.clone()
    if reduction == "max":
        for i in range(indices.shape[0]):
            idx = tuple(indices_t[j][i] for j in range(indices_t.shape[0]))
            outputs[idx] = paddle.maximum(outputs[idx], updates[i])
    elif reduction == "min":
        for i in range(indices.shape[0]):
            idx = tuple(indices_t[j][i] for j in range(indices_t.shape[0]))
            outputs[idx] = paddle.minimum(outputs[idx], updates[i])
    elif reduction == "mul":
        for i in range(indices.shape[0]):
            idx = tuple(indices_t[j][i] for j in range(indices_t.shape[0]))
            outputs[idx] = outputs[idx] * updates[i]
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
    return outputs


def slice(inputs, start_indices, shape):
    inputs = convert_to_tensor(inputs)
    if isinstance(start_indices, (list, tuple)) and isinstance(
        shape, (list, tuple)
    ):
        if all(isinstance(s, int) for s in start_indices) and all(
            isinstance(s, int) for s in shape
        ):
            slices = [
                builtins.slice(start_index, start_index + length)
                for start_index, length in zip(start_indices, shape)
            ]
            return inputs[tuple(slices)]

    shape_dtype = paddle.int64
    start_indices = convert_to_tensor(start_indices).cast(shape_dtype)
    shape = convert_to_tensor(shape).cast(shape_dtype)
    axes = list(range(int(start_indices.shape[0])))
    return paddle.slice(
        inputs,
        axes=axes,
        starts=start_indices,
        ends=start_indices + shape,
    )


def slice_update(inputs, start_indices, updates):
    inputs = convert_to_tensor(inputs)
    updates = convert_to_tensor(updates)

    if isinstance(start_indices, (list, tuple)) and all(
        isinstance(s, int) for s in start_indices
    ):
        slices = [
            builtins.slice(start_index, start_index + update_length)
            for start_index, update_length in zip(start_indices, updates.shape)
        ]
        outputs = inputs.clone()
        outputs[tuple(slices)] = updates
        return outputs

    start_indices = convert_to_tensor(start_indices, dtype="int64")
    if hasattr(start_indices, "tolist"):
        try:
            start_indices_list = start_indices.tolist()
            if all(isinstance(s, int) for s in start_indices_list):
                slices = [
                    builtins.slice(s, s + u)
                    for s, u in zip(start_indices_list, updates.shape)
                ]
                outputs = inputs.clone()
                outputs[tuple(slices)] = updates
                return outputs
        except Exception:
            pass
    outputs = inputs.clone()
    update_shape = list(updates.shape)
    dims = len(update_shape)
    indices_list = []
    for dim in range(dims):
        dim_indices = paddle.arange(
            update_shape[dim],
            dtype=start_indices.dtype,
        )
        dim_indices = dim_indices + start_indices[dim]
        indices_list.append(dim_indices)

    grids = paddle.meshgrid(*indices_list)
    flat_indices = [g.flatten() for g in grids]
    indices = paddle.stack(flat_indices, axis=-1)
    current_values = paddle.gather_nd(inputs, indices)
    return paddle.scatter_nd_add(
        inputs, indices, updates.flatten() - current_values
    )


def switch(index, branches, *operands):
    index = convert_to_tensor(index, "int32")
    index = paddle.clip(index, 0, len(branches) - 1)
    return branches[index](*operands)


def while_loop(
    cond,
    body,
    loop_vars,
    maximum_iterations=None,
):
    current_iter = 0
    iteration_check = lambda iter: (
        maximum_iterations is None or iter < maximum_iterations
    )
    is_tuple = isinstance(loop_vars, (tuple, list))
    loop_vars = tuple(loop_vars) if is_tuple else (loop_vars,)
    loop_vars = tree.map_structure(convert_to_tensor, loop_vars)

    def get_cond_val(vars):
        val = cond(*vars)
        if is_tensor(val):
            return bool(val.item())
        return bool(val)

    while get_cond_val(loop_vars) and iteration_check(current_iter):
        loop_vars = body(*loop_vars)
        if not isinstance(loop_vars, (list, tuple)):
            loop_vars = (loop_vars,)
        loop_vars = tuple(loop_vars)
        current_iter += 1
    return loop_vars if is_tuple else loop_vars[0]


def fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def stop_gradient(variable):
    if isinstance(variable, Variable):
        variable = variable.value
    return variable.detach()


def unstack(x, num=None, axis=0):
    return paddle.unbind(x, axis)


def random_seed_dtype():
    return "int32"


def remat(f):
    """Implementation of rematerialization.

    Args:
        f: The function or operation to rematerialize.
    Returns:
        A function wrapping f that recomputes f on the backwards pass.
    """
    return f
