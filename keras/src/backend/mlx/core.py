import builtins
import contextlib
import functools
import warnings

import mlx.core as mx
import numpy as np

from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import slice_along_axis
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.symbolic_scope import SymbolicScope
from keras.src.backend.config import floatx

SUPPORTS_SPARSE_TENSORS = False
SUPPORTS_RAGGED_TENSORS = False
SUPPORTS_COMPLEX_DTYPES = True
# MLX lazy arrays are pinned to the stream of the thread that built them and
# cannot be materialized (`mx.eval`/`float`) from another thread. Keras
# dispatches some callbacks asynchronously on a ThreadPoolExecutor, which would
# materialize arrays off the main thread and raise
# "There is no Stream(...) in current thread". Run callbacks synchronously
# (like the tensorflow backend) so all materialization stays on the main thread.
IS_THREAD_SAFE = False


def _ensure_default_device():
    """Ensure the current thread has a default MLX device/stream.

    MLX streams are thread-local and are NOT inherited by worker threads (e.g.
    Keras' async callback dispatch via a ThreadPoolExecutor). Materializing a
    lazy array (`mx.eval` / `float()` / `.item()`) from such a thread raises
    ``RuntimeError: There is no Stream(...) in current thread``. Establishing
    the default device for the thread fixes this; it is idempotent and a no-op
    on threads that already have a default stream.
    """
    try:
        mx.set_default_device(mx.default_device())
    except Exception:
        pass

# Keras dtype string -> mlx dtype.
_DTYPE_BY_NAME = {
    "bool": mx.bool_,
    "uint8": mx.uint8,
    "uint16": mx.uint16,
    "uint32": mx.uint32,
    "uint64": mx.uint64,
    "int8": mx.int8,
    "int16": mx.int16,
    "int32": mx.int32,
    "int64": mx.int64,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
    "float64": mx.float64,
    "complex64": mx.complex64,
}

# MLX does not support these dtypes (raise a clear error rather than coerce).
_UNSUPPORTED_DTYPES = ("complex128", "float8_e4m3fn", "float8_e5m2")


def _mlx_dtype(dtype):
    """Map a keras dtype string to a `mlx.core.Dtype`."""
    dtype = standardize_dtype(dtype)
    if dtype in _UNSUPPORTED_DTYPES:
        raise ValueError(
            f"`{dtype}` dtype is not supported by the MLX backend."
        )
    # MLX has no float64 on the GPU. Promote to float32 to avoid silent
    # device-placement failures. float64 is CPU-only in MLX.
    if dtype == "float64":
        dtype = "float32"
    return _DTYPE_BY_NAME[dtype]


def _mlx_dtype_to_str(dtype):
    """Map a `mlx.core.Dtype` (e.g. `mx.float32`) to a keras dtype string.

    `str(mx.float32)` -> `"mlx.core.float32"`.
    """
    return str(dtype).rsplit(".", 1)[-1]


def _cast(x, dtype):
    """Cast an mlx tensor to a keras dtype string (or mlx/numpy dtype object)."""
    return x.astype(_mlx_dtype(dtype))


class Variable(KerasVariable):
    def _initialize(self, value):
        # Rebind (MLX arrays are immutable in the autograd sense). We do not
        # mutate; the functional-grad trainer relies on this being a fresh leaf.
        self._value = convert_to_tensor(value, dtype=self._dtype)

    def _direct_assign(self, value):
        self._value = convert_to_tensor(value, dtype=self._dtype)

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)

    # Overload native accessor for `np.array(variable)` / `variable.numpy()`.
    def __array__(self, dtype=None):
        arr = self.value
        if str(arr.dtype) == "mlx.core.bfloat16":
            # numpy has no native bfloat16 and `ml_dtypes` may be absent.
            arr = arr.astype(mx.float32)
        arr = np.asarray(arr)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr


def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    if sparse:
        raise ValueError("`sparse=True` is not supported with the MLX backend")
    if ragged:
        raise ValueError("`ragged=True` is not supported with the MLX backend")

    if isinstance(x, Variable):
        if dtype is not None and standardize_dtype(dtype) != x.dtype:
            return x.value.astype(_mlx_dtype(dtype))
        return x.value

    if dtype is not None:
        return mx.array(x, dtype=_mlx_dtype(dtype))

    # dtype is None: infer.
    if isinstance(x, mx.array):
        return x
    if isinstance(x, bool):
        return mx.array(x, dtype=mx.bool_)
    if isinstance(x, int):
        return mx.array(x, dtype=mx.int32)
    if isinstance(x, float):
        return mx.array(x, dtype=_mlx_dtype(floatx()))

    # Lists / tuples / numpy arrays / nested structures.
    flat = tree.flatten(x)
    # Build a keras result dtype, mapping mlx/numpy dtypes to strings.
    def _kind(item):
        d = getattr(item, "dtype", type(item))
        if isinstance(d, mx.Dtype):
            d = _mlx_dtype_to_str(d)
        return d

    inferred = result_type(*[_kind(item) for item in flat])
    inferred = standardize_dtype(inferred)
    return mx.array(x, dtype=_mlx_dtype(inferred))


def convert_to_numpy(x):
    if isinstance(x, Variable):
        x = x.value
    else:
        x = convert_to_tensor(x)
    # bfloat16 can't round-trip to numpy without `ml_dtypes`.
    if str(x.dtype) == "mlx.core.bfloat16":
        x = x.astype(mx.float32)
    # May be called from a worker thread (async callbacks) lacking an MLX
    # stream; ensure the thread has a default device before materializing.
    _ensure_default_device()
    mx.eval(x)
    return np.asarray(x)


def is_tensor(x):
    return isinstance(x, mx.array)


def shape(x):
    return tuple(x.shape)


def cast(x, dtype):
    return convert_to_tensor(x, dtype=dtype)


def cond(pred, true_fn, false_fn):
    if bool(pred):
        return true_fn()
    return false_fn()


def vectorized_map(function, elements):
    if not isinstance(elements, (list, tuple)):
        return mx.stack([function(x) for x in elements])
    else:
        batch_size = elements[0].shape[0]
        output_store = [
            function([x[index] for x in elements])
            for index in range(batch_size)
        ]
        return mx.stack(output_store)


# Shape / dtype inference util
def compute_output_spec(fn, *args, **kwargs):
    with StatelessScope(), SymbolicScope():

        def has_none_shape(x):
            if isinstance(x, KerasTensor):
                return None in x.shape
            return False

        none_in_shape = any(
            builtins.map(has_none_shape, tree.flatten((args, kwargs)))
        )

        def convert_keras_tensor_to_mlx(x, fill_value=None):
            if isinstance(x, KerasTensor):
                shape = list(x.shape)
                if fill_value:
                    for i, e in enumerate(shape):
                        if e is None:
                            shape[i] = fill_value
                return mx.zeros(
                    shape=tuple(shape),
                    dtype=_mlx_dtype(x.dtype),
                )
            return x

        args_1, kwargs_1 = tree.map_structure(
            lambda x: convert_keras_tensor_to_mlx(x, fill_value=83),
            (args, kwargs),
        )
        outputs_1 = fn(*args_1, **kwargs_1)

        outputs = outputs_1

        if none_in_shape:
            args_2, kwargs_2 = tree.map_structure(
                lambda x: convert_keras_tensor_to_mlx(x, fill_value=89),
                (args, kwargs),
            )
            outputs_2 = fn(*args_2, **kwargs_2)

            flat_out_1 = tree.flatten(outputs_1)
            flat_out_2 = tree.flatten(outputs_2)

            flat_out = []
            for x1, x2 in zip(flat_out_1, flat_out_2):
                shape = list(x1.shape)
                for i, e in enumerate(x2.shape):
                    if e != shape[i]:
                        shape[i] = None
                flat_out.append(
                    KerasTensor(shape, standardize_dtype(_mlx_dtype_to_str(x1.dtype)))
                )
            outputs = tree.pack_sequence_as(outputs_1, flat_out)

        def convert_mlx_to_keras_tensor(x):
            if is_tensor(x):
                return KerasTensor(
                    x.shape, standardize_dtype(_mlx_dtype_to_str(x.dtype))
                )
            return x

        output_spec = tree.map_structure(convert_mlx_to_keras_tensor, outputs)
    return output_spec


def map(f, xs):
    def g(_, x):
        return (), f(x)

    _, ys = scan(g, (), xs)
    return ys


def scan(f, init, xs=None, length=None, reverse=False, unroll=1):
    # Ref: jax.lax.scan
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
    init_flat = [convert_to_tensor(init) for init in init_flat]
    init = pack_output(init_flat)
    dummy_y = [mx.zeros_like(init) for init in init_flat]

    carry = init
    ys = []
    maybe_reversed = reversed if reverse else lambda x: x
    for i in maybe_reversed(range(n)):
        xs_slice = [x[i] for x in xs_flat]
        packed_xs = pack_input(xs_slice) if len(xs_slice) > 0 else None
        carry, y = f(carry, packed_xs)
        ys.append(y if y is not None else dummy_y)
    stacked_y = tree.map_structure(
        lambda *ys: mx.stack(ys), *maybe_reversed(ys)
    )
    return carry, stacked_y


def associative_scan(f, elems, reverse=False, axis=0):
    # Ref: jax.lax.associative_scan
    if not callable(f):
        raise TypeError(f"`f` should be a callable. Received: f={f}")
    elems_flat = tree.flatten(elems)
    elems_flat = [convert_to_tensor(elem) for elem in elems_flat]
    if reverse:
        elems_flat = [mx.flip(elem, (axis,)) for elem in elems_flat]

    def _combine(a_flat, b_flat):
        a = tree.pack_sequence_as(elems, a_flat)
        b = tree.pack_sequence_as(elems, b_flat)
        c = f(a, b)
        c_flat = tree.flatten(c)
        return c_flat

    num_elems = int(elems_flat[0].shape[axis])
    if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
        raise ValueError(
            "Array inputs to associative_scan must have the same "
            "first dimension. (saw: {})".format(
                [elem.shape for elem in elems_flat]
            )
        )

    def _interleave(a, b, axis):
        """Given two Tensors of static shape, interleave them along axis."""
        if not (
            a.shape[axis] == b.shape[axis] or a.shape[axis] == b.shape[axis] + 1
        ):
            raise ValueError(
                "Shapes are incompatible for associative_scan interleaving. "
                f"a.shape[{axis}]={a.shape[axis]}, "
                f"b.shape[{axis}]={b.shape[axis]}"
            )

        La = a.shape[axis]
        Lb = b.shape[axis]
        out_len = La + Lb
        out_shape = list(a.shape)
        out_shape[axis] = out_len
        out = mx.zeros(out_shape, dtype=a.dtype)

        nd = a.ndim

        def _idx(positions):
            full = [slice(None)] * nd
            full[axis] = positions
            return tuple(full)

        # `a` lands at even positions (0, 2, 4, ...), `b` at odd positions.
        a_positions = mx.arange(0, out_len, 2)  # length La
        b_positions = mx.arange(1, out_len, 2)  # length Lb
        out = out.at[_idx(a_positions)].add(a)
        out = out.at[_idx(b_positions)].add(b)
        return out

    def _scan(elems):
        num_elems = elems[0].shape[axis]
        if num_elems < 2:
            return elems

        reduced_elems = _combine(
            [
                slice_along_axis(elem, 0, -1, step=2, axis=axis)
                for elem in elems
            ],
            [
                slice_along_axis(elem, 1, None, step=2, axis=axis)
                for elem in elems
            ],
        )
        odd_elems = _scan(reduced_elems)
        if num_elems % 2 == 0:
            even_elems = _combine(
                [slice_along_axis(e, 0, -1, axis=axis) for e in odd_elems],
                [
                    slice_along_axis(e, 2, None, step=2, axis=axis)
                    for e in elems
                ],
            )
        else:
            even_elems = _combine(
                odd_elems,
                [
                    slice_along_axis(e, 2, None, step=2, axis=axis)
                    for e in elems
                ],
            )

        even_elems = [
            mx.concatenate(
                [slice_along_axis(elem, 0, 1, axis=axis), result],
                axis=axis,
            )
            for (elem, result) in zip(elems, even_elems)
        ]
        return list(
            builtins.map(
                functools.partial(_interleave, axis=axis), even_elems, odd_elems
            )
        )

    scans = _scan(elems_flat)
    if reverse:
        scans = [mx.flip(scanned, (axis,)) for scanned in scans]

    return tree.pack_sequence_as(elems, scans)


def scatter(indices, values, shape):
    indices = convert_to_tensor(indices)
    values = convert_to_tensor(values)
    zeros = mx.zeros(tuple(shape), dtype=values.dtype)

    index_length = indices.shape[-1]
    value_shape = tuple(shape)[index_length:]
    indices = mx.reshape(indices, [-1, index_length])
    values = mx.reshape(values, [-1] + list(value_shape))

    # Per-dimension index arrays for fancy (tuple) indexing.
    idx = tuple(indices[:, i] for i in range(index_length))
    # Accumulate (handles duplicate indices correctly via `array.at`).
    return zeros.at[idx].add(values)


def scatter_update(inputs, indices, updates, reduction=None):
    inputs = convert_to_tensor(inputs)
    updates = convert_to_tensor(updates)
    indices = convert_to_tensor(indices)
    # `indices` is (num_dims, num_updates) -> transpose to per-dim tuple.
    indices = mx.transpose(indices)
    idx = tuple(indices)

    if reduction is None:
        # Overwrite: out[idx] = updates. Achieved by adding (updates - inputs[idx]).
        return inputs.at[idx].add(updates - inputs[idx])
    elif reduction == "add" or reduction == "sum":
        return inputs.at[idx].add(updates)
    elif reduction == "subtract" or reduction == "sub":
        return inputs.at[idx].subtract(updates)
    elif reduction == "mul" or reduction == "multiply":
        return inputs.at[idx].multiply(updates)
    elif reduction == "min":
        return inputs.at[idx].minimum(updates)
    elif reduction == "max":
        return inputs.at[idx].maximum(updates)
    elif reduction == "div" or reduction == "divide":
        return inputs.at[idx].divide(updates)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def slice(inputs, start_indices, shape):
    # Validate inputs
    if len(start_indices) != len(shape):
        raise ValueError(
            "Length of `start_indices` must match length of `shape`. "
            f"Received: start_indices={start_indices}, shape={shape}"
        )
    slices = tuple(
        builtins.slice(int(start), int(start) + int(length))
        for start, length in zip(start_indices, shape)
    )
    return inputs[slices]


def slice_update(inputs, start_indices, updates):
    inputs = convert_to_tensor(inputs)
    updates = convert_to_tensor(updates)
    start = mx.array([int(s) for s in start_indices])
    axes = list(range(len(start_indices)))
    return mx.slice_update(inputs, updates, start, axes)


def switch(index, branches, *operands):
    index = convert_to_tensor(index, "int32")
    index = mx.clip(index, 0, len(branches) - 1)
    return branches[int(index)](*operands)


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
    while cond(*loop_vars) and iteration_check(current_iter):
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
    elif not is_tensor(variable):
        variable = convert_to_tensor(variable)
    return mx.stop_gradient(variable)


def unstack(x, num=None, axis=0):
    x = mx.moveaxis(x, axis, 0)
    if num is None:
        num = x.shape[0]
    return [x[i] for i in range(num)]


def random_seed_dtype():
    return "uint32"


class custom_gradient:
    """Decorator for custom gradients.

    Args:
        fun: Forward pass function.
    """

    def __init__(self, fun):
        warnings.warn(
            "`custom_gradient` for the MLX backend acts as a pass-through to "
            "support the forward pass. No gradient computation or modification "
            "takes place."
        )
        self.fun = fun

    def __call__(self, *args, **kwargs):
        outputs, _ = self.fun(*args, **kwargs)
        return outputs


@contextlib.contextmanager
def device_scope(device_name):
    """Context manager for device placement.

    Maps the backend-agnostic device string (`"cpu:0"`, `"gpu:0"`, `"mps:0"`)
    to an MLX device. MLX uses unified memory; placement controls which
    stream/dispatch the ops run on. By default MLX uses the GPU.
    """
    device_type = str(device_name).split(":")[0].lower()
    if device_type in ("gpu", "mps", "metal"):
        dev = mx.Device(mx.DeviceType.gpu, 0)
    elif device_type == "cpu":
        dev = mx.Device(mx.DeviceType.cpu, 0)
    else:
        # Unknown device string: fall back to the default device.
        yield
        return
    with mx.stream(dev):
        yield


def remat(f):
    # MLX has `mx.checkpoint` for rematerialization; for now act as a
    # pass-through (the graph is already lazily evaluated).
    return f
