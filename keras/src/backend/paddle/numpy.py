import builtins

import numpy as np
import paddle

from keras.src.backend.common import standardize_dtype
from keras.src.backend.paddle.core import convert_to_tensor
from keras.src.backend.paddle.core import shape
from keras.src.backend.paddle.core import to_paddle_dtype
from keras.src.backend.paddle.core import _weak_tensors


_CPU_UNSUPPORTED_DTYPES = {paddle.float16, paddle.bfloat16}
_FLOAT_TYPES = {"float16", "float32", "float64", "bfloat16",
                "float8_e4m3fn", "float8_e5m2"}


def _maybe_upcast(x):
    """Cast float16/bfloat16 to float32 on CPU for ops that don't support them."""
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        return x.cast("float32"), True
    return x, False


def _maybe_downcast(result, needs_downcast, original_dtype):
    """Cast result back to original dtype if it was upcast."""
    if needs_downcast:
        return result.cast(original_dtype)
    return result


def _promote_dtypes(x1, x2):
    """Cast two tensors to a common dtype for cross-type operations.

    Matches Keras/JAX behavior: when mixing float and int/bool types,
    promote to the float type (not widen to float64).
    Python scalars are treated as "weak" types (JAX semantics):
    int8 + python_int → int8, int8 + python_float → float32.
    """
    dt1 = standardize_dtype(x1.dtype)
    dt2 = standardize_dtype(x2.dtype)
    if dt1 == dt2:
        return x1, x2
    float_types = {"float16", "float32", "float64", "bfloat16"}
    complex_types = {"complex64", "complex128"}
    int_types = {
        "bool", "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
    }
    is_f1 = dt1 in float_types
    is_f2 = dt2 in float_types
    is_c1 = dt1 in complex_types
    is_c2 = dt2 in complex_types
    is_i1 = dt1 in int_types
    is_i2 = dt2 in int_types
    # Weak type: Python scalar converted to default dtype
    w1 = id(x1) in _weak_tensors
    w2 = id(x2) in _weak_tensors
    # If weak type is same category as the other, defer to the other
    if w1 and not w2 and ((is_f1 and is_f2) or (is_i1 and is_i2)):
        common_dtype = dt2
    elif w2 and not w1 and ((is_f1 and is_f2) or (is_i1 and is_i2)):
        common_dtype = dt1
    elif is_f1 and not is_f2:
        common_dtype = dt1
    elif is_f2 and not is_f1:
        common_dtype = dt2
    elif (is_c1 or is_c2) and not (is_c1 and is_c2):
        common_dtype = dt1 if is_c1 else dt2
    else:
        try:
            common = np.result_type(
                np.zeros(1, dtype=dt1), np.zeros(1, dtype=dt2)
            )
            common_dtype = standardize_dtype(common)
        except (TypeError, np.exceptions.DTypePromotionError):
            common_dtype = "float32"
    if dt1 != common_dtype:
        x1 = paddle.cast(x1, to_paddle_dtype(common_dtype))
    if dt2 != common_dtype:
        x2 = paddle.cast(x2, to_paddle_dtype(common_dtype))
    # CPU doesn't support float16/bfloat16 for most ops
    if common_dtype in ("float16", "bfloat16"):
        x1 = x1.cast("float32")
        x2 = x2.cast("float32")
    return x1, x2


def _binary_op_with_dtype(op, x1, x2):
    """Run a binary op with _promote_dtypes and cast result back if needed.

    Handles float16/bfloat16 CPU upcasting.
    """
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dt1 = standardize_dtype(x1.dtype)
    dt2 = standardize_dtype(x2.dtype)
    w1 = id(x1) in _weak_tensors
    w2 = id(x2) in _weak_tensors
    low_precision = {"float16", "bfloat16"}
    int_types = {
        "bool", "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
    }
    # Determine if result should be low precision float
    target = None
    if w1 and not w2:
        if dt2 in low_precision:
            target = dt2
    elif w2 and not w1:
        if dt1 in low_precision:
            target = dt1
    elif not w1 and not w2:
        if dt1 in low_precision and dt2 in int_types:
            target = dt1
        elif dt2 in low_precision and dt1 in int_types:
            target = dt2
        elif dt1 in low_precision and dt2 in low_precision:
            target = dt1 if dt1 == dt2 else None
    x1, x2 = _promote_dtypes(x1, x2)
    result = op(x1, x2)
    if target in low_precision:
        result_dtype = standardize_dtype(result.dtype)
        if result_dtype != target:
            result = result.cast(to_paddle_dtype(target))
    return result


def _unary_op(op, x):
    """Run a unary op with float16/bfloat16 CPU upcasting."""
    x = convert_to_tensor(x)
    orig_dtype = x.dtype
    needs_cast = x.dtype in _CPU_UNSUPPORTED_DTYPES
    if needs_cast:
        x = x.cast("float32")
    result = op(x)
    if needs_cast:
        result = result.cast(orig_dtype)
    return result


_CPU_UNSUPPORTED_INT = {"int8", "int16", "uint8", "bool"}


def _binary_op_with_int(op, x1, x2):
    """Run a binary op with full CPU dtype support (float16 + int types)."""
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dt1 = standardize_dtype(x1.dtype)
    dt2 = standardize_dtype(x2.dtype)
    w1 = id(x1) in _weak_tensors
    w2 = id(x2) in _weak_tensors
    # Determine target dtype
    if w1 and not w2:
        target = dt2
    elif w2 and not w1:
        target = dt1
    else:
        try:
            common = np.result_type(
                np.zeros(1, dtype=dt1), np.zeros(1, dtype=dt2)
            )
            target = standardize_dtype(common)
        except (TypeError, np.exceptions.DTypePromotionError):
            target = "float32"
    # Cast unsupported types
    if dt1 in _CPU_UNSUPPORTED_INT or dt1 in ("float16", "bfloat16"):
        cast_to = "float32" if dt1 in ("float16", "bfloat16") else "int32"
        x1 = x1.cast(cast_to)
    if dt2 in _CPU_UNSUPPORTED_INT or dt2 in ("float16", "bfloat16"):
        cast_to = "float32" if dt2 in ("float16", "bfloat16") else "int32"
        x2 = x2.cast(cast_to)
    result = op(x1, x2)
    result_dtype = standardize_dtype(result.dtype)
    if target != result_dtype:
        result = result.cast(to_paddle_dtype(target))
    return result


def add(x1, x2):
    return _binary_op_with_dtype(paddle.add, x1, x2)


def subtract(x1, x2):
    return _binary_op_with_dtype(paddle.subtract, x1, x2)


def multiply(x1, x2):
    return _binary_op_with_dtype(paddle.multiply, x1, x2)


def divide(x1, x2):
    return _binary_op_with_dtype(paddle.divide, x1, x2)


def true_divide(x1, x2):
    x1 = convert_to_tensor(x1, "float32")
    x2 = convert_to_tensor(x2, "float32")
    return paddle.divide(x1, x2)


def floor_divide(x1, x2):
    return _binary_op_with_dtype(paddle.floor_divide, x1, x2)


def mod(x1, x2):
    return _binary_op_with_dtype(paddle.remainder, x1, x2)


def negative(x):
    return _unary_op(paddle.neg, x)


def abs(x):
    return _unary_op(paddle.abs, x)


def absolute(x):
    return abs(x)


def sign(x):
    return _unary_op(paddle.sign, x)


def log(x):
    return _unary_op(paddle.log, x)


def log2(x):
    return _unary_op(paddle.log2, x)


def log10(x):
    return _unary_op(paddle.log10, x)


def log1p(x):
    return _unary_op(paddle.log1p, x)


def exp(x):
    return _unary_op(paddle.exp, x)


def expm1(x):
    return _unary_op(paddle.expm1, x)


def sqrt(x):
    return _unary_op(paddle.sqrt, x)


def square(x):
    return _unary_op(paddle.square, x)


def pow(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return _binary_op_with_dtype(paddle.pow, x1, x2)


def power(x1, x2):
    return _binary_op_with_dtype(paddle.pow, x1, x2)


def maximum(x1, x2):
    return _binary_op_with_dtype(paddle.maximum, x1, x2)


def minimum(x1, x2):
    return _binary_op_with_dtype(paddle.minimum, x1, x2)


def round(x, decimals=0):
    return _unary_op(lambda t: paddle.round(t), x)


def clip(x, x_min, x_max):
    return _unary_op(lambda t: paddle.clip(t, x_min, x_max), x)


def clip_by_value(x, clip_value_min, clip_value_max):
    return clip(x, clip_value_min, clip_value_max)


def floor(x):
    return _unary_op(paddle.floor, x)


def ceil(x):
    return _unary_op(paddle.ceil, x)


def dot(x, y):
    return _binary_op_with_dtype(paddle.dot, x, y)


def tensordot(x1, x2, axes=2):
    return _binary_op_with_dtype(
        lambda a, b: paddle.tensordot(a, b, axes), x1, x2
    )


def einsum(subscripts, *operands, **kwargs):
    operands = [convert_to_tensor(x) for x in operands]
    # Cast unsupported dtypes to float32/int32
    needs_cast = False
    cast_dtype = None
    for i, op in enumerate(operands):
        dt = standardize_dtype(op.dtype)
        if dt in ("float16", "bfloat16"):
            operands[i] = op.cast("float32")
            needs_cast = True
            cast_dtype = to_paddle_dtype(dt)
        elif dt in ("int8", "int16", "uint8", "bool"):
            operands[i] = op.cast("int32")
            needs_cast = True
            cast_dtype = to_paddle_dtype(dt)
    result = paddle.einsum(subscripts, *operands)
    if needs_cast and cast_dtype is not None:
        result = result.cast(cast_dtype)
    return result


def get_item(x, i):
    x = convert_to_tensor(x)
    return x[i]


def slice_count(x):
    x = convert_to_tensor(x)
    return len(x.shape)


def shape_equal(x, y):
    x_shape = shape(x)
    y_shape = shape(y)
    return x_shape == y_shape


def where(condition, x1, x2):
    condition = convert_to_tensor(condition, dtype="bool")
    if x1 is not None and x2 is not None:
        x1 = convert_to_tensor(x1)
        x2 = convert_to_tensor(x2)
        x1, x2 = _promote_dtypes(x1, x2)
        # Cast unsupported int/bool types for CPU
        orig_dtype = x1.dtype
        dt = standardize_dtype(x1.dtype)
        if dt in ("int8", "int16", "uint8", "bool"):
            x1 = x1.cast("int32")
            x2 = x2.cast("int32")
        result = paddle.where(condition, x1, x2)
        if result.dtype != orig_dtype:
            result = result.cast(orig_dtype)
        return result
    return paddle.where(condition)


def mean(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return x.clone()
    orig_dtype = x.dtype
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    result = paddle.mean(x, axis=axis, keepdim=keepdims)
    return result.cast(orig_dtype)


def variance(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return paddle.zeros_like(x)
    orig_dtype = x.dtype
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    result = paddle.var(x, axis=axis, keepdim=keepdims)
    return result.cast(orig_dtype)


def std(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return paddle.zeros_like(x)
    orig_dtype = x.dtype
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    result = paddle.std(x, axis=axis, keepdim=keepdims)
    return result.cast(orig_dtype)


def sum(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return x.clone()
    return paddle.sum(x, axis=axis, keepdim=keepdims)


def prod(x, axis=None, keepdims=False, dtype=None):
    if dtype is not None:
        x = convert_to_tensor(x, dtype=dtype)
    else:
        x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return x.clone()
    orig_dtype = x.dtype
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    result = paddle.prod(x, axis=axis, keepdim=keepdims)
    return result.cast(orig_dtype)


def logsumexp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return paddle.logsumexp(x, axis=axis, keepdim=keepdims)


def cumsum(x, axis=None, dtype=None):
    x = convert_to_tensor(x)
    if axis is None:
        x = x.flatten()
        axis = 0
    dtype = standardize_dtype(dtype or x.dtype)
    if dtype == "bool":
        dtype = "int32"
    orig_dtype = x.dtype
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    result = paddle.cumsum(x, axis=axis, dtype=x.dtype)
    return paddle.cast(result, dtype)


def cumprod(x, axis=None, dtype=None):
    x = convert_to_tensor(x)
    if axis is None:
        x = x.flatten()
        axis = 0
    dtype = standardize_dtype(dtype or x.dtype)
    if dtype == "bool":
        dtype = "int32"
    orig_dtype = x.dtype
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    result = paddle.cumprod(x, dim=axis, dtype=x.dtype)
    return paddle.cast(result, dtype)


def argmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    return paddle.argmax(x, axis=axis, keepdim=keepdims)


def argmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    return paddle.argmin(x, axis=axis, keepdim=keepdims)


def argsort(x, axis=-1):
    x = convert_to_tensor(x)
    if axis is None:
        axis = -1
        x = x.reshape([-1])
    orig_dtype = x.dtype
    # Paddle CPU does not support argsort for bool/int8
    if orig_dtype in (paddle.bool, paddle.int8):
        x = x.cast(paddle.int32)
    # Paddle returns int64, but JAX returns int32 — match JAX
    return paddle.argsort(x, axis=axis).cast(paddle.int32)


def sort(x, axis=-1):
    x = convert_to_tensor(x)
    if axis is None:
        x = x.flatten()
        axis = 0
    orig_dtype = x.dtype
    # Paddle CPU does not support argsort (used internally by sort) for bool/int8
    if orig_dtype in (paddle.bool, paddle.int8):
        x = x.cast(paddle.int32)
    result = paddle.sort(x, axis=axis)
    if orig_dtype != result.dtype:
        result = result.cast(orig_dtype)
    return result


def searchsorted(sorted_sequence, values, side="left"):
    sorted_sequence = convert_to_tensor(sorted_sequence)
    values = convert_to_tensor(values)
    # Paddle CPU searchsorted only supports float32/float64/int32/int64
    _supported = {paddle.float32, paddle.float64, paddle.int32, paddle.int64}
    if sorted_sequence.dtype not in _supported:
        sorted_sequence = sorted_sequence.cast(paddle.float32)
    if values.dtype not in _supported:
        values = values.cast(paddle.float32)
    # Paddle returns int64, but JAX/NumPy return int32 — match them
    return paddle.searchsorted(sorted_sequence, values).cast(paddle.int32)


def top_k(x, k, sorted=False):
    return paddle.topk(convert_to_tensor(x), k)


def in_top_k(targets, predictions, k):
    targets = convert_to_tensor(targets, "int64")
    predictions = convert_to_tensor(predictions)
    topk_indices = paddle.topk(predictions, k, axis=-1)[1]
    return paddle.any(topk_indices == targets.unsqueeze(-1), axis=-1)


def flip(x, axis=None):
    x = convert_to_tensor(x)
    if axis is None:
        axis = list(range(x.ndim))
    orig_dtype = x.dtype
    # Paddle CPU does not support flip for many dtypes — cast to float32/int32
    _float_unsupported = {paddle.float16, paddle.bfloat16}
    _int_unsupported = {paddle.int8, paddle.int16, paddle.uint8}
    if orig_dtype in _float_unsupported:
        x = x.cast(paddle.float32)
    elif orig_dtype in _int_unsupported:
        x = x.cast(paddle.int32)
    result = paddle.flip(x, axis=axis)
    if result.dtype != orig_dtype:
        result = result.cast(orig_dtype)
    return result


def roll(x, shift, axis=None):
    x = convert_to_tensor(x)
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        orig_dtype = x.dtype
        return paddle.roll(x.cast("float32"), shift, axis=axis).cast(orig_dtype)
    return paddle.roll(x, shift, axis=axis)


def pad(x, pad_width, mode="constant", constant_values=0):
    x = convert_to_tensor(x)
    orig_dtype = x.dtype
    # Cast unsupported dtypes for CPU
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    elif standardize_dtype(x.dtype) in ("int8", "uint8", "bool"):
        x = x.cast("int32")
    pad_list = []
    for left, right in reversed(pad_width):
        pad_list.extend([left, right])
    if mode == "constant":
        if constant_values is None:
            constant_values = 0
        result = paddle.nn.functional.pad(
            x, pad_list, mode="constant", value=constant_values
        )
    elif mode == "reflect":
        # paddle only supports reflect for 3D with exactly 6 padding values
        # Use numpy for general case
        x_np = x.numpy()
        pad_width_np = [(int(l), int(r)) for l, r in pad_width]
        result_np = np.pad(x_np, pad_width_np, mode="reflect")
        result = paddle.to_tensor(result_np, dtype=x.dtype)
    elif mode == "symmetric":
        x_np = x.numpy()
        pad_width_np = [(int(l), int(r)) for l, r in pad_width]
        result_np = np.pad(x_np, pad_width_np, mode="symmetric")
        result = paddle.to_tensor(result_np, dtype=x.dtype)
    elif mode == "edge":
        result = paddle.nn.functional.pad(x, pad_list, mode="replicate")
    else:
        raise NotImplementedError(
            f"`pad` with mode='{mode}' is not supported with paddle backend"
        )
    return result.cast(orig_dtype)


def _promote_dtypes_list(tensors):
    """Promote a list of tensors to a common dtype."""
    if len(tensors) <= 1:
        return tensors
    result = [tensors[0]]
    for t in tensors[1:]:
        _, promoted = _promote_dtypes(result[0], t)
        result.append(promoted)
    common = result[0]
    for i, t in enumerate(result):
        if t.dtype != common.dtype:
            result[i] = t.cast(common.dtype)
    if tensors[0].dtype != common.dtype:
        result[0] = tensors[0].cast(common.dtype)
    return result


def concatenate(xs, axis=0):
    xs = [convert_to_tensor(x) for x in xs]
    xs = _promote_dtypes_list(xs)
    return paddle.concat(xs, axis=axis)


def append(x, values, axis=None):
    x = convert_to_tensor(x)
    values = convert_to_tensor(values)
    x, values = _promote_dtypes(x, values)
    if axis is None:
        return paddle.concat([x.flatten(), values.flatten()])
    return paddle.concat([x, values], axis=axis)


def stack(x, axis=0):
    x = [convert_to_tensor(xi) for xi in x]
    x = _promote_dtypes_list(x)
    return paddle.stack(x, axis=axis)


def unstack(x, num=None, axis=0):
    return paddle.unbind(x, axis)


def split(x, indices_or_sections, axis=0):
    return paddle.split(convert_to_tensor(x), indices_or_sections, axis=axis)


def swapaxes(x, axis1, axis2):
    perm = list(range(len(shape(x))))
    perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
    return paddle.transpose(convert_to_tensor(x), perm)


def moveaxis(x, source, destination):
    perm = list(range(len(shape(x))))
    dim = perm.pop(source)
    perm.insert(destination, dim)
    return paddle.transpose(convert_to_tensor(x), perm)


def transpose(x, axes=None):
    x = convert_to_tensor(x)
    if axes is None:
        axes = list(range(len(x.shape)))[::-1]
    return paddle.transpose(x, axes)


def squeeze(x, axis=None):
    return paddle.squeeze(convert_to_tensor(x), axis=axis)


def expand_dims(x, axis):
    return paddle.unsqueeze(convert_to_tensor(x), axis)


def reshape(x, newshape):
    return paddle.reshape(convert_to_tensor(x), newshape)


def eye(N, M=None, k=0, dtype="float32"):
    if M is None:
        M = N
    dtype = to_paddle_dtype(dtype)
    if dtype in _CPU_UNSUPPORTED_DTYPES:
        return paddle.eye(N, M, dtype=paddle.float32).cast(dtype)
    return paddle.eye(N, M, dtype=dtype)


def linspace(start, stop, num, dtype=None, endpoint=True, retstep=False, axis=0):
    if dtype is not None:
        dtype = to_paddle_dtype(dtype)
    needs_downcast = dtype in _CPU_UNSUPPORTED_DTYPES
    compute_dtype = paddle.float32 if needs_downcast else dtype
    result = paddle.linspace(start, stop, num, dtype=compute_dtype)
    if needs_downcast:
        result = result.cast(dtype)
    if retstep:
        step = (stop - start) / max(num - 1 if endpoint else num, 1)
        return result, step
    return result


def arange(start, stop=None, step=1, dtype=None):
    if dtype is not None:
        dtype = to_paddle_dtype(dtype)
    if stop is None:
        return paddle.arange(start, dtype=dtype)
    return paddle.arange(start, stop, step, dtype=dtype)


def full(shape, fill_value, dtype=None):
    if dtype is not None:
        dtype = to_paddle_dtype(dtype)
    return paddle.full(shape, fill_value, dtype=dtype)


def full_like(x, fill_value, dtype=None):
    if dtype is not None:
        dtype = to_paddle_dtype(dtype)
    if isinstance(fill_value, np.ndarray):
        fill_value = fill_value.flat[0]
    return paddle.full_like(convert_to_tensor(x), fill_value, dtype=dtype)


def zeros(shape, dtype="float32"):
    dtype = to_paddle_dtype(dtype)
    return paddle.zeros(shape, dtype=dtype)


def zeros_like(x, dtype=None):
    if dtype is not None:
        dtype = to_paddle_dtype(dtype)
    return paddle.zeros_like(convert_to_tensor(x), dtype=dtype)


def ones(shape, dtype="float32"):
    dtype = to_paddle_dtype(dtype)
    return paddle.ones(shape, dtype=dtype)


def ones_like(x, dtype=None):
    if dtype is not None:
        dtype = to_paddle_dtype(dtype)
    return paddle.ones_like(convert_to_tensor(x), dtype=dtype)


def identity(n, dtype="float32"):
    return eye(n, dtype=dtype)


def tri(N, M=None, k=0, dtype="float32"):
    if M is None:
        M = N
    dtype = to_paddle_dtype(dtype)
    return paddle.tril(paddle.ones([N, M], dtype=dtype), diagonal=k)


def tril(x, k=0):
    x = convert_to_tensor(x)
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        orig_dtype = x.dtype
        return paddle.tril(x.cast("float32"), diagonal=k).cast(orig_dtype)
    return paddle.tril(x, diagonal=k)


def triu(x, k=0):
    x = convert_to_tensor(x)
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        orig_dtype = x.dtype
        return paddle.triu(x.cast("float32"), diagonal=k).cast(orig_dtype)
    return paddle.triu(x, diagonal=k)


def diagonal(x, offset=0, axis1=0, axis2=1):
    return paddle.diagonal(convert_to_tensor(x), offset, axis1, axis2)


def trace(x, offset=0, axis1=0, axis2=1):
    return paddle.trace(convert_to_tensor(x), offset)


def meshgrid(*x, indexing="xy"):
    x = [convert_to_tensor(xi) for xi in x]
    return paddle.meshgrid(*x)


def histogram(x, bins=10, range=None):
    x = convert_to_tensor(x)
    x_np = x.numpy()
    if range is not None:
        hist, bin_edges = np.histogram(x_np, bins=bins, range=range)
    else:
        hist, bin_edges = np.histogram(x_np, bins=bins)
    return (
        paddle.to_tensor(hist, dtype="int64"),
        paddle.to_tensor(bin_edges, dtype=x.dtype),
    )


def tile(x, repeats):
    return paddle.tile(convert_to_tensor(x), repeats)


def repeat(x, repeats, axis=None):
    return paddle.repeat_interleave(convert_to_tensor(x), repeats, axis=axis)


def take(x, indices, axis=None):
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices, dtype="int64")
    if axis is None:
        return paddle.gather(x.flatten(), indices.flatten())
    # Paddle's gather requires indices to be 1D or have last dim == 1
    # Reshape indices to work with paddle.gather
    axis = axis + x.ndim if axis < 0 else axis
    indices_flat = indices.flatten()
    result = paddle.gather(x, indices_flat, axis=axis)
    new_shape = (
        list(x.shape[:axis]) + list(indices.shape) + list(x.shape[axis + 1 :])
    )
    return paddle.reshape(result, new_shape)


def take_along_axis(x, indices, axis=None):
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices, dtype="int64")
    orig_dtype = x.dtype
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    if axis is None:
        result = paddle.take_along_axis(x.flatten(), indices.flatten(), 0)
    else:
        # Ensure indices has same ndim as x
        while indices.ndim < x.ndim:
            indices = indices.unsqueeze(-1)
        result = paddle.take_along_axis(x, indices, axis)
    return result.cast(orig_dtype)


def put_along_axis(x, indices, values, axis=None):
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices, dtype="int64")
    values = convert_to_tensor(values, dtype=x.dtype)
    if axis is None:
        return paddle.put_along_axis(
            x.flatten(), indices.flatten(), values.flatten(), axis=0
        ).reshape(x.shape)
    return paddle.put_along_axis(x, indices, values, axis=axis)


def block_diag(inputs):
    inputs = [convert_to_tensor(x) for x in inputs]
    rows = []
    for i, x in enumerate(inputs):
        row = []
        for j, y in enumerate(inputs):
            if i == j:
                row.append(x)
            else:
                row.append(
                    paddle.zeros([x.shape[0], y.shape[1]], dtype=x.dtype)
                )
        rows.append(paddle.concat(row, axis=1))
    return paddle.concat(rows, axis=0)


def conjugate(x):
    return paddle.conj(convert_to_tensor(x))


def conj(x):
    return conjugate(x)


def real(x):
    return paddle.real(convert_to_tensor(x))


def imag(x):
    return paddle.imag(convert_to_tensor(x))


def angle(x):
    return paddle.angle(convert_to_tensor(x))


def isclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    if x1.dtype in _CPU_UNSUPPORTED_DTYPES:
        x1 = x1.cast("float32")
    if x2.dtype in _CPU_UNSUPPORTED_DTYPES:
        x2 = x2.cast("float32")
    return paddle.isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)


def allclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    if x1.dtype in _CPU_UNSUPPORTED_DTYPES:
        x1 = x1.cast("float32")
    if x2.dtype in _CPU_UNSUPPORTED_DTYPES:
        x2 = x2.cast("float32")
    return paddle.allclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)


def equal(x1, x2):
    return _binary_op_with_dtype(paddle.equal, x1, x2)


def not_equal(x1, x2):
    return _binary_op_with_dtype(paddle.not_equal, x1, x2)


def greater(x1, x2):
    return _binary_op_with_dtype(paddle.greater_than, x1, x2)


def greater_equal(x1, x2):
    return _binary_op_with_dtype(paddle.greater_equal, x1, x2)


def less(x1, x2):
    return _binary_op_with_dtype(paddle.less_than, x1, x2)


def less_equal(x1, x2):
    return _binary_op_with_dtype(paddle.less_equal, x1, x2)


def all(x, axis=None, keepdims=False):
    return paddle.all(convert_to_tensor(x), axis=axis, keepdim=keepdims)


def any(x, axis=None, keepdims=False):
    return paddle.any(convert_to_tensor(x), axis=axis, keepdim=keepdims)


def logical_and(x1, x2):
    return _binary_op_with_dtype(paddle.logical_and, x1, x2)


def logical_or(x1, x2):
    return _binary_op_with_dtype(paddle.logical_or, x1, x2)


def logical_not(x):
    return paddle.logical_not(convert_to_tensor(x))


def logical_xor(x1, x2):
    return _binary_op_with_dtype(paddle.logical_xor, x1, x2)


def bitwise_and(x1, x2):
    return _binary_op_with_dtype(paddle.bitwise_and, x1, x2)


def bitwise_or(x1, x2):
    return _binary_op_with_dtype(paddle.bitwise_or, x1, x2)


def bitwise_xor(x1, x2):
    return _binary_op_with_dtype(paddle.bitwise_xor, x1, x2)


def bitwise_not(x):
    return paddle.bitwise_not(convert_to_tensor(x))


def isfinite(x):
    x = convert_to_tensor(x)
    if not standardize_dtype(x.dtype) in _FLOAT_TYPES:
        x = x.cast("float32")
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    return paddle.isfinite(x)


def isinf(x):
    x = convert_to_tensor(x)
    if not standardize_dtype(x.dtype) in _FLOAT_TYPES:
        x = x.cast("float32")
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    return paddle.isinf(x)


def isnan(x):
    x = convert_to_tensor(x)
    if not standardize_dtype(x.dtype) in _FLOAT_TYPES:
        x = x.cast("float32")
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    return paddle.isnan(x)


def isneginf(x):
    x = convert_to_tensor(x)
    return paddle.logical_and(paddle.isinf(x), x < 0)


def isposinf(x):
    x = convert_to_tensor(x)
    return paddle.logical_and(paddle.isinf(x), x > 0)


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    x = convert_to_tensor(x)
    return paddle.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dt1 = standardize_dtype(x1.dtype)
    x1, x2 = _promote_dtypes(x1, x2)
    if axis is not None:
        x1 = paddle.moveaxis(x1, axisa, axis)
        x2 = paddle.moveaxis(x2, axisb, axis)
    # Broadcast to same shape (paddle.cross doesn't support broadcasting)
    target_shape = paddle.broadcast_shape(x1.shape, x2.shape)
    x1 = paddle.broadcast_to(x1, target_shape)
    x2 = paddle.broadcast_to(x2, target_shape)
    result = paddle.cross(x1, x2, axis=axisc if axis is not None else axisa)
    if dt1 in ("float16", "bfloat16"):
        result = result.cast(to_paddle_dtype(dt1))
    return result


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids, "int64")
    if num_segments is None:
        num_segments = paddle.max(segment_ids) + 1
    zeros = paddle.zeros(
        [num_segments] + list(data.shape[1:]), dtype=data.dtype
    )
    return paddle.scatter_nd_add(zeros, segment_ids.unsqueeze(-1), data)


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids, "int64")
    if num_segments is None:
        num_segments = int(segment_ids.max().item()) + 1
    result = paddle.zeros(
        [num_segments] + list(data.shape[1:]), dtype=data.dtype
    )
    for i in range(num_segments):
        mask = segment_ids == i
        if mask.any():
            result[i] = paddle.max(data[mask], axis=0)
    return result


def gamma(shape, alpha, dtype=None, seed=None):
    alpha = convert_to_tensor(alpha)
    return paddle.distribution.gamma.Gamma(
        alpha, paddle.ones_like(alpha)
    ).sample(shape)


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    counts = convert_to_tensor(counts)
    probabilities = convert_to_tensor(probabilities)
    return paddle.distribution.binomial.Binomial(counts, probabilities).sample(
        shape
    )


def beta(shape, alpha, beta_param, dtype=None, seed=None):
    alpha = convert_to_tensor(alpha)
    beta_param = convert_to_tensor(beta_param)
    return paddle.distribution.beta.Beta(alpha, beta_param).sample(shape)


# --- Additional ops needed for layers/models ---


def matmul(x1, x2):
    return _binary_op_with_dtype(paddle.matmul, x1, x2)


def copy(x):
    return convert_to_tensor(x).clone()


def broadcast_to(x, shape):
    return paddle.broadcast_to(convert_to_tensor(x), shape)


def array(x, dtype=None):
    if dtype is not None:
        return convert_to_tensor(x, dtype=dtype)
    return convert_to_tensor(x)


def ndim(x):
    return len(convert_to_tensor(x).shape)


def size(x):
    return convert_to_tensor(x).size


def ravel(x):
    return paddle.flatten(convert_to_tensor(x))


def trunc(x):
    return _unary_op(paddle.trunc, x)


def inner(a, b):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    dt1 = standardize_dtype(a.dtype)
    a, b = _promote_dtypes(a, b)
    # inner product: sum over last axis
    # output shape: a.shape[:-1] + b.shape[:-1]
    if a.ndim == 1 and b.ndim == 1:
        result = paddle.dot(a, b)
    else:
        # Reshape: a -> [..., 1, K], b -> [1, ..., K], then multiply and sum
        a_expanded = a.reshape(list(a.shape[:-1]) + [1] + [a.shape[-1]])
        b_expanded = b.reshape([1] * (a.ndim - 1) + list(b.shape))
        result = paddle.sum(a_expanded * b_expanded, axis=-1)
    if dt1 in ("float16", "bfloat16"):
        result = result.cast(to_paddle_dtype(dt1))
    return result


def outer(a, b):
    a = convert_to_tensor(a).flatten()
    b = convert_to_tensor(b).flatten()
    dt1 = standardize_dtype(a.dtype)
    a, b = _promote_dtypes(a, b)
    result = paddle.mm(a.unsqueeze(1), b.unsqueeze(0))
    if dt1 in ("float16", "bfloat16"):
        result = result.cast(to_paddle_dtype(dt1))
    return result


def reciprocal(x):
    return _unary_op(paddle.reciprocal, x)


def cos(x):
    return _unary_op(paddle.cos, x)


def sin(x):
    return _unary_op(paddle.sin, x)


def tan(x):
    return _unary_op(paddle.tan, x)


def cosh(x):
    return _unary_op(paddle.cosh, x)


def sinh(x):
    return _unary_op(paddle.sinh, x)


def arccos(x):
    return _unary_op(paddle.acos, x)


def arcsin(x):
    return _unary_op(paddle.asin, x)


def arctan(x):
    return _unary_op(paddle.atan, x)


def arctan2(x1, x2):
    return _binary_op_with_dtype(paddle.atan2, x1, x2)


def arccosh(x):
    return _unary_op(paddle.acosh, x)


def arcsinh(x):
    return _unary_op(paddle.asinh, x)


def arctanh(x):
    return _unary_op(paddle.atanh, x)


def deg2rad(x):
    return _unary_op(paddle.deg2rad, x)


def rad2deg(x):
    return _unary_op(paddle.rad2deg, x)


def hypot(x1, x2):
    return _binary_op_with_dtype(paddle.hypot, x1, x2)


def fmod(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    # numpy.fmod uses truncation division, not floor division
    # Cast to float for division, then cast back
    orig_dtype = x1.dtype
    x1f = x1.cast("float64")
    x2f = x2.cast("float64")
    return (x1f - paddle.trunc(x1f / x2f) * x2f).cast(orig_dtype)


def ldexp(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dt1 = standardize_dtype(x1.dtype)
    x1, x2 = _promote_dtypes(x1, x2)
    result = x1 * paddle.pow(
        paddle.to_tensor(2.0, dtype=x1.dtype), x2
    )
    if dt1 in ("float16", "bfloat16"):
        result = result.cast(to_paddle_dtype(dt1))
    return result


def left_shift(x1, x2):
    return paddle.bitwise_left_shift(
        convert_to_tensor(x1), convert_to_tensor(x2)
    )


def right_shift(x1, x2):
    return paddle.bitwise_right_shift(
        convert_to_tensor(x1), convert_to_tensor(x2)
    )


def bitwise_left_shift(x1, x2):
    return left_shift(x1, x2)


def bitwise_right_shift(x1, x2):
    return right_shift(x1, x2)


def bitwise_invert(x):
    return paddle.bitwise_not(convert_to_tensor(x))


def signbit(x):
    x = convert_to_tensor(x)
    return paddle.sign(x) < 0


def heaviside(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    x1, x2 = _promote_dtypes(x1, x2)
    return paddle.where(
        x1 > 0,
        paddle.ones_like(x1),
        paddle.where(x1 == 0, x2, paddle.zeros_like(x1)),
    )


def i0(x):
    x = convert_to_tensor(x, "float32")
    return paddle.i0(x)


def sinc(x):
    x = convert_to_tensor(x, "float32")
    return paddle.where(
        x == 0, paddle.ones_like(x), paddle.sin(np.pi * x) / (np.pi * x)
    )


def count_nonzero(x, axis=None):
    x = convert_to_tensor(x)
    return paddle.sum(paddle.cast(x != 0, "int32"), axis=axis)


def nanargmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    x = paddle.where(paddle.isnan(x), paddle.full_like(x, float("-inf")), x)
    return paddle.argmax(x, axis=axis, keepdim=keepdims)


def nanargmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    x = paddle.where(paddle.isnan(x), paddle.full_like(x, float("inf")), x)
    return paddle.argmin(x, axis=axis, keepdim=keepdims)


def nanmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    x = paddle.where(paddle.isnan(x), paddle.full_like(x, float("-inf")), x)
    return paddle.max(x, axis=axis, keepdim=keepdims)


def nanmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    x = paddle.where(paddle.isnan(x), paddle.full_like(x, float("inf")), x)
    return paddle.min(x, axis=axis, keepdim=keepdims)


def nansum(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    x = paddle.where(paddle.isnan(x), paddle.zeros_like(x), x)
    return paddle.sum(x, axis=axis, keepdim=keepdims)


def nanmean(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if not standardize_dtype(x.dtype) in _FLOAT_TYPES:
        x = x.cast("float32")
    mask = ~paddle.isnan(x)
    x = paddle.where(mask, x, paddle.zeros_like(x))
    count = paddle.sum(
        paddle.cast(mask, "float32"), axis=axis, keepdim=keepdims
    )
    return paddle.sum(x, axis=axis, keepdim=keepdims) / count


def nanvar(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if not standardize_dtype(x.dtype) in _FLOAT_TYPES:
        x = x.cast("float32")
    mask = ~paddle.isnan(x)
    m = nanmean(x, axis=axis, keepdims=True)
    x = paddle.where(mask, x, paddle.zeros_like(x))
    diff = (x - m) * mask.cast(x.dtype)
    count = paddle.sum(mask.cast("float32"), axis=axis, keepdim=True)
    return paddle.sum(diff**2, axis=axis, keepdim=keepdims) / count


def nanstd(x, axis=None, keepdims=False):
    return paddle.sqrt(nanvar(x, axis=axis, keepdims=keepdims))


def nanprod(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    x = paddle.where(paddle.isnan(x), paddle.ones_like(x), x)
    return paddle.prod(x, axis=axis, keepdim=keepdims)


def nancumsum(x, axis=None, dtype=None):
    x = convert_to_tensor(x)
    x = paddle.where(paddle.isnan(x), paddle.zeros_like(x), x)
    result = paddle.cumsum(x, axis=axis)
    if dtype is not None:
        result = result.cast(to_paddle_dtype(dtype))
    return result


def nancumprod(x, axis=None, dtype=None):
    x = convert_to_tensor(x)
    x = paddle.where(paddle.isnan(x), paddle.ones_like(x), x)
    result = paddle.cumprod(x, axis=axis)
    if dtype is not None:
        result = result.cast(to_paddle_dtype(dtype))
    return result


def select(condlist, choicelist, default=0):
    result = paddle.full_like(convert_to_tensor(choicelist[-1]), default)
    for cond, choice in reversed(zip(condlist, choicelist)):
        result = paddle.where(
            convert_to_tensor(cond), convert_to_tensor(choice), result
        )
    return result


def unique(x, **kwargs):
    x = convert_to_tensor(x)
    return paddle.unique(x, **kwargs)


def unravel_index(indices, shape):
    indices = convert_to_tensor(indices, "int64")
    result = []
    for s in reversed(shape):
        result.append(indices % s)
        indices = indices // s
    return tuple(reversed(result))


def kron(a, b):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    dt1 = standardize_dtype(a.dtype)
    a, b = _promote_dtypes(a, b)
    result = paddle.kron(a, b)
    if dt1 in ("float16", "bfloat16"):
        result = result.cast(to_paddle_dtype(dt1))
    return result


def vdot(a, b):
    a = convert_to_tensor(a).flatten()
    b = convert_to_tensor(b).flatten()
    if a.is_complex():
        return paddle.dot(a.conj(), b)
    return paddle.dot(a, b)


def vectorize(pyfunc, **kwargs):
    import functools

    @functools.wraps(pyfunc)
    def wrapper(*args):
        return pyfunc(*args)

    return wrapper


def view(x, dtype=None):
    x = convert_to_tensor(x)
    if dtype is None:
        return x
    return x.view(dtype=to_paddle_dtype(dtype))


def diff(x, n=1, axis=-1, prepend=None, append=None):
    x = convert_to_tensor(x)
    if prepend is not None:
        prepend = convert_to_tensor(prepend)
        x = paddle.concat([prepend, x], axis=axis)
    if append is not None:
        append = convert_to_tensor(append)
        x = paddle.concat([x, append], axis=axis)
    for _ in range(n):
        x = paddle.diff(x, axis=axis)
    return x


def digitize(x, bins):
    x = convert_to_tensor(x)
    bins = convert_to_tensor(bins)
    # numpy.digitize is equivalent to searchsorted with side='right'
    return paddle.searchsorted(bins, x, right=True)


def bincount(x, weights=None, minlength=0, sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with paddle backend")
    x = convert_to_tensor(x, "int64")
    if weights is not None:
        weights = convert_to_tensor(weights)
    n = paddle.maximum(x.max() + 1, paddle.to_tensor(minlength, dtype="int64"))
    if weights is None:
        ones = paddle.ones([x.shape[0]], dtype="int64")
        return paddle.scatter_nd_add(
            paddle.zeros([n], dtype="int64"), x.unsqueeze(-1), ones
        )
    return paddle.scatter_nd_add(
        paddle.zeros([n], dtype=weights.dtype), x.unsqueeze(-1), weights
    )


def corrcoef(x):
    x = convert_to_tensor(x)
    std = standardize_dtype(x.dtype)
    if std == "bool":
        x = x.cast("float32")
    elif std == "int64":
        x = x.cast("float64")
    # Flatten to 2D if needed
    if x.ndim < 2:
        x = x.unsqueeze(0)
    # Center the data
    x = x - x.mean(axis=1, keepdim=True)
    # Compute covariance
    n = x.shape[1]
    c = paddle.matmul(x, x.t()) / n
    # Compute correlation
    d = paddle.diag(c).clip(min=1e-12).sqrt()
    c = c / (d.unsqueeze(1) * d.unsqueeze(0))
    return c


def correlate(x1, x2, mode="valid"):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    if mode == "valid":
        return paddle.nn.functional.conv1d(
            x1.reshape([1, 1, -1]), x2.reshape([1, 1, -1]), padding=0
        )
    elif mode == "same":
        return paddle.nn.functional.conv1d(
            x1.reshape([1, 1, -1]),
            x2.reshape([1, 1, -1]),
            padding=x2.shape[0] // 2,
        )
    elif mode == "full":
        return paddle.nn.functional.conv1d(
            x1.reshape([1, 1, -1]),
            x2.reshape([1, 1, -1]),
            padding=x2.shape[0] - 1,
        )
    raise ValueError(f"Mode {mode} not supported")


def median(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if not standardize_dtype(x.dtype) in _FLOAT_TYPES:
        x = x.cast("float32")
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    return paddle.median(x, axis=axis, keepdim=keepdims)


def quantile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    q = convert_to_tensor(q, "float32")
    if not standardize_dtype(x.dtype) in _FLOAT_TYPES:
        x = x.cast("float32")
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        x = x.cast("float32")
    if isinstance(axis, tuple):
        axis = list(axis)
    return paddle.quantile(
        x, q, axis=axis, keepdim=keepdims, interpolation=method
    )


def percentile(x, q, axis=None, method="linear", keepdims=False):
    return quantile(x, q / 100.0, axis=axis, method=method, keepdims=keepdims)


def nanmedian(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    mask = paddle.isnan(x)
    if axis is None:
        x = x.flatten()
        mask = mask.flatten()
    x_no_nan = paddle.where(mask, paddle.zeros_like(x), x)
    sorted_x = paddle.sort(x_no_nan, axis=axis if axis is not None else -1)
    if axis is None:
        valid_count = paddle.sum((~mask).cast("int64")).item()
        mid = valid_count // 2
        if valid_count % 2 == 0:
            result = (sorted_x[mid - 1] + sorted_x[mid]) / 2.0
        else:
            result = sorted_x[mid]
    else:
        valid_count = paddle.sum((~mask).cast("int64"), axis=axis)
        mid = valid_count // 2
        result = paddle.gather(sorted_x, mid, axis=axis)
    if keepdims:
        if axis is not None:
            result = paddle.unsqueeze(result, axis=axis)
    return result


def nanquantile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    if not standardize_dtype(x.dtype) in _FLOAT_TYPES:
        x = x.cast("float32")
    mask = paddle.isnan(x)
    if axis is None:
        x = x.flatten()
        mask = mask.flatten()
    x_no_nan = paddle.where(mask, paddle.zeros_like(x), x)
    sorted_x = paddle.sort(x_no_nan, axis=axis if axis is not None else -1)
    if isinstance(q, (list, tuple)):
        q = convert_to_tensor(q, "float32")
    else:
        q = convert_to_tensor([q], "float32")
    if axis is None:
        valid_count = paddle.sum((~mask).cast("int64")).item()
        indices = q * (valid_count - 1)
        lower = indices.cast("int64")
        upper = paddle.minimum(
            lower + 1, paddle.to_tensor(valid_count - 1, dtype="int64")
        )
        frac = indices - lower.cast("float32")
        result = sorted_x[lower] * (1 - frac) + sorted_x[upper] * frac
    else:
        valid_count = paddle.sum((~mask).cast("int64"), axis=axis)
        indices = q * (valid_count - 1)
        lower = indices.cast("int64")
        upper = paddle.minimum(
            lower + 1, paddle.to_tensor(valid_count - 1, dtype="int64")
        )
        frac = indices - lower.cast("float32")
        lower_vals = paddle.gather(sorted_x, lower, axis=axis)
        upper_vals = paddle.gather(sorted_x, upper, axis=axis)
        result = lower_vals * (1 - frac) + upper_vals * frac
    if keepdims:
        if axis is not None:
            result = paddle.unsqueeze(result, axis=axis)
    return result


def nanpercentile(x, q, axis=None, method="linear", keepdims=False):
    return nanquantile(x, q / 100.0, axis=axis, method=method, keepdims=keepdims)


def ptp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return paddle.max(x, axis=axis, keepdim=keepdims) - paddle.min(
        x, axis=axis, keepdim=keepdims
    )


def logaddexp(x1, x2):
    return paddle.logaddexp(convert_to_tensor(x1), convert_to_tensor(x2))


def logaddexp2(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return paddle.logaddexp(x1 * np.log(2), x2 * np.log(2)) / np.log(2)


def logspace(start, stop, num, base=10.0, dtype=None, endpoint=True, axis=0):
    result = linspace(start, stop, num, endpoint=endpoint, dtype=dtype)
    orig_dtype = result.dtype
    if result.dtype in _CPU_UNSUPPORTED_DTYPES:
        result = result.cast("float32")
    base_t = paddle.to_tensor(base, dtype=result.dtype)
    return paddle.pow(base_t, result).cast(orig_dtype)


def geomspace(start, stop, num, endpoint=True, dtype=None, axis=0):
    start = convert_to_tensor(start, "float32")
    stop = convert_to_tensor(stop, "float32")
    result = paddle.exp(
        paddle.linspace(paddle.log(start), paddle.log(stop), num)
    )
    if dtype is not None:
        target_dtype = to_paddle_dtype(dtype)
        if target_dtype in _CPU_UNSUPPORTED_DTYPES:
            result = result.cast(target_dtype)
        else:
            result = result.cast(target_dtype)
    return result


def empty(shape, dtype="float32"):
    return zeros(shape, dtype=dtype)


def empty_like(x, dtype=None):
    return zeros_like(x, dtype=dtype)


def nextafter(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    # Cast to float for computation
    orig_dtype = x1.dtype
    if not standardize_dtype(x1.dtype) in _FLOAT_TYPES:
        x1 = x1.cast("float32")
        x2 = x2.cast("float32")
    if x1.dtype in _CPU_UNSUPPORTED_DTYPES:
        x1 = x1.cast("float32")
        x2 = x2.cast("float32")
    # Approximate nextafter using bit manipulation
    eps = np.finfo(standardize_dtype(x1.dtype)).eps
    direction = paddle.sign(x2 - x1)
    result = x1 + direction * eps
    return result.cast(orig_dtype)


def isreal(x):
    x = convert_to_tensor(x)
    return paddle.isreal(x)


def isin(elements, test_elements, assume_unique=False, invert=False):
    elements = convert_to_tensor(elements)
    test_elements = convert_to_tensor(test_elements).flatten()
    elements, test_elements = _promote_dtypes(elements, test_elements)
    # Broadcasting comparison
    result = paddle.any(elements.unsqueeze(-1) == test_elements, axis=-1)
    if invert:
        result = ~result
    return result


def nonzero(x):
    x = convert_to_tensor(x)
    indices = paddle.nonzero(x)
    return tuple(indices.T)


def array_split(x, indices_or_sections, axis=0):
    x = convert_to_tensor(x)
    if isinstance(indices_or_sections, int):
        n = indices_or_sections
        size = x.shape[axis]
        base_size = size // n
        remainder = size % n
        splits = []
        start = 0
        for i in range(n):
            end = start + base_size + (1 if i < remainder else 0)
            splits.append(paddle.slice(x, [axis], [start], [end]))
            start = end
        return splits
    else:
        indices = list(indices_or_sections)
        splits = []
        start = 0
        for idx in indices:
            splits.append(paddle.slice(x, [axis], [start], [idx]))
            start = idx
        splits.append(paddle.slice(x, [axis], [start], [x.shape[axis]]))
        return splits


def dsplit(x, indices_or_sections):
    return split(x, indices_or_sections, axis=2)


def hsplit(x, indices_or_sections):
    return split(x, indices_or_sections, axis=1)


def vsplit(x, indices_or_sections):
    return split(x, indices_or_sections, axis=0)


def dstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    xs = _promote_dtypes_list(xs)
    xs = [paddle.unsqueeze(x, axis=2) if len(x.shape) < 3 else x for x in xs]
    return paddle.concat(xs, axis=2)


def hstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    xs = _promote_dtypes_list(xs)
    if len(xs[0].shape) == 1:
        return paddle.concat(xs, axis=0)
    return paddle.concat(xs, axis=1)


def vstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    xs = _promote_dtypes_list(xs)
    return paddle.concat(xs, axis=0)


def diagflat(x, k=0):
    x = convert_to_tensor(x).flatten()
    return paddle.diag(x, offset=k)


def fliplr(x):
    return paddle.flip(convert_to_tensor(x), axis=[1])


def flipud(x):
    return paddle.flip(convert_to_tensor(x), axis=[0])


def rot90(x, k=1, axes=(0, 1)):
    x = convert_to_tensor(x)
    if x.ndim < 2:
        raise ValueError(
            f"rot90 requires at least 2 dimensions, got {x.ndim}"
        )
    if axes[0] == axes[1] or axes[0] % x.ndim == axes[1] % x.ndim:
        raise ValueError("axes must be different")
    k = k % 4
    if k == 0:
        return x
    # Build the permutation that swaps axes[0] and axes[1]
    perm = list(range(x.ndim))
    perm[axes[0]], perm[axes[1]] = axes[1], axes[0]
    if k == 1:
        return paddle.transpose(paddle.flip(x, [axes[1]]), perm)
    elif k == 2:
        return paddle.flip(x, [axes[0], axes[1]])
    elif k == 3:
        return paddle.transpose(paddle.flip(x, [axes[0]]), perm)
    return x


def average(x, axis=None, weights=None, returned=False, keepdims=False):
    x = convert_to_tensor(x)
    if weights is None:
        result = paddle.mean(x, axis=axis, keepdim=keepdims)
    else:
        weights = convert_to_tensor(weights)
        x, weights = _promote_dtypes(x, weights)
        result = paddle.sum(
            x * weights, axis=axis, keepdim=keepdims
        ) / paddle.sum(weights, axis=axis, keepdim=keepdims)
    if returned:
        weights_sum = paddle.sum(
            weights if weights is not None else paddle.ones_like(x),
            axis=axis,
            keepdim=keepdims,
        )
        return result, weights_sum
    return result


def cbrt(x):
    return paddle.pow(convert_to_tensor(x, "float32"), 1.0 / 3.0)


def exp2(x):
    return paddle.pow(2.0, convert_to_tensor(x, "float32"))


def divide_no_nan(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    x1, x2 = _promote_dtypes(x1, x2)
    safe_x2 = paddle.where(x2 == 0, paddle.ones_like(x2), x2)
    return paddle.where(x2 == 0, paddle.zeros_like(x1), x1 / safe_x2)


def slogdet(x):
    x = convert_to_tensor(x)
    sign, logabsdet = paddle.linalg.slogdet(x)
    return sign, logabsdet


def argpartition(x, kth, axis=-1):
    x = convert_to_tensor(x)
    if axis is None:
        x = x.flatten()
        axis = 0
    sorted_indices = paddle.argsort(x, axis=axis)
    return sorted_indices


def gcd(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    # Cast to same int type for gcd
    dt1 = standardize_dtype(x1.dtype)
    dt2 = standardize_dtype(x2.dtype)
    if dt1 != dt2:
        # Use the wider type
        int_order = {"bool": 0, "int8": 1, "int16": 2, "int32": 3, "int64": 4,
                     "uint8": 1, "uint16": 2, "uint32": 3}
        w1 = int_order.get(dt1, 3)
        w2 = int_order.get(dt2, 3)
        target = dt1 if w1 >= w2 else dt2
        x1 = x1.cast(to_paddle_dtype(target))
        x2 = x2.cast(to_paddle_dtype(target))
    return paddle.gcd(x1, x2)


def lcm(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dt1 = standardize_dtype(x1.dtype)
    dt2 = standardize_dtype(x2.dtype)
    if dt1 != dt2:
        int_order = {"bool": 0, "int8": 1, "int16": 2, "int32": 3, "int64": 4,
                     "uint8": 1, "uint16": 2, "uint32": 3}
        w1 = int_order.get(dt1, 3)
        w2 = int_order.get(dt2, 3)
        target = dt1 if w1 >= w2 else dt2
        x1 = x1.cast(to_paddle_dtype(target))
        x2 = x2.cast(to_paddle_dtype(target))
    return paddle.lcm(x1, x2)


def max(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        orig_dtype = x.dtype
        x = x.cast("float32")
        result = paddle.max(x, axis=axis, keepdim=keepdims)
        if initial is not None:
            result = paddle.maximum(result, convert_to_tensor(initial))
        return result.cast(orig_dtype)
    result = paddle.max(x, axis=axis, keepdim=keepdims)
    if initial is not None:
        result = paddle.maximum(result, convert_to_tensor(initial))
    return result


def min(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        orig_dtype = x.dtype
        x = x.cast("float32")
        result = paddle.min(x, axis=axis, keepdim=keepdims)
        if initial is not None:
            result = paddle.minimum(result, convert_to_tensor(initial))
        return result.cast(orig_dtype)
    result = paddle.min(x, axis=axis, keepdim=keepdims)
    if initial is not None:
        result = paddle.minimum(result, convert_to_tensor(initial))
    return result


def amin(x, axis=None, keepdims=False, initial=None):
    return min(x, axis=axis, keepdims=keepdims, initial=initial)


def amax(x, axis=None, keepdims=False, initial=None):
    return max(x, axis=axis, keepdims=keepdims, initial=initial)


def var(x, axis=None, keepdims=False):
    return paddle.var(convert_to_tensor(x), axis=axis, keepdim=keepdims)


def tanh(x):
    return paddle.tanh(convert_to_tensor(x))


def fabs(x):
    return paddle.abs(convert_to_tensor(x))


def diag(x, k=0):
    x = convert_to_tensor(x)
    if x.dtype in _CPU_UNSUPPORTED_DTYPES:
        orig_dtype = x.dtype
        return paddle.diag(x.cast("float32"), offset=k).cast(orig_dtype)
    return paddle.diag(x, offset=k)


def trapezoid(y, x=None, dx=1.0, axis=-1):
    y = convert_to_tensor(y)
    if x is not None:
        x = convert_to_tensor(x)
        dx_tensor = x[..., 1:] - x[..., :-1]
        avg = (y[..., 1:] + y[..., :-1]) / 2.0
        return paddle.sum(avg * dx_tensor, axis=axis)

    ndim = y.ndim
    axis = axis + ndim if axis < 0 else axis
    slice_left = [builtins.slice(None)] * ndim
    slice_left[axis] = builtins.slice(None, -1)
    slice_right = [builtins.slice(None)] * ndim
    slice_right[axis] = builtins.slice(1, None)
    avg = (y[tuple(slice_right)] + y[tuple(slice_left)]) / 2.0
    return paddle.sum(avg * dx, axis=axis)


def bartlett(M):
    if M < 1:
        return paddle.zeros([0])
    if M == 1:
        return paddle.ones([1])
    n = paddle.arange(M, dtype="float32")
    return paddle.where(
        n <= (M - 1) / 2.0, 2 * n / (M - 1), 2 - 2 * n / (M - 1)
    )


def blackman(M):
    if M < 1:
        return paddle.zeros([0])
    n = paddle.arange(M, dtype="float32")
    return (
        0.42
        - 0.5 * paddle.cos(2 * np.pi * n / (M - 1))
        + 0.08 * paddle.cos(4 * np.pi * n / (M - 1))
    )


def hamming(M):
    if M < 1:
        return paddle.zeros([0])
    n = paddle.arange(M, dtype="float32")
    return 0.54 - 0.46 * paddle.cos(2 * np.pi * n / (M - 1))


def hanning(M):
    if M < 1:
        return paddle.zeros([0])
    n = paddle.arange(M, dtype="float32")
    return 0.5 - 0.5 * paddle.cos(2 * np.pi * n / (M - 1))


def kaiser(M, beta):
    if M < 1:
        return paddle.zeros([0])
    n = paddle.arange(M, dtype="float32")
    alpha = (M - 1) / 2.0
    return paddle.i0(
        beta * paddle.sqrt(1 - ((n - alpha) / alpha) ** 2)
    ) / paddle.i0(paddle.to_tensor(beta, dtype="float32"))


def vander(x, N=None, increasing=False):
    x = convert_to_tensor(x)
    if N is None:
        N = x.shape[0]
    if not increasing:
        powers = paddle.arange(N - 1, -1, -1, dtype="float32")
    else:
        powers = paddle.arange(0, N, dtype="float32")
    x, powers = _promote_dtypes(x, powers)
    return x.unsqueeze(1) ** powers.unsqueeze(0)
