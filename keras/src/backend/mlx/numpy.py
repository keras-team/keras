import builtins
import math
from copy import copy as builtin_copy

import mlx.core as mx
import numpy as np

from keras.src.backend import config
from keras.src.backend import result_type
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.common.backend_utils import vectorize_impl
from keras.src.backend.mlx.core import cast
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.backend.mlx.core import convert_to_tensors
from keras.src.backend.mlx.core import is_tensor
from keras.src.backend.mlx.core import slice
from keras.src.backend.mlx.core import to_mlx_dtype


def _promote(*xs):
    # mlx promotes uint32 + signed to int32, keras/jax want int64. Cast to
    # result_type so the output dtype matches the other backends.
    dtype = result_type(*[getattr(x, "dtype", type(x)) for x in xs])
    return [convert_to_tensor(x, dtype) for x in xs]


def add(x1, x2):
    x1, x2 = _promote(x1, x2)
    return mx.add(x1, x2)


def einsum(subscripts, *operands, **kwargs):
    operands = [convert_to_tensor(x) for x in operands]
    dtypes_to_resolve = list({standardize_dtype(x.dtype) for x in operands})
    # int8-only einsum accumulates into int32 to align with jax.
    if len(dtypes_to_resolve) == 1 and dtypes_to_resolve[0] == "int8":
        result_dtype = "int32"
    else:
        result_dtype = result_type(*[x.dtype for x in operands])
    # mlx einsum only supports floating point, so integer contractions run in
    # float32 and are cast back to the integer result dtype.
    if "int" in result_dtype or result_dtype == "bool":
        compute_dtype = mx.float32
    else:
        compute_dtype = to_mlx_dtype(result_dtype)
    operands = [x.astype(compute_dtype) for x in operands]
    out = mx.einsum(subscripts, *operands)
    return out.astype(to_mlx_dtype(result_dtype))


def subtract(x1, x2):
    x1, x2 = _promote(x1, x2)
    return mx.subtract(x1, x2)


def matmul(x1, x2):
    x1, x2 = convert_to_tensors(x1, x2)
    x1_dtype = standardize_dtype(x1.dtype)
    x2_dtype = standardize_dtype(x2.dtype)
    # int8 @ int8 accumulates into int32 to align with jax.
    if x1_dtype == "int8" and x2_dtype == "int8":
        result_dtype = "int32"
    else:
        result_dtype = result_type(x1.dtype, x2.dtype)
    # mlx matmul only supports floating point, so integer matmuls run in
    # float32 and are cast back to the integer result dtype.
    if "int" in result_dtype or result_dtype == "bool":
        compute_dtype = mx.float32
    else:
        compute_dtype = to_mlx_dtype(result_dtype)
    out = mx.matmul(x1.astype(compute_dtype), x2.astype(compute_dtype))
    return out.astype(to_mlx_dtype(result_dtype))


def multiply(x1, x2):
    x1, x2 = _promote(x1, x2)
    return mx.multiply(x1, x2)


def mean(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    # `mx.mean` does not handle low precision (e.g., float16) overflow
    # correctly, so we compute with float32 and cast back to the original type.
    compute_dtype = result_type(x.dtype, "float32")
    if "int" in ori_dtype or ori_dtype == "bool":
        result_dtype = compute_dtype
    else:
        result_dtype = ori_dtype

    compute_dtype = to_mlx_dtype(compute_dtype)
    result_dtype = to_mlx_dtype(result_dtype)
    x = x.astype(compute_dtype)
    output = mx.mean(x, axis=axis, keepdims=keepdims)
    return cast(output, result_dtype)


def _reduced_shape(shape, axis, keepdims):
    # The output shape of a reduction over `axis`, honoring keepdims.
    if axis is None:
        return (1,) * len(shape) if keepdims else ()
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes = tuple(a % len(shape) for a in axes)
    out = []
    for i, s in enumerate(shape):
        if i in axes:
            if keepdims:
                out.append(1)
        else:
            out.append(s)
    return tuple(out)


def _reduces_empty_axis(shape, axis):
    # True when the reduction covers an axis of size 0. Use builtins.any since
    # this module defines its own `any`.
    if axis is None:
        return 0 in shape
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    return builtins.any(shape[a % len(shape)] == 0 for a in axes)


def max(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)
    if 0 in x.shape:
        # mlx cannot reduce a zero size array, so build the result directly.
        # An initial value is only required when a reduced axis is empty and
        # the output is non-empty, matching numpy.
        out_shape = _reduced_shape(x.shape, axis, keepdims)
        needs_initial = (
            _reduces_empty_axis(x.shape, axis) and 0 not in out_shape
        )
        if needs_initial and initial is None:
            raise ValueError("Cannot compute the max of an empty tensor.")
        fill = initial if needs_initial else 0
        return mx.full(out_shape, fill, dtype=x.dtype)

    result = mx.max(x, axis=axis, keepdims=keepdims)
    if initial is not None:
        result = mx.maximum(result, initial)

    return result.astype(x.dtype)


def ones(shape, dtype=None):
    dtype = to_mlx_dtype(dtype or config.floatx())
    return mx.ones(shape, dtype=dtype)


def zeros(shape, dtype=None):
    dtype = to_mlx_dtype(dtype or config.floatx())
    return mx.zeros(shape, dtype=dtype)


def zeros_like(x, dtype=None):
    x = convert_to_tensor(x)
    dtype = to_mlx_dtype(dtype or x.dtype)
    return mx.zeros(x.shape, dtype=dtype)


def absolute(x):
    x = convert_to_tensor(x)
    return mx.abs(x)


def abs(x):
    x = convert_to_tensor(x)
    return absolute(x)


def all(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return mx.all(x, axis=axis, keepdims=keepdims)


def any(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return mx.any(x, axis=axis, keepdims=keepdims)


def amax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return mx.max(x, axis=axis, keepdims=keepdims)


def amin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return mx.min(x, axis=axis, keepdims=keepdims)


def append(x1, x2, axis=None):
    x1, x2 = _promote(x1, x2)
    return mx.concatenate([x1, x2], axis=axis)


def arange(start, stop=None, step=None, dtype=None):
    if dtype is None:
        dtypes_to_resolve = [getattr(start, "dtype", type(start))]
        if stop is not None:
            dtypes_to_resolve.append(getattr(stop, "dtype", type(stop)))
        if step is not None:
            dtypes_to_resolve.append(getattr(step, "dtype", type(step)))
        dtype = result_type(*dtypes_to_resolve)
    dtype = to_mlx_dtype(dtype)
    if stop is None:
        stop = start
        start = 0
    if step is None:
        step = 1
    # mlx arange only accepts python scalars, not 0-d arrays.
    start = start.item() if hasattr(start, "item") else start
    stop = stop.item() if hasattr(stop, "item") else stop
    step = step.item() if hasattr(step, "item") else step
    return mx.arange(start, stop, step=step, dtype=dtype)


def arccos(x):
    x = convert_to_tensor(x)
    return mx.arccos(x)


def arccosh(x):
    x = convert_to_tensor(x)
    return mx.arccosh(x)


def arcsin(x):
    x = convert_to_tensor(x)
    return mx.arcsin(x)


def arcsinh(x):
    x = convert_to_tensor(x)
    return mx.arcsinh(x)


def arctan(x):
    x = convert_to_tensor(x)
    return mx.arctan(x)


def arctan2(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.arctan2(x1, x2)


def arctanh(x):
    x = convert_to_tensor(x)
    return mx.arctanh(x)


def argmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    # cast to int32 to align with other backends
    return mx.argmax(x, axis=axis, keepdims=keepdims).astype(mx.int32)


def argmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    # cast to int32 to align with other backends
    return mx.argmin(x, axis=axis, keepdims=keepdims).astype(mx.int32)


def argsort(x, axis=-1):
    x = convert_to_tensor(x)
    axis = None if x.ndim == 0 else axis
    # cast to int32 to align with other backends
    return mx.argsort(x, axis=axis).astype(mx.int32)


def argpartition(x, kth, axis=-1):
    x = convert_to_tensor(x)
    # cast to int32 to align with other backends
    return mx.argpartition(x, kth, axis).astype(mx.int32)


def array(x, dtype=None):
    return convert_to_tensor(x, dtype=dtype)


def average(x, axis=None, weights=None):
    x = convert_to_tensor(x)
    dtypes_to_resolve = [x.dtype, float]
    if weights is not None:
        weights = convert_to_tensor(weights)
        dtypes_to_resolve.append(weights.dtype)
    dtype = result_type(*dtypes_to_resolve)
    x = cast(x, dtype)

    # Early exit
    if axis == () or axis == []:
        return x

    # Weighted average
    if weights is not None:
        weights = cast(weights, dtype)
        if len(weights.shape) < len(x.shape):
            s = [1] * len(x.shape)
            s[axis] = x.shape[axis]
            weights = weights.reshape(s)

        # TODO: mean(a * b) / mean(b) is more numerically stable in case a is
        #       large
        return mx.sum(mx.multiply(x, weights), axis=axis) / mx.sum(
            weights, axis=axis
        )

    # Plain average
    return mx.mean(x, axis=axis)


def bitwise_and(x, y):
    x, y = _promote(x, y)
    return mx.bitwise_and(x, y)


def bitwise_invert(x):
    x = convert_to_tensor(x)
    return ~x


def bitwise_not(x):
    return bitwise_invert(x)


def bitwise_or(x, y):
    x, y = _promote(x, y)
    return mx.bitwise_or(x, y)


def bitwise_xor(x, y):
    x, y = _promote(x, y)
    return mx.bitwise_xor(x, y)


def bitwise_left_shift(x, y):
    x = convert_to_tensor(x)
    if not isinstance(y, int):
        y = convert_to_tensor(y)

    # handle result dtype to match other backends
    types = [x.dtype]
    if is_tensor(y):
        types.append(y.dtype)
    result_dtype = result_type(*types)
    mlx_result_dtype = to_mlx_dtype(result_dtype)

    result = mx.left_shift(x, y)
    if result.dtype != mlx_result_dtype:
        return result.astype(mlx_result_dtype)
    return result


def left_shift(x, y):
    return bitwise_left_shift(x, y)


def bitwise_right_shift(x, y):
    x = convert_to_tensor(x)
    if not isinstance(y, int):
        y = convert_to_tensor(y)

    # handle result dtype to match other backends
    types = [x.dtype]
    if is_tensor(y):
        types.append(y.dtype)
    result_dtype = result_type(*types)
    mlx_result_dtype = to_mlx_dtype(result_dtype)

    result = mx.right_shift(x, y)
    if result.dtype != mlx_result_dtype:
        return result.astype(mlx_result_dtype)
    return result


def right_shift(x, y):
    return bitwise_right_shift(x, y)


def _bincount_1d(x, weights=None, minlength=0):
    x_max = mx.max(x)
    length = mx.maximum(x_max + 1, minlength)

    counts = mx.zeros(length)
    if weights is None:
        counts = counts.at[x].add(1)
    else:
        counts = counts.at[x].add(weights)

    return counts


def bincount(x, weights=None, minlength=0, sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with mlx backend")

    x = convert_to_tensor(x)
    if mx.any(x < 0):
        raise ValueError(
            "`bincount` does not support negative values in `x`. "
            f"Received: x={x}"
        )
    dtypes_to_resolve = [x.dtype]
    if weights is not None:
        weights = convert_to_tensor(weights)
        dtypes_to_resolve.append(weights.dtype)
        dtype = dtypes.result_type(*dtypes_to_resolve)
    else:
        dtype = "int32"
    mlx_dtype = to_mlx_dtype(dtype)

    if len(x.shape) == 2:
        batch_size = x.shape[0]
        results = []
        for i in range(batch_size):
            w = None if weights is None else weights[i]
            results.append(_bincount_1d(x[i], w, minlength))
        return mx.stack(results).astype(mlx_dtype)

    return _bincount_1d(x, weights, minlength).astype(mlx_dtype)


def broadcast_to(x, shape):
    x = convert_to_tensor(x)
    return mx.broadcast_to(x, shape)


def ceil(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    mlx_dtype = to_mlx_dtype(dtype)
    x = x.astype(mlx_dtype)
    return mx.ceil(x)


def clip(x, x_min, x_max):
    x, x_min, x_max = convert_to_tensors(x, x_min, x_max)
    # Match numpy and jax: when x_min > x_max the upper bound wins.
    return mx.minimum(mx.maximum(x, x_min), x_max)


def concatenate(xs, axis=0):
    xs = _promote(*xs)
    return mx.concatenate(xs, axis=axis)


def conjugate(x):
    x = convert_to_tensor(x)
    return mx.conjugate(x)


def conj(x):
    x = convert_to_tensor(x)
    return conjugate(x)


def copy(x):
    x = convert_to_tensor(x)
    return builtin_copy(x)


def cos(x):
    x = convert_to_tensor(x)
    return mx.cos(x)


def cosh(x):
    x = convert_to_tensor(x)
    return mx.cosh(x)


def count_nonzero(x, axis=None):
    x = convert_to_tensor(x)
    return (x != 0).astype(mx.int32).sum(axis=axis)


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    x1, x2 = _promote(x1, x2)

    if axis is not None:
        axisa = axisb = axisc = axis

    if axisa != -1:
        x1 = mx.moveaxis(x1, axisa, -1)
    if axisb != -1:
        x2 = mx.moveaxis(x2, axisb, -1)

    result = mx.linalg.cross(x1, x2)

    if x1.shape[-1] == 2:
        result = result[
            ..., 2
        ]  # if inputs are 2D vectors, take only scalar result

    if axisc != -1:
        result = mx.moveaxis(result, -1, axisc)

    return result


def cumprod(x, axis=None, dtype=None):
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(dtype or x.dtype)
    if dtype == "bool":
        dtype = "int32"
    x = cast(x, dtype)
    if x.dtype in [mx.int64, mx.uint64]:
        return mx.cumprod(
            x, axis=axis, stream=mx.Device(type=mx.DeviceType.cpu)
        )
    return mx.cumprod(x, axis=axis)


def cumsum(x, axis=None, dtype=None):
    x = convert_to_tensor(x)
    if dtype is not None:
        x = cast(x, dtype)
    if x.dtype in [mx.int64, mx.uint64]:
        return mx.cumsum(x, axis=axis, stream=mx.Device(type=mx.DeviceType.cpu))
    return mx.cumsum(x, axis=axis)


def diag(x, k=0):
    x = convert_to_tensor(x)
    if x.dtype in [mx.int64, mx.uint64]:
        return mx.diag(x, k=k, stream=mx.Device(type=mx.DeviceType.cpu))
    return mx.diag(x, k=k)


def diagonal(x, offset=0, axis1=0, axis2=1):
    x = convert_to_tensor(x)
    return mx.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)


def diff(a, n=1, axis=-1):
    a = convert_to_tensor(a)
    if n <= 0:
        return a

    if axis < 0:
        axis = a.ndim + axis

    start1 = [0] * a.ndim
    start2 = [0] * a.ndim
    shape = list(a.shape)
    shape[axis] -= 1

    start1[axis] = 1
    out = slice(a, start1, shape) - slice(a, start2, shape)

    return diff(out, n - 1, axis) if n > 1 else out


def digitize(x, bins):
    # TODO: This is quite inefficient but we don't have native support yet
    x = convert_to_tensor(x)
    bins = convert_to_tensor(bins)

    return (x[..., None] >= bins).sum(axis=-1)


def dot(x1, x2):
    x = convert_to_tensor(x1)
    y = convert_to_tensor(x2)

    ndimx = x.ndim
    ndimy = y.ndim

    if ndimx == ndimy == 1:
        return (x[None] @ y[:, None]).reshape(())

    if ndimx == ndimy == 2:
        return x @ y

    if ndimx == 0 or ndimy == 0:
        return x * y

    if ndimy == 1:
        r = x @ y
        return r

    # else if ndimy >= 2:
    x = x.reshape(x.shape[:-1] + (x.shape[-1],) + (1,) * (ndimy - 2))
    r = x @ y
    return r


def empty(shape, dtype=None):
    dtype = to_mlx_dtype(dtype or config.floatx())
    return mx.zeros(shape, dtype=dtype)


def equal(x1, x2):
    x1 = maybe_convert_to_tensor(x1)
    x2 = maybe_convert_to_tensor(x2)
    return mx.equal(x1, x2)


def exp(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = cast(x, config.floatx())
    return mx.exp(x)


def exp2(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = cast(x, config.floatx())
    return mx.power(2, x)


def expand_dims(x, axis):
    x = convert_to_tensor(x)
    return mx.expand_dims(x, axis=axis)


def expm1(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = cast(x, config.floatx())
    # mlx's native GPU `expm1` kernel loses precision away from zero, while
    # `exp(x) - 1` catastrophically cancels near zero. Use whichever is
    # accurate for the input's magnitude.
    return mx.where(mx.abs(x) < 0.1, mx.expm1(x), mx.exp(x) - 1)


def flip(x, axis=None):
    x = convert_to_tensor(x)
    if axis is None:
        indexer = tuple(builtins.slice(None, None, -1) for _ in range(x.ndim))
        return x[indexer]
    if isinstance(axis, int):
        axis = (axis,)
    indexer = [builtins.slice(None)] * x.ndim
    for ax in axis:
        if ax < 0:
            ax = x.ndim + ax
        if not 0 <= ax < x.ndim:
            raise ValueError(
                f"axis {ax} is out of bounds for array of dimension {x.ndim}"
            )
        indexer[ax] = builtins.slice(None, None, -1)

    return x[tuple(indexer)]


def floor(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = cast(x, dtype)
    return mx.floor(x)


def full(shape, fill_value, dtype=None):
    dtype = to_mlx_dtype(dtype)
    fill_value = convert_to_tensor(fill_value, dtype=dtype)
    return mx.full(shape, fill_value)


def full_like(x, fill_value, dtype=None):
    dtype = dtype or x.dtype
    return full(shape=x.shape, fill_value=fill_value, dtype=dtype)


def greater(x1, x2):
    x1 = maybe_convert_to_tensor(x1)
    x2 = maybe_convert_to_tensor(x2)
    return mx.greater(x1, x2)


def greater_equal(x1, x2):
    x1 = maybe_convert_to_tensor(x1)
    x2 = maybe_convert_to_tensor(x2)
    return mx.greater_equal(x1, x2)


def hstack(xs):
    xs = _promote(*xs)
    if xs[0].ndim <= 1:
        xs = [mx.atleast_1d(x) for x in xs]
        return mx.concatenate(xs, axis=0)
    else:
        return mx.concatenate(xs, axis=1)


def identity(n, dtype=None):
    dtype = to_mlx_dtype(dtype or config.floatx())
    return mx.eye(n, dtype=dtype)


def imag(x):
    x = convert_to_tensor(x)
    return mx.imag(x)


def isclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)


def isfinite(x):
    x = convert_to_tensor(x)
    return mx.isfinite(x)


def isinf(x):
    x = convert_to_tensor(x)
    return mx.isinf(x)


def isnan(x):
    x = convert_to_tensor(x)
    return mx.isnan(x)


def less(x1, x2):
    x1 = maybe_convert_to_tensor(x1)
    x2 = maybe_convert_to_tensor(x2)
    return mx.less(x1, x2)


def less_equal(x1, x2):
    x1 = maybe_convert_to_tensor(x1)
    x2 = maybe_convert_to_tensor(x2)
    return mx.less_equal(x1, x2)


def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    if axis != 0:
        raise NotImplementedError(
            "MLX doesn't support linspace with `axis` argument"
            f"Received axis={axis}"
        )

    start = convert_to_tensor(start)
    stop = convert_to_tensor(stop)
    if dtype is None:
        dtypes_to_resolve = [
            getattr(start, "dtype", type(start)),
            getattr(stop, "dtype", type(stop)),
            float,
        ]
        dtype = dtypes.result_type(*dtypes_to_resolve)
    mlx_dtype = to_mlx_dtype(dtype)

    if start.ndim == 0 and stop.ndim == 0:
        result = mx.linspace(
            start, stop, num=num if endpoint else num + 1, dtype=mlx_dtype
        )
    else:
        start = start.astype(mlx_dtype)
        stop = stop.astype(mlx_dtype)
        zero_one = mx.linspace(
            0, 1, num=num if endpoint else num + 1, dtype=mlx_dtype
        )
        zero_one = zero_one.reshape([-1] + [1] * start.ndim)
        result = zero_one * (stop - start)[None] + start[None]

    if not endpoint:
        result = result[:-1]

    if retstep:
        step = (stop - start) / (num - 1 if endpoint else num)
        return result, step

    return result


def log(x):
    x = convert_to_tensor(x)
    return mx.log(x)


def log10(x):
    x = convert_to_tensor(x)
    return mx.log10(x)


def log1p(x):
    x = convert_to_tensor(x)
    return mx.log1p(x)


def log2(x):
    x = convert_to_tensor(x)
    return mx.log2(x)


def logaddexp(x1, x2):
    x1 = maybe_convert_to_tensor(x1)
    x2 = maybe_convert_to_tensor(x2)
    return mx.logaddexp(x1, x2)


def logical_and(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return mx.logical_and(x1, x2)


def logical_not(x):
    x = convert_to_tensor(x)
    return mx.logical_not(x)


def logical_or(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return mx.logical_or(x1, x2)


def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    if axis != 0:
        raise NotImplementedError(
            "MLX logspace does not support an `axis` argument. "
            f"Received axis={axis}"
        )
    points = linspace(start, stop, num, endpoint=endpoint, dtype=dtype)
    return mx.power(base, points)


def maximum(x1, x2):
    x1, x2 = _promote(x1, x2)
    return mx.maximum(x1, x2)


def median(x, axis=None, keepdims=False):
    # mlx's median axis must be an int, a sequence of ints, or None.
    if isinstance(axis, list):
        axis = tuple(axis)
    x = convert_to_tensor(x)
    result_dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_mlx_result_dtype(result_dtype))
    return mx.median(x, axis=axis, keepdims=keepdims)


def meshgrid(*x, indexing="xy"):
    x = [convert_to_tensor(xi) for xi in x]
    return mx.meshgrid(*x, indexing=indexing)


def min(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)
    if 0 in x.shape:
        # mlx cannot reduce a zero size array, so build the result directly.
        # An initial value is only required when a reduced axis is empty and
        # the output is non-empty, matching numpy.
        out_shape = _reduced_shape(x.shape, axis, keepdims)
        needs_initial = (
            _reduces_empty_axis(x.shape, axis) and 0 not in out_shape
        )
        if needs_initial and initial is None:
            raise ValueError("Cannot compute the min of an empty tensor.")
        fill = initial if needs_initial else 0
        return mx.full(out_shape, fill, dtype=x.dtype)

    result = mx.min(x, axis=axis, keepdims=keepdims)
    if initial is not None:
        result = mx.minimum(result, initial)

    return result.astype(x.dtype)


def minimum(x1, x2):
    x1, x2 = _promote(x1, x2)
    return mx.minimum(x1, x2)


def mod(x1, x2):
    x1, x2 = _promote(x1, x2)
    return mx.remainder(x1, x2)


def moveaxis(x, source, destination):
    x = convert_to_tensor(x)

    if not isinstance(source, (list, tuple)):
        source = [source]
    if not isinstance(destination, (list, tuple)):
        destination = [destination]

    source = [axis if axis >= 0 else x.ndim + axis for axis in source]
    destination = [axis if axis >= 0 else x.ndim + axis for axis in destination]

    perm = [i for i in range(x.ndim)]
    for s, d in zip(source, destination):
        perm.remove(s)
        perm.insert(d, s)

    return mx.transpose(x, perm)


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    x = convert_to_tensor(x)
    return mx.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def ndim(x):
    x = convert_to_tensor(x)
    return x.ndim


def nonzero(x):
    # TODO: swap to mlx when nonzero is implemented
    x = convert_to_tensor(x)
    if x.dtype == mx.bfloat16:
        x = x.astype(mx.float32)
    x = np.array(x)
    output = np.nonzero(x)
    return tuple(mx.array(x).astype(mx.int32) for x in output)


def not_equal(x1, x2):
    x1 = maybe_convert_to_tensor(x1)
    x2 = maybe_convert_to_tensor(x2)
    # Filter out None (scalar operands) before the membership test, since
    # `mx.Dtype.__eq__(None)` raises instead of returning False.
    operand_dtypes = [
        d
        for d in (getattr(x1, "dtype", None), getattr(x2, "dtype", None))
        if d is not None
    ]
    if mx.float64 in operand_dtypes:
        # float64 is only supported on the cpu stream.
        with mx.stream(mx.cpu):
            return mx.not_equal(x1, x2)
    return x1 != x2


def ones_like(x, dtype=None):
    x = convert_to_tensor(x)
    dtype = to_mlx_dtype(dtype or x.dtype)
    return mx.ones(x.shape, dtype=dtype)


def outer(x1, x2):
    x1, x2 = _promote(x1, x2)
    x1 = x1.reshape(-1)
    x2 = x2.reshape(-1)

    return x1[:, None] * x2[None, :]


def pad(x, pad_width, mode="constant", constant_values=None):
    if isinstance(pad_width, mx.array):
        pad_width = pad_width.tolist()
    x = convert_to_tensor(x)

    if constant_values is not None:
        if mode != "constant":
            raise ValueError(
                "Argument `constant_values` can only be "
                "provided when `mode == 'constant'`. "
                f"Received: mode={mode}"
            )
    elif mode == "constant":
        constant_values = 0

    if mode == "constant":
        return mx.pad(x, pad_width, constant_values=constant_values)

    if mode in ["symmetric", "reflect"]:
        result = x
        for axis, (pad_before, pad_after) in enumerate(pad_width):
            if pad_before == 0 and pad_after == 0:
                continue

            size = x.shape[axis]
            if mode == "symmetric":
                before_idx = mx.arange(pad_before - 1, -1, -1) % size
                after_idx = mx.arange(size - 1, size - pad_after - 1, -1) % size
            else:  # reflect
                before_idx = mx.arange(pad_before - 1, -1, -1) % (size - 1)
                after_idx = mx.arange(size - 2, size - pad_after - 2, -1) % (
                    size - 1
                )

            indices = mx.concatenate([before_idx, mx.arange(size), after_idx])
            result = mx.take(result, indices, axis=axis)

        return result

    raise ValueError(f"Unsupported padding mode: {mode}")


def prod(x, axis=None, keepdims=False, dtype=None):
    x = convert_to_tensor(x)
    if dtype is None:
        dtype = _widen_reduce_int_dtype(dtypes.result_type(x.dtype))
    mlx_dtype = to_mlx_dtype(dtype)
    output = mx.prod(x, axis=axis, keepdims=keepdims)
    return output.astype(mlx_dtype)


def quantile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    q = convert_to_tensor(q)

    # TODO: swap to mlx when quantile is supported
    ori_dtype = standardize_dtype(x.dtype)
    # np.quantile doesn't support bool
    if ori_dtype == "bool":
        default_dtype = to_mlx_dtype(config.floatx())
        x = x.astype(default_dtype)
    if ori_dtype == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    mlx_dtype = to_mlx_dtype(dtype)

    # problem casting mlx bfloat16 array to numpy
    if ori_dtype == "bfloat16":
        default_dtype = to_mlx_dtype(config.floatx())
        x = x.astype(default_dtype)
    x = np.array(x)
    q = np.array(q)
    result = np.quantile(x, q, axis=axis, method=method, keepdims=keepdims)
    return mx.array(result).astype(mlx_dtype)


def ravel(x):
    x = convert_to_tensor(x)
    return x.reshape(-1)


def real(x):
    x = convert_to_tensor(x)
    return mx.real(x)


def reciprocal(x):
    x = convert_to_tensor(x)
    return mx.reciprocal(x)


def repeat(x, repeats, axis=None):
    x = convert_to_tensor(x)
    repeats = convert_to_tensor(repeats)

    if repeats.size == 1:
        return mx.repeat(x, repeats, axis=axis)

    if axis is None:
        x = mx.reshape(x, (-1,))
        axis = 0

    if repeats.size != x.shape[axis]:
        raise ValueError(
            "repeats must have same length as axis: "
            f"got {repeats.size} vs {x.shape[axis]}"
        )

    indices = mx.concatenate([mx.full(r, i) for i, r in enumerate(repeats)])
    return mx.take(x, indices, axis=axis)


def reshape(x, newshape):
    if not isinstance(newshape, (list, tuple)):
        newshape = (newshape,)
    x = convert_to_tensor(x)
    return mx.reshape(x, newshape)


def roll(x, shift, axis=None):
    x = convert_to_tensor(x)
    return mx.roll(x, shift, axis=axis)


def sign(x):
    x = convert_to_tensor(x)
    return mx.sign(x)


def sin(x):
    x = convert_to_tensor(x)
    return mx.sin(x)


def sinh(x):
    x = convert_to_tensor(x)
    return mx.sinh(x)


def size(x):
    x = convert_to_tensor(x)
    return x.size


def sort(x, axis=-1):
    x = convert_to_tensor(x)
    return mx.sort(x, axis=axis)


def split(x, indices_or_sections, axis=0):
    x = convert_to_tensor(x)
    return mx.split(x, indices_or_sections, axis=axis)


def stack(x, axis=0):
    xs = _promote(*x)
    return mx.stack(xs, axis=axis)


def std(x, axis=None, keepdims=False):
    # Reuse var so std inherits the float32 compute precision safeguard.
    return mx.sqrt(var(x, axis=axis, keepdims=keepdims))


def swapaxes(x, axis1, axis2):
    x = convert_to_tensor(x)
    return mx.swapaxes(x, axis1=axis1, axis2=axis2)


def take(x, indices, axis=None):
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices)
    return mx.take(x, indices, axis=axis)


def take_along_axis(x, indices, axis=None):
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices)
    return mx.take_along_axis(x, indices, axis=axis)


def tan(x):
    x = convert_to_tensor(x)
    return mx.tan(x)


def tanh(x):
    x = convert_to_tensor(x)
    return mx.tanh(x)


def tensordot(x1, x2, axes=2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    if isinstance(axes, int):
        return mx.tensordot(x1, x2, axes)
    elif isinstance(axes, (list, tuple)):
        if not isinstance(axes[0], (list, tuple)):
            axes = [[axes[0]], [axes[1]]]
        return mx.tensordot(x1, x2, axes)

    raise ValueError(
        f"`axes` must be an integer or sequence Received: axes={axes}"
    )


def round(x, decimals=0):
    x = convert_to_tensor(x)
    return mx.round(x, decimals=decimals)


def tile(x, repeats):
    x = convert_to_tensor(x)
    return mx.tile(x, repeats)


def trace(x, offset=0, axis1=0, axis2=1):
    x = convert_to_tensor(x)
    dtype = _widen_reduce_int_dtype(standardize_dtype(x.dtype))
    mlx_dtype = to_mlx_dtype(dtype)
    return diagonal(x, offset, axis1, axis2).sum(-1).astype(mlx_dtype)


def tri(N, M=None, k=0, dtype=None):
    dtype = to_mlx_dtype(dtype or config.floatx())
    M = N if M is None else M
    x = mx.ones((N, M), dtype=dtype)

    return tril(x, k=k)


def tril(x, k=0):
    x = convert_to_tensor(x)

    idx_y = mx.arange(x.shape[-2])
    idx_x = mx.arange(x.shape[-1])
    mask = idx_y[:, None] >= idx_x[None] - k

    return mx.where(mask, x, mx.zeros_like(x))


def triu(x, k=0):
    x = convert_to_tensor(x)

    idx_y = mx.arange(x.shape[-2])
    idx_x = mx.arange(x.shape[-1])
    mask = idx_y[:, None] <= idx_x[None] - k

    return mx.where(mask, x, mx.zeros_like(x))


def trunc(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or "bool" == dtype:
        return x
    return mx.where(x < 0, mx.ceil(x), mx.floor(x))


def vdot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    result_dtype = to_mlx_dtype(dtype)
    x1_conj = mx.conj(mx.reshape(x1, (x1.size,)))
    result = mx.sum(x1_conj * mx.reshape(x2, (x2.size,)))
    return result.astype(result_dtype)


def inner(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.inner(x1, x2)


def vstack(xs):
    xs = _promote(*xs)
    if xs[0].ndim <= 1:
        xs = [mx.atleast_1d(x)[None] for x in xs]
    return mx.concatenate(xs, axis=0)


def where(condition, x1=None, x2=None):
    if x1 is None and x2 is None:
        return nonzero(condition)
    elif x1 is None or x2 is None:
        raise ValueError("`x1` and `x2` either both should be `None`")

    # handling of python dtypes similar to numpy's backend
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    condition = convert_to_tensor(condition)
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return mx.where(condition, x1, x2)


def divide(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    return mx.divide(x1, x2)


def divide_no_nan(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.where(x2 == 0, 0, mx.divide(x1, x2))


def true_divide(x1, x2):
    x1 = maybe_convert_to_tensor(x1)
    x2 = maybe_convert_to_tensor(x2)
    return divide(x1, x2)


def power(x1, x2):
    x1, x2 = _promote(x1, x2)
    return mx.power(x1, x2)


def negative(x):
    x = convert_to_tensor(x)
    return mx.negative(x)


def square(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "bool":
        x = x.astype(mx.int32)
    return mx.square(x)


def sqrt(x):
    x = convert_to_tensor(x)
    return mx.sqrt(x)


def squeeze(x, axis=None):
    x = convert_to_tensor(x)
    return mx.squeeze(x, axis=axis)


def transpose(x, axes=None):
    x = convert_to_tensor(x)
    return mx.transpose(x, axes)


def var(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    # mlx computes the variance in the input dtype, which overflows in low
    # precision such as float16. Compute in float32 and cast back.
    compute_dtype = dtypes.result_type(x.dtype, "float32")
    result_dtype = dtypes.result_type(x.dtype, float)
    result = mx.var(
        x.astype(to_mlx_dtype(compute_dtype)), axis=axis, keepdims=keepdims
    )
    return result.astype(_mlx_result_dtype(result_dtype))


def sum(x, axis=None, keepdims=False):
    if isinstance(x, (list, tuple)):
        x = stack(x)
    x = convert_to_tensor(x)
    return mx.sum(x, axis=axis, keepdims=keepdims)


def eye(N, M=None, k=0, dtype=None):
    # mlx silently converts a float-valued 0-d array to int, numpy raises.
    for arg in (N, M):
        arg_dtype = getattr(arg, "dtype", None)
        if arg_dtype is not None and "float" in standardize_dtype(arg_dtype):
            raise TypeError(
                f"Argument to `eye` must be an integer, received dtype "
                f"{standardize_dtype(arg_dtype)}."
            )
    dtype = to_mlx_dtype(dtype or config.floatx())
    M = N if M is None else M
    k = 0 if k is None else k
    return mx.eye(N, M, k, dtype=dtype)


def floor_divide(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)), getattr(x2, "dtype", type(x2))
    )
    x1 = convert_to_tensor(x1, dtype=dtype)
    x2 = convert_to_tensor(x2, dtype=dtype)
    if "int" in standardize_dtype(dtype):
        # mlx `//` truncates toward zero for integers, so correct the quotient
        # down by one when the remainder forces a floor toward -inf.
        q = x1 // x2
        r = x1 - q * x2
        return mx.where((r != 0) & ((r < 0) != (x2 < 0)), q - 1, q)
    return mx.floor(x1 / x2)


def logical_xor(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return x1.astype(mx.bool_) - x2.astype(mx.bool_)


def maybe_convert_to_tensor(x):
    if isinstance(x, (int, float, bool)):
        return x
    return convert_to_tensor(x)


def correlate(x1, x2, mode="valid"):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype == "int64":
        dtype = "float64"
    elif dtype not in ["bfloat16", "float16", "float64"]:
        dtype = "float32"
    mlx_dtype = to_mlx_dtype(dtype)

    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError("correlate() only supports 1-dimensional inputs")
    if len(x1) == 0 or len(x2) == 0:
        raise ValueError(
            f"inputs cannot be empty, got shapes {x1.shape} and {x2.shape}"
        )

    x2 = mx.conj(x2)
    if len(x1) < len(x2):
        x1, x2 = x2, x1
        reverse_output = True
    else:
        reverse_output = False

    if mode == "valid":
        pad_width = [(0, 0)]
    elif mode == "same":
        pad_size = x2.shape[0] // 2
        pad_width = [(pad_size, x2.shape[0] - pad_size - 1)]
    elif mode == "full":
        pad_size = x2.shape[0] - 1
        pad_width = [(pad_size, pad_size)]
    else:
        raise ValueError("mode must be one of ['full', 'same', 'valid']")

    if mode != "valid":
        x1 = mx.pad(x1, pad_width)

    output_size = len(x1) - len(x2) + 1
    window_indices = mx.arange(output_size)[:, None] + mx.arange(len(x2))[None, :]
    windows = x1[window_indices]
    result = mx.sum(windows * x2, axis=1).astype(mlx_dtype)

    return result[::-1] if reverse_output else result


def select(condlist, choicelist, default=0):
    x = convert_to_tensor(default)

    for condition, choice in zip(reversed(condlist), reversed(choicelist)):
        x = mx.where(condition, choice, x)

    return x


def slogdet(x):
    # TODO: Swap to mlx.linalg.slogdet when supported (or with determinant)
    x = convert_to_tensor(x)
    # numpy cannot consume bfloat16 buffers, so compute in float32.
    if x.dtype == mx.bfloat16:
        x = x.astype(mx.float32)
    x = np.array(x)
    output = np.linalg.slogdet(x)
    return (mx.array(output[0]), mx.array(output[1]))


def vectorize(pyfunc, *, excluded=None, signature=None):
    return vectorize_impl(
        pyfunc, mx.vmap, excluded=excluded, signature=signature
    )


def histogram_bin_edges(a, bins=10, range=None):
    # Ref: jax.numpy.histogram
    # infer range if None
    if range is None:
        range = (mx.min(a).item(), mx.max(a).item())

    if range[0] == range[1]:
        range = (range[0] - 0.5, range[1] + 0.5)

    bin_edges = mx.linspace(range[0], range[1], bins + 1, dtype=mx.float32)
    # due to the way mlx currently handles linspace
    # with fp32 precision it is not always right edge inclusive
    # manually set the right edge for now
    bin_edges[-1] = range[-1]
    return bin_edges


def histogram(x, bins=10, range=None):
    # Ref: jax.numpy.histogram
    x = convert_to_tensor(x)
    if range is not None:
        if not isinstance(range, tuple) or len(range) != 2:
            raise ValueError(
                "Invalid value for argument `range`. Only `None` or "
                "a tuple of the lower and upper range of bins is supported. "
                f"Received: range={range}"
            )

    bin_edges = histogram_bin_edges(x, bins, range)

    bin_idx = searchsorted(bin_edges, x, side="right")
    bin_idx = mx.where(x == bin_edges[-1], len(bin_edges) - 1, bin_idx)

    counts = mx.zeros(len(bin_edges))
    counts = counts.at[bin_idx].add(mx.ones_like(x))

    return counts[1:], bin_edges


def unravel_index(indices, shape):
    x = convert_to_tensor(indices)
    input_dtype = x.dtype

    if None in shape:
        raise ValueError(
            f"`shape` argument cannot contain `None`. Received: shape={shape}"
        )

    if x.ndim == 1:
        coords = []
        for dim in reversed(shape):
            coords.append((x % dim).astype(input_dtype))
            x = x // dim
        return tuple(reversed(coords))

    x_shape = x.shape
    coords = []
    for dim in shape:
        coords.append(mx.reshape((x % dim).astype(input_dtype), x_shape))
        x = x // dim

    return tuple(reversed(coords))


def searchsorted_binary(a, b, side="left"):
    original_shape = b.shape
    b_flat = b.reshape(-1)

    size = a.shape[0]
    steps = math.ceil(math.log2(size))
    indices = mx.full(b_flat.shape, vals=size // 2, dtype=mx.int32)

    if side == "left":
        comparison = lambda x, y: x <= y
    else:
        comparison = lambda x, y: x < y

    upper = size
    lower = 0
    for _ in range(steps):
        comp = comparison(b_flat, a[indices])
        new_indices = mx.where(
            comp, (lower + indices) // 2, (indices + upper) // 2
        )
        lower = mx.where(comp, lower, indices)
        upper = mx.where(comp, indices, upper)
        indices = new_indices

    result = mx.where(comparison(b_flat, a[indices]), indices, indices + 1)
    return result.reshape(original_shape)


def searchsorted_linear(a, b, side="left"):
    original_shape = b.shape
    b_flat = b.reshape(-1)
    b_flat_broadcast = b_flat.reshape(-1, 1)
    if side == "left":
        result = (a[None, :] < b_flat_broadcast).sum(axis=1)
    else:
        result = (a[None, :] <= b_flat_broadcast).sum(axis=1)

    return result.reshape(original_shape)


def searchsorted(sorted_sequence, values, side="left"):
    if side not in ("left", "right"):
        raise ValueError(f"Invalid side `{side}`, must be `left` or `right`.")
    sorted_sequence = convert_to_tensor(sorted_sequence)
    values = convert_to_tensor(values)
    if sorted_sequence.ndim != 1:
        raise ValueError(
            "Invalid sorted_sequence, should be 1-dimensional. "
            f"Received sorted_sequence.shape={sorted_sequence.shape}"
        )
    if values.ndim == 0:
        raise ValueError(
            "Invalid values, should be N-dimensional. Received "
            f"scalar array values.shape={values.shape}"
        )

    sorted_size = sorted_sequence.size
    search_size = values.size

    # TODO: swap to mlx implementation if exists in the future
    # current implementation and search choice based on discussion:
    # https://github.com/ml-explore/mlx/issues/1255
    use_linear = sorted_size <= 1024 or (
        sorted_size <= 16384 and search_size <= 256
    )

    if use_linear:
        return searchsorted_linear(sorted_sequence, values, side=side)
    else:
        return searchsorted_binary(sorted_sequence, values, side=side)


def diagflat(x, k=0):
    x = convert_to_tensor(x)

    # GPU scatter does not yet support int64 or complex64
    # for the input or updates.
    stream = mx.cpu if x.dtype in [mx.int64, mx.complex64] else None

    return mx.diag(mx.reshape(x, [-1]), k, stream=stream)


def rot90(array, k=1, axes=(0, 1)):
    array = convert_to_tensor(array)

    if array.ndim < 2:
        raise ValueError(
            f"Input array must have at least 2 dimensions. "
            f"Received: array.ndim={array.ndim}"
        )
    if len(axes) != 2 or axes[0] == axes[1]:
        raise ValueError(
            f"Invalid axes: {axes}. Axes must be a tuple of "
            "two different dimensions."
        )

    array_axes = list(range(array.ndim))
    # Swap axes
    array_axes[axes[0]], array_axes[axes[1]] = (
        array_axes[axes[1]],
        array_axes[axes[0]],
    )

    if k < 0:
        axes = (axes[1], axes[0])
        k *= -1

    k = k % 4

    if k > 0:
        slices = [builtins.slice(None) for _ in range(array.ndim)]
        if k == 2:
            # 180 deg rotation => reverse elements along both axes
            slices[axes[0]] = builtins.slice(None, None, -1)
            slices[axes[1]] = builtins.slice(None, None, -1)
        else:
            # 90 or 270 deg rotation => transpose and reverse along one axis
            array = mx.transpose(array, axes=array_axes)
            if k == 1:
                slices[axes[0]] = builtins.slice(None, None, -1)
            else:
                slices[axes[1]] = builtins.slice(None, None, -1)

        array = array[tuple(slices)]

    return array


def signbit(x):
    x = convert_to_tensor(x)

    if x.dtype in (
        mx.float16,
        mx.float32,
        mx.float64,
        mx.bfloat16,
        mx.complex64,
    ):
        if x.dtype == mx.complex64:
            # check sign of real part for complex numbers
            real_part = mx.real(x)
            return signbit(real_part)
        zeros = x == 0
        # this works because in mlx 1/0=inf and 1/-0=-inf
        neg_zeros = (1 / x == mx.array(float("-inf"))) & zeros
        return mx.where(zeros, neg_zeros, x < 0)
    elif x.dtype in (mx.uint8, mx.uint16, mx.uint32, mx.uint64):
        # unsigned integers never negative
        return mx.zeros_like(x).astype(mx.bool_)
    elif x.dtype in (mx.int8, mx.int16, mx.int32, mx.int64):
        # for integers, simple negative check
        return x < 0
    elif x.dtype == mx.bool_:
        # for boolean array, return false
        return mx.zeros_like(x).astype(mx.bool_)
    else:
        raise ValueError(f"Unsupported dtype in `signbit`: {x.dtype}")


def bartlett(x):
    # ref: jax.numpy.bartlett
    dtype = to_mlx_dtype(config.floatx())
    if x <= 1:
        return mx.ones(x, dtype=dtype)

    # note: mx.arange cannot take mx.array as input
    n = mx.arange(x, dtype=dtype)
    return 1 - mx.abs(2 * n + 1 - x) / (x - 1)


def blackman(x):
    # ref: jax.numpy.blackman
    dtype = to_mlx_dtype(config.floatx())
    if x <= 1:
        return mx.ones(x, dtype=dtype)

    # note: mx.arange cannot take mx.array as input
    n = mx.arange(x, dtype=dtype)
    return (
        0.42
        - 0.5 * mx.cos(2 * mx.pi * n / (x - 1))
        + 0.08 * mx.cos(4 * mx.pi * n / (x - 1))
    )


def angle(x):
    x = convert_to_tensor(x)
    re = real(x)
    im = imag(x)

    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    re = cast(re, dtype)
    im = cast(im, dtype)
    return mx.arctan2(im, re)


def _to_numpy(*xs):
    # numpy cannot read mlx bfloat16 arrays, so upcast to float32 first. This
    # is only used by ops that fall back to numpy for unsupported primitives.
    out = []
    for x in xs:
        if isinstance(x, mx.array) and x.dtype == mx.bfloat16:
            x = x.astype(mx.float32)
        out.append(np.asarray(x))
    return out[0] if len(out) == 1 else out


def _np_axis(axis):
    if isinstance(axis, list):
        return tuple(axis)
    return axis


def _mlx_result_dtype(dtype):
    # mlx has no float64 on the default (GPU) device, so the backend never
    # materializes float64. Mirror the rest of the backend by downcasting any
    # float64 result to the configured floatx. 64-bit dtypes are excluded from
    # the dtype test matrix, so this only affects host int64/float64 inputs.
    if standardize_dtype(dtype) == "float64":
        return to_mlx_dtype(config.floatx())
    return to_mlx_dtype(dtype)


def _widen_reduce_int_dtype(dtype):
    # Match numpy integer accumulation, where small integer types widen to
    # 32-bit so the reduction does not overflow.
    if dtype in ("bool", "int8", "int16"):
        return "int32"
    if dtype in ("uint8", "uint16"):
        return "uint32"
    return dtype


def allclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.allclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)


def array_split(x, indices_or_sections, axis=0):
    x = convert_to_tensor(x)
    if axis < 0:
        axis += x.ndim
    if isinstance(indices_or_sections, int):
        n = indices_or_sections
        if n == 1:
            return [x]
        size = x.shape[axis]
        each, rem = divmod(size, n)
        split_points = []
        pos = 0
        for i in range(n - 1):
            pos += each + (1 if i < rem else 0)
            split_points.append(pos)
        return mx.split(x, split_points, axis=axis)
    return mx.split(x, indices_or_sections, axis=axis)


def cbrt(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype in ("bool", "int8", "int16", "int32", "uint8", "uint16", "uint32"):
        dtype = config.floatx()
    elif dtype == "int64":
        dtype = "float64"
    return mx.array(np.cbrt(_to_numpy(x))).astype(_mlx_result_dtype(dtype))


def corrcoef(x):
    x = convert_to_tensor(x)
    sd = standardize_dtype(x.dtype)
    if sd in ("int64", "float64"):
        dtype = "float64"
    elif sd in ("bfloat16", "float16"):
        dtype = sd
    else:
        dtype = config.floatx()
    return mx.array(np.corrcoef(_to_numpy(x))).astype(_mlx_result_dtype(dtype))


def deg2rad(x):
    x = convert_to_tensor(x)
    sd = standardize_dtype(x.dtype)
    if sd in ("int64", "float64"):
        dtype = "float64"
    elif sd in ("bfloat16", "float16"):
        dtype = sd
    else:
        dtype = config.floatx()
    mlx_dtype = _mlx_result_dtype(dtype)
    x = x.astype(mlx_dtype)
    return (x * (math.pi / 180.0)).astype(mlx_dtype)


def dsplit(x, indices_or_sections):
    x = convert_to_tensor(x)
    return mx.split(x, indices_or_sections, axis=2)


def _atleast_3d(x):
    if x.ndim == 0:
        return x.reshape(1, 1, 1)
    if x.ndim == 1:
        return x.reshape(1, x.shape[0], 1)
    if x.ndim == 2:
        return x.reshape(x.shape[0], x.shape[1], 1)
    return x


def dstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    dtype = _mlx_result_dtype(
        dtypes.result_type(*[standardize_dtype(x.dtype) for x in xs])
    )
    xs = [_atleast_3d(x).astype(dtype) for x in xs]
    return mx.concatenate(xs, axis=2)


def empty_like(x, dtype=None):
    x = convert_to_tensor(x)
    dtype = _mlx_result_dtype(dtype or x.dtype)
    return mx.zeros(x.shape, dtype=dtype)


def fabs(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        x = x.astype(_mlx_result_dtype(config.floatx()))
    return mx.abs(x)


def fliplr(x):
    x = convert_to_tensor(x)
    return flip(x, axis=1)


def flipud(x):
    x = convert_to_tensor(x)
    return flip(x, axis=0)


def fmax(x1, x2):
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)), getattr(x2, "dtype", type(x2))
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    if "float" in dtype:
        return mx.where(
            mx.isnan(x1), x2, mx.where(mx.isnan(x2), x1, mx.maximum(x1, x2))
        )
    return mx.maximum(x1, x2)


def fmin(x1, x2):
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)), getattr(x2, "dtype", type(x2))
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    if "float" in dtype:
        return mx.where(
            mx.isnan(x1), x2, mx.where(mx.isnan(x2), x1, mx.minimum(x1, x2))
        )
    return mx.minimum(x1, x2)


def fmod(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype == "bool":
        dtype = "int32"
    mlx_dtype = _mlx_result_dtype(dtype)
    x1 = x1.astype(mlx_dtype)
    x2 = x2.astype(mlx_dtype)
    return mx.array(np.fmod(_to_numpy(x1), _to_numpy(x2))).astype(mlx_dtype)


def gcd(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    return mx.array(np.gcd(_to_numpy(x1), _to_numpy(x2))).astype(
        _mlx_result_dtype(dtype)
    )


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    dtype = dtype or config.floatx()
    result = np.geomspace(
        start, stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis
    )
    return mx.array(result).astype(_mlx_result_dtype(standardize_dtype(dtype)))


def hamming(x):
    x = convert_to_tensor(x)
    return mx.array(np.hamming(int(_to_numpy(x)))).astype(
        _mlx_result_dtype(config.floatx())
    )


def hanning(x):
    x = convert_to_tensor(x)
    return mx.array(np.hanning(int(_to_numpy(x)))).astype(
        _mlx_result_dtype(config.floatx())
    )


def heaviside(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype in ("int8", "int16", "int32", "uint8", "uint16", "uint32"):
        dtype = config.floatx()
    elif dtype == "int64":
        dtype = "float64"
    return mx.array(np.heaviside(_to_numpy(x1), _to_numpy(x2))).astype(
        _mlx_result_dtype(dtype)
    )


def hsplit(x, indices_or_sections):
    x = convert_to_tensor(x)
    axis = 0 if x.ndim == 1 else 1
    return mx.split(x, indices_or_sections, axis=axis)


def hypot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype in ("int8", "int16", "int32", "uint8", "uint16", "uint32"):
        dtype = config.floatx()
    elif dtype == "int64":
        dtype = "float64"
    return mx.array(np.hypot(_to_numpy(x1), _to_numpy(x2))).astype(
        _mlx_result_dtype(dtype)
    )


def i0(x):
    x = convert_to_tensor(x)
    sd = standardize_dtype(x.dtype)
    dtype = (
        "float64"
        if sd in ("int64", "float64")
        else dtypes.result_type(x.dtype, float)
    )
    x = x.astype(_mlx_result_dtype(dtype))
    return mx.array(np.i0(_to_numpy(x))).astype(_mlx_result_dtype(dtype))


def isin(x1, x2, assume_unique=False, invert=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.array(
        np.isin(
            _to_numpy(x1),
            _to_numpy(x2),
            assume_unique=assume_unique,
            invert=invert,
        )
    )


def isneginf(x):
    x = convert_to_tensor(x)
    return mx.array(np.isneginf(_to_numpy(x)))


def isposinf(x):
    x = convert_to_tensor(x)
    return mx.array(np.isposinf(_to_numpy(x)))


def isreal(x):
    x = convert_to_tensor(x)
    return mx.array(np.isreal(_to_numpy(x)))


def kaiser(x, beta):
    x = convert_to_tensor(x)
    beta = float(_to_numpy(convert_to_tensor(beta)))
    return mx.array(np.kaiser(int(_to_numpy(x)), beta)).astype(
        _mlx_result_dtype(config.floatx())
    )


def kron(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    return mx.array(np.kron(_to_numpy(x1), _to_numpy(x2))).astype(
        _mlx_result_dtype(dtype)
    )


def lcm(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    return mx.array(np.lcm(_to_numpy(x1), _to_numpy(x2))).astype(
        _mlx_result_dtype(dtype)
    )


def ldexp(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    if standardize_dtype(x2.dtype) not in dtypes.INT_TYPES:
        raise TypeError(
            "ldexp exponent must be an integer type. "
            f"Received: x2 dtype={x2.dtype}"
        )
    return mx.array(np.ldexp(_to_numpy(x1), _to_numpy(x2))).astype(
        _mlx_result_dtype(dtype)
    )


def logaddexp2(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    return mx.array(np.logaddexp2(_to_numpy(x1), _to_numpy(x2))).astype(
        _mlx_result_dtype(dtype)
    )


def nanargmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    xn = _to_numpy(x)
    if not np.issubdtype(xn.dtype, np.floating):
        return argmax(x, axis=axis, keepdims=keepdims)
    nan_mask = np.isnan(xn)
    result = np.where(
        np.all(nan_mask, axis=axis, keepdims=keepdims),
        -1,
        np.nanargmax(
            np.where(nan_mask, -np.inf, xn), axis=axis, keepdims=keepdims
        ).astype("int32"),
    )
    return mx.array(result)


def nanargmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    xn = _to_numpy(x)
    if not np.issubdtype(xn.dtype, np.floating):
        return argmin(x, axis=axis, keepdims=keepdims)
    nan_mask = np.isnan(xn)
    result = np.where(
        np.all(nan_mask, axis=axis, keepdims=keepdims),
        -1,
        np.nanargmin(
            np.where(nan_mask, np.inf, xn), axis=axis, keepdims=keepdims
        ).astype("int32"),
    )
    return mx.array(result)


def nancumprod(x, axis=None, dtype=None):
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(dtype or x.dtype)
    if dtype == "bool":
        dtype = "int32"
    result = np.nancumprod(_to_numpy(x), axis=_np_axis(axis), dtype=dtype)
    return mx.array(result).astype(_mlx_result_dtype(dtype))


def nancumsum(x, axis=None, dtype=None):
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(dtype or x.dtype)
    if dtype == "bool":
        dtype = "int32"
    result = np.nancumsum(_to_numpy(x), axis=_np_axis(axis), dtype=dtype)
    return mx.array(result).astype(_mlx_result_dtype(dtype))


def nanmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    result = np.nanmax(_to_numpy(x), axis=_np_axis(axis), keepdims=keepdims)
    return mx.array(result).astype(x.dtype)


def nanmean(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(standardize_dtype(x.dtype), float)
    result = np.nanmean(_to_numpy(x), axis=_np_axis(axis), keepdims=keepdims)
    return mx.array(result).astype(_mlx_result_dtype(dtype))


def nanmedian(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(standardize_dtype(x.dtype), float)
    result = np.nanmedian(_to_numpy(x), axis=_np_axis(axis), keepdims=keepdims)
    return mx.array(result).astype(_mlx_result_dtype(dtype))


def nanmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    result = np.nanmin(_to_numpy(x), axis=_np_axis(axis), keepdims=keepdims)
    return mx.array(result).astype(x.dtype)


def nanpercentile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    q = convert_to_tensor(q)
    if standardize_dtype(x.dtype) == "bool":
        x = x.astype(_mlx_result_dtype(config.floatx()))
    if "float" in standardize_dtype(x.dtype):
        dtype = standardize_dtype(x.dtype)
    else:
        dtype = config.floatx()
    result = np.nanpercentile(
        _to_numpy(x),
        _to_numpy(q),
        axis=_np_axis(axis),
        method=method,
        keepdims=keepdims,
    )
    return mx.array(result).astype(_mlx_result_dtype(dtype))


def nanprod(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = _widen_reduce_int_dtype(dtypes.result_type(x.dtype))
    result = np.nanprod(
        _to_numpy(x), axis=_np_axis(axis), keepdims=keepdims, dtype=dtype
    )
    return mx.array(result).astype(_mlx_result_dtype(dtype))


def nanquantile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    q = convert_to_tensor(q)
    if standardize_dtype(x.dtype) == "bool":
        x = x.astype(_mlx_result_dtype(config.floatx()))
    dtype = dtypes.result_type(x.dtype, float)
    result = np.nanquantile(
        _to_numpy(x),
        _to_numpy(q),
        axis=_np_axis(axis),
        method=method,
        keepdims=keepdims,
    )
    return mx.array(result).astype(_mlx_result_dtype(dtype))


def nanstd(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    compute_dtype = dtypes.result_type(x.dtype, "float32")
    result_dtype = dtypes.result_type(x.dtype, float)
    result = np.nanstd(
        _to_numpy(x),
        axis=_np_axis(axis),
        keepdims=keepdims,
        dtype=compute_dtype,
    )
    return mx.array(result).astype(_mlx_result_dtype(result_dtype))


def nansum(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = _widen_reduce_int_dtype(standardize_dtype(x.dtype))
    result = np.nansum(_to_numpy(x), axis=_np_axis(axis), keepdims=keepdims)
    return mx.array(result).astype(_mlx_result_dtype(dtype))


def nanvar(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    compute_dtype = dtypes.result_type(x.dtype, "float32")
    result_dtype = dtypes.result_type(x.dtype, float)
    result = np.nanvar(
        _to_numpy(x),
        axis=_np_axis(axis),
        keepdims=keepdims,
        dtype=compute_dtype,
    )
    return mx.array(result).astype(_mlx_result_dtype(result_dtype))


def nextafter(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    return mx.array(np.nextafter(_to_numpy(x1), _to_numpy(x2))).astype(
        _mlx_result_dtype(dtype)
    )


def percentile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    q = convert_to_tensor(q)
    if standardize_dtype(x.dtype) == "bool":
        x = x.astype(_mlx_result_dtype(config.floatx()))
    dtype = dtypes.result_type(x.dtype, float)
    result = np.percentile(
        _to_numpy(x),
        _to_numpy(q),
        axis=_np_axis(axis),
        method=method,
        keepdims=keepdims,
    )
    return mx.array(result).astype(_mlx_result_dtype(dtype))


def ptp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    result = np.ptp(_to_numpy(x), axis=_np_axis(axis), keepdims=keepdims)
    return mx.array(result).astype(x.dtype)


def rad2deg(x):
    x = convert_to_tensor(x)
    sd = standardize_dtype(x.dtype)
    if sd in ("int64", "float64"):
        dtype = "float64"
    elif sd in ("bfloat16", "float16"):
        dtype = sd
    else:
        dtype = config.floatx()
    mlx_dtype = _mlx_result_dtype(dtype)
    x = x.astype(mlx_dtype)
    return (x * (180.0 / math.pi)).astype(mlx_dtype)


def sinc(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_mlx_result_dtype(dtype))
    return mx.array(np.sinc(_to_numpy(x))).astype(_mlx_result_dtype(dtype))


def trapezoid(y, x=None, dx=1.0, axis=-1):
    y = convert_to_tensor(y)
    result_dtype = dtypes.result_type(y.dtype, float)
    yn = _to_numpy(y)
    xn = _to_numpy(convert_to_tensor(x)) if x is not None else None
    result = np.trapezoid(yn, xn, dx=dx, axis=axis)
    return mx.array(result).astype(_mlx_result_dtype(result_dtype))


def unique(
    x,
    sorted=True,
    return_index=False,
    return_inverse=False,
    return_counts=False,
    axis=None,
    size=None,
    fill_value=None,
):
    x = convert_to_tensor(x)
    xn = _to_numpy(x)
    # np.unique always sorts in versions < 2.3.0. We accept the `sorted`
    # argument for API consistency but do not forward it to np.unique to
    # avoid a TypeError on older numpy.
    output = np.unique(
        xn,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
        equal_nan=False,
    )

    if not (return_index or return_inverse or return_counts):
        output = [output]
    else:
        output = list(output)

    values = output[0]

    if size is not None:
        dim = 0 if axis is None else (axis % x.ndim)
        values_count = values.shape[dim]
        if values_count > size:
            indices = [builtins.slice(None)] * values.ndim
            indices[dim] = builtins.slice(0, size)
            values = values[tuple(indices)]
            if return_counts:
                output[-1] = output[-1][indices[dim]]
            if return_index:
                output[1] = output[1][indices[dim]]
        elif values_count < size:
            pad_width = [(0, 0)] * values.ndim
            pad_width[dim] = (0, size - values_count)
            fill = 0 if fill_value is None else fill_value
            values = np.pad(values, pad_width, constant_values=fill)
            if return_counts:
                output[-1] = np.pad(
                    output[-1], pad_width[dim], constant_values=0
                )
            if return_index:
                output[1] = np.pad(output[1], pad_width[dim], constant_values=1)

    output[0] = values
    output = [mx.array(o) for o in output]
    return output[0] if len(output) == 1 else tuple(output)


def vander(x, N=None, increasing=False):
    x = convert_to_tensor(x)
    result_dtype = dtypes.result_type(x.dtype)
    compute_dtype = dtypes.result_type(x.dtype, config.floatx())
    x = x.astype(_mlx_result_dtype(compute_dtype))
    result = np.vander(_to_numpy(x), N=N, increasing=increasing)
    return mx.array(result).astype(_mlx_result_dtype(result_dtype))


def view(x, dtype=None):
    x = convert_to_tensor(x)
    xn = _to_numpy(x)
    if dtype is None:
        return mx.array(xn)
    target = standardize_dtype(dtype)
    return mx.array(xn.view(np.dtype(target)))


def vsplit(x, indices_or_sections):
    x = convert_to_tensor(x)
    return mx.split(x, indices_or_sections, axis=0)
