import builtins
from copy import copy as builtin_copy

import jax.numpy as jnp
import mlx.core as mx

from keras.src.backend import config
from keras.src.backend import result_type
from keras.src.backend import standardize_dtype
from keras.src.backend.mlx.core import cast
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.backend.mlx.core import convert_to_tensors
from keras.src.backend.mlx.core import slice
from keras.src.backend.mlx.core import to_mlx_dtype


def add(x1, x2):
    x1 = maybe_convert_to_tensor(x1)
    x2 = maybe_convert_to_tensor(x2)
    return mx.add(x1, x2)


def einsum(subscripts, *operands, **kwargs):
    operands = [convert_to_tensor(x) for x in operands]
    return mx.einsum(subscripts, *operands)


def subtract(x1, x2):
    x1 = maybe_convert_to_tensor(x1)
    x2 = maybe_convert_to_tensor(x2)
    return mx.subtract(x1, x2)


def matmul(x1, x2):
    x1, x2 = convert_to_tensors(x1, x2)
    return mx.matmul(x1, x2)


def multiply(x1, x2):
    x1 = maybe_convert_to_tensor(x1)
    x2 = maybe_convert_to_tensor(x2)
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


def max(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)
    if 0 in x.shape:
        if initial is None:
            raise ValueError("Cannot compute the max of an empty tensor.")
        elif keepdims:
            return mx.full((1,) * len(x.shape), initial)
        else:
            return mx.array(initial)

    result = mx.max(x, axis=axis, keepdims=keepdims)
    if initial is not None:
        result = mx.maximum(result, initial)

    return result


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
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.concatenate([x1, x2], axis=axis)


def arange(start, stop=None, step=1, dtype=None):
    if dtype is None:
        dtypes_to_resolve = [
            getattr(start, "dtype", type(start)),
            getattr(step, "dtype", type(step)),
        ]
        if stop is not None:
            dtypes_to_resolve.append(getattr(stop, "dtype", type(stop)))
        dtype = result_type(*dtypes_to_resolve)
    dtype = to_mlx_dtype(dtype)
    if stop is None:
        stop = start
        start = 0
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
    return mx.argmax(x, axis=axis, keepdims=keepdims)


def argmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return mx.argmin(x, axis=axis, keepdims=keepdims)


def argsort(x, axis=-1):
    x = convert_to_tensor(x)
    if x.ndim == 0:
        return mx.argsort(x, axis=None)
    return mx.argsort(x, axis=axis)


def argpartition(x, kth, axis=-1):
    x = convert_to_tensor(x)
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


def _bincount_1d(x, weights=None, minlength=0):
    length = builtins.max(builtins.max(x) + 1, minlength or 0)
    counts = mx.zeros(length)
    w = weights if weights is not None else mx.ones_like(x)
    return counts.at[x].add(w)


def bincount(x, weights=None, minlength=0, sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with mlx backend")

    x = convert_to_tensor(x)

    if len(x.shape) == 2:
        batch_size = x.shape[0]
        results = []
        for i in range(batch_size):
            w = None if weights is None else weights[i]
            results.append(_bincount_1d(x[i], w, minlength))
        return mx.stack(results)

    return _bincount_1d(x, weights, minlength)


def broadcast_to(x, shape):
    x = convert_to_tensor(x)
    return mx.broadcast_to(x, shape)


def ceil(x):
    x = convert_to_tensor(x)
    return mx.ceil(x)


def clip(x, x_min, x_max):
    x, x_min, x_max = convert_to_tensors(x, x_min, x_max)
    return mx.maximum(x_min, mx.minimum(x, x_max))


def concatenate(xs, axis=0):
    xs = convert_to_tensors(*xs)
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
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

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
    if dtype is not None:
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
    # TODO: This is quite inefficient but we don't have natice support yet
    x = convert_to_tensor(x)
    bins = convert_to_tensor(bins)

    return (x[..., None] >= bins).sum(axis=-1)


def dot(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)

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


def expand_dims(x, axis):
    x = convert_to_tensor(x)
    return mx.expand_dims(x, axis=axis)


def expm1(x):
    # TODO: Add the numerically stable version
    x = convert_to_tensor(x)
    return mx.exp(x) - 1


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
    xs = [convert_to_tensor(x) for x in xs]
    if xs[0].ndim == 1:
        return mx.concatenate(xs, axis=0)
    else:
        return mx.concatenate(xs, axis=1)


def identity(n, dtype=None):
    dtype = to_mlx_dtype(dtype or config.floatx())

    zeros = mx.zeros((n, n), dtype=dtype)
    idx = mx.arange(n)
    zeros[idx, idx] = 1

    return zeros


def imag(x):
    x = convert_to_tensor(x)
    return mx.imag(x)


def isclose(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.isclose(x1, x2)


def isfinite(x):
    x = convert_to_tensor(x)
    return True - (isinf(x) + isnan(x))


def isinf(x):
    x = convert_to_tensor(x)
    return mx.abs(x) == float("inf")


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
    if dtype is not None:
        dtype = to_mlx_dtype(dtype)

    if start.ndim == 0 and stop.ndim == 0:
        result = mx.linspace(
            start, stop, num=num if endpoint else num + 1, dtype=dtype
        )
    else:
        zero_one = mx.linspace(
            0, 1, num=num if endpoint else num + 1, dtype=dtype
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
    points = linspace(start, stop, num, endpoint=endpoint)
    return mx.power(base, points)


def maximum(x1, x2):
    x1 = maybe_convert_to_tensor(x1)
    x2 = maybe_convert_to_tensor(x2)
    return mx.maximum(x1, x2)


def median(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    axis_arg = axis
    x_dim = x.ndim

    if axis is None:
        x = x.flatten()
        axis = (0,)
    elif isinstance(axis, int):
        axis = (axis,)

    axis = tuple(sorted(ax if ax >= 0 else ax + x.ndim for ax in axis))

    transposed_axes = [i for i in range(x.ndim) if i not in axis] + list(axis)
    x = x.transpose(*transposed_axes)

    shape_without_axes = tuple(x.shape[i] for i in range(x.ndim - len(axis)))
    x = x.reshape(shape_without_axes + (-1,))

    x_sorted = mx.sort(x, axis=-1)
    mid_index = x_sorted.shape[-1] // 2
    if x_sorted.shape[-1] % 2 == 0:
        lower = mx.take(x_sorted, mx.array([mid_index - 1]), axis=-1)
        upper = mx.take(x_sorted, mx.array([mid_index]), axis=-1)
        medians = (lower + upper) / 2
    else:
        medians = mx.take(x_sorted, mx.array([mid_index]), axis=-1)

    if keepdims:
        final_shape = list(shape_without_axes) + [1] * len(axis)
        medians = medians.reshape(final_shape)
        index_value_pairs = [
            (i, transposed_axes[i]) for i in range(len(transposed_axes))
        ]
        index_value_pairs.sort(key=lambda pair: pair[1])
        sorted_indices = [pair[0] for pair in index_value_pairs]
        medians = medians.transpose(*sorted_indices)
    else:
        medians = medians.squeeze()

    if keepdims and axis_arg is None:
        while medians.ndim < x_dim:
            medians = mx.expand_dims(medians, axis=-1)

    return medians


def meshgrid(*x, indexing="xy"):
    x = [convert_to_tensor(xi) for xi in x]
    return mx.meshgrid(*x, indexing=indexing)


def min(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)
    if 0 in x.shape:
        if initial is None:
            raise ValueError("Cannot compute the min of an empty tensor.")
        elif keepdims:
            return mx.full((1,) * len(x.shape), initial)
        else:
            return mx.array(initial)

    result = mx.min(x, axis=axis, keepdims=keepdims)
    if initial is not None:
        result = mx.minimum(result, initial)

    return result


def minimum(x1, x2):
    x1 = maybe_convert_to_tensor(x1)
    x2 = maybe_convert_to_tensor(x2)
    return mx.minimum(x1, x2)


def mod(x1, x2):
    x1 = maybe_convert_to_tensor(x1)
    x2 = maybe_convert_to_tensor(x2)
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
    x = jnp.array(x)
    output = jnp.nonzero(x)
    return tuple(mx.array(x) for x in output)


def not_equal(x1, x2):
    x1 = maybe_convert_to_tensor(x1)
    x2 = maybe_convert_to_tensor(x2)
    return x1 != x2


def ones_like(x, dtype=None):
    x = convert_to_tensor(x)
    dtype = to_mlx_dtype(dtype or x.dtype)
    return mx.ones(x.shape, dtype=dtype)


def outer(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
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
    if dtype is not None:
        x = cast(x, dtype)
    output = mx.prod(x, axis=axis, keepdims=keepdims)
    return output


def quantile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    q = convert_to_tensor(q)
    if standardize_dtype(x.dtype) == "int64":
        x = cast(x, config.floatx())

    # TODO: swap to mlx when quantile is supported
    x = jnp.array(x)
    q = jnp.array(q)
    result = jnp.quantile(x, q, axis=axis, method=method, keepdims=keepdims)

    # TODO: with jax < 0.4.26 jnp.quantile failed to keepdims when axis is None
    if keepdims is True and axis is None:
        result_ndim = x.ndim + (1 if len(q.shape) > 0 else 0)
        while result.ndim < result_ndim:
            result = jnp.expand_dims(result, axis=-1)

    result = mx.array(result)
    return result


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


def reshape(x, new_shape):
    if not isinstance(new_shape, (list, tuple)):
        new_shape = (new_shape,)
    x = convert_to_tensor(x)
    return mx.reshape(x, new_shape)


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


def stack(xs, axis=0):
    xs = [convert_to_tensor(x) for x in xs]
    return mx.stack(xs, axis=axis)


def std(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return mx.sqrt(mx.var(x, axis=axis, keepdims=keepdims))


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


def trace(x, offset=None, axis1=None, axis2=None):
    x = convert_to_tensor(x)
    return diagonal(x, offset, axis1, axis2).sum(-1)


def tri(N, M=None, k=0, dtype=None):
    dtype = to_mlx_dtype(dtype or config.floatx())
    M = M or N
    x = mx.ones((N, M), dtype=dtype)

    return tril(x, k=k)


def tril(x, k=0):
    x = convert_to_tensor(x)

    idx_y = mx.arange(x.shape[-2])
    idx_x = mx.arange(x.shape[-1])
    mask = idx_y[:, None] >= idx_x[None] - k

    return x * mask


def triu(x, k=0):
    x = convert_to_tensor(x)

    idx_y = mx.arange(x.shape[-2])
    idx_x = mx.arange(x.shape[-1])
    mask = idx_y[:, None] <= idx_x[None] - k

    return x * mask


def vdot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    x1_conj = mx.conj(mx.reshape(x1, (x1.size,)))
    return mx.sum(x1_conj * mx.reshape(x2, (x2.size,)))


def vstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    if xs[0].ndim == 1:
        xs = [x[None] for x in xs]
    return mx.concatenate(xs, axis=0)


def where(condition, x1=None, x2=None):
    condition = convert_to_tensor(condition)

    if x1 is None and x2 is None:
        return nonzero(condition)
    elif x1 is None or x2 is None:
        raise ValueError("`x1` and `x2` either both should be `None`")

    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
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
    x1 = maybe_convert_to_tensor(x1)
    x2 = maybe_convert_to_tensor(x2)
    return mx.power(x1, x2)


def negative(x):
    x = convert_to_tensor(x)
    return mx.negative(x)


def square(x):
    x = convert_to_tensor(x)
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
    return mx.var(x, axis=axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
    if isinstance(x, (list, tuple)):
        x = stack(x)
    x = convert_to_tensor(x)
    return mx.sum(x, axis=axis, keepdims=keepdims)


def eye(N, M=None, k=None, dtype=None):
    dtype = to_mlx_dtype(dtype or config.floatx())
    M = N if M is None else M
    k = 0 if k is None else k
    return mx.eye(N, M, k, dtype=dtype)


def floor_divide(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return mx.floor(x1 // x2)


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
    result = mx.zeros(output_size)

    for i in range(output_size):
        result = result.at[i].add(mx.sum(x1[i : i + len(x2)] * x2))

    return result[::-1] if reverse_output else result


def select(condlist, choicelist, default=0):
    x = convert_to_tensor(default)

    for condition, choice in zip(reversed(condlist), reversed(choicelist)):
        x = mx.where(condition, choice, x)

    return x


def slogdet(x):
    # TODO: Swap to mlx.linalg.slogdet when supported (or with determinant)
    x = convert_to_tensor(x)
    x = jnp.array(x)
    output = jnp.linalg.slogdet(x)
    return (mx.array(output[0]), mx.array(output[1]))


def vectorize(pyfunc, *, excluded=None, signature=None):
    if excluded is not None:
        raise NotImplementedError("excluded parameter not supported yet")

    if signature is None:
        return lambda *args: mx.vmap(pyfunc)(*args)

    def wrapped(*args):
        array_args = [
            mx.array(arg) if not isinstance(arg, mx.array) else arg
            for arg in args
        ]
        if signature == "(d,d)->()":
            return pyfunc(*array_args)
        elif signature == "(d,d)->(d)":
            return pyfunc(*array_args)
        vmapped = mx.vmap(pyfunc, in_axes=0, out_axes=0)
        return vmapped(*array_args)

    return wrapped
