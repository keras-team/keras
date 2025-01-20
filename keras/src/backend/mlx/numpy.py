import mlx.core as mx

from keras.src.backend import config
from keras.src.backend import result_type
from keras.src.backend import standardize_dtype
from keras.src.backend.mlx.core import cast
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.backend.mlx.core import convert_to_tensors
from keras.src.backend.mlx.core import to_mlx_dtype


def add(x1, x2):
    x1 = maybe_convert_to_tensor(x1)
    x2 = maybe_convert_to_tensor(x2)
    return mx.add(x1, x2)


def einsum(subscripts, *operands, **kwargs):
    raise NotImplementedError()


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

    # TODO: decide if we need special low precision handling

    return mx.mean(x, axis=axis, keepdims=keepdims)


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


def bincount(x, weights=None, minlength=0, sparse=False):
    raise NotImplementedError("The MLX backend doesn't support bincount yet")


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
    raise NotImplementedError("The MLX backend doesn't support conjugate yet")


def conj(x):
    x = convert_to_tensor(x)
    return conjugate(x)


def copy(x):
    raise NotImplementedError("The MLX backend doesn't support copy yet")


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
    # TODO: Write it inline if necessary
    raise NotImplementedError(
        "The MLX backend doesn't support cross product yet"
    )


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


def _diagonal_indices(H, W, k):
    if k >= 0:
        N = min(W - k, H)
        idx1 = mx.arange(0, N)
        idx2 = mx.arange(k, k + N)
    elif k < 0:
        k = -k
        N = min(H - k, W)
        idx1 = mx.arange(k, k + N)
        idx2 = mx.arange(0, N)
    return idx1, idx2


def diag(x, k=0):
    x = convert_to_tensor(x)
    if x.dtype in [mx.int64, mx.uint64]:
        return mx.diag(x, k=k, stream=mx.Device(type=mx.DeviceType.cpu))
    return mx.diag(x, k=k)


def diagonal(x, offset=0, axis1=0, axis2=1):
    x = convert_to_tensor(x)
    return mx.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)


def diff(x, n=1, axis=-1):
    x = convert_to_tensor(x)
    ndim = x.ndim
    axis = (ndim + axis) % ndim
    indices = [slice(None) for _ in range(axis)]
    index_a = indices + [slice(None, -1)]
    index_b = indices + [slice(1)]

    y = x
    for i in range(n):
        y = y[index_b] - y[index_a]
    return y


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
        return (x[None] @ y[:, None]).reshape()

    if ndimx == ndimy == 2:
        return x @ y

    if ndimx == 0 or ndimy == 0:
        return x * y

    if ndimy == 1:
        r = x @ y
        return r.squeeze(-1)

    if ndimy >= 2:
        x = x.reshape(x.shape + [1] * ndimy - 1)
        r = x @ y
        return r.squeeze(-2)

    raise RuntimeError("This should be unreachable")


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
        indexer = tuple(slice(None, None, -1) for _ in range(x.ndim))
        return x[indexer]
    if isinstance(axis, int):
        axis = (axis,)
    indexer = [slice(None)] * x.ndim
    for ax in axis:
        if ax < 0:
            ax = x.ndim + ax
        if not 0 <= ax < x.ndim:
            raise ValueError(
                f"axis {ax} is out of bounds for array of dimension {x.ndim}"
            )
        indexer[ax] = slice(None, None, -1)

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
    raise NotImplementedError("MLX doesn't support imag yet")


def isclose(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    result_dtype = result_type(x1.dtype, x2.dtype)
    x1 = cast(x1, result_dtype)
    x2 = cast(x2, result_dtype)

    rtol = 1e-5
    atol = 1e-8
    return absolute(x1 - x2) <= (atol + rtol * absolute(x2))


def isfinite(x):
    x = convert_to_tensor(x)
    return True - (isinf(x) + isnan(x))


def isinf(x):
    x = convert_to_tensor(x)
    return mx.abs(x) == float("inf")


def isnan(x):
    x = convert_to_tensor(x)
    return x != x


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
            "MLX doesn't support linspace with an `axis` argument. "
            f"Received axis={axis}"
        )
    start = convert_to_tensor(start)
    stop = convert_to_tensor(stop)
    zero_one = mx.arange(num) / ((num - 1) if endpoint else num)
    direction = stop - start
    zero_one = zero_one.reshape([-1] + [1] * direction.ndim)
    rs = zero_one * direction[None] + start[None]

    if retstep:
        return rs, rs[1] - rs[0]
    else:
        return rs


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
    return x1.astype(mx.bool_) * x2.astype(mx.bool_)


def logical_not(x):
    x = convert_to_tensor(x)
    return True - x.astype(mx.bool_)


def logical_or(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return x1.astype(mx.bool_) + x2.astype(mx.bool_)


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


def median(x, axis=-1, keepdims=False):
    x = convert_to_tensor(x)

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
            return mx.tensor(initial)

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

    ndim = x.ndim
    axes = list(range(ndim))
    for s, d in zip(source, destination):
        s = (ndim + s) % ndim
        d = (ndim + d) % ndim
        axes.insert(d, axes.pop(s))

    return mx.transpose(x, axes)


def nan_to_num(x):
    raise NotImplementedError("The MLX backend doesn't support nan_to_num yet")


def ndim(x):
    x = convert_to_tensor(x)
    return x.ndim


def nonzero(x):
    raise NotImplementedError("The MLX backend doesn't support nonzero yet")


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
    if mode != "constant":
        raise NotImplementedError(
            "MLX pad supports only `mode == 'constant'`"
            f"Received: mode={mode}"
        )

    if isinstance(pad_width, mx.array):
        pad_width = pad_width.tolist()

    x = convert_to_tensor(x)
    return mx.pad(x, pad_width, constant_values=constant_values or 0)


def prod(x, axis=None, keepdims=False, dtype=None):
    x = convert_to_tensor(x)
    if dtype is not None:
        x = cast(x, dtype)
    return mx.prod(x, axis=axis, keepdims=keepdims)


def quantile(x, q, axis=None, method="linear", keepdims=False):
    raise NotImplementedError("MLX doesn't support quantile yet")


def ravel(x):
    x = convert_to_tensor(x)
    return x.reshape(-1)


def real(x):
    raise NotImplementedError("MLX doesn't support real yet")


def reciprocal(x):
    x = convert_to_tensor(x)
    return mx.reciprocal(x)


def repeat(x, repeats, axis=None):
    x = convert_to_tensor(x)
    return mx.repeat(x, repeats, axis=axis)


def reshape(x, new_shape):
    if not isinstance(new_shape, (list, tuple)):
        new_shape = (new_shape,)
    x = convert_to_tensor(x)
    return mx.reshape(x, new_shape)


def roll(x, shift, axis=None):
    # TODO: Implement using concatenate
    raise NotImplementedError("The MLX backend doesn't support roll yet")


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
    xs = [mx.expand_dims(x, axis) for x in xs]
    return mx.concatenate(xs, axis=axis)


def std(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return mx.sqrt(mx.var(x, axis=axis, keepdims=keepdims))


def swapaxes(x, axis1, axis2):
    x = convert_to_tensor(x)
    axes = list(range(x.ndim))
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
    return x.transpose(axes)


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
    return mx.tensordot(x1, x2, axes=axes)


def round(x, decimals=0):
    x = convert_to_tensor(x)
    return mx.round(x, decimals=decimals)


def tile(x, repeats):
    x = convert_to_tensor(x)
    ndim = x.ndim
    if not isinstance(repeats, (tuple, list)):
        repeats = [repeats]

    if ndim > len(repeats):
        repeats = [1] * (ndim - len(repeats))
    elif ndim < len(repeats):
        shape = [1] * (len(repeats) - ndim) + x.shape
        x = x.reshape(shape)

    shape = []
    for s in x.shape:
        shape.append(s)
        shape.append(1)
    x = x.reshape(shape)
    for i, r in enumerate(repeats):
        shape[2 * i] = r
    x = mx.broadcast_to(x, shape)
    final_shape = []
    for i in range(len(shape) // 2):
        final_shape.append(shape[i] * shape[i + 1])
    x = x.reshape(final_shape)

    return x


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
    raise NotImplementedError("The MLX backend doesn't support vdot yet")


def vstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    if xs[0].ndim == 1:
        xs = [x[None] for x in xs]
    return mx.concatenate(xs, axis=0)


def where(condition, x1, x2):
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
