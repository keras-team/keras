import builtins

import numpy as np
import paddle

from keras.src.backend.paddle.core import convert_to_tensor
from keras.src.backend.paddle.core import shape
from keras.src.backend.paddle.core import to_paddle_dtype


def add(x1, x2):
    return paddle.add(convert_to_tensor(x1), convert_to_tensor(x2))


def subtract(x1, x2):
    return paddle.subtract(convert_to_tensor(x1), convert_to_tensor(x2))


def multiply(x1, x2):
    return paddle.multiply(convert_to_tensor(x1), convert_to_tensor(x2))


def divide(x1, x2):
    return paddle.divide(convert_to_tensor(x1), convert_to_tensor(x2))


def true_divide(x1, x2):
    x1 = convert_to_tensor(x1, "float32")
    x2 = convert_to_tensor(x2, "float32")
    return paddle.divide(x1, x2)


def floor_divide(x1, x2):
    return paddle.floor_divide(convert_to_tensor(x1), convert_to_tensor(x2))


def mod(x1, x2):
    return paddle.remainder(convert_to_tensor(x1), convert_to_tensor(x2))


def negative(x):
    return paddle.neg(convert_to_tensor(x))


def abs(x):
    return paddle.abs(convert_to_tensor(x))


def absolute(x):
    return abs(x)


def sign(x):
    return paddle.sign(convert_to_tensor(x))


def log(x):
    return paddle.log(convert_to_tensor(x))


def log2(x):
    return paddle.log2(convert_to_tensor(x))


def log10(x):
    return paddle.log10(convert_to_tensor(x))


def log1p(x):
    return paddle.log1p(convert_to_tensor(x))


def exp(x):
    return paddle.exp(convert_to_tensor(x))


def expm1(x):
    return paddle.expm1(convert_to_tensor(x))


def sqrt(x):
    return paddle.sqrt(convert_to_tensor(x))


def square(x):
    return paddle.square(convert_to_tensor(x))


def pow(x1, x2):
    return paddle.pow(convert_to_tensor(x1), convert_to_tensor(x2))


def power(x1, x2):
    return pow(x1, x2)


def maximum(x1, x2):
    return paddle.maximum(convert_to_tensor(x1), convert_to_tensor(x2))


def minimum(x1, x2):
    return paddle.minimum(convert_to_tensor(x1), convert_to_tensor(x2))


def round(x, decimals=0):
    return paddle.round(convert_to_tensor(x))


def clip(x, x_min, x_max):
    return paddle.clip(convert_to_tensor(x), x_min, x_max)


def clip_by_value(x, clip_value_min, clip_value_max):
    return clip(x, clip_value_min, clip_value_max)


def floor(x):
    return paddle.floor(convert_to_tensor(x))


def ceil(x):
    return paddle.ceil(convert_to_tensor(x))


def dot(x, y):
    return paddle.dot(convert_to_tensor(x), convert_to_tensor(y))


def tensordot(x1, x2, axes=2):
    return paddle.tensordot(convert_to_tensor(x1), convert_to_tensor(x2), axes)


def einsum(subscripts, *operands, **kwargs):
    operands = [convert_to_tensor(x) for x in operands]
    return paddle.einsum(subscripts, *operands)


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
        return paddle.where(condition, x1, x2)
    return paddle.where(condition)


def mean(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return x.clone()
    return paddle.mean(x, axis=axis, keepdim=keepdims)


def variance(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return paddle.zeros_like(x)
    return paddle.var(x, axis=axis, keepdim=keepdims)


def std(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if isinstance(axis, tuple) and len(axis) == 0:
        return paddle.zeros_like(x)
    return paddle.std(x, axis=axis, keepdim=keepdims)


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
    return paddle.prod(x, axis=axis, keepdim=keepdims)


def logsumexp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return paddle.logsumexp(x, axis=axis, keepdim=keepdims)


def cumsum(x, axis=None):
    return paddle.cumsum(convert_to_tensor(x), axis=axis)


def cumprod(x, axis=None):
    return paddle.cumprod(convert_to_tensor(x), axis=axis)


def argmax(x, axis=None, keepdims=False):
    return paddle.argmax(convert_to_tensor(x), axis=axis, keepdim=keepdims)


def argmin(x, axis=None, keepdims=False):
    return paddle.argmin(convert_to_tensor(x), axis=axis, keepdim=keepdims)


def argsort(x, axis=-1):
    return paddle.argsort(convert_to_tensor(x), axis=axis)


def sort(x, axis=-1):
    return paddle.sort(convert_to_tensor(x), axis=axis)


def searchsorted(sorted_sequence, values, side="left"):
    sorted_sequence = convert_to_tensor(sorted_sequence)
    values = convert_to_tensor(values)
    return paddle.searchsorted(sorted_sequence, values)


def top_k(x, k, sorted=False):
    return paddle.topk(convert_to_tensor(x), k)


def in_top_k(targets, predictions, k):
    targets = convert_to_tensor(targets, "int64")
    predictions = convert_to_tensor(predictions)
    topk_indices = paddle.topk(predictions, k, axis=-1)
    return paddle.any(topk_indices == targets.unsqueeze(-1), axis=-1)


def flip(x, axis=None):
    return paddle.flip(convert_to_tensor(x), axis=axis)


def roll(x, shift, axis=None):
    return paddle.roll(convert_to_tensor(x), shift, axis=axis)


def pad(x, pad_width, mode="constant", constant_values=0):
    x = convert_to_tensor(x)
    if mode == "constant":
        # paddle.nn.functional.pad expects padding in reverse order
        pad_list = []
        for left, right in reversed(pad_width):
            pad_list.extend([left, right])
        return paddle.nn.functional.pad(
            x, pad_list, mode="constant", value=constant_values
        )
    elif mode == "reflect":
        pad_list = []
        for left, right in reversed(pad_width):
            pad_list.extend([left, right])
        return paddle.nn.functional.pad(x, pad_list, mode="reflect")
    elif mode == "symmetric":
        pad_list = []
        for left, right in reversed(pad_width):
            pad_list.extend([left, right])
        return paddle.nn.functional.pad(x, pad_list, mode="replicate")
    elif mode == "edge":
        pad_list = []
        for left, right in reversed(pad_width):
            pad_list.extend([left, right])
        return paddle.nn.functional.pad(x, pad_list, mode="replicate")
    raise NotImplementedError(
        f"`pad` with mode='{mode}' is not supported with paddle backend"
    )


def concatenate(xs, axis=0):
    xs = [convert_to_tensor(x) for x in xs]
    return paddle.concat(xs, axis=axis)


def append(x, values, axis=None):
    x = convert_to_tensor(x)
    values = convert_to_tensor(values)
    if axis is None:
        return paddle.concat([x.flatten(), values.flatten()])
    return paddle.concat([x, values], axis=axis)


def stack(x, axis=0):
    x = [convert_to_tensor(xi) for xi in x]
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
    return paddle.eye(N, M, dtype=dtype)


def linspace(start, stop, num, dtype=None, endpoint=True):
    if dtype is not None:
        dtype = to_paddle_dtype(dtype)
    return paddle.linspace(start, stop, num, dtype=dtype)


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
    return paddle.tril(convert_to_tensor(x), diagonal=k)


def triu(x, k=0):
    return paddle.triu(convert_to_tensor(x), diagonal=k)


def diagonal(x, offset=0, axis1=0, axis2=1):
    return paddle.diagonal(convert_to_tensor(x), offset, axis1, axis2)


def trace(x, offset=0, axis1=0, axis2=1):
    return paddle.trace(convert_to_tensor(x), offset)


def meshgrid(*x, indexing="xy"):
    x = [convert_to_tensor(xi) for xi in x]
    return paddle.meshgrid(*x)


def histogram(x, bins=10):
    x = convert_to_tensor(x)
    if isinstance(bins, int):
        min_val = paddle.min(x)
        max_val = paddle.max(x)
        bin_edges = paddle.linspace(min_val, max_val, bins + 1)
    else:
        bin_edges = convert_to_tensor(bins)
        bins = len(bin_edges) - 1
    hist = paddle.zeros([bins], dtype="int64")
    for i in range(bins):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        if i == bins - 1:
            mask = mask | (x == bin_edges[i + 1])
        hist[i] = paddle.sum(mask.cast("int64"))
    return hist, bin_edges


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
    orig_shape = paddle.shape(indices)
    indices_flat = indices.flatten()
    result = paddle.gather(x, indices_flat, axis=axis)
    x_shape = paddle.shape(x)
    new_shape = paddle.concat([x_shape[:axis], orig_shape, x_shape[axis + 1 :]])
    return paddle.reshape(result, new_shape)


def take_along_axis(x, indices, axis=None):
    return paddle.take_along_axis(
        convert_to_tensor(x), convert_to_tensor(indices, dtype="int64"), axis
    )


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
    return paddle.isclose(
        convert_to_tensor(x1), convert_to_tensor(x2), rtol=rtol, atol=atol
    )


def allclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    return paddle.allclose(
        convert_to_tensor(x1), convert_to_tensor(x2), rtol=rtol, atol=atol
    )


def equal(x1, x2):
    return paddle.equal(convert_to_tensor(x1), convert_to_tensor(x2))


def not_equal(x1, x2):
    return paddle.not_equal(convert_to_tensor(x1), convert_to_tensor(x2))


def greater(x1, x2):
    return paddle.greater_than(convert_to_tensor(x1), convert_to_tensor(x2))


def greater_equal(x1, x2):
    return paddle.greater_equal(convert_to_tensor(x1), convert_to_tensor(x2))


def less(x1, x2):
    return paddle.less_than(convert_to_tensor(x1), convert_to_tensor(x2))


def less_equal(x1, x2):
    return paddle.less_equal(convert_to_tensor(x1), convert_to_tensor(x2))


def all(x, axis=None, keepdims=False):
    return paddle.all(convert_to_tensor(x), axis=axis, keepdim=keepdims)


def any(x, axis=None, keepdims=False):
    return paddle.any(convert_to_tensor(x), axis=axis, keepdim=keepdims)


def logical_and(x1, x2):
    return paddle.logical_and(convert_to_tensor(x1), convert_to_tensor(x2))


def logical_or(x1, x2):
    return paddle.logical_or(convert_to_tensor(x1), convert_to_tensor(x2))


def logical_not(x):
    return paddle.logical_not(convert_to_tensor(x))


def logical_xor(x1, x2):
    return paddle.logical_xor(convert_to_tensor(x1), convert_to_tensor(x2))


def bitwise_and(x1, x2):
    return paddle.bitwise_and(convert_to_tensor(x1), convert_to_tensor(x2))


def bitwise_or(x1, x2):
    return paddle.bitwise_or(convert_to_tensor(x1), convert_to_tensor(x2))


def bitwise_xor(x1, x2):
    return paddle.bitwise_xor(convert_to_tensor(x1), convert_to_tensor(x2))


def bitwise_not(x):
    return paddle.bitwise_not(convert_to_tensor(x))


def isfinite(x):
    return paddle.isfinite(convert_to_tensor(x))


def isinf(x):
    return paddle.isinf(convert_to_tensor(x))


def isnan(x):
    return paddle.isnan(convert_to_tensor(x))


def isneginf(x):
    x = convert_to_tensor(x)
    return paddle.logical_and(paddle.isinf(x), x < 0)


def isposinf(x):
    x = convert_to_tensor(x)
    return paddle.logical_and(paddle.isinf(x), x > 0)


def nan_to_num(x):
    x = convert_to_tensor(x)
    return paddle.nan_to_num(x)


def cross(x1, x2):
    return paddle.cross(convert_to_tensor(x1), convert_to_tensor(x2))


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
    return paddle.matmul(convert_to_tensor(x1), convert_to_tensor(x2))


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
    return paddle.trunc(convert_to_tensor(x))


def inner(a, b):
    return paddle.dot(
        convert_to_tensor(a).flatten(), convert_to_tensor(b).flatten()
    )


def outer(a, b):
    a = convert_to_tensor(a).flatten()
    b = convert_to_tensor(b).flatten()
    return paddle.mm(a.unsqueeze(1), b.unsqueeze(0))


def reciprocal(x):
    return paddle.reciprocal(convert_to_tensor(x))


def cos(x):
    return paddle.cos(convert_to_tensor(x))


def sin(x):
    return paddle.sin(convert_to_tensor(x))


def tan(x):
    return paddle.tan(convert_to_tensor(x))


def cosh(x):
    return paddle.cosh(convert_to_tensor(x))


def sinh(x):
    return paddle.sinh(convert_to_tensor(x))


def arccos(x):
    return paddle.acos(convert_to_tensor(x))


def arcsin(x):
    return paddle.asin(convert_to_tensor(x))


def arctan(x):
    return paddle.atan(convert_to_tensor(x))


def arctan2(x1, x2):
    return paddle.atan2(convert_to_tensor(x1), convert_to_tensor(x2))


def arccosh(x):
    return paddle.acosh(convert_to_tensor(x))


def arcsinh(x):
    return paddle.asinh(convert_to_tensor(x))


def arctanh(x):
    return paddle.atanh(convert_to_tensor(x))


def deg2rad(x):
    return paddle.deg2rad(convert_to_tensor(x))


def rad2deg(x):
    return paddle.rad2deg(convert_to_tensor(x))


def hypot(x1, x2):
    return paddle.hypot(convert_to_tensor(x1), convert_to_tensor(x2))


def fmod(x1, x2):
    return paddle.mod(convert_to_tensor(x1), convert_to_tensor(x2))


def ldexp(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return x1 * (2.0**x2)


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


def nanargmax(x, axis=None):
    x = convert_to_tensor(x)
    x = paddle.where(paddle.isnan(x), paddle.full_like(x, float("-inf")), x)
    return paddle.argmax(x, axis=axis)


def nanargmin(x, axis=None):
    x = convert_to_tensor(x)
    x = paddle.where(paddle.isnan(x), paddle.full_like(x, float("inf")), x)
    return paddle.argmin(x, axis=axis)


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
    mask = ~paddle.isnan(x)
    x = paddle.where(mask, x, paddle.zeros_like(x))
    count = paddle.sum(
        paddle.cast(mask, "float32"), axis=axis, keepdim=keepdims
    )
    return paddle.sum(x, axis=axis, keepdim=keepdims) / count


def nanvar(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
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


def nancumsum(x, axis=None):
    x = convert_to_tensor(x)
    x = paddle.where(paddle.isnan(x), paddle.zeros_like(x), x)
    return paddle.cumsum(x, axis=axis)


def nancumprod(x, axis=None):
    x = convert_to_tensor(x)
    x = paddle.where(paddle.isnan(x), paddle.ones_like(x), x)
    return paddle.cumprod(x, axis=axis)


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
    return paddle.kron(convert_to_tensor(a), convert_to_tensor(b))


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


def view(x, dtype):
    raise NotImplementedError("`view` is not supported with paddle backend")


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
    return paddle.searchsorted(bins, x)


def bincount(x, weights=None, minlength=0):
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
    return paddle.corrcoef(x)


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
    return paddle.median(x, axis=axis, keepdim=keepdims)


def quantile(x, q, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if axis is None:
        x = x.flatten()
    sorted_x = paddle.sort(x, axis=axis if axis is not None else -1)
    if isinstance(q, (list, tuple)):
        q = convert_to_tensor(q, "float32")
    else:
        q = convert_to_tensor([q], "float32")
    if axis is None:
        n = int(np.prod(sorted_x.shape))
        indices = q * (n - 1)
        lower = indices.cast("int64")
        upper = paddle.minimum(
            lower + 1, paddle.to_tensor(n - 1, dtype="int64")
        )
        frac = indices - lower.cast("float32")
        result = sorted_x[lower] * (1 - frac) + sorted_x[upper] * frac
    else:
        n = sorted_x.shape[axis]
        indices = q * (n - 1)
        lower = indices.cast("int64")
        upper = paddle.minimum(
            lower + 1, paddle.to_tensor(n - 1, dtype="int64")
        )
        frac = indices - lower.cast("float32")
        if axis is not None:
            frac_shape = [1] * sorted_x.ndim
            frac_shape[axis] = -1
            frac = paddle.reshape(frac, frac_shape)
        lower_vals = paddle.gather(sorted_x, lower, axis=axis)
        upper_vals = paddle.gather(sorted_x, upper, axis=axis)
        result = lower_vals * (1 - frac) + upper_vals * frac
    if keepdims:
        if axis is not None:
            result = paddle.unsqueeze(result, axis=axis)
    return result


def percentile(x, q, axis=None, keepdims=False):
    return quantile(x, q / 100.0, axis=axis, keepdims=keepdims)


def nanmedian(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    mask = paddle.isnan(x)
    x_no_nan = paddle.where(mask, paddle.zeros_like(x), x)
    sorted_x = paddle.sort(x_no_nan, axis=axis)
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


def nanquantile(x, q, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    mask = paddle.isnan(x)
    x_no_nan = paddle.where(mask, paddle.zeros_like(x), x)
    sorted_x = paddle.sort(x_no_nan, axis=axis)
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


def nanpercentile(x, q, axis=None, keepdims=False):
    return nanquantile(x, q / 100.0, axis=axis, keepdims=keepdims)


def ptp(x, axis=None):
    x = convert_to_tensor(x)
    return paddle.max(x, axis=axis) - paddle.min(x, axis=axis)


def logaddexp(x1, x2):
    return paddle.logaddexp(convert_to_tensor(x1), convert_to_tensor(x2))


def logaddexp2(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return paddle.logaddexp(x1 * np.log(2), x2 * np.log(2)) / np.log(2)


def logspace(start, stop, num, base=10.0, dtype=None, endpoint=True):
    result = linspace(start, stop, num, endpoint=endpoint)
    return base**result


def geomspace(start, stop, num, endpoint=True, dtype=None):
    start = convert_to_tensor(start, "float32")
    stop = convert_to_tensor(stop, "float32")
    return paddle.exp(paddle.linspace(paddle.log(start), paddle.log(stop), num))


def empty(shape, dtype="float32"):
    return zeros(shape, dtype=dtype)


def empty_like(x, dtype=None):
    return zeros_like(x, dtype=dtype)


def nextafter(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    # Approximate nextafter using bit manipulation
    eps = paddle.finfo(x1.dtype).eps
    direction = paddle.sign(x2 - x1)
    return x1 + direction * eps


def isreal(x):
    x = convert_to_tensor(x)
    return paddle.isreal(x)


def isin(elements, test_elements, assume_unique=False, invert=False):
    elements = convert_to_tensor(elements)
    test_elements = convert_to_tensor(test_elements).flatten()
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
    xs = [paddle.unsqueeze(x, axis=2) if len(x.shape) < 3 else x for x in xs]
    return paddle.concat(xs, axis=2)


def hstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    if len(xs[0].shape) == 1:
        return paddle.concat(xs, axis=0)
    return paddle.concat(xs, axis=1)


def vstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
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
    k = k % 4
    if k == 0:
        return x
    elif k == 1:
        return paddle.flip(paddle.transpose(x, axes), [axes[1]])
    elif k == 2:
        return paddle.flip(x, [axes[0], axes[1]])
    elif k == 3:
        return paddle.transpose(paddle.flip(x, [axes[1]]), axes)
    return x


def average(x, axis=None, weights=None, returned=False, keepdims=False):
    x = convert_to_tensor(x)
    if weights is None:
        result = paddle.mean(x, axis=axis, keepdim=keepdims)
    else:
        weights = convert_to_tensor(weights)
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
    safe_x2 = paddle.where(x2 == 0, paddle.ones_like(x2), x2)
    return paddle.where(x2 == 0, paddle.zeros_like(x1), x1 / safe_x2)


def slogdet(x):
    x = convert_to_tensor(x)
    sign, logabsdet = paddle.linalg.slogdet(x)
    return sign, logabsdet


def argpartition(x, kth, axis=-1):
    x = convert_to_tensor(x)
    sorted_indices = paddle.argsort(x, axis=axis)
    return sorted_indices


def gcd(x1, x2):
    return paddle.gcd(convert_to_tensor(x1), convert_to_tensor(x2))


def lcm(x1, x2):
    return paddle.lcm(convert_to_tensor(x1), convert_to_tensor(x2))


def max(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)
    result = paddle.max(x, axis=axis, keepdim=keepdims)
    if initial is not None:
        result = paddle.maximum(result, convert_to_tensor(initial))
    return result


def min(x, axis=None, keepdims=False, initial=None):
    x = convert_to_tensor(x)
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
    return paddle.diag(convert_to_tensor(x), offset=k)


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
    return x.unsqueeze(1) ** powers.unsqueeze(0)
