import numpy as np
import paddle

from keras.src.backend.paddle.core import PADDLE_DTYPES
from keras.src.backend.paddle.core import convert_to_numpy
from keras.src.backend.paddle.core import convert_to_tensor
from keras.src.backend.paddle.core import is_tensor
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
    raise NotImplementedError(
        "`slice_count` is not supported with paddle backend"
    )


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
    return paddle.mean(convert_to_tensor(x), axis=axis, keepdim=keepdims)


def variance(x, axis=None, keepdims=False):
    return paddle.var(convert_to_tensor(x), axis=axis, keepdim=keepdims)


def std(x, axis=None, keepdims=False):
    return paddle.std(convert_to_tensor(x), axis=axis, keepdim=keepdims)


def sum(x, axis=None, keepdims=False):
    return paddle.sum(convert_to_tensor(x), axis=axis, keepdim=keepdims)


def prod(x, axis=None, keepdims=False, dtype=None):
    if dtype is not None:
        x = convert_to_tensor(x, dtype=dtype)
    else:
        x = convert_to_tensor(x)
    return paddle.prod(x, axis=axis, keepdim=keepdims)


def logsumexp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return paddle.logsumexp(x, axis=axis, keepdim=keepdims)


def cumsum(x, axis=None):
    return paddle.cumsum(convert_to_tensor(x), axis=axis)


def cumprod(x, axis=None):
    return paddle.cumprod(convert_to_tensor(x), axis=axis)


def argmax(x, axis=None):
    return paddle.argmax(convert_to_tensor(x), axis=axis)


def argmin(x, axis=None):
    return paddle.argmin(convert_to_tensor(x), axis=axis)


def argsort(x, axis=-1):
    return paddle.argsort(convert_to_tensor(x), axis=axis)


def sort(x, axis=-1):
    return paddle.sort(convert_to_tensor(x), axis=axis)


def searchsorted(sorted_sequence, values, side="left"):
    raise NotImplementedError(
        "`searchsorted` is not supported with paddle backend"
    )


def top_k(x, k, sorted=False):
    return paddle.topk(convert_to_tensor(x), k)


def in_top_k(targets, predictions, k):
    raise NotImplementedError(
        "`in_top_k` is not supported with paddle backend"
    )


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
    return paddle.transpose(convert_to_tensor(x), axes)


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
    raise NotImplementedError(
        "`histogram` is not supported with paddle backend"
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
    orig_shape = indices.shape
    indices_flat = indices.flatten().unsqueeze(1)
    result = paddle.gather(x, indices_flat, axis=axis)
    # Restore the original index shape in the result
    new_shape = list(orig_shape) + list(x.shape[axis+1:])
    return paddle.reshape(result, new_shape)


def take_along_axis(x, indices, axis=None):
    return paddle.take_along_axis(
        convert_to_tensor(x), convert_to_tensor(indices, dtype="int64"), axis
    )


def put_along_axis(x, indices, values, axis=None):
    raise NotImplementedError(
        "`put_along_axis` is not supported with paddle backend"
    )


def block_diag(inputs):
    raise NotImplementedError(
        "`block_diag` is not supported with paddle backend"
    )


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
    return paddle.isclose(convert_to_tensor(x1), convert_to_tensor(x2), rtol=rtol, atol=atol)


def allclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    return paddle.allclose(convert_to_tensor(x1), convert_to_tensor(x2), rtol=rtol, atol=atol)


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


def binary_crossentropy(target, output, from_logits=False):
    raise NotImplementedError(
        "`binary_crossentropy` is not supported with paddle backend"
    )


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    raise NotImplementedError(
        "`sparse_categorical_crossentropy` is not supported with paddle backend"
    )


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    raise NotImplementedError(
        "`categorical_crossentropy` is not supported with paddle backend"
    )


def conv(inputs, kernel, strides=1, padding="valid", data_format=None, dilation_rate=1):
    raise NotImplementedError(
        "`conv` is not supported with paddle backend"
    )


def depthwise_conv(inputs, kernel, strides=1, padding="valid", data_format=None, dilation_rate=1):
    raise NotImplementedError(
        "`depthwise_conv` is not supported with paddle backend"
    )


def separable_conv(inputs, depthwise_kernel, pointwise_kernel, strides=1, padding="valid", data_format=None, dilation_rate=1):
    raise NotImplementedError(
        "`separable_conv` is not supported with paddle backend"
    )


def conv_transpose(inputs, kernel, strides, padding="valid", output_padding=None, data_format=None, dilation_rate=1):
    raise NotImplementedError(
        "`conv_transpose` is not supported with paddle backend"
    )


def avg_pool(inputs, pool_size, strides, padding="valid", data_format=None):
    raise NotImplementedError(
        "`avg_pool` is not supported with paddle backend"
    )


def max_pool(inputs, pool_size, strides, padding="valid", data_format=None):
    raise NotImplementedError(
        "`max_pool` is not supported with paddle backend"
    )


def adaptive_avg_pool(inputs, output_size, data_format=None):
    raise NotImplementedError(
        "`adaptive_avg_pool` is not supported with paddle backend"
    )


def adaptive_max_pool(inputs, output_size, data_format=None):
    raise NotImplementedError(
        "`adaptive_max_pool` is not supported with paddle backend"
    )


def average_pool(inputs, pool_size, strides, padding="valid", data_format=None):
    raise NotImplementedError(
        "`average_pool` is not supported with paddle backend"
    )


def global_average_pool(inputs, data_format=None):
    raise NotImplementedError(
        "`global_average_pool` is not supported with paddle backend"
    )


def global_max_pool(inputs, data_format=None):
    raise NotImplementedError(
        "`global_max_pool` is not supported with paddle backend"
    )


def moments(inputs, axes, keepdims=False, synchronized=False):
    raise NotImplementedError(
        "`moments` is not supported with paddle backend"
    )


def batch_normalization(x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3):
    raise NotImplementedError(
        "`batch_normalization` is not supported with paddle backend"
    )


def ctc_decode(inputs, input_lengths, strategy="greedy", beam_width=100, top_paths=1, merge_repeated=False, mask_value=-1):
    raise NotImplementedError(
        "`ctc_decode` is not supported with paddle backend"
    )


def psnr(x1, x2, max_val):
    raise NotImplementedError(
        "`psnr` is not supported with paddle backend"
    )


def dot_product_attention(query, key, value, bias=None, mask=None, scale=None, is_causal=False, flash_attention=None):
    raise NotImplementedError(
        "`dot_product_attention` is not supported with paddle backend"
    )


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    raise NotImplementedError(
        "`segment_sum` is not supported with paddle backend"
    )


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    raise NotImplementedError(
        "`segment_max` is not supported with paddle backend"
    )


def gamma(shape, alpha, dtype=None, seed=None):
    raise NotImplementedError(
        "`gamma` is not supported with paddle backend"
    )


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    raise NotImplementedError(
        "`binomial` is not supported with paddle backend"
    )


def beta(shape, alpha, beta, dtype=None, seed=None):
    raise NotImplementedError(
        "`beta` is not supported with paddle backend"
    )


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
    return paddle.dot(convert_to_tensor(a).flatten(), convert_to_tensor(b).flatten())


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
    return x1 * (2.0 ** x2)


def left_shift(x1, x2):
    return paddle.bitwise_left_shift(convert_to_tensor(x1), convert_to_tensor(x2))


def right_shift(x1, x2):
    return paddle.bitwise_right_shift(convert_to_tensor(x1), convert_to_tensor(x2))


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
    return paddle.where(x1 > 0, paddle.ones_like(x1), paddle.where(x1 == 0, x2, paddle.zeros_like(x1)))


def i0(x):
    x = convert_to_tensor(x, "float32")
    return paddle.i0(x)


def sinc(x):
    x = convert_to_tensor(x, "float32")
    return paddle.where(x == 0, paddle.ones_like(x), paddle.sin(np.pi * x) / (np.pi * x))


def count_nonzero(x, axis=None):
    x = convert_to_tensor(x)
    return paddle.sum(paddle.cast(x != 0, "int32"), axis=axis)


def nanargmax(x, axis=None):
    x = convert_to_tensor(x)
    x = paddle.where(paddle.isnan(x), paddle.full_like(x, float('-inf')), x)
    return paddle.argmax(x, axis=axis)


def nanargmin(x, axis=None):
    x = convert_to_tensor(x)
    x = paddle.where(paddle.isnan(x), paddle.full_like(x, float('inf')), x)
    return paddle.argmin(x, axis=axis)


def nanmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    x = paddle.where(paddle.isnan(x), paddle.full_like(x, float('-inf')), x)
    return paddle.max(x, axis=axis, keepdim=keepdims)


def nanmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    x = paddle.where(paddle.isnan(x), paddle.full_like(x, float('inf')), x)
    return paddle.min(x, axis=axis, keepdim=keepdims)


def nansum(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    x = paddle.where(paddle.isnan(x), paddle.zeros_like(x), x)
    return paddle.sum(x, axis=axis, keepdim=keepdims)


def nanmean(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    mask = ~paddle.isnan(x)
    x = paddle.where(mask, x, paddle.zeros_like(x))
    count = paddle.sum(paddle.cast(mask, "float32"), axis=axis, keepdim=keepdims)
    return paddle.sum(x, axis=axis, keepdim=keepdims) / count


def nanvar(x, axis=None, keepdims=False):
    m = nanmean(x, axis=axis, keepdims=True)
    x = convert_to_tensor(x)
    x = paddle.where(paddle.isnan(x), paddle.zeros_like(x), x)
    mask = ~paddle.isnan(x)
    count = paddle.sum(paddle.cast(mask, "float32"), axis=axis, keepdim=keepdims)
    return paddle.sum((x - m) ** 2, axis=axis, keepdim=keepdims) / count


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
        result = paddle.where(convert_to_tensor(cond), convert_to_tensor(choice), result)
    return result


def unique(x, **kwargs):
    raise NotImplementedError("`unique` is not supported with paddle backend")


def unravel_index(indices, shape):
    raise NotImplementedError(
        "`unravel_index` is not supported with paddle backend"
    )


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
    raise NotImplementedError(
        "`digitize` is not supported with paddle backend"
    )


def bincount(x, weights=None, minlength=0):
    raise NotImplementedError(
        "`bincount` is not supported with paddle backend"
    )


def corrcoef(x):
    raise NotImplementedError(
        "`corrcoef` is not supported with paddle backend"
    )


def correlate(x1, x2, mode="valid"):
    raise NotImplementedError(
        "`correlate` is not supported with paddle backend"
    )


def median(x, axis=None, keepdims=False):
    raise NotImplementedError(
        "`median` is not supported with paddle backend"
    )


def quantile(x, q, axis=None, keepdims=False):
    raise NotImplementedError(
        "`quantile` is not supported with paddle backend"
    )


def percentile(x, q, axis=None, keepdims=False):
    raise NotImplementedError(
        "`percentile` is not supported with paddle backend"
    )


def nanmedian(x, axis=None, keepdims=False):
    raise NotImplementedError(
        "`nanmedian` is not supported with paddle backend"
    )


def nanquantile(x, q, axis=None, keepdims=False):
    raise NotImplementedError(
        "`nanquantile` is not supported with paddle backend"
    )


def nanpercentile(x, q, axis=None, keepdims=False):
    raise NotImplementedError(
        "`nanpercentile` is not supported with paddle backend"
    )


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
    return base ** result


def geomspace(start, stop, num, endpoint=True, dtype=None):
    raise NotImplementedError(
        "`geomspace` is not supported with paddle backend"
    )


def empty(shape, dtype="float32"):
    return zeros(shape, dtype=dtype)


def empty_like(x, dtype=None):
    return zeros_like(x, dtype=dtype)


def nextafter(x1, x2):
    raise NotImplementedError(
        "`nextafter` is not supported with paddle backend"
    )


def isreal(x):
    x = convert_to_tensor(x)
    return paddle.isreal(x)


def isin(elements, test_elements, assume_unique=False, invert=False):
    raise NotImplementedError(
        "`isin` is not supported with paddle backend"
    )


def nonzero(x):
    x = convert_to_tensor(x)
    indices = paddle.nonzero(x)
    return tuple(indices.T)


def array_split(x, indices_or_sections, axis=0):
    raise NotImplementedError(
        "`array_split` is not supported with paddle backend"
    )


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
    n = x.shape[0]
    return paddle.diag(x, offset=k)


def fliplr(x):
    return paddle.flip(convert_to_tensor(x), axis=[1])


def flipud(x):
    return paddle.flip(convert_to_tensor(x), axis=[0])


def rot90(x, k=1, axes=(0, 1)):
    raise NotImplementedError(
        "`rot90` is not supported with paddle backend"
    )


def average(x, axis=None, weights=None, returned=False, keepdims=False):
    x = convert_to_tensor(x)
    if weights is None:
        result = paddle.mean(x, axis=axis, keepdim=keepdims)
    else:
        weights = convert_to_tensor(weights)
        result = paddle.sum(x * weights, axis=axis, keepdim=keepdims) / paddle.sum(weights, axis=axis, keepdim=keepdims)
    if returned:
        weights_sum = paddle.sum(weights if weights is not None else paddle.ones_like(x), axis=axis, keepdim=keepdims)
        return result, weights_sum
    return result


def cbrt(x):
    return paddle.pow(convert_to_tensor(x, "float32"), 1.0 / 3.0)


def exp2(x):
    return paddle.pow(2.0, convert_to_tensor(x, "float32"))


def divide_no_nan(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return paddle.where(x2 == 0, paddle.zeros_like(x1), x1 / x2)


def slogdet(x):
    raise NotImplementedError(
        "`slogdet` is not supported with paddle backend"
    )


def argpartition(x, kth, axis=-1):
    raise NotImplementedError(
        "`argpartition` is not supported with paddle backend"
    )


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
