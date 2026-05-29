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
    return paddle.gather(x, indices, axis=axis)


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
