import math

import mlx.core as mx

from keras.src.backend.mlx.core import convert_to_tensor


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids)
    segment_max = mx.max(segment_ids)
    segment_max = segment_max.item() + 1
    num_segments = num_segments or segment_max

    segment_ids = mx.where(segment_ids >= 0, segment_ids, segment_max)

    data_shape = (num_segments + 1,) + tuple(data.shape[1:])
    result = mx.zeros(data_shape, dtype=data.dtype)

    result = result.at[segment_ids].add(data)
    result = result[:-1, ...]
    return result


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids)
    segment_max = mx.max(segment_ids)
    segment_max = segment_max.item() + 1
    num_segments = num_segments or segment_max

    segment_ids = mx.where(segment_ids >= 0, segment_ids, segment_max)

    data_shape = (num_segments + 1,) + tuple(data.shape[1:])
    result = mx.zeros(data_shape, dtype=data.dtype)

    result = result.at[segment_ids].maximum(data)
    result = result[:-1, ...]
    return result


def top_k(x, k, sorted=True):
    x = convert_to_tensor(x)
    indices = mx.argpartition(mx.negative(x), k, axis=-1)[..., :k]
    values = mx.take_along_axis(x, indices, axis=-1)
    return values, indices


def in_top_k(targets, predictions, k):
    targets = convert_to_tensor(targets)
    predictions = convert_to_tensor(predictions)
    targets = targets[..., None]
    topk_values = top_k(predictions, k)[0]
    targets_values = mx.take_along_axis(predictions, targets, axis=-1)
    mask = targets_values >= topk_values
    return mx.any(mask, axis=-1)


def logsumexp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return mx.logsumexp(x, axis, keepdims)


def qr(x, mode="reduced"):
    # TODO https://ml-explore.github.io/mlx/build/html/python/linalg.html
    raise NotImplementedError("QR decomposition not supported in mlx yet")


def extract_sequences(x, sequence_length, sequence_stride):
    x = convert_to_tensor(x)

    *batch_shape, signal_length = x.shape
    frames = (signal_length - sequence_length) // sequence_stride + 1
    N = math.prod(batch_shape)
    x = mx.as_strided(
        x,
        shape=(N, frames, sequence_length),
        strides=(signal_length, sequence_stride, 1),
    )
    return x.reshape(*batch_shape, frames, sequence_length)


def fft(x):
    # TODO: https://ml-explore.github.io/mlx/build/html/python/fft.html#fft
    raise NotImplementedError("fft not yet implemented in mlx")


def fft2(x):
    # TODO: https://ml-explore.github.io/mlx/build/html/python/fft.html#fft
    raise NotImplementedError("fft not yet implemented in mlx")


def rfft(x, fft_length=None):
    # TODO: https://ml-explore.github.io/mlx/build/html/python/fft.html#fft
    raise NotImplementedError("fft not yet implemented in mlx")


def irfft(x, fft_length=None):
    # TODO: https://ml-explore.github.io/mlx/build/html/python/fft.html#fft
    raise NotImplementedError("fft not yet implemented in mlx")


def stft(
    x, sequence_length, sequence_stride, fft_length, window="hann", center=True
):
    raise NotImplementedError("fft not yet implemented in mlx")


def istft(
    x,
    sequence_length,
    sequence_stride,
    fft_length,
    length=None,
    window="hann",
    center=True,
):
    raise NotImplementedError("fft not yet implemented in mlx")


def rsqrt(x):
    x = convert_to_tensor(x)
    return mx.rsqrt(x)


def erf(x):
    x = convert_to_tensor(x)
    return mx.erf(x)


def erfinv(x):
    x = convert_to_tensor(x)
    return mx.erfinv(x)


def solve(a, b):
    raise NotImplementedError(
        "Linear system solving not yet implemented in mlx"
    )
