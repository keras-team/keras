import math

import mlx.core as mx

from keras.src.backend.mlx.core import convert_to_tensor


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    raise NotImplementedError("segment_sum is not implemented for mlx")


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    raise NotImplementedError("segment_max is not implemented for mlx")


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


def _get_complex_tensor_from_tuple(x):
    if not isinstance(x, (tuple, list)) or len(x) != 2:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            f"Received: x={x}"
        )
    real, imag = x
    real = convert_to_tensor(real)
    imag = convert_to_tensor(imag)
    # Check shapes.
    if real.shape != imag.shape:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            "Both the real and imaginary parts should have the same shape. "
            f"Received: x[0].shape = {real.shape}, x[1].shape = {imag.shape}"
        )
    # Ensure dtype is float.
    if not mx.issubdtype(real.dtype, mx.floating) or not mx.issubdtype(
        imag.dtype, mx.floating
    ):
        raise ValueError(
            "At least one tensor in input `x` is not of type float."
            f"Received: x={x}."
        )
    complex_input = mx.add(real, 1j * imag)
    return complex_input


def fft(x):
    x = _get_complex_tensor_from_tuple(x)
    complex_output = mx.fft.fft(x)
    return mx.real(complex_output), mx.imag(complex_output)


def fft2(x):
    x = _get_complex_tensor_from_tuple(x)
    complex_output = mx.fft.fft2(x)
    return mx.real(complex_output), mx.imag(complex_output)


def rfft(x, fft_length=None):
    x = convert_to_tensor(x)
    complex_output = mx.fft.rfft(x, n=fft_length)
    return mx.real(complex_output), mx.imag(complex_output)


def irfft(x, fft_length=None):
    x = _get_complex_tensor_from_tuple(x)
    real_output = mx.fft.irfft(x, n=fft_length)
    return real_output


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
