import paddle

from keras.src.backend.paddle.core import convert_to_tensor
from keras.src.backend.paddle.core import is_tensor


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    raise NotImplementedError(
        "`segment_sum` is not supported with paddle backend"
    )


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    raise NotImplementedError(
        "`segment_max` is not supported with paddle backend"
    )


def top_k(x, k, sorted=False):
    raise NotImplementedError("`top_k` is not supported with paddle backend")


def in_top_k(targets, predictions, k):
    raise NotImplementedError(
        "`in_top_k` is not supported with paddle backend"
    )


def logsumexp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return paddle.logsumexp(x, axis=axis, keepdim=keepdims)


def qr(x, mode="reduced"):
    raise NotImplementedError("`qr` is not supported with paddle backend")


def extract_sequences(x, sequence_length, sequence_stride):
    raise NotImplementedError(
        "`extract_sequences` is not supported with paddle backend"
    )


def fft(x):
    raise NotImplementedError("`fft` is not supported with paddle backend")


def fft2(x):
    raise NotImplementedError("`fft2` is not supported with paddle backend")


def rfft(x, fft_length=None):
    raise NotImplementedError("`rfft` is not supported with paddle backend")


def irfft(x, fft_length=None):
    raise NotImplementedError(
        "`irfft` is not supported with paddle backend"
    )


def stft(
    x, sequence_length, sequence_stride, fft_length, window="hann", center=True
):
    raise NotImplementedError("`stft` is not supported with paddle backend")


def istft(
    x,
    sequence_length,
    sequence_stride,
    fft_length,
    length=None,
    window="hann",
    center=True,
):
    raise NotImplementedError("`istft` is not supported with paddle backend")


def rsqrt(x):
    x = convert_to_tensor(x)
    return paddle.rsqrt(x)


def erf(x):
    x = convert_to_tensor(x)
    return paddle.erf(x)


def erfinv(x):
    raise NotImplementedError(
        "`erfinv` is not supported with paddle backend"
    )


def solve(a, b):
    raise NotImplementedError("`solve` is not supported with paddle backend")


def norm(x, ord=None, axis=None, keepdims=False):
    raise NotImplementedError("`norm` is not supported with paddle backend")
