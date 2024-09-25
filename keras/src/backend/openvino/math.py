from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.utils.module_utils import scipy


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    raise NotImplementedError(
        "`segment_sum` is not supported with openvino backend"
    )


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    raise NotImplementedError(
        "`segment_max` is not supported with openvino backend"
    )


def top_k(x, k, sorted=False):
    raise NotImplementedError(
        "`top_k` is not supported with openvino backend"
    )


def in_top_k(targets, predictions, k):
    raise NotImplementedError(
        "`in_top_k` is not supported with openvino backend"
    )


def logsumexp(x, axis=None, keepdims=False):
    raise NotImplementedError(
        "`logsumexp` is not supported with openvino backend"
    )


def qr(x, mode="reduced"):
    raise NotImplementedError(
        "`qr` is not supported with openvino backend"
    )


def extract_sequences(x, sequence_length, sequence_stride):
    raise NotImplementedError(
        "`extract_sequences` is not supported with openvino backend"
    )


def fft(x):
    raise NotImplementedError(
        "`fft` is not supported with openvino backend"
    )


def fft2(x):
    raise NotImplementedError(
        "`fft2` is not supported with openvino backend"
    )


def rfft(x, fft_length=None):
    raise NotImplementedError(
        "`rfft` is not supported with openvino backend"
    )


def irfft(x, fft_length=None):
    raise NotImplementedError(
        "`irfft` is not supported with openvino backend"
    )


def stft(
        x, sequence_length, sequence_stride, fft_length, window="hann", center=True
):
    raise NotImplementedError(
        "`stft` is not supported with openvino backend"
    )


def istft(
        x,
        sequence_length,
        sequence_stride,
        fft_length,
        length=None,
        window="hann",
        center=True,
):
    raise NotImplementedError(
        "`istft` is not supported with openvino backend"
    )


def rsqrt(x):
    raise NotImplementedError(
        "`rsqrt` is not supported with openvino backend"
    )


def erf(x):
    raise NotImplementedError(
        "`erf` is not supported with openvino backend"
    )


def erfinv(x):
    raise NotImplementedError(
        "`erfinv` is not supported with openvino backend"
    )


def solve(a, b):
    raise NotImplementedError(
        "`solve` is not supported with openvino backend"
    )


def norm(x, ord=None, axis=None, keepdims=False):
    raise NotImplementedError(
        "`norm` is not supported with openvino backend"
    )
