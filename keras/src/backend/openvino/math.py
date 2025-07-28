import openvino.runtime.opset14 as ov_opset
from openvino import Type

from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import get_ov_output


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    raise NotImplementedError(
        "`segment_sum` is not supported with openvino backend"
    )


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    raise NotImplementedError(
        "`segment_max` is not supported with openvino backend"
    )


def top_k(x, k, sorted=True):
    x = get_ov_output(x)
    k_tensor = ov_opset.constant(k, dtype=Type.i32)
    axis = -1
    sort_type = "value" if sorted else "none"
    topk_node = ov_opset.topk(x, k_tensor, axis, "max", sort_type)
    values = topk_node.output(0)
    indices = topk_node.output(1)
    return OpenVINOKerasTensor(values), OpenVINOKerasTensor(indices)


def in_top_k(targets, predictions, k):
    raise NotImplementedError(
        "`in_top_k` is not supported with openvino backend"
    )


def logsumexp(x, axis=None, keepdims=False):
    x = get_ov_output(x)
    if axis is None:
        flatten_shape = ov_opset.constant([-1], Type.i32).output(0)
        x = ov_opset.reshape(x, flatten_shape, False).output(0)
        axis = 0
    if isinstance(axis, tuple):
        axis = list(axis)
    axis = ov_opset.constant(axis, Type.i32).output(0)
    const_zero = ov_opset.constant(0, x.get_element_type()).output(0)
    # Use keepdims=True for reduce_max to ensure proper broadcasting
    reduce_max = ov_opset.reduce_max(x, axis, True).output(0)
    is_finite = ov_opset.is_finite(reduce_max).output(0)
    norm_max = ov_opset.select(is_finite, reduce_max, const_zero).output(0)
    norm_max_sub = ov_opset.subtract(x, norm_max).output(0)
    exp_norm_max = ov_opset.exp(norm_max_sub).output(0)
    sum_exp = ov_opset.reduce_sum(exp_norm_max, axis, keepdims).output(0)
    log_sum_exp = ov_opset.log(sum_exp).output(0)
    # Squeeze norm_max if needed to match dimensions
    if not keepdims:
        norm_max = ov_opset.squeeze(norm_max, axis).output(0)
    log_sum_exp = ov_opset.add(norm_max, log_sum_exp).output(0)
    return OpenVINOKerasTensor(log_sum_exp)


def qr(x, mode="reduced"):
    raise NotImplementedError("`qr` is not supported with openvino backend")


def extract_sequences(x, sequence_length, sequence_stride):
    raise NotImplementedError(
        "`extract_sequences` is not supported with openvino backend"
    )


def fft(x):
    raise NotImplementedError("`fft` is not supported with openvino backend")


def fft2(x):
    raise NotImplementedError("`fft2` is not supported with openvino backend")


def rfft(x, fft_length=None):
    raise NotImplementedError("`rfft` is not supported with openvino backend")


def irfft(x, fft_length=None):
    raise NotImplementedError("`irfft` is not supported with openvino backend")


def stft(
    x, sequence_length, sequence_stride, fft_length, window="hann", center=True
):
    raise NotImplementedError("`stft` is not supported with openvino backend")


def istft(
    x,
    sequence_length,
    sequence_stride,
    fft_length,
    length=None,
    window="hann",
    center=True,
):
    raise NotImplementedError("`istft` is not supported with openvino backend")


def rsqrt(x):
    x = get_ov_output(x)
    const_one = ov_opset.constant(1, x.get_element_type()).output(0)
    sqrt = ov_opset.sqrt(x).output(0)
    return OpenVINOKerasTensor(ov_opset.divide(const_one, sqrt).output(0))


def erf(x):
    x = get_ov_output(x)
    erf = ov_opset.erf(x).output(0)
    return OpenVINOKerasTensor(erf)


def erfinv(x):
    raise NotImplementedError("`erfinv` is not supported with openvino backend")


def solve(a, b):
    raise NotImplementedError("`solve` is not supported with openvino backend")


def norm(x, ord=None, axis=None, keepdims=False):
    raise NotImplementedError("`norm` is not supported with openvino backend")
