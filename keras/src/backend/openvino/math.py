import numpy as np
import openvino.opset15 as ov_opset
import scipy.signal
from openvino import Type

from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import cast
from keras.src.backend.openvino.core import get_ov_output
from keras.src.backend.openvino.core import standardize_dtype
from keras.src.backend.openvino.numpy import stack


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
    from keras.src.backend.openvino.numpy import take_along_axis

    # Expand targets: (batch,) → (batch, 1) for use with take_along_axis
    targets = ov_opset.unsqueeze(
        get_ov_output(targets), ov_opset.constant(1, Type.i32)
    ).output(0)
    predictions = get_ov_output(predictions)

    # top_k returns (batch, k) sorted descending; last col is the k-th largest
    topk_values = top_k(predictions, k)[0]
    # Grab only the last column (index k-1): threshold value, shape (batch,)
    k_minus_1_idx = ov_opset.constant([k - 1], dtype=Type.i32).output(0)
    topk_values_axis = ov_opset.constant(1, dtype=Type.i32).output(0)
    topk_min = ov_opset.gather(
        topk_values, k_minus_1_idx, topk_values_axis
    ).output(0)

    # Gather the prediction score at each true class index → shape (batch, 1)
    targets_values = take_along_axis(predictions, targets, axis=-1)
    # target score >= k-th largest score means it belongs in the top-k
    mask = ov_opset.greater_equal(targets_values, topk_min).output(0)
    return OpenVINOKerasTensor(mask)


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


def _dft(x, axes_offsets, inverse=False):
    """Shared helper for fft, fft2, and ifft2.

    Args:
        x: Tuple of (real, imag) KerasTensors.
        axes_offsets: List of negative axis offsets relative to the
            complex-data rank (e.g. [-2] for fft, [-3, -2] for fft2/ifft2).
        inverse: If True, use ov_opset.idft; otherwise use ov_opset.dft.
    """
    ori_dtype = x[0].dtype
    x0 = cast(x[0], "float32") if ori_dtype == "float64" else x[0]
    x1 = cast(x[1], "float32") if ori_dtype == "float64" else x[1]

    real = ov_opset.unsqueeze(
        get_ov_output(x0), ov_opset.constant([-1], Type.i32)
    ).output(0)
    imag = ov_opset.unsqueeze(
        get_ov_output(x1), ov_opset.constant([-1], Type.i32)
    ).output(0)
    complex_data = ov_opset.concat([real, imag], -1).output(0)

    rank = len(x[0].shape) + 1
    axes = ov_opset.constant(
        [rank + off for off in axes_offsets], Type.i32
    ).output(0)

    op = ov_opset.idft if inverse else ov_opset.dft
    result = op(complex_data, axes).output(0)

    out_real = ov_opset.gather(
        result, ov_opset.constant(0, Type.i32), ov_opset.constant(-1, Type.i32)
    ).output(0)
    out_imag = ov_opset.gather(
        result, ov_opset.constant(1, Type.i32), ov_opset.constant(-1, Type.i32)
    ).output(0)

    if ori_dtype == "float64":
        out_real = ov_opset.convert(out_real, Type.f64).output(0)
        out_imag = ov_opset.convert(out_imag, Type.f64).output(0)

    return OpenVINOKerasTensor(out_real), OpenVINOKerasTensor(out_imag)


def fft(x):
    # axes_offsets=[-2]: last axis of complex data (rank = input_rank + 1)
    return _dft(x, axes_offsets=[-2], inverse=False)


def fft2(x):
    # axes_offsets=[-3, -2]: two trailing axes of complex data
    return _dft(x, axes_offsets=[-3, -2], inverse=False)


def ifft2(x):
    # Same axes as fft2 but with the inverse DFT
    return _dft(x, axes_offsets=[-3, -2], inverse=True)


def rfft(x, fft_length=None):
    ori_dtype = x.dtype
    x = cast(x, "float32") if x.dtype == "float64" else x

    x_node = get_ov_output(x)
    rank = len(x_node.shape)
    axes = ov_opset.constant([rank - 1], Type.i32).output(0)

    if fft_length is not None:
        signal_size = ov_opset.constant([fft_length], Type.i32).output(0)
        # Pad input if signal_size > input_size (OpenVINO limitation)
        last_dim = x_node.shape[-1]
        if isinstance(last_dim, int) and last_dim < fft_length:
            pad_begin = [0] * rank
            pad_end = [0] * rank
            pad_end[-1] = fft_length - last_dim
            pad_begin_node = ov_opset.constant(pad_begin, Type.i32).output(0)
            pad_end_node = ov_opset.constant(pad_end, Type.i32).output(0)
            x_node = ov_opset.pad(
                x_node, pad_begin_node, pad_end_node, "constant"
            ).output(0)

        rdft = ov_opset.rdft(x_node, axes, signal_size).output(0)
    else:
        rdft = ov_opset.rdft(x_node, axes).output(0)

    out_real = ov_opset.gather(
        rdft, ov_opset.constant(0, Type.i32), ov_opset.constant(-1, Type.i32)
    ).output(0)
    out_imag = ov_opset.gather(
        rdft, ov_opset.constant(1, Type.i32), ov_opset.constant(-1, Type.i32)
    ).output(0)

    if ori_dtype == "float64":
        out_real = ov_opset.convert(out_real, Type.f64).output(0)
        out_imag = ov_opset.convert(out_imag, Type.f64).output(0)

    return OpenVINOKerasTensor(out_real), OpenVINOKerasTensor(out_imag)


def irfft(x, fft_length=None):
    ori_dtype = x[0].dtype
    if ori_dtype == "float64":
        x = (cast(x[0], "float32"), cast(x[1], "float32"))

    complex_data = get_ov_output(stack(x, axis=-1))
    rank = len(complex_data.shape)
    axes = ov_opset.constant([rank - 2], Type.i32).output(0)

    if fft_length is not None:
        signal_size = ov_opset.constant([fft_length], Type.i32).output(0)
        irdft = ov_opset.irdft(complex_data, axes, signal_size).output(0)
    else:
        irdft = ov_opset.irdft(complex_data, axes).output(0)

    if ori_dtype == "float64":
        irdft = ov_opset.convert(irdft, Type.f64).output(0)

    return OpenVINOKerasTensor(irdft)


def stft(
    x, sequence_length, sequence_stride, fft_length, window="hann", center=True
):
    if standardize_dtype(x.dtype) not in {"float32", "float64"}:
        raise TypeError(
            "Invalid input type. Expected `float32` or `float64`. "
            f"Received: input type={x.dtype}"
        )
    if fft_length < sequence_length:
        raise ValueError(
            "`fft_length` must equal or larger than `sequence_length`. "
            f"Received: sequence_length={sequence_length}, "
            f"fft_length={fft_length}"
        )
    if isinstance(window, str):
        if window not in {"hann", "hamming"}:
            raise ValueError(
                "If a string is passed to `window`, it must be one of "
                f'`"hann"`, `"hamming"`. Received: window={window}'
            )

    ori_dtype = x.dtype
    x = get_ov_output(x)

    ori_shape = x.shape
    num_dims = len(ori_shape)

    if num_dims > 2:
        flatten_shape = ov_opset.constant([-1, ori_shape[-1]], Type.i32).output(
            0
        )
        x = ov_opset.reshape(x, flatten_shape, False).output(0)

    if center:
        # pad x with reflect mode
        pad_begin = [0] * len(x.shape)
        pad_end = [0] * len(x.shape)
        pad_begin[-1] = fft_length // 2
        pad_end[-1] = fft_length // 2
        pad_begin_node = ov_opset.constant(pad_begin, Type.i32).output(0)
        pad_end_node = ov_opset.constant(pad_end, Type.i32).output(0)
        x = ov_opset.pad(x, pad_begin_node, pad_end_node, "reflect").output(0)

    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    element_type = x.get_element_type()
    if element_type == Type.f64:
        x = ov_opset.convert(x, Type.f32).output(0)
        element_type = Type.f32

    if window is not None:
        if isinstance(window, str):
            win = scipy.signal.get_window(window, sequence_length)
        else:
            win = window
        if len(win.shape) != 1 or win.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win.shape}"
            )
        win = np.pad(win, [[l_pad, r_pad]])
        win_node = ov_opset.constant(win, element_type).output(0)
    else:
        win = np.ones((sequence_length + l_pad + r_pad))
        win_node = ov_opset.constant(win, element_type).output(0)

    frame_size_node = ov_opset.constant(fft_length, Type.i32).output(0)
    frame_step_node = ov_opset.constant(sequence_stride, Type.i32).output(0)

    stft_node = ov_opset.stft(
        x, win_node, frame_size_node, frame_step_node, transpose_frames=False
    ).output(0)

    out_real = ov_opset.gather(
        stft_node,
        ov_opset.constant(0, Type.i32),
        ov_opset.constant(-1, Type.i32),
    ).output(0)
    out_imag = ov_opset.gather(
        stft_node,
        ov_opset.constant(1, Type.i32),
        ov_opset.constant(-1, Type.i32),
    ).output(0)

    if num_dims > 2:
        target_shape = list(ori_shape[:-1]) + [-1, fft_length // 2 + 1]
        target_shape_node = ov_opset.constant(target_shape, Type.i32).output(0)
        out_real = ov_opset.reshape(out_real, target_shape_node, False).output(
            0
        )
        out_imag = ov_opset.reshape(out_imag, target_shape_node, False).output(
            0
        )

    if ori_dtype == "float64":
        out_real = ov_opset.convert(out_real, Type.f64).output(0)
        out_imag = ov_opset.convert(out_imag, Type.f64).output(0)

    return OpenVINOKerasTensor(out_real), OpenVINOKerasTensor(out_imag)


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
