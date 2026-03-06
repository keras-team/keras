import numpy as np
import openvino.opset15 as ov_opset
import scipy.signal
from openvino import Type

from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import cast
from keras.src.backend.openvino.core import get_ov_output
from keras.src.backend.openvino.core import standardize_dtype
from keras.src.backend.openvino.numpy import stack

INT32_MAX = 2**31 - 1


def _segment_reduction_fn(
    data, segment_ids, reduction_method, num_segments, sorted
):
    data = get_ov_output(data)
    segment_ids = get_ov_output(segment_ids)

    if num_segments is None:
        max_id = ov_opset.reduce_max(
            segment_ids, ov_opset.constant([0], Type.i32), keep_dims=False
        ).output(0)
        num_segments = ov_opset.add(
            max_id, ov_opset.constant(1, max_id.get_element_type())
        ).output(0)
    else:
        num_segments = ov_opset.constant(
            num_segments, segment_ids.get_element_type()
        ).output(0)

    is_negative = ov_opset.less(
        segment_ids, ov_opset.constant(0, segment_ids.get_element_type())
    ).output(0)
    safe_segment_ids = ov_opset.select(
        is_negative, num_segments, segment_ids
    ).output(0)
    indices = ov_opset.unsqueeze(
        safe_segment_ids, ov_opset.constant(-1, Type.i32)
    ).output(0)

    num_segments_plus_1 = ov_opset.add(
        num_segments, ov_opset.constant(1, num_segments.get_element_type())
    ).output(0)

    data_shape = data.get_partial_shape()
    rank = data_shape.rank.get_length() if data_shape.rank.is_static else -1

    if rank > 1:
        data_shape_node = ov_opset.shape_of(data, output_type=Type.i32).output(
            0
        )
        rest_shape = ov_opset.slice(
            data_shape_node,
            start=ov_opset.constant([1], Type.i32),
            stop=ov_opset.constant([2147483647], Type.i32),
            step=ov_opset.constant([1], Type.i32),
            axes=ov_opset.constant([0], Type.i32),
        ).output(0)
        num_seg_node = ov_opset.unsqueeze(
            num_segments_plus_1, ov_opset.constant(0, Type.i32)
        ).output(0)
        buffer_shape = ov_opset.concat(
            [num_seg_node, rest_shape], axis=0
        ).output(0)
    else:
        buffer_shape = ov_opset.unsqueeze(
            num_segments_plus_1, ov_opset.constant(0, Type.i32)
        ).output(0)

    if reduction_method == "max":
        from keras.src.backend.openvino.core import DTYPES_MIN

        data_type = data.get_element_type()
        if data_type.is_real():
            init_val = np.array(-np.inf, dtype=np.float32)
        else:
            init_val = DTYPES_MIN[data_type]
    else:
        init_val = 0

    init_val_node = ov_opset.constant(init_val, data.get_element_type()).output(
        0
    )
    buffer = ov_opset.broadcast(init_val_node, buffer_shape).output(0)

    scattered = ov_opset.scatter_nd_update(
        buffer, indices, data, reduction=reduction_method
    ).output(0)

    start = ov_opset.constant([0], Type.i32).output(0)
    end = ov_opset.unsqueeze(
        num_segments, ov_opset.constant(0, Type.i32)
    ).output(0)
    axes = ov_opset.constant([0], Type.i32).output(0)
    step = ov_opset.constant([1], Type.i32).output(0)
    result = ov_opset.slice(scattered, start, end, step, axes).output(0)

    return OpenVINOKerasTensor(result)


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    return _segment_reduction_fn(data, segment_ids, "sum", num_segments, sorted)


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    return _segment_reduction_fn(data, segment_ids, "max", num_segments, sorted)


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
    x = get_ov_output(x)
    x_shape = x.partial_shape
    ndim = len(x_shape)

    # Define common constants for reuse
    zero_const_1d = ov_opset.constant([0], Type.i32)
    shape_tensor = ov_opset.shape_of(x, output_type=Type.i32).output(0)

    last_idx = ov_opset.constant([ndim - 1], Type.i32)
    axis0 = ov_opset.constant(0, Type.i32)
    signal_len_1d = ov_opset.gather(shape_tensor, last_idx, axis0).output(0)
    signal_len_scalar = ov_opset.squeeze(signal_len_1d, zero_const_1d).output(0)

    minus_one = ov_opset.constant([-1], Type.i32).output(0)
    shape_2d = ov_opset.concat([minus_one, signal_len_1d], axis=0).output(0)
    x_2d = ov_opset.reshape(x, shape_2d, False).output(0)

    seq_len_c = ov_opset.constant(sequence_length, Type.i32).output(0)
    stride_c = ov_opset.constant(sequence_stride, Type.i32).output(0)
    diff = ov_opset.subtract(signal_len_scalar, seq_len_c).output(0)
    num_seq_scalar = ov_opset.add(
        ov_opset.divide(diff, stride_c).output(0),
        ov_opset.constant(1, Type.i32).output(0),
    ).output(0)

    row_stop = ov_opset.multiply(num_seq_scalar, stride_c).output(0)
    row_idx = ov_opset.range(
        ov_opset.constant(0, Type.i32).output(0),
        row_stop,
        stride_c,
        output_type=Type.i32,
    ).output(0)
    row_idx_2d = ov_opset.unsqueeze(
        row_idx, ov_opset.constant([1], Type.i32)
    ).output(0)

    col_idx = ov_opset.constant(
        np.arange(sequence_length, dtype=np.int32)
    ).output(0)
    col_idx_2d = ov_opset.unsqueeze(col_idx, zero_const_1d).output(0)

    indices = ov_opset.add(row_idx_2d, col_idx_2d).output(0)

    gathered = ov_opset.gather(
        x_2d, indices, ov_opset.constant(1, Type.i32)
    ).output(0)

    batch_shape = ov_opset.slice(
        shape_tensor,
        start=zero_const_1d,
        stop=ov_opset.constant([ndim - 1], Type.i32),
        step=ov_opset.constant([1], Type.i32),
        axes=zero_const_1d,
    ).output(0)
    num_seq_1d = ov_opset.unsqueeze(num_seq_scalar, zero_const_1d).output(0)
    seq_len_1d = ov_opset.constant([sequence_length], Type.i32).output(0)
    out_shape = ov_opset.concat(
        [batch_shape, num_seq_1d, seq_len_1d], axis=0
    ).output(0)
    result = ov_opset.reshape(gathered, out_shape, False).output(0)

    return OpenVINOKerasTensor(result)


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


def _overlap_sequences_ov(x, sequence_stride, fft_length):
    """Perform overlap-and-add using OpenVINO ops.

    Takes a tensor x of shape [batch, num_sequences, fft_length] and
    reconstructs a time-domain signal by striding each frame by
    `sequence_stride` and summing the overlapping contributions.

    Returns the reconstructed signal and its length as an OV scalar node.
    """
    nstep = 1 + (fft_length - 1) // sequence_stride
    padded_len = nstep * sequence_stride
    pad_amount = padded_len - fft_length

    zero = ov_opset.constant(0, Type.i32).output(0)

    # Extract dynamic batch size and number of sequences from input shape.
    x_shape = ov_opset.shape_of(x, output_type=Type.i32).output(0)
    flat_batch = ov_opset.gather(
        x_shape, ov_opset.constant(0, Type.i32), zero
    ).output(0)
    num_sequences = ov_opset.gather(
        x_shape, ov_opset.constant(1, Type.i32), zero
    ).output(0)

    # Compute expected output length: (num_sequences - 1) * stride + fft_length
    output_size = ov_opset.add(
        ov_opset.multiply(
            ov_opset.constant(sequence_stride, Type.i32),
            ov_opset.subtract(num_sequences, ov_opset.constant(1, Type.i32)),
        ),
        ov_opset.constant(fft_length, Type.i32),
    ).output(0)

    # Pad each frame along the last axis so its length is a multiple of stride.
    if pad_amount > 0:
        pad_begin = ov_opset.constant([0, 0, 0], Type.i32).output(0)
        pad_end = ov_opset.constant([0, 0, pad_amount], Type.i32).output(0)
        x = ov_opset.pad(x, pad_begin, pad_end, "constant").output(0)

    # Reshape to [batch, num_sequences, nstep, sequence_stride] to expose
    # the overlap structure within each frame.
    overlap_shape = ov_opset.concat(
        [
            ov_opset.unsqueeze(
                flat_batch, ov_opset.constant(0, Type.i32)
            ).output(0),
            ov_opset.unsqueeze(
                num_sequences, ov_opset.constant(0, Type.i32)
            ).output(0),
            ov_opset.constant([nstep, sequence_stride], Type.i32).output(0),
        ],
        0,
    ).output(0)
    x = ov_opset.reshape(x, overlap_shape, False).output(0)

    # Transpose to [batch, nstep, num_sequences, sequence_stride] so that
    # overlapping frames become adjacent along axis 2.
    x = ov_opset.transpose(x, ov_opset.constant([0, 2, 1, 3], Type.i32)).output(
        0
    )

    # Pad num_sequences axis by num_sequences to create interleaved zeros,
    # enabling a flat reshape that places each frame at its strided offset.
    pad_begin_n = ov_opset.constant([0, 0, 0, 0], Type.i32).output(0)
    pad_end_n = ov_opset.concat(
        [
            ov_opset.constant([0, 0], Type.i32).output(0),
            ov_opset.unsqueeze(
                num_sequences, ov_opset.constant(0, Type.i32)
            ).output(0),
            ov_opset.constant([0], Type.i32).output(0),
        ],
        0,
    ).output(0)
    x = ov_opset.pad(x, pad_begin_n, pad_end_n, "constant").output(0)

    # overlapping_dim_size = 2 * num_sequences - 1: the number of distinct
    # stride-aligned positions covered by all frames after interleaving.
    overlapping_dim_size = ov_opset.subtract(
        ov_opset.multiply(ov_opset.constant(2, Type.i32), num_sequences),
        ov_opset.constant(1, Type.i32),
    ).output(0)

    # Flatten to [batch, nstep * 2 * sequence_stride * num_sequences] so the
    # interleaved frame data forms a contiguous sequence.
    total_inner = ov_opset.multiply(
        ov_opset.constant(nstep * 2 * sequence_stride, Type.i32),
        num_sequences,
    ).output(0)
    flatten_shape = ov_opset.concat(
        [
            ov_opset.unsqueeze(
                flat_batch, ov_opset.constant(0, Type.i32)
            ).output(0),
            ov_opset.unsqueeze(
                total_inner, ov_opset.constant(0, Type.i32)
            ).output(0),
        ],
        0,
    ).output(0)
    x = ov_opset.reshape(x, flatten_shape, False).output(0)

    # Slice away the trailing padding introduced by the interleaving step.
    slice_len = ov_opset.multiply(
        overlapping_dim_size,
        ov_opset.constant(nstep * sequence_stride, Type.i32),
    ).output(0)
    x = ov_opset.slice(
        x,
        ov_opset.constant([0], Type.i32).output(0),
        ov_opset.unsqueeze(slice_len, ov_opset.constant(0, Type.i32)).output(0),
        ov_opset.constant([1], Type.i32).output(0),
        ov_opset.constant([1], Type.i32).output(0),
    ).output(0)

    # Reshape to [batch, nstep, overlapping_dim_size * sequence_stride] and
    # reduce-sum over the nstep axis to accumulate overlapping frame values.
    inner_size = ov_opset.multiply(
        overlapping_dim_size, ov_opset.constant(sequence_stride, Type.i32)
    ).output(0)
    sum_shape = ov_opset.concat(
        [
            ov_opset.unsqueeze(
                flat_batch, ov_opset.constant(0, Type.i32)
            ).output(0),
            ov_opset.constant([nstep], Type.i32).output(0),
            ov_opset.unsqueeze(
                inner_size, ov_opset.constant(0, Type.i32)
            ).output(0),
        ],
        0,
    ).output(0)
    x = ov_opset.reshape(x, sum_shape, False).output(0)

    x = ov_opset.reduce_sum(
        x,
        ov_opset.constant([1], Type.i32).output(0),
        keep_dims=False,
    ).output(0)

    # Trim the result to the expected signal length.
    x = ov_opset.slice(
        x,
        ov_opset.constant([0], Type.i32).output(0),
        ov_opset.unsqueeze(output_size, ov_opset.constant(0, Type.i32)).output(
            0
        ),
        ov_opset.constant([1], Type.i32).output(0),
        ov_opset.constant([1], Type.i32).output(0),
    ).output(0)

    return x, output_size


def istft(
    x,
    sequence_length,
    sequence_stride,
    fft_length,
    length=None,
    window="hann",
    center=True,
):
    if isinstance(window, str):
        if window not in {"hann", "hamming"}:
            raise ValueError(
                "If a string is passed to `window`, it must be one of "
                f'`"hann"`, `"hamming"`. Received: window={window}'
            )

    ori_dtype = x[0].dtype

    x0 = get_ov_output(x[0])
    ori_partial_shape = x0.get_partial_shape()
    num_dims = ori_partial_shape.rank.get_length()
    ori_shape_list = [
        None if dim.is_dynamic else dim.get_length()
        for dim in ori_partial_shape
    ]

    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    if window is not None:
        if isinstance(window, str):
            win = scipy.signal.get_window(window, sequence_length)
        else:
            win = np.asarray(window, dtype=np.float64)
        if len(win.shape) != 1 or win.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win.shape}"
            )
        win = np.pad(win, [[l_pad, r_pad]])

        denom = np.square(win)
        overlaps = -(-fft_length // sequence_stride)
        denom = np.pad(denom, [(0, overlaps * sequence_stride - fft_length)])
        denom = denom.reshape([overlaps, sequence_stride])
        denom = denom.sum(axis=0, keepdims=True)
        denom = np.tile(denom, [overlaps, 1])
        denom = denom.reshape([overlaps * sequence_stride])
        win = win / denom[:fft_length]
    else:
        win = None

    frames = irfft(x, fft_length)
    frames = get_ov_output(frames)

    element_type = frames.get_element_type()
    if element_type == Type.f64:
        frames = ov_opset.convert(frames, Type.f32).output(0)
        element_type = Type.f32

    if win is not None:
        win_node = ov_opset.constant(win.astype(np.float32), Type.f32).output(0)
        if element_type != Type.f32:
            win_node = ov_opset.convert(win_node, element_type).output(0)
        frames = ov_opset.multiply(frames, win_node).output(0)

    if num_dims == 2:
        frames = ov_opset.unsqueeze(
            frames, ov_opset.constant(0, Type.i32)
        ).output(0)
    elif num_dims > 2:
        frames_shp = ov_opset.shape_of(frames, output_type=Type.i32).output(0)
        num_seq_node = ov_opset.gather(
            frames_shp,
            ov_opset.constant(num_dims - 2, Type.i32),
            ov_opset.constant(0, Type.i32),
        ).output(0)
        flatten_shp = ov_opset.concat(
            [
                ov_opset.constant([-1], Type.i32).output(0),
                ov_opset.unsqueeze(
                    num_seq_node, ov_opset.constant(0, Type.i32)
                ).output(0),
                ov_opset.constant([fft_length], Type.i32).output(0),
            ],
            0,
        ).output(0)
        frames = ov_opset.reshape(frames, flatten_shp, False).output(0)

    frames, output_size = _overlap_sequences_ov(
        frames, sequence_stride, fft_length
    )

    start_val = fft_length // 2 if center else 0

    if length is not None:
        frames = ov_opset.slice(
            frames,
            ov_opset.constant([start_val], Type.i32).output(0),
            ov_opset.constant([start_val + length], Type.i32).output(0),
            ov_opset.constant([1], Type.i32).output(0),
            ov_opset.constant([1], Type.i32).output(0),
        ).output(0)
    else:
        if start_val > 0:
            frames = ov_opset.slice(
                frames,
                ov_opset.constant([start_val], Type.i32).output(0),
                ov_opset.constant([INT32_MAX], Type.i32).output(0),
                ov_opset.constant([1], Type.i32).output(0),
                ov_opset.constant([1], Type.i32).output(0),
            ).output(0)
        if center:
            cur_len = ov_opset.gather(
                ov_opset.shape_of(frames, output_type=Type.i32).output(0),
                ov_opset.constant(1, Type.i32),
                ov_opset.constant(0, Type.i32),
            ).output(0)
            end_node = ov_opset.subtract(
                cur_len,
                ov_opset.constant(fft_length // 2, Type.i32),
            ).output(0)
            frames = ov_opset.slice(
                frames,
                ov_opset.constant([0], Type.i32).output(0),
                ov_opset.unsqueeze(
                    end_node, ov_opset.constant(0, Type.i32)
                ).output(0),
                ov_opset.constant([1], Type.i32).output(0),
                ov_opset.constant([1], Type.i32).output(0),
            ).output(0)

    if num_dims == 2:
        frames = ov_opset.squeeze(
            frames, ov_opset.constant([0], Type.i32)
        ).output(0)
    elif num_dims > 2:
        batch_dims = ori_shape_list[:-2]
        target_shape = [d if d is not None else -1 for d in batch_dims] + [-1]
        target_shape_node = ov_opset.constant(target_shape, Type.i32).output(0)
        frames = ov_opset.reshape(frames, target_shape_node, False).output(0)

    if ori_dtype == "float64":
        frames = ov_opset.convert(frames, Type.f64).output(0)

    return OpenVINOKerasTensor(frames)


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
    # TODO: Float64 infinity values are clamped on CPU backend,
    # breaking erfinv(±1) = ±inf
    # See https://github.com/openvinotoolkit/openvino/issues/34138
    # Tests excluded: test_erfinv_operation_basic, test_erfinv_operation_dtype
    x = get_ov_output(x)
    dtype = x.get_element_type()

    a = 0.147
    two_over_pi_a = 2.0 / (np.pi * a)
    two_over_sqrt_pi = 2.0 / np.sqrt(np.pi)

    one = ov_opset.constant(1.0, dtype).output(0)
    half = ov_opset.constant(0.5, dtype).output(0)

    x_sq = ov_opset.multiply(x, x).output(0)
    log_term = ov_opset.log(ov_opset.subtract(one, x_sq).output(0)).output(0)

    k = ov_opset.add(
        ov_opset.constant(two_over_pi_a, dtype).output(0),
        ov_opset.multiply(half, log_term).output(0),
    ).output(0)

    inner = ov_opset.subtract(
        ov_opset.multiply(k, k).output(0),
        ov_opset.multiply(
            ov_opset.constant(1.0 / a, dtype).output(0), log_term
        ).output(0),
    ).output(0)

    y0 = ov_opset.multiply(
        ov_opset.sign(x).output(0),
        ov_opset.sqrt(
            ov_opset.subtract(ov_opset.sqrt(inner).output(0), k).output(0)
        ).output(0),
    ).output(0)

    erf_err = ov_opset.subtract(ov_opset.erf(y0).output(0), x).output(0)

    y0_sq = ov_opset.multiply(y0, y0).output(0)
    exp_term = ov_opset.exp(ov_opset.negative(y0_sq).output(0)).output(0)
    deriv = ov_opset.multiply(
        ov_opset.constant(two_over_sqrt_pi, dtype).output(0),
        exp_term,
    ).output(0)
    y1 = ov_opset.subtract(
        y0, ov_opset.divide(erf_err, deriv).output(0)
    ).output(0)

    return OpenVINOKerasTensor(y1)
