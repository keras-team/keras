import functools
import math

import paddle

from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.paddle.core import convert_to_tensor
from keras.src.backend.paddle.core import needs_reduced_precision_upcast
from keras.src.backend.paddle.core import to_paddle_dtype


def _segment_reduction_fn(data, segment_ids, reduction, num_segments):
    segment_ids = convert_to_tensor(segment_ids, dtype="int64")
    if num_segments is None:
        num_segments = int(paddle.max(segment_ids).item()) + 1

    if reduction == "amax":
        reduce_op = lambda x: paddle.max(x, axis=0)
        initial_value = -float("Inf")
    elif reduction == "amin":
        reduce_op = lambda x: paddle.min(x, axis=0)
        initial_value = float("Inf")
    elif reduction == "prod":
        reduce_op = lambda x: paddle.prod(x, axis=0)
        initial_value = 1.0
    else:
        reduce_op = lambda x: paddle.sum(x, axis=0)
        initial_value = 0.0

    data_f = data.cast("float32")
    flat_data = data_f.reshape([-1] + list(data.shape[1:]))
    flat_ids = segment_ids.flatten()

    outputs = []
    for i in range(num_segments):
        mask = flat_ids == i
        if paddle.any(mask).item():
            outputs.append(reduce_op(flat_data[mask]))
        else:
            outputs.append(
                paddle.full(data.shape[1:], initial_value, dtype="float32")
            )

    return paddle.stack(outputs, axis=0).cast(data.dtype)


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids, dtype="int64")
    if num_segments is None:
        num_segments = int(paddle.max(segment_ids).item()) + 1
    # Redirect negative and out-of-range segment ids to an extra bucket
    # (index `num_segments`), which is discarded before returning the result.
    valid = paddle.logical_and(segment_ids >= 0, segment_ids < num_segments)
    segment_ids = paddle.where(
        valid, segment_ids, paddle.full_like(segment_ids, num_segments)
    )
    zeros = paddle.zeros(
        [num_segments + 1] + list(data.shape[1:]), dtype=data.dtype
    )
    out = paddle.scatter_nd_add(zeros, segment_ids.unsqueeze(-1), data)
    return out[:num_segments]


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids)
    return _segment_reduction_fn(data, segment_ids, "amax", num_segments)


def segment_min(data, segment_ids, num_segments=None, sorted=False):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids)
    return _segment_reduction_fn(data, segment_ids, "amin", num_segments)


def segment_prod(data, segment_ids, num_segments=None, sorted=False):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids)
    return _segment_reduction_fn(data, segment_ids, "prod", num_segments)


def top_k(x, k, sorted=True):
    x = convert_to_tensor(x)
    return paddle.topk(x, k, sorted=sorted)


def in_top_k(targets, predictions, k):
    targets = convert_to_tensor(targets).cast("int64")
    targets = targets.unsqueeze(-1)
    predictions = convert_to_tensor(predictions)
    topk_values = paddle.topk(predictions, k)[0]
    targets_values = paddle.take_along_axis(predictions, targets, axis=-1)
    mask = targets_values >= topk_values
    return paddle.any(mask, axis=-1)


def logsumexp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    # `paddle.logsumexp` only accepts inputs of rank <= 4 and returns NaN
    # when a whole reduced slice is `-inf`, so always use the numerically
    # stable max-shift formulation instead.
    orig_dtype = x.dtype
    # The computation relies on `max`/`exp`, which have no CPU kernels for
    # float16/bfloat16.
    upcast = needs_reduced_precision_upcast(x)
    if upcast:
        x = x.cast("float32")
    if axis is None:
        reduce_axis = list(range(len(x.shape)))
    elif isinstance(axis, (tuple, list)):
        reduce_axis = [a if a >= 0 else a + len(x.shape) for a in axis]
    else:
        reduce_axis = [axis if axis >= 0 else axis + len(x.shape)]
    max_x = paddle.max(x, axis=reduce_axis, keepdim=True)
    # Replace non-finite maxima with 0 so that `x - max_x` cannot produce
    # NaN when a whole slice is -inf.
    max_x = paddle.where(
        paddle.isfinite(max_x), max_x, paddle.zeros_like(max_x)
    )
    shifted = x - max_x
    exp_shifted = paddle.exp(shifted)
    # Paddle's CPU `exp` kernel clamps subnormal results to the smallest
    # normal float instead of flushing them to zero, which would turn the
    # log-sum-exp of a fully masked (all -inf) slice into a finite number.
    exp_shifted = paddle.where(
        shifted == float("-inf"),
        paddle.zeros_like(exp_shifted),
        exp_shifted,
    )
    result = (
        paddle.log(paddle.sum(exp_shifted, axis=reduce_axis, keepdim=True))
        + max_x
    )
    if not keepdims:
        result = paddle.squeeze(result, axis=reduce_axis)
    if upcast:
        result = result.cast(orig_dtype)
    return result


def qr(x, mode="reduced"):
    x = convert_to_tensor(x)
    return paddle.linalg.qr(x, mode=mode)


def extract_sequences(x, sequence_length, sequence_stride):
    x = convert_to_tensor(x)
    *batch_shape, signal_length = x.shape
    need_squeeze = False
    if not batch_shape:
        x = x.unsqueeze(0)
        need_squeeze = True
    elif len(batch_shape) > 1:
        any_dynamic = any(not isinstance(d, int) for d in batch_shape)
        flat = -1 if any_dynamic else math.prod(batch_shape)
        x = x.reshape([flat, signal_length])
    # frame returns [batch, frame_length, num_frames]
    x = paddle.signal.frame(
        x, frame_length=sequence_length, hop_length=sequence_stride
    )
    # transpose to [batch, num_frames, frame_length]
    x = paddle.transpose(x, [0, 2, 1])
    num_frames = x.shape[1]
    if need_squeeze:
        x = x.squeeze(0)
    elif len(batch_shape) > 1:
        x = x.reshape([*batch_shape, num_frames, sequence_length])
    return x


def _overlap_sequences(x, sequence_stride):
    x = convert_to_tensor(x)
    *batch_shape, num_sequences, sequence_length = x.shape
    if sequence_stride > sequence_length:
        raise ValueError(
            "`sequence_stride` must equal or less than x.shape[-1]. "
            f"Received: sequence_stride={sequence_stride}, "
            f"x.shape[-1]={sequence_length}"
        )
    if sequence_stride < (sequence_length / num_sequences):
        raise ValueError(
            "`sequence_stride` must equal or greater than "
            "x.shape[-1] / x.shape[-2]. "
            f"Received: sequence_stride={sequence_stride}, "
            f"x.shape[-1]={sequence_length}, x.shape[-2]={num_sequences}"
        )
    any_dynamic = any(not isinstance(d, int) for d in batch_shape)
    flat_batchsize = -1 if any_dynamic else math.prod(batch_shape)
    x = paddle.reshape(x, (flat_batchsize, num_sequences, sequence_length))
    output_size = sequence_stride * (num_sequences - 1) + sequence_length
    nstep_per_segment = 1 + (sequence_length - 1) // sequence_stride
    padded_segment_len = nstep_per_segment * sequence_stride
    # Paddle's `F.pad` with a 2*ndim list pads from the first dimension to
    # the last one, so the last spatial dimension comes last in the list.
    x = paddle.nn.functional.pad(
        x, [0, 0, 0, 0, 0, padded_segment_len - sequence_length]
    )
    x = paddle.reshape(
        x, (flat_batchsize, num_sequences, nstep_per_segment, sequence_stride)
    )
    x = paddle.transpose(x, [0, 2, 1, 3])
    x = paddle.nn.functional.pad(x, [0, 0, 0, 0, 0, num_sequences, 0, 0])
    shrinked = x.shape[2] - 1
    x = paddle.reshape(x, (flat_batchsize, -1))
    x = x[:, : (nstep_per_segment * shrinked * sequence_stride)]
    x = paddle.reshape(
        x, (flat_batchsize, nstep_per_segment, shrinked * sequence_stride)
    )
    x = paddle.sum(x, axis=1)[:, :output_size]
    return paddle.reshape(x, tuple(batch_shape) + (-1,))


def _get_complex_tensor_from_tuple(x):
    if not isinstance(x, (tuple, list)) or len(x) != 2:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            f"Received: x={x}"
        )
    real, imag = x
    real = convert_to_tensor(real)
    imag = convert_to_tensor(imag)
    if real.shape != imag.shape:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            "Both the real and imaginary parts should have the same shape. "
            f"Received: x[0].shape = {real.shape}, x[1].shape = {imag.shape}"
        )
    if not real.is_floating_point() or not imag.is_floating_point():
        raise ValueError(
            "At least one tensor in input `x` is not of type float."
            f"Received: x={x}."
        )
    return paddle.complex(real, imag)


def fft(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = paddle.fft.fft(complex_input)
    return complex_output.real(), complex_output.imag()


def fft2(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = paddle.fft.fft2(complex_input)
    return complex_output.real(), complex_output.imag()


def ifft2(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = paddle.fft.ifft2(complex_input)
    return complex_output.real(), complex_output.imag()


def rfft(x, fft_length=None):
    x = convert_to_tensor(x)
    complex_output = paddle.fft.rfft(x, n=fft_length, axis=-1, norm="backward")
    return complex_output.real(), complex_output.imag()


def irfft(x, fft_length=None):
    complex_input = _get_complex_tensor_from_tuple(x)
    return paddle.fft.irfft(
        complex_input, n=fft_length, axis=-1, norm="backward"
    )


def _get_window(name, length, dtype):
    """Generate window tensor using paddle.audio."""
    from paddle.audio.functional import get_window

    try:
        return get_window(name, length, dtype=dtype)
    except (ValueError, RuntimeError):
        raise ValueError(
            "If a string is passed to `window`, it must be one of "
            f'`"hann"`, `"hamming"`. Received: window={name}'
        )


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
    x = convert_to_tensor(x)

    if window is not None:
        if isinstance(window, str):
            win = _get_window(window, sequence_length, x.dtype)
        else:
            win = convert_to_tensor(window, dtype=x.dtype)
        if len(win.shape) != 1 or win.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win.shape}"
            )
    else:
        win = paddle.ones([sequence_length], dtype=x.dtype)

    need_unpack = False
    *batch_shape, samples = x.shape
    if len(x.shape) > 2:
        need_unpack = True
        any_dynamic = any(not isinstance(d, int) for d in batch_shape)
        flat_batchsize = -1 if any_dynamic else math.prod(batch_shape)
        x = paddle.reshape(x, (flat_batchsize, samples))

    x = paddle.signal.stft(
        x,
        n_fft=fft_length,
        hop_length=sequence_stride,
        win_length=sequence_length,
        window=win,
        center=center,
    )
    if need_unpack:
        fft_unique_bins, num_sequences = x.shape[-2:]
        x = paddle.reshape(x, (*batch_shape, fft_unique_bins, num_sequences))

    x = paddle.transpose(x, [*range(len(x.shape) - 2), -1, -2])
    return x.real(), x.imag()


def istft(
    x,
    sequence_length,
    sequence_stride,
    fft_length,
    length=None,
    window="hann",
    center=True,
):
    complex_input = _get_complex_tensor_from_tuple(x)
    dtype = complex_input.real().dtype
    win = None
    if window is not None:
        if isinstance(window, str):
            win = _get_window(window, sequence_length, dtype)
        else:
            win = convert_to_tensor(window, dtype=dtype)
        if len(win.shape) != 1 or win.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win.shape}"
            )

    # `paddle.signal.istft` normalizes by the true window envelope, which
    # does not match the reference implementation at the signal edges, so
    # reconstruct with `irfft` + overlap-add instead.
    x_out = irfft(x, fft_length)

    expected_output_len = fft_length + sequence_stride * (x_out.shape[-2] - 1)

    if win is not None:
        l_pad = (fft_length - sequence_length) // 2
        r_pad = fft_length - sequence_length - l_pad
        from keras.src.backend.paddle.numpy import pad as _pad

        win = _pad(win, [[l_pad, r_pad]], "constant")

        _sequence_length = sequence_length + l_pad + r_pad
        denom = paddle.square(win)
        overlaps = -(-_sequence_length // sequence_stride)
        denom = _pad(
            denom, [[0, overlaps * sequence_stride - _sequence_length]]
        )
        denom = paddle.reshape(denom, [overlaps, sequence_stride])
        denom = paddle.sum(denom, 0, keepdim=True)
        denom = paddle.tile(denom, [overlaps, 1])
        denom = paddle.reshape(denom, [overlaps * sequence_stride])
        win = paddle.divide(win, denom[:_sequence_length])
        x_out = paddle.multiply(x_out, win)

    x_out = _overlap_sequences(x_out, sequence_stride)

    start = 0 if center is False else fft_length // 2
    if length is not None:
        end = start + length
    elif center is True:
        end = expected_output_len - (fft_length // 2)
    else:
        end = expected_output_len
    return x_out[..., start:end]


def _upcast_reduced_precision(fn):
    """Run `fn` in float32 when paddle has no reduced precision kernel."""

    @functools.wraps(fn)
    def wrapper(x, *args, **kwargs):
        x = convert_to_tensor(x)
        if needs_reduced_precision_upcast(x):
            return fn(x.cast("float32"), *args, **kwargs).cast(x.dtype)
        return fn(x, *args, **kwargs)

    return wrapper


@_upcast_reduced_precision
def rsqrt(x):
    x = convert_to_tensor(x)
    return paddle.rsqrt(x)


@_upcast_reduced_precision
def erf(x):
    x = convert_to_tensor(x)
    return paddle.erf(x)


@_upcast_reduced_precision
def erfc(x):
    x = convert_to_tensor(x)
    return 1.0 - paddle.erf(x)


@_upcast_reduced_precision
def erfinv(x):
    x = convert_to_tensor(x)
    result = paddle.erfinv(x)
    # paddle returns ±inf for |x|>=1, JAX returns nan for |x|>1, ±inf for ±1
    gt_mask = x > 1.0
    lt_mask = x < -1.0
    result = paddle.where(
        gt_mask, paddle.full_like(result, float("nan")), result
    )
    result = paddle.where(
        lt_mask, paddle.full_like(result, float("nan")), result
    )
    return result


def logdet(x):
    x = convert_to_tensor(x)
    return paddle.log(paddle.linalg.det(x))


def cdist(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError("`cdist` inputs must have rank >= 2")
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("Last dimension of inputs to `cdist` must match")
    result_dtype = dtypes.result_type(
        standardize_dtype(x.dtype), standardize_dtype(y.dtype), float
    )
    # Integer inputs have to be accumulated in floating point, and
    # `subtract`/`multiply` have no float16 or bfloat16 CPU kernel.
    compute_dtype = result_dtype
    if compute_dtype not in ("float32", "float64"):
        compute_dtype = "float32"
    x = x.cast(to_paddle_dtype(compute_dtype))
    y = y.cast(to_paddle_dtype(compute_dtype))
    diff = paddle.unsqueeze(x, axis=-2) - paddle.unsqueeze(y, axis=-3)
    dist = paddle.sqrt(paddle.sum(diff * diff, axis=-1))
    return dist.cast(to_paddle_dtype(result_dtype))
