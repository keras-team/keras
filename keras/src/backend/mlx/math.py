import math

import mlx.core as mx

from keras.src.backend import standardize_dtype
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.backend.mlx.linalg import det
from keras.src.utils.module_utils import scipy


def _segment_reduction_fn(
    data, segment_ids, reduction_method, num_segments, sorted
):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids)

    if data.dtype == mx.int64:
        # GPU scatter does not yet support int64 for the input or updates.
        data = data.astype(mx.int32)

    if num_segments is None:
        num_segments = mx.max(segment_ids) + 1

    mask = segment_ids >= 0
    # pack segment_ids < 0 into index 0 and then handle below
    safe_segment_ids = mx.where(mask, segment_ids, 0)

    if not sorted:
        sort_indices = mx.argsort(safe_segment_ids)
        safe_segment_ids = mx.take(safe_segment_ids, sort_indices)
        data = mx.take(data, sort_indices, axis=0)
        mask = mx.take(mask, sort_indices)

    # expand mask dimensions to match data dimensions
    for i in range(1, len(data.shape)):
        mask = mx.expand_dims(mask, axis=i)

    data_shape = list(data.shape)
    data_shape[0] = num_segments

    if reduction_method == "max":
        masked_data = mx.where(mask, data, -mx.inf)
        result = mx.ones(data_shape, dtype=data.dtype) * -mx.inf
        result = result.at[safe_segment_ids].maximum(masked_data)
    else:  # sum
        masked_data = mx.where(mask, data, 0)
        result = mx.zeros(data_shape, dtype=data.dtype)
        result = result.at[safe_segment_ids].add(masked_data)

    return result


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    return _segment_reduction_fn(data, segment_ids, "sum", num_segments, sorted)


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    return _segment_reduction_fn(data, segment_ids, "max", num_segments, sorted)


def top_k(x, k, sorted=True):
    # default to sorted=True to match other backends
    x = convert_to_tensor(x)
    indices = mx.argpartition(mx.negative(x), k, axis=-1)[..., :k]
    values = mx.take_along_axis(x, indices, axis=-1)
    
    if sorted:
        sort_indices = mx.argsort(mx.negative(values), axis=-1)
        values = mx.take_along_axis(values, sort_indices, axis=-1)
        indices = mx.take_along_axis(indices, sort_indices, axis=-1)
    
    return values, indices


def in_top_k(targets, predictions, k):
    targets = convert_to_tensor(targets)
    predictions = convert_to_tensor(predictions)
    targets = targets[..., None]
    topk_values = top_k(predictions, k, sorted=False)[0]
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


def ifft2(x):
    x = _get_complex_tensor_from_tuple(x)
    complex_output = mx.fft.ifft2(x)
    return mx.real(complex_output), mx.imag(complex_output)


def rfft(x, fft_length=None):
    x = convert_to_tensor(x)
    complex_output = mx.fft.rfft(x, n=fft_length)
    return mx.real(complex_output), mx.imag(complex_output)


def irfft(x, fft_length=None):
    x = _get_complex_tensor_from_tuple(x)
    real_output = mx.fft.irfft(x, n=fft_length)
    return real_output


def _create_sliding_windows(x, window_size, step):
    batch_size, signal_length, _ = x.shape
    num_windows = (signal_length - window_size) // step + 1

    # create indices for all windows
    indices = mx.arange(window_size)
    window_starts = mx.arange(num_windows) * step

    # create a mesh of indices for all windows
    # shape: (num_windows, window_size)
    indices_mesh = indices[None, :] + window_starts[:, None]

    batch_idx = mx.arange(batch_size)[:, None, None]
    indices_mesh = indices_mesh[None, :, :]

    return x[batch_idx, indices_mesh]


def _stft(x, window, nperseg, noverlap, nfft, axis=-1):
    # Ref: jax.scipy.signal.stft
    axis = canonicalize_axis(axis, x.ndim)
    result_dtype = mx.complex64

    if x.size == 0:
        return mx.zeros(x.shape, result_dtype)

    x = mx.moveaxis(x, axis, -1)

    if nfft < nperseg:
        raise ValueError("nfft must be greater than or equal to nperseg.")
    if noverlap >= nperseg:
        raise ValueError("noverlap must be less than nperseg.")

    *batch_shape, signal_length = x.shape
    if nperseg == 1 and noverlap == 0:
        result = x[..., None]
    else:
        step = nperseg - noverlap
        batch_shape = list(batch_shape)
        x = x.reshape((math.prod(batch_shape), signal_length, 1))

        result = _create_sliding_windows(x, nperseg, step)
        result = result.reshape(*batch_shape, result.shape[1], result.shape[2])

    win_shape = (1,) * len(batch_shape) + (1, nperseg)
    result = window.reshape(win_shape) * result
    result = mx.fft.rfft(mx.real(result), n=nfft, axis=-1)

    result *= mx.sqrt(1.0 / window.sum() ** 2)
    result = result.astype(result_dtype)
    result = mx.moveaxis(result, -1, axis)
    return result


def _reflect_pad(x, pad_width, axis=-1):
    left_pad, right_pad = pad_width

    if left_pad > 0:
        indices = mx.arange(1, left_pad + 1, dtype=mx.int32)[::-1]
        prefix = mx.take(x, indices, axis=axis)
    else:
        prefix = None

    if right_pad > 0:
        indices = mx.arange(
            x.shape[axis] - 2, x.shape[axis] - right_pad - 2, -1, dtype=mx.int32
        )
        suffix = mx.take(x, indices, axis=axis)
    else:
        suffix = None

    if prefix is not None and suffix is not None:
        return mx.concatenate([prefix, x, suffix], axis=axis)
    elif prefix is not None:
        return mx.concatenate([prefix, x], axis=axis)
    elif suffix is not None:
        return mx.concatenate([x, suffix], axis=axis)
    return x


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
    x = convert_to_tensor(x)

    if center:
        pad_width = (fft_length // 2, fft_length // 2)
        x = _reflect_pad(x, pad_width)

    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    if window is not None:
        if isinstance(window, str):
            # use scipy window for now to match precision with jax
            win = convert_to_tensor(
                scipy.signal.get_window(window, sequence_length), dtype=x.dtype
            )
        else:
            win = convert_to_tensor(window, dtype=x.dtype)
        if len(win.shape) != 1 or win.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win.shape}"
            )
        win = mx.pad(win, [[l_pad, r_pad]])
    else:
        win = mx.ones((sequence_length + l_pad + r_pad), dtype=x.dtype)

    result = _stft(
        x,
        window=win,
        nperseg=(sequence_length + l_pad + r_pad),
        noverlap=(sequence_length + l_pad + r_pad - sequence_stride),
        nfft=fft_length,
    )
    scale = mx.sqrt(1.0 / win.sum() ** 2)
    result = result / scale
    result = mx.swapaxes(result, -2, -1)
    return mx.real(result), mx.imag(result)


def _overlap_and_add(x, step_size):
    # Ref: jax.scipy.signal.istft
    """Utility function compatible with tf.signal.overlap_and_add.

    Args:
        x: An array with `(..., frames, frame_length)`-shape.
        step_size: An integer denoting overlap offsets. Must be less than
        `frame_length`.

    Returns:
        An array with `(..., output_size)`-shape containing overlapped signal.
    """
    if x.ndim < 2:
        raise ValueError("Input must have (..., frames, frame_length) shape.")

    *batch_shape, nframes, segment_len = x.shape
    flat_batchsize = math.prod(batch_shape)
    x = x.reshape((flat_batchsize, nframes, segment_len))
    output_size = step_size * (nframes - 1) + segment_len
    nstep_per_segment = 1 + (segment_len - 1) // step_size

    # Here, we use shorter notation for axes.
    # B: batch_size, N: nframes, S: nstep_per_segment,
    # T: segment_len divided by S

    padded_segment_len = nstep_per_segment * step_size
    x = mx.pad(x, ((0, 0), (0, 0), (0, padded_segment_len - segment_len)))
    x = x.reshape((flat_batchsize, nframes, nstep_per_segment, step_size))

    # For obtaining shifted signals, this routine reinterprets flattened
    # array with a shrinked axis.  With appropriate truncation/ padding,
    # this operation pushes the last padded elements of the previous row
    # to the head of the current row.
    # See implementation of `overlap_and_add` in Tensorflow for details.
    x = x.transpose((0, 2, 1, 3))  # x: (B, S, N, T)
    x = mx.pad(x, ((0, 0), (0, 0), (0, nframes), (0, 0)))  # x: (B, S, N*2, T)
    shrinked = x.shape[2] - 1
    x = x.reshape((flat_batchsize, -1))
    x = x[:, : (nstep_per_segment * shrinked * step_size)]
    x = x.reshape((flat_batchsize, nstep_per_segment, shrinked * step_size))

    # Finally, sum shifted segments, and truncate results to the output_size.
    x = x.sum(axis=1)[:, :output_size]
    return x.reshape(tuple(batch_shape) + (-1,))


def _istft(
    Zxx,
    window,
    nperseg,
    noverlap,
    nfft,
    time_axis=-1,
    freq_axis=-2,
):
    # Ref: jax.scipy.signal.istft
    if Zxx.ndim < 2:
        raise ValueError("Input stft must be at least 2d!")
    freq_axis = canonicalize_axis(freq_axis, Zxx.ndim)
    time_axis = canonicalize_axis(time_axis, Zxx.ndim)

    if freq_axis == time_axis:
        raise ValueError("Must specify differing time and frequency axes!")
    if nperseg < 1:
        raise ValueError("nperseg must be a positive integer")
    if nfft < nperseg:
        raise ValueError(
            f"FFT length ({nfft}) must be longer than nperseg ({nperseg})."
        )
    if noverlap >= nperseg:
        raise ValueError("noverlap must be less than nperseg.")

    nstep = nperseg - noverlap

    # Rearrange axes if necessary
    if time_axis != Zxx.ndim - 1 or freq_axis != Zxx.ndim - 2:
        outer_idxs = tuple(
            idx for idx in range(Zxx.ndim) if idx not in {time_axis, freq_axis}
        )
        Zxx = mx.transpose(Zxx, outer_idxs + (freq_axis, time_axis))

    # Perform IFFT
    xsubs = mx.fft.irfft(Zxx, axis=-2, n=nfft)[..., :nperseg, :]

    # Assumes window is already an array
    if len(window.shape) != 1:
        raise ValueError("window must be 1-D")
    if window.shape[0] != nperseg:
        raise ValueError(f"window must have length of {nperseg}")
    xsubs *= window.sum()  # This takes care of the 'spectrum' scaling

    # make window broadcastable over xsubs
    window = mx.expand_dims(window, (*range(xsubs.ndim - 2), -1))
    x = _overlap_and_add((xsubs * window).swapaxes(-2, -1), nstep)
    window_squared = mx.repeat((window * window), xsubs.shape[-1], axis=-1)
    norm = _overlap_and_add(window_squared.swapaxes(-2, -1), nstep)

    x /= mx.where(norm > 1e-10, norm, 1.0)

    # Put axes back
    if x.ndim > 1:
        if time_axis != Zxx.ndim - 1:
            if freq_axis < time_axis:
                time_axis -= 1
            x = mx.moveaxis(x, -1, time_axis)

    return x


def istft(
    x,
    sequence_length,
    sequence_stride,
    fft_length,
    length=None,
    window="hann",
    center=True,
):
    x = _get_complex_tensor_from_tuple(x)
    dtype = mx.real(x).dtype

    if len(x.shape) < 2:
        raise ValueError(
            f"Input `x` must have at least 2 dimensions. "
            f"Received shape: {x.shape}"
        )

    expected_output_len = fft_length + sequence_stride * (x.shape[-2] - 1)
    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    if window is not None:
        if isinstance(window, str):
            win = convert_to_tensor(
                scipy.signal.get_window(window, sequence_length), dtype=dtype
            )
        else:
            win = convert_to_tensor(window, dtype=dtype)
        if len(win.shape) != 1 or win.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win.shape}"
            )
        win = mx.pad(win, [[l_pad, r_pad]])
    else:
        win = mx.ones((sequence_length + l_pad + r_pad), dtype=dtype)

    x = _istft(
        x,
        window=win,
        nperseg=(sequence_length + l_pad + r_pad),
        noverlap=(sequence_length + l_pad + r_pad - sequence_stride),
        nfft=fft_length,
        time_axis=-2,
        freq_axis=-1,
    )

    # scale
    x = x / win.sum() if window is not None else x / sequence_stride

    start = 0 if center is False else fft_length // 2
    if length is not None:
        end = start + length
    elif center is True:
        end = -(fft_length // 2)
    else:
        end = expected_output_len
    return x[..., start:end]


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


def logdet(x):
    x = convert_to_tensor(x)
    det_x = det(x)
    return mx.log(det_x)
