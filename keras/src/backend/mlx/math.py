import mlx.core as mx
import numpy as np

from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.mlx.core import _cast
from keras.src.backend.mlx.core import _mlx_dtype
from keras.src.backend.mlx.core import convert_to_numpy
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.utils.module_utils import scipy


def _to_np(x):
    if hasattr(x, "dtype") and "mlx" in str(type(x)):
        return np.asarray(convert_to_numpy(x))
    return np.asarray(x)


def _segment_reduction_fn(
    data, segment_ids, reduction_method, num_segments, sorted
):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids).astype(mx.int32)
    if num_segments is None:
        num_segments = (
            int(np.amax(np.asarray(convert_to_numpy(segment_ids)))) + 1
        )

    # MLX has no boolean-mask indexing. Route rows whose `segment_id == -1`
    # (ignored per Keras semantics) into a scratch bucket past the end, then
    # slice it off. `arr.at[idx].<op>` accumulates over duplicate indices, so
    # the result is correct regardless of whether `segment_ids` is sorted.
    safe_ids = mx.where(segment_ids >= 0, segment_ids, num_segments)
    out_shape = (num_segments + 1,) + tuple(data.shape[1:])

    if reduction_method == "max":
        result = mx.full(out_shape, -np.inf, dtype=data.dtype)
        result = result.at[safe_ids].maximum(data)
    elif reduction_method == "min":
        result = mx.full(out_shape, np.inf, dtype=data.dtype)
        result = result.at[safe_ids].minimum(data)
    elif reduction_method == "prod":
        result = mx.ones(out_shape, dtype=data.dtype)
        result = result.at[safe_ids].multiply(data)
    else:  # sum
        result = mx.zeros(out_shape, dtype=data.dtype)
        result = result.at[safe_ids].add(data)
    return result[:num_segments]


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    return _segment_reduction_fn(data, segment_ids, "sum", num_segments, sorted)


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    return _segment_reduction_fn(data, segment_ids, "max", num_segments, sorted)


def segment_min(data, segment_ids, num_segments=None, sorted=False):
    return _segment_reduction_fn(data, segment_ids, "min", num_segments, sorted)


def segment_prod(data, segment_ids, num_segments=None, sorted=False):
    return _segment_reduction_fn(
        data, segment_ids, "prod", num_segments, sorted
    )


def top_k(x, k, sorted=True):
    x = convert_to_tensor(x)
    if sorted:
        # Take the k largest values.
        sorted_indices = mx.argsort(x, axis=-1)[..., ::-1]
        sorted_values = mx.take_along_axis(x, sorted_indices, axis=-1)
        top_k_values = sorted_values[..., :k]
        top_k_indices = sorted_indices[..., :k]
    else:
        # Partition the array such that all values larger than the k-th
        # largest value are to the right of it.
        top_k_indices = mx.argpartition(x, -k, axis=-1)[..., -k:]
        top_k_values = mx.take_along_axis(x, top_k_indices, axis=-1)
    return top_k_values, _cast(top_k_indices, "int32")


def in_top_k(targets, predictions, k):
    targets = convert_to_tensor(targets).astype(mx.int32)
    predictions = convert_to_tensor(predictions)
    topk_values = top_k(predictions, k)[0]
    targets_values = mx.take_along_axis(predictions, targets[:, None], axis=-1)
    mask = targets_values >= topk_values
    return mx.any(mask, axis=-1)


def logsumexp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype == "bool":
        x = _cast(x, "int32")
    return mx.logsumexp(x, axis=axis, keepdims=keepdims)


def cdist(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    if x.ndim < 2 or y.ndim < 2:
        raise ValueError("`cdist` inputs must have rank >= 2")
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("Last dimension of inputs to `cdist` must match")
    diff = x[..., :, None, :] - y[..., None, :, :]
    return mx.sqrt(mx.sum(diff * diff, axis=-1))


def extract_sequences(x, sequence_length, sequence_stride):
    x = convert_to_tensor(x)
    *batch_shape, seq_len = x.shape
    num_seq = (seq_len - (sequence_length - sequence_stride)) // sequence_stride
    starts = mx.arange(num_seq, dtype=mx.int32) * sequence_stride
    offsets = mx.arange(sequence_length, dtype=mx.int32)
    idx = starts[:, None] + offsets[None, :]  # (num_seq, sequence_length)
    return x[..., idx]  # (*batch, num_seq, sequence_length)


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
    if not mx.issubdtype(real.dtype, mx.floating) or not mx.issubdtype(
        imag.dtype, mx.floating
    ):
        raise ValueError(
            "At least one tensor in input `x` is not of type float."
            f"Received: x={x}."
        )
    return real.astype(mx.complex64) + imag.astype(mx.complex64) * 1j


def _split_complex(c):
    return mx.real(c), mx.imag(c)


def fft(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    out = mx.fft.fft(complex_input)
    return mx.real(out), mx.imag(out)


def fft2(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    out = mx.fft.fft2(complex_input)
    return mx.real(out), mx.imag(out)


def rfft(x, fft_length=None):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    complex_output = mx.fft.rfft(x, n=fft_length, axis=-1, norm="backward")
    return (
        mx.real(complex_output).astype(_mlx_dtype(dtype)),
        mx.imag(complex_output).astype(_mlx_dtype(dtype)),
    )


def ifft2(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = mx.fft.ifft2(complex_input)
    return mx.real(complex_output), mx.imag(complex_output)


def irfft(x, fft_length=None):
    complex_input = _get_complex_tensor_from_tuple(x)
    # numpy always outputs float64, so we need to recast the dtype
    dtype = standardize_dtype(convert_to_tensor(x[0]).dtype)
    out = mx.fft.irfft(complex_input, n=fft_length, axis=-1, norm="backward")
    return out.astype(_mlx_dtype(dtype))


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
    ori_dtype = standardize_dtype(x.dtype)
    x_np = np.asarray(convert_to_numpy(x))

    if center:
        pad_width = [(0, 0) for _ in range(len(x_np.shape))]
        pad_width[-1] = (fft_length // 2, fft_length // 2)
        x_np = np.pad(x_np, pad_width, mode="reflect")

    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    if window is not None:
        if isinstance(window, str):
            win = scipy.signal.get_window(window, sequence_length)
        else:
            win = _to_np(convert_to_tensor(window))
        if len(win.shape) != 1 or win.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win.shape}"
            )
        win = np.pad(win, [[l_pad, r_pad]])
    else:
        win = np.ones((sequence_length + l_pad + r_pad), dtype=x_np.dtype)

    x_np = scipy.signal.stft(
        x_np,
        fs=1.0,
        window=win,
        nperseg=(sequence_length + l_pad + r_pad),
        noverlap=(sequence_length + l_pad + r_pad - sequence_stride),
        nfft=fft_length,
        boundary=None,
        padded=False,
    )[-1]

    # scale and swap to (..., num_sequences, fft_bins)
    x_np = x_np / np.sqrt(1.0 / win.sum() ** 2)
    x_np = np.swapaxes(x_np, -2, -1)
    return convert_to_tensor(
        np.real(x_np).astype(np.dtype(ori_dtype))
    ), convert_to_tensor(np.imag(x_np).astype(np.dtype(ori_dtype)))


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
    x_np = np.asarray(convert_to_numpy(x))

    expected_output_len = fft_length + sequence_stride * (x_np.shape[-2] - 1)
    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    if window is not None:
        if isinstance(window, str):
            win = scipy.signal.get_window(window, sequence_length)
        else:
            win = _to_np(convert_to_tensor(window))
        if len(win.shape) != 1 or win.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win.shape}"
            )
        win = np.pad(win, [[l_pad, r_pad]])
    else:
        win = np.ones((sequence_length + l_pad + r_pad), dtype=x_np.dtype)

    x_np = scipy.signal.istft(
        x_np,
        fs=1.0,
        window=win,
        nperseg=(sequence_length + l_pad + r_pad),
        noverlap=(sequence_length + l_pad + r_pad - sequence_stride),
        nfft=fft_length,
        boundary=False,
        time_axis=-2,
        freq_axis=-1,
    )[-1]

    # scale
    x_np = x_np / win.sum() if window is not None else x_np / sequence_stride

    start = 0 if center is False else fft_length // 2
    if length is not None:
        end = start + length
    elif center is True:
        end = -(fft_length // 2)
    else:
        end = expected_output_len
    return convert_to_tensor(x_np[..., start:end])


def rsqrt(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        x = _cast(x, dtypes.result_type(x.dtype, float))
    return mx.rsqrt(x)


def erf(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        x = _cast(x, dtypes.result_type(x.dtype, float))
    return mx.erf(x)


def erfc(x):
    # MLX has no `erfc`; use scipy (rarely on the autograd path).
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(x.dtype, float)
    result = scipy.special.erfc(_to_np(_cast(x, dtype)))
    return convert_to_tensor(result).astype(_mlx_dtype(dtype))


def erfinv(x):
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(x.dtype, float)
    if standardize_dtype(x.dtype) in ("int8", "int16", "int32", "int64"):
        x = _cast(x, dtype)
    return mx.erfinv(x).astype(_mlx_dtype(dtype))


def logdet(x):
    from keras.src.backend.mlx.numpy import slogdet

    # slogdet is more stable than log(det). See numpy backend.
    return slogdet(x)[1]
