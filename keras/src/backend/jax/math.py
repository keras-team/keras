import math

import jax
import jax.numpy as jnp

from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.jax.core import cast
from keras.src.backend.jax.core import convert_to_tensor
from keras.src.utils.module_utils import scipy


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    if num_segments is None:
        raise ValueError(
            "Argument `num_segments` must be set when using the JAX backend. "
            "Received: num_segments=None"
        )
    return jax.ops.segment_sum(
        data, segment_ids, num_segments, indices_are_sorted=sorted
    )


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    if num_segments is None:
        raise ValueError(
            "Argument `num_segments` must be set when using the JAX backend. "
            "Received: num_segments=None"
        )
    return jax.ops.segment_max(
        data, segment_ids, num_segments, indices_are_sorted=sorted
    )


def top_k(x, k, sorted=True):
    # Jax does not supported `sorted`, but in the case where `sorted=False`,
    # order is not guaranteed, so OK to return sorted output.
    return jax.lax.top_k(x, k)


def in_top_k(targets, predictions, k):
    preds_at_label = jnp.take_along_axis(
        predictions, jnp.expand_dims(targets, axis=-1), axis=-1
    )
    # `nan` shouldn't be considered as large probability.
    preds_at_label = jnp.where(
        jnp.isnan(preds_at_label), -jnp.inf, preds_at_label
    )
    rank = 1 + jnp.sum(jnp.greater(predictions, preds_at_label), axis=-1)
    return jnp.less_equal(rank, k)


def logsumexp(x, axis=None, keepdims=False):
    max_x = jnp.max(x, axis=axis, keepdims=True)
    result = (
        jnp.log(jnp.sum(jnp.exp(x - max_x), axis=axis, keepdims=True)) + max_x
    )
    return jnp.squeeze(result) if not keepdims else result


def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    return jnp.linalg.qr(x, mode=mode)


def extract_sequences(x, sequence_length, sequence_stride):
    *batch_shape, signal_length = x.shape
    batch_shape = list(batch_shape)
    x = jnp.reshape(x, (math.prod(batch_shape), signal_length, 1))
    x = jax.lax.conv_general_dilated_patches(
        x,
        (sequence_length,),
        (sequence_stride,),
        "VALID",
        dimension_numbers=("NTC", "OIT", "NTC"),
    )
    return jnp.reshape(x, (*batch_shape, *x.shape[-2:]))


def _get_complex_tensor_from_tuple(x):
    if not isinstance(x, (tuple, list)) or len(x) != 2:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            f"Received: x={x}"
        )
    # `convert_to_tensor` does not support passing complex tensors. We separate
    # the input out into real and imaginary and convert them separately.
    real, imag = x
    # Check shapes.
    if real.shape != imag.shape:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            "Both the real and imaginary parts should have the same shape. "
            f"Received: x[0].shape = {real.shape}, x[1].shape = {imag.shape}"
        )
    # Ensure dtype is float.
    if not jnp.issubdtype(real.dtype, jnp.floating) or not jnp.issubdtype(
        imag.dtype, jnp.floating
    ):
        raise ValueError(
            "At least one tensor in input `x` is not of type float."
            f"Received: x={x}."
        )
    complex_input = jax.lax.complex(real, imag)
    return complex_input


def fft(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = jnp.fft.fft(complex_input)
    return jnp.real(complex_output), jnp.imag(complex_output)


def fft2(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = jnp.fft.fft2(complex_input)
    return jnp.real(complex_output), jnp.imag(complex_output)


def rfft(x, fft_length=None):
    complex_output = jnp.fft.rfft(x, n=fft_length, axis=-1, norm="backward")
    return jnp.real(complex_output), jnp.imag(complex_output)


def irfft(x, fft_length=None):
    complex_input = _get_complex_tensor_from_tuple(x)
    return jnp.fft.irfft(complex_input, n=fft_length, axis=-1, norm="backward")


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
        pad_width = [(0, 0) for _ in range(len(x.shape))]
        pad_width[-1] = (fft_length // 2, fft_length // 2)
        x = jnp.pad(x, pad_width, mode="reflect")

    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    if window is not None:
        if isinstance(window, str):
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
        win = jnp.pad(win, [[l_pad, r_pad]])
    else:
        win = jnp.ones((sequence_length + l_pad + r_pad), dtype=x.dtype)

    result = jax.scipy.signal.stft(
        x,
        fs=1.0,
        window=win,
        nperseg=(sequence_length + l_pad + r_pad),
        noverlap=(sequence_length + l_pad + r_pad - sequence_stride),
        nfft=fft_length,
        boundary=None,
        padded=False,
    )[-1]
    # scale and swap to (..., num_sequences, fft_bins)
    scale = jnp.sqrt(1.0 / win.sum() ** 2)
    result = result / scale
    result = jnp.swapaxes(result, -2, -1)
    return jnp.real(result), jnp.imag(result)


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
    dtype = jnp.real(x).dtype

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
        win = jnp.pad(win, [[l_pad, r_pad]])
    else:
        win = jnp.ones((sequence_length + l_pad + r_pad), dtype=dtype)

    x = jax.scipy.signal.istft(
        x,
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
    return jax.lax.rsqrt(x)


def erf(x):
    return jax.lax.erf(x)


def erfinv(x):
    return jax.lax.erf_inv(x)


def solve(a, b):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    return jnp.linalg.solve(a, b)


def norm(x, ord=None, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)
    return jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def logdet(x):
    from keras.src.backend.jax.numpy import slogdet

    # In JAX (like in NumPy) slogdet is more stable than
    # `np.log(np.linalg.det(x))`. See
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.slogdet.html
    return slogdet(x)[1]
