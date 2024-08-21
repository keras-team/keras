import tensorflow as tf

from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.tensorflow.core import cast
from keras.src.backend.tensorflow.core import convert_to_tensor


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    if sorted:
        if num_segments is not None:
            raise ValueError(
                "Argument `num_segments` cannot be set when sorted is True "
                "when using the tensorflow backend."
                f"Received: num_segments={num_segments}, sorted={sorted}."
            )
        return tf.math.segment_sum(data, segment_ids)
    else:
        if num_segments is None:
            unique_segment_ids, _ = tf.unique(segment_ids)
            num_segments = tf.shape(unique_segment_ids)[0]
        return tf.math.unsorted_segment_sum(data, segment_ids, num_segments)


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    if sorted:
        if num_segments is not None:
            raise ValueError(
                "Argument `num_segments` cannot be set when sorted is True "
                "when using the tensorflow backend."
                f"Received: num_segments={num_segments}, sorted={sorted}."
            )
        return tf.math.segment_max(data, segment_ids)
    else:
        if num_segments is None:
            unique_segment_ids, _ = tf.unique(segment_ids)
            num_segments = tf.shape(unique_segment_ids)[0]
        return tf.math.unsorted_segment_max(data, segment_ids, num_segments)


def top_k(x, k, sorted=True):
    return tf.math.top_k(x, k, sorted=sorted)


def in_top_k(targets, predictions, k):
    return tf.math.in_top_k(targets, predictions, k)


def logsumexp(x, axis=None, keepdims=False):
    return tf.math.reduce_logsumexp(x, axis=axis, keepdims=keepdims)


def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    if mode == "reduced":
        return tf.linalg.qr(x)
    return tf.linalg.qr(x, full_matrices=True)


def extract_sequences(x, sequence_length, sequence_stride):
    return tf.signal.frame(
        x,
        frame_length=sequence_length,
        frame_step=sequence_stride,
        axis=-1,
        pad_end=False,
    )


def _get_complex_tensor_from_tuple(x):
    if not isinstance(x, (tuple, list)) or len(x) != 2:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            f"Received: x={x}"
        )
    # `convert_to_tensor` does not support passing complex tensors. We separate
    # the input out into real and imaginary and convert them separately.
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
    if not real.dtype.is_floating or not imag.dtype.is_floating:
        raise ValueError(
            "At least one tensor in input `x` is not of type float."
            f"Received: x={x}."
        )
    complex_input = tf.dtypes.complex(real, imag)
    return complex_input


def fft(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = tf.signal.fft(complex_input)
    return tf.math.real(complex_output), tf.math.imag(complex_output)


def fft2(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = tf.signal.fft2d(complex_input)
    return tf.math.real(complex_output), tf.math.imag(complex_output)


def rfft(x, fft_length=None):
    if fft_length is not None:
        fft_length = [fft_length]
    complex_output = tf.signal.rfft(x, fft_length=fft_length)
    return tf.math.real(complex_output), tf.math.imag(complex_output)


def irfft(x, fft_length=None):
    complex_input = _get_complex_tensor_from_tuple(x)
    if fft_length is not None:
        fft_length = [fft_length]
    return tf.signal.irfft(complex_input, fft_length)


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
        x = tf.pad(x, pad_width, mode="reflect")

    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    if window is not None:
        if isinstance(window, str):
            if window == "hann":
                win_array = tf.signal.hann_window(
                    sequence_length, periodic=True, dtype=x.dtype
                )
            else:
                win_array = tf.signal.hamming_window(
                    sequence_length, periodic=True, dtype=x.dtype
                )
        else:
            win_array = convert_to_tensor(window, dtype=x.dtype)
        if len(win_array.shape) != 1 or win_array.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win_array.shape}"
            )
        win_array = tf.pad(win_array, [[l_pad, r_pad]])

        def win(frame_step, dtype):
            return win_array

    else:
        win = None

    result = tf.signal.stft(
        x,
        frame_length=(sequence_length + l_pad + r_pad),
        frame_step=sequence_stride,
        fft_length=fft_length,
        window_fn=win,
    )
    return tf.math.real(result), tf.math.imag(result)


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
    dtype = tf.math.real(complex_input).dtype

    expected_output_len = fft_length + sequence_stride * (
        tf.shape(complex_input)[-2] - 1
    )
    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    if window is not None:
        if isinstance(window, str):
            if window == "hann":
                win_array = tf.signal.hann_window(
                    sequence_length, periodic=True, dtype=dtype
                )
            else:
                win_array = tf.signal.hamming_window(
                    sequence_length, periodic=True, dtype=dtype
                )
        else:
            win_array = convert_to_tensor(window, dtype=dtype)
        if len(win_array.shape) != 1 or win_array.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win_array.shape}"
            )
        win_array = tf.pad(win_array, [[l_pad, r_pad]])
        win = tf.signal.inverse_stft_window_fn(
            sequence_stride, lambda frame_step, dtype: win_array
        )
    else:
        win = None

    x = tf.signal.inverse_stft(
        complex_input,
        frame_length=(sequence_length + l_pad + r_pad),
        frame_step=sequence_stride,
        fft_length=fft_length,
        window_fn=win,
    )

    start = 0 if center is False else fft_length // 2
    if length is not None:
        end = start + length
    elif center is True:
        end = -(fft_length // 2)
    else:
        end = expected_output_len
    return x[..., start:end]


def rsqrt(x):
    return tf.math.rsqrt(x)


def erf(x):
    return tf.math.erf(x)


def erfinv(x):
    return tf.math.erfinv(x)


def solve(a, b):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    return tf.linalg.solve(a, b)


def norm(x, ord=None, axis=None, keepdims=False):
    from keras.src.backend.tensorflow.numpy import moveaxis

    x = convert_to_tensor(x)
    x_shape = x.shape
    ndim = x_shape.rank

    if axis is None:
        axis = tuple(range(ndim))
    elif isinstance(axis, int):
        axis = (axis,)

    axis = axis[0] if len(axis) == 1 else axis
    num_axes = 1 if isinstance(axis, int) else len(axis)

    if num_axes == 1 and ord is None:
        ord = "euclidean"
    elif num_axes == 2 and ord is None:
        ord = "fro"

    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = cast(x, dtype)

    # Fast path to utilze `tf.linalg.norm`
    if (num_axes == 1 and ord in ("euclidean", 1, 2, float("inf"))) or (
        num_axes == 2 and ord in ("euclidean", "fro", 1, 2, float("inf"))
    ):
        return tf.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

    # Ref: jax.numpy.linalg.norm
    if num_axes == 1 and ord not in ("fro", "nuc"):
        if ord == float("-inf"):
            return tf.math.reduce_min(
                tf.math.abs(x), axis=axis, keepdims=keepdims
            )
        elif ord == 0:
            return tf.math.reduce_sum(
                tf.cast(tf.not_equal(x, 0), dtype=x.dtype),
                axis=axis,
                keepdims=keepdims,
            )
        else:
            ord = convert_to_tensor(ord, dtype=x.dtype)
            out = tf.math.reduce_sum(
                tf.pow(tf.math.abs(x), ord), axis=axis, keepdims=keepdims
            )
            return tf.pow(out, 1.0 / ord)
    elif num_axes == 2 and ord in ("nuc", float("-inf"), -2, -1):
        row_axis, col_axis = axis[0], axis[1]
        row_axis = row_axis + ndim if row_axis < 0 else row_axis
        col_axis = col_axis + ndim if col_axis < 0 else col_axis
        if ord == float("-inf"):
            if not keepdims and row_axis > col_axis:
                row_axis -= 1
            x = tf.math.reduce_min(
                tf.reduce_sum(tf.math.abs(x), axis=col_axis, keepdims=keepdims),
                axis=row_axis,
                keepdims=keepdims,
            )
        elif ord == -1:
            if not keepdims and col_axis > row_axis:
                col_axis -= 1
            x = tf.math.reduce_min(
                tf.reduce_sum(tf.math.abs(x), axis=row_axis, keepdims=keepdims),
                axis=col_axis,
                keepdims=keepdims,
            )
        else:
            x = moveaxis(x, axis, (-2, -1))
            if ord == -2:
                x = tf.math.reduce_min(
                    tf.linalg.svd(x, compute_uv=False), axis=-1
                )
            else:
                x = tf.math.reduce_sum(
                    tf.linalg.svd(x, compute_uv=False), axis=-1
                )
            if keepdims:
                x = tf.expand_dims(x, axis[0])
                x = tf.expand_dims(x, axis[1])
        return x

    if num_axes == 1:
        raise ValueError(
            f"Invalid `ord` argument for vector norm. Received: ord={ord}"
        )
    elif num_axes == 2:
        raise ValueError(
            f"Invalid `ord` argument for matrix norm. Received: ord={ord}"
        )
    else:
        raise ValueError(f"Invalid axis values. Received: axis={axis}")
