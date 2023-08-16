import torch

from keras_core.backend import standardize_dtype
from keras_core.backend.torch.core import convert_to_tensor
from keras_core.backend.torch.core import get_device
from keras_core.backend.torch.numpy import pad


def segment_sum(data, segment_ids, num_segments=None, **kwargs):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids)
    num_repeats = torch.prod(
        torch.tensor(data.shape[1:], device=get_device())
    ).long()
    # To use `scatter_add` in torch, we need to replicate `segment_ids` into the
    # shape of `data`.
    segment_ids = (
        segment_ids.repeat_interleave(num_repeats)
        .view(*data.shape)
        .type(torch.int64)
    )
    num_segments = num_segments or len(torch.unique(segment_ids))

    # .scatter_add does not support -1 in the indices.
    # Add all out-of-bound indices value to an extra dimension after
    # num_segments, which is removed before returning the result.

    # Replacing the out-of-bound indices.
    segment_ids = torch.where(segment_ids >= 0, segment_ids, num_segments)
    segment_ids = torch.where(
        segment_ids < num_segments, segment_ids, num_segments
    )

    # Add one more dimension to the result shape with the "+1".
    shape = (num_segments + 1,) + tuple(data.shape[1:])

    result = torch.zeros(*shape, device=get_device()).scatter_add(
        0, segment_ids, data.float()
    )

    # Removing the extra dimension.
    result = result[:-1, ...]

    return result.type(data.dtype)


def segment_max(data, segment_ids, num_segments=None, **kwargs):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids)
    num_repeats = torch.prod(
        torch.tensor(data.shape[1:], device=get_device())
    ).long()
    # To use `scatter_reduce` in torch, we need to replicate `segment_ids` into
    # the shape of `data`.
    segment_ids = (
        segment_ids.repeat_interleave(num_repeats)
        .view(*data.shape)
        .type(torch.int64)
    )
    num_segments = num_segments or len(torch.unique(segment_ids))

    # .scatter_reduce does not support -1 in the indices.
    # Add all out-of-bound indices value to an extra dimension after
    # num_segments, which is removed before returning the result.

    # Replacing the out-of-bound indices.
    segment_ids = torch.where(segment_ids >= 0, segment_ids, num_segments)
    segment_ids = torch.where(
        segment_ids < num_segments, segment_ids, num_segments
    )

    # Add one more dimension to the result shape with the "+1".
    shape = (num_segments + 1,) + tuple(data.shape[1:])

    result = torch.zeros(*shape, device=get_device()).scatter_reduce(
        0, segment_ids, data.float(), "amax"
    )

    # Removing the extra dimension.
    result = result[:-1, ...]

    return result.type(data.dtype)


def top_k(x, k, sorted=True):
    x = convert_to_tensor(x)
    return torch.topk(x, k, sorted=sorted)


def in_top_k(targets, predictions, k):
    targets = convert_to_tensor(targets).type(torch.int64)
    targets = targets[:, None]
    predictions = convert_to_tensor(predictions)
    topk_values = top_k(predictions, k).values
    targets_values = torch.take_along_dim(predictions, targets, dim=-1)
    mask = targets_values >= topk_values
    return torch.any(mask, axis=-1)


def logsumexp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if axis is None:
        max_x = torch.max(x)
        return torch.log(torch.sum(torch.exp(x - max_x))) + max_x

    max_x = torch.amax(x, dim=axis, keepdim=True)
    result = (
        torch.log(torch.sum(torch.exp(x - max_x), dim=axis, keepdim=True))
        + max_x
    )
    return torch.squeeze(result, dim=axis) if not keepdims else result


def qr(x, mode="reduced"):
    x = convert_to_tensor(x)
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    x = convert_to_tensor(x)
    return torch.linalg.qr(x, mode=mode)


def extract_sequences(x, sequence_length, sequence_stride):
    x = convert_to_tensor(x)
    return torch.unfold_copy(
        x, dimension=-1, size=sequence_length, step=sequence_stride
    )


def _get_complex_tensor_from_tuple(a):
    if not isinstance(a, (tuple, list)) or len(a) != 2:
        raise ValueError(
            "Input `a` should be a tuple of two tensors - real and imaginary."
            f"Received: a={a}"
        )
    # `convert_to_tensor` does not support passing complex tensors. We separate
    # the input out into real and imaginary and convert them separately.
    real, imag = a
    real = convert_to_tensor(real)
    imag = convert_to_tensor(imag)
    # Check shape.
    if real.shape != imag.shape:
        raise ValueError(
            "Input `a` should be a tuple of two tensors - real and imaginary."
            "Both the real and imaginary parts should have the same shape. "
            f"Received: a[0].shape = {real.shape}, a[1].shape = {imag.shape}"
        )
    # Ensure dtype is float.
    if not torch.is_floating_point(real) or not torch.is_floating_point(imag):
        raise ValueError(
            "At least one tensor in input `a` is not of type float."
            f"Received: a={a}."
        )

    complex_input = torch.complex(real, imag)
    return complex_input


def fft(a):
    complex_input = _get_complex_tensor_from_tuple(a)
    complex_output = torch.fft.fft(complex_input)
    return complex_output.real, complex_output.imag


def fft2(a):
    complex_input = _get_complex_tensor_from_tuple(a)
    complex_output = torch.fft.fft2(complex_input)
    return complex_output.real, complex_output.imag


def rfft(x, fft_length=None):
    x = convert_to_tensor(x)
    complex_output = torch.fft.rfft(x, n=fft_length, dim=-1, norm="backward")
    return complex_output.real, complex_output.imag


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
        pad_width = [(0, 0) for _ in range(x.ndim)]
        pad_width[-1] = (fft_length // 2, fft_length // 2)
        # torch does not support reflect padding when x.ndim >= 3
        if x.ndim < 3:
            x = pad(x, pad_width, "reflect")
        else:
            x = pad(x, pad_width, "constant")

    x = extract_sequences(x, fft_length, sequence_stride)

    if window is not None:
        if isinstance(window, str):
            if window == "hann":
                win = torch.hann_window(
                    sequence_length,
                    periodic=True,
                    dtype=x.dtype,
                    device=get_device(),
                )
            else:
                win = torch.hamming_window(
                    sequence_length,
                    periodic=True,
                    dtype=x.dtype,
                    device=get_device(),
                )
        else:
            win = convert_to_tensor(window, dtype=x.dtype)
        if len(win.shape) != 1 or win.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win.shape}"
            )
        l_pad = (fft_length - sequence_length) // 2
        r_pad = fft_length - sequence_length - l_pad
        win = pad(win, [[l_pad, r_pad]], "constant")
        x = torch.multiply(x, win)

    return rfft(x, fft_length)


def rsqrt(x):
    x = convert_to_tensor(x)
    return torch.rsqrt(x)
