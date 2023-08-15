import torch

from keras_core.backend.torch.core import convert_to_tensor
from keras_core.backend.torch.core import get_device


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


def rsqrt(x):
    x = convert_to_tensor(x)
    return torch.rsqrt(x)
