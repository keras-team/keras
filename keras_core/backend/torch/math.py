import torch

from keras_core.backend.torch.core import convert_to_tensor


def segment_sum(data, segment_ids, num_segments=None, **kwargs):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids)
    num_repeats = torch.prod(torch.tensor(data.shape[1:])).long()
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

    result = torch.zeros(*shape).scatter_add(0, segment_ids, data.float())

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
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    return torch.linalg.qr(x, mode=mode)
