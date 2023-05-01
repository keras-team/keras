import jax


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    return jax.ops.segment_sum(
        data, segment_ids, num_segments, indices_are_sorted=sorted
    )


def top_k(x, k, sorted=True):
    if not sorted:
        return ValueError(
            "Jax backend does not support `sorted=False` for `ops.top_k`"
        )
    return jax.lax.top_k(x, k)


def in_top_k(targets, predictions, k):
    topk_indices = top_k(predictions, k)[1]
    targets = targets[..., None]
    mask = targets == topk_indices
    return jax.numpy.any(mask, axis=1)
