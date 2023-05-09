import jax
import jax.numpy as jnp


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


def logsumexp(x, axis=None, keepdims=False):
    max_x = jnp.max(x, axis=axis, keepdims=True)
    result = (
        jnp.log(jnp.sum(jnp.exp(x - max_x), axis=axis, keepdims=keepdims))
        + max_x
    )
    return jnp.squeeze(result) if not keepdims else result
