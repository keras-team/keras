import jax
import jax.numpy as jnp


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    if num_segments is None:
        raise ValueError(
            "Argument `num_segments` must be set when using the JAX backend. "
            "Received: num_segments=None"
        )
    return jax.ops.segment_sum(
        data, segment_ids, num_segments, indices_are_sorted=sorted
    )


def top_k(x, k, sorted=True):
    # Jax does not supported `sorted`, but in the case where `sorted=False`,
    # order is not guaranteed, so OK to return sorted output.
    return jax.lax.top_k(x, k)


def in_top_k(targets, predictions, k):
    targets = targets[..., None]
    topk_values = top_k(predictions, k)[0]
    targets_values = jnp.take_along_axis(predictions, targets, axis=-1)
    mask = targets_values >= topk_values
    return jax.numpy.any(mask, axis=1)


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
    return jax.numpy.linalg.qr(x, mode=mode)
