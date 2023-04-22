import jax


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    return jax.ops.segment_sum(
        data, segment_ids, num_segments, indices_are_sorted=sorted
    )


def top_k(x, k, sorted=False):
    if sorted:
        return ValueError(
            "Jax backend does not support `sorted=True` for `ops.top_k`"
        )
    return jax.lax.top_k(x, k)
