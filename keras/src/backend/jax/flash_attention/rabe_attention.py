import math
from functools import partial

import jax
from jax import lax, numpy as jnp, jit

# constants

HIGHEST_PRECISION = jax.lax.Precision.HIGHEST

einsum = partial(jnp.einsum, precision = HIGHEST_PRECISION)

# Figure 1 from https://arxiv.org/abs/2112.05682
# cleaned up

def _query_chunk_attention(q, k, v, k_chunk_size = 4096):
    q_len, k_len, dim, v_dim = q.shape[-2], *k.shape, v.shape[-1]

    k_chunk_size = min(k_chunk_size, k_len)
    q = q / jnp.sqrt(dim)

    @partial(jax.checkpoint, prevent_cse = False)
    def summarize_chunk(q, k, v):
        attn_weights = einsum('qd, kd -> qk', q, k)
        max_score = jnp.max(attn_weights, axis = -1, keepdims = True)
        max_score = jax.lax.stop_gradient(max_score)
        exp_weights = jnp.exp(attn_weights - max_score)
        exp_values = einsum('vf, qv -> qf', v, exp_weights)
        return (exp_values, exp_weights.sum(axis = -1), max_score.reshape((q_len,)))

    def chunk_scanner(chunk_idx):
        k_chunk = lax.dynamic_slice(k, (chunk_idx, 0), slice_sizes=(k_chunk_size, dim))
        v_chunk = lax.dynamic_slice(v, (chunk_idx, 0), slice_sizes=(k_chunk_size, v_dim))
        return summarize_chunk(q, k_chunk, v_chunk)

    chunk_values, chunk_weights, chunk_max = jax.lax.map(chunk_scanner, xs = jnp.arange(0, k_len, k_chunk_size))
    global_max = jnp.max(chunk_max, axis = 0, keepdims = True)
    max_diffs = jnp.exp(chunk_max - global_max)
    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(axis = 0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis = 0)
    return all_values / all_weights

@jit
def rabe_attention(q, k, v, q_chunk_size = 1024, k_chunk_size = 4096):
    q_len, dim, v_dim = *q.shape, v.shape[-1]

    def chunk_scanner(chunk_idx, _):
        q_chunk = lax.dynamic_slice(q, (chunk_idx, 0), slice_sizes = (min(q_chunk_size, q_len), dim))
        return (chunk_idx + q_chunk_size, _query_chunk_attention(q_chunk, k, v, k_chunk_size = k_chunk_size))

    _, res = jax.lax.scan(chunk_scanner, init = 0, xs = None, length = math.ceil(q_len / q_chunk_size))
    return res.reshape(q_len, v_dim)
