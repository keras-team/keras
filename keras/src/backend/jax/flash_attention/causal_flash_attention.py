import math
import jax
from functools import partial
from jax import nn
from jax import custom_vjp
from jax import numpy as jnp, lax, jit
from jax.numpy import einsum

from einops import rearrange

# constants

EPSILON = 1e-10
MASK_VALUE = -1e10

Q_CHUNK_SIZE = 1024
K_CHUNK_SIZE = 1024

# flash attention

def _query_chunk_flash_attention(q_range_chunk, k_range, q, k, v):
    q_len, k_len, bh, dim, v_dim = q.shape[0], *k.shape, v.shape[-1]
    scale = 1 / jnp.sqrt(dim)
    q_scaled  = q * scale

    def chunk_scanner(carries, _):
        key_chunk_idx, out, row_sum, row_max = carries
        k_chunk_sizes = min(K_CHUNK_SIZE, k_len)

        k_chunk = lax.dynamic_slice(k, (key_chunk_idx, 0, 0), slice_sizes=(k_chunk_sizes, bh, dim))
        v_chunk = lax.dynamic_slice(v, (key_chunk_idx, 0, 0), slice_sizes=(k_chunk_sizes, bh, v_dim))

        k_range_chunk = lax.dynamic_slice(k_range, (0, key_chunk_idx), slice_sizes=(1, k_chunk_sizes))

        causal_mask = q_range_chunk < k_range_chunk

        attn_weights = einsum('i ... d, j ... d -> i ... j', q_scaled, k_chunk)

        causal_mask = rearrange(causal_mask, 'i j -> i 1 j')
        attn_weights = jnp.where(causal_mask, MASK_VALUE, attn_weights)

        block_row_max = jnp.max(attn_weights, axis = -1, keepdims = True)

        exp_weights = jnp.exp(attn_weights - block_row_max)

        exp_weights = jnp.where(causal_mask, 0., exp_weights)

        block_row_sum = jnp.sum(exp_weights, axis = -1, keepdims = True) + EPSILON

        exp_values = einsum('i ... j, j ... d -> i ... d', exp_weights, v_chunk)

        new_row_max = jnp.maximum(block_row_max, row_max)

        exp_row_max_diff = jnp.exp(row_max - new_row_max)
        exp_block_row_max_diff = jnp.exp(block_row_max - new_row_max)

        new_row_sum = exp_row_max_diff * row_sum + exp_block_row_max_diff * block_row_sum

        out = (row_sum / new_row_sum) * exp_row_max_diff * out + \
              (exp_block_row_max_diff / new_row_sum) * exp_values

        return (key_chunk_idx + k_chunk_sizes, out, new_row_sum, new_row_max), None

    out = jnp.zeros((q_len, bh, dim))
    row_sum = jnp.zeros((q_len, bh, 1))
    row_max = jnp.ones((q_len, bh, 1)) * -1e6

    (_, out, row_sum, row_max), _ = lax.scan(chunk_scanner, init = (0, out, row_sum, row_max), xs = None, length = math.ceil(k_len / K_CHUNK_SIZE))

    out = out.reshape(q_len, bh, v_dim)
    row_sum = row_sum.reshape(q_len, bh)
    row_max = row_max.reshape(q_len, bh)

    return out, row_sum, row_max

def _causal_flash_attention(q, k, v):
    batch, heads, q_len, dim, k_len, v_dim = *q.shape, *v.shape[-2:]

    bh = batch * heads

    q, k, v = map(lambda t: rearrange(t, 'b h n d -> n (b h) d'), (q, k, v))

    q_range = jnp.arange(q_len).reshape(q_len, 1) + (k_len - q_len)
    k_range = jnp.arange(k_len).reshape(1, k_len)

    def chunk_scanner(chunk_idx, _):
        chunk_sizes = min(Q_CHUNK_SIZE, q_len)

        q_chunk = lax.dynamic_slice(q, (chunk_idx, 0, 0), slice_sizes = (chunk_sizes, bh, dim))
        q_range_chunk = lax.dynamic_slice(q_range, (chunk_idx, 0), slice_sizes = (chunk_sizes, 1))

        return (chunk_idx + chunk_sizes, _query_chunk_flash_attention(q_range_chunk, k_range, q_chunk, k, v))

    _, (out, row_sum, row_max) = lax.scan(chunk_scanner, init = 0, xs = None, length = math.ceil(q_len / Q_CHUNK_SIZE))

    out = out.reshape(q_len, bh, v_dim)
    row_sum = row_sum.reshape(q_len, bh)
    row_max = row_max.reshape(q_len, bh)

    out = rearrange(out, 'n (b h) d -> b h n d', b = batch)
    return out, (row_sum, row_max)

@custom_vjp
@jit
def causal_flash_attention(q, k, v):
  out, _ = _causal_flash_attention(q, k, v)
  return out

@jit
def flash_attention_forward(q, k, v):
    out, (row_sum, row_max) = _causal_flash_attention(q, k, v)
    return out, (q, k, v, out, row_sum, row_max)

def _query_chunk_flash_attention_backward(query_range_chunk, key_range, q, k, v, o, do, l, m):
    q_len, bh, dim, k_len, _, v_dim = *q.shape, *v.shape

    scale = 1 / jnp.sqrt(dim)
    q_scaled = q * scale

    def chunk_scanner(carries, _):
        key_chunk_idx, dq = carries
        k_chunk_sizes = min(K_CHUNK_SIZE, k_len)

        k_chunk = lax.dynamic_slice(k, (key_chunk_idx, 0, 0), slice_sizes=(k_chunk_sizes, bh, dim))
        v_chunk = lax.dynamic_slice(v, (key_chunk_idx, 0, 0), slice_sizes=(k_chunk_sizes, bh, v_dim))

        key_range_chunk = lax.dynamic_slice(key_range, (0, key_chunk_idx), slice_sizes=(1, k_chunk_sizes))

        causal_mask = query_range_chunk < key_range_chunk

        attn_weights = einsum('i ... d, j ... d -> i ... j', q_scaled, k_chunk)

        causal_mask = rearrange(causal_mask, 'i j -> i 1 j')
        attn_weights = jnp.where(causal_mask, MASK_VALUE, attn_weights)

        exp_attn_weights = jnp.exp(attn_weights - m)

        exp_attn_weights = jnp.where(causal_mask, 0., exp_attn_weights)

        p = exp_attn_weights / l

        dv_chunk = einsum('i ... j, i ... d -> j ... d', p, do)
        dp = einsum('i ... d, j ... d -> i ... j', do, v_chunk)

        D = jnp.sum(do * o, axis = -1, keepdims = True)
        ds = p * scale * (dp - D)

        dq_chunk = einsum('i ... j, j ... d -> i ... d', ds, k_chunk)
        dk_chunk = einsum('i ... j, i ... d -> j ... d', ds, q)

        return (key_chunk_idx + k_chunk_sizes, dq + dq_chunk), (dk_chunk, dv_chunk)

    dq = jnp.zeros_like(q)

    (_, dq), (dk, dv) = lax.scan(chunk_scanner, init = (0, dq), xs = None, length = math.ceil(k_len / K_CHUNK_SIZE))

    dq = dq.reshape(q_len, bh, dim)
    dk = dk.reshape(k_len, bh, v_dim)
    dv = dv.reshape(k_len, bh, v_dim)

    return dq, dk, dv

@jit
def flash_attention_backward(res, do):
    q, k, v, o, l, m = res

    batch, heads, q_len, dim, k_len, v_dim = *q.shape, *v.shape[-2:]

    bh = batch * heads

    m = m.reshape(q_len, bh, 1)
    l = l.reshape(q_len, bh, 1)

    q, k, v, o, do = map(lambda t: rearrange(t, 'b h n d -> n (b h) d'), (q, k, v, o, do))

    dk = jnp.zeros_like(k)
    dv = jnp.zeros_like(v)

    q_range = jnp.arange(q_len).reshape(q_len, 1) + (k_len - q_len)
    k_range = jnp.arange(k_len).reshape(1, k_len)

    def chunk_scanner(carries, _):
        chunk_idx, dk, dv = carries

        chunk_sizes = min(Q_CHUNK_SIZE, q_len)

        q_chunk = lax.dynamic_slice(q, (chunk_idx, 0, 0), slice_sizes = (chunk_sizes, bh, q.shape[-1]))
        q_range_chunk = lax.dynamic_slice(q_range, (chunk_idx, 0), slice_sizes = (chunk_sizes, 1))

        m_chunk = lax.dynamic_slice(m, (chunk_idx, 0, 0), slice_sizes = (chunk_sizes, bh, 1))
        l_chunk = lax.dynamic_slice(l, (chunk_idx, 0, 0), slice_sizes = (chunk_sizes, bh, 1))
        o_chunk = lax.dynamic_slice(o, (chunk_idx, 0, 0), slice_sizes = (chunk_sizes, bh, o.shape[-1]))
        do_chunk = lax.dynamic_slice(do, (chunk_idx, 0, 0), slice_sizes = (chunk_sizes, bh, do.shape[-1]))

        dq_chunk, dk_chunk, dv_chunk = _query_chunk_flash_attention_backward(q_range_chunk, k_range, q_chunk, k, v, o_chunk, do_chunk, l_chunk, m_chunk)
        return (chunk_idx + chunk_sizes, dk + dk_chunk, dv + dv_chunk), dq_chunk

    (_, dk, dv), dq = lax.scan(chunk_scanner, init = (0, dk, dv), xs = None, length = math.ceil(q_len / Q_CHUNK_SIZE))

    dq = dq.reshape(q_len, bh, dim)

    dq, dk, dv = map(lambda t: rearrange(t, 'n (b h) d -> b h n d', b = batch), (dq, dk, dv))

    return dq, dk, dv

causal_flash_attention.defvjp(flash_attention_forward, flash_attention_backward)
