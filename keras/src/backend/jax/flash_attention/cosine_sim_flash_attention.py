import math
import jax
from functools import partial
from jax import nn
from jax import custom_vjp
from jax import numpy as jnp, lax, jit

# constants

EPSILON = 1e-10
MASK_VALUE = -1e10

Q_CHUNK_SIZE = 1024
K_CHUNK_SIZE = 1024
COSINE_SIM_SCALE = 10 # this may need to be a function of log(sequence length), but 16 was sufficient for 2048 and 4096 in my tests

# flash attention

def _query_chunk_flash_attention(chunk_idx, q, k, v, key_mask):
    q_len, k_len, dim, v_dim = q.shape[-2], *k.shape, v.shape[-1]

    def chunk_scanner(carries, _):
        chunk_idx, out, row_sum = carries
        k_chunk_sizes = min(K_CHUNK_SIZE, k_len)

        k_chunk = lax.dynamic_slice(k, (chunk_idx, 0), slice_sizes=(k_chunk_sizes, dim))
        v_chunk = lax.dynamic_slice(v, (chunk_idx, 0), slice_sizes=(k_chunk_sizes, v_dim))
        key_mask_chunk = lax.dynamic_slice(key_mask, (chunk_idx,), slice_sizes=(k_chunk_sizes,))

        attn_weights = (q @ k_chunk.transpose() * COSINE_SIM_SCALE) - COSINE_SIM_SCALE  # the output of this will range from [-2 * scale, 0], and the row sums are now bounded by key/value sequence length - you can also shift this more if you wish to tailor the normalization constant (in the case of extreme sequence lengths)

        attn_weights = jnp.where(key_mask_chunk, attn_weights, MASK_VALUE)

        exp_weights = jnp.exp(attn_weights)
        exp_weights = jnp.where(key_mask_chunk, exp_weights, 0.)

        block_row_sum = jnp.sum(exp_weights, axis = -1, keepdims = True)

        exp_values = exp_weights @ v_chunk

        chunk_out = exp_values / k_len

        return (chunk_idx + k_chunk_sizes, out + chunk_out, row_sum + block_row_sum), None

    out = jnp.zeros((q_len, dim))
    row_sum = jnp.zeros((q_len, 1))

    (_, out, row_sum), _ = lax.scan(chunk_scanner, init = (0, out, row_sum), xs = None, length = math.ceil(k_len / K_CHUNK_SIZE))

    out = out * (k_len / (row_sum + EPSILON)) # renormalize after acquiring all the correct row sums

    out = out.reshape(q_len, v_dim)
    row_sum = row_sum.reshape(q_len)

    return out, row_sum

@jit
def l2norm(t):
    return t / (jnp.linalg.norm(t) + EPSILON)

@jit
def cosine_sim_flash_attention(q, k, v, key_mask):
    q, k = map(l2norm, (q, k))
    return cosine_sim_flash_attention_after_l2norm(q, k, v, key_mask)

def _cosine_sim_flash_attention_after_l2norm(q, k, v, key_mask):
    q_len, dim, v_dim = *q.shape, v.shape[-1]

    def chunk_scanner(chunk_idx, _):
        chunk_sizes = min(Q_CHUNK_SIZE, q_len)

        q_chunk = lax.dynamic_slice(q, (chunk_idx, 0), slice_sizes = (chunk_sizes, dim))

        return (chunk_idx + chunk_sizes, _query_chunk_flash_attention(chunk_idx, q_chunk, k, v, key_mask))

    _, (out, row_sum) = lax.scan(chunk_scanner, init = 0, xs = None, length = math.ceil(q_len / Q_CHUNK_SIZE))

    out = out.reshape(q_len, v_dim)
    row_sum = row_sum.reshape(q_len)

    return out, (row_sum,)

@custom_vjp
def cosine_sim_flash_attention_after_l2norm(q, k, v, key_mask):
  out, _ = _cosine_sim_flash_attention_after_l2norm(q, k, v, key_mask)
  return out

@jit
def flash_attention_forward(q, k, v, key_mask):
    out, (row_sum,) = _cosine_sim_flash_attention_after_l2norm(q, k, v, key_mask)
    return out, (q, k, v, key_mask, out, row_sum)

def _query_chunk_flash_attention_backward(q, k, v, key_mask,o, do, l):
    q_len, dim, k_len, v_dim = *q.shape, *v.shape

    def chunk_scanner(carries, _):
        chunk_idx, dq = carries
        k_chunk_sizes = min(K_CHUNK_SIZE, k_len)

        k_chunk = lax.dynamic_slice(k, (chunk_idx, 0), slice_sizes=(k_chunk_sizes, dim))
        v_chunk = lax.dynamic_slice(v, (chunk_idx, 0), slice_sizes=(k_chunk_sizes, v_dim))
        key_mask_chunk = lax.dynamic_slice(key_mask, (chunk_idx,), slice_sizes=(k_chunk_sizes,))

        attn_weights = q @ k_chunk.transpose() * COSINE_SIM_SCALE - COSINE_SIM_SCALE

        exp_attn_weights = jnp.exp(attn_weights)

        exp_attn_weights = jnp.where(key_mask_chunk, exp_attn_weights, 0.)

        p = exp_attn_weights / (l + EPSILON)

        dv_chunk = p.transpose() @ do
        dp = do @ v_chunk.transpose()

        D = jnp.sum(do * o, axis = -1, keepdims = True)
        ds = p * COSINE_SIM_SCALE * (dp - D)

        dq_chunk = ds @ k_chunk
        dk_chunk = ds.transpose() @ q

        return (chunk_idx + k_chunk_sizes, dq + dq_chunk), (dk_chunk, dv_chunk)

    dq = jnp.zeros_like(q)

    (_, dq), (dk, dv) = lax.scan(chunk_scanner, init = (0, dq), xs = None, length = math.ceil(k_len / K_CHUNK_SIZE))

    dq = dq.reshape(q_len, dim)
    dk = dk.reshape(k_len, v_dim)
    dv = dv.reshape(k_len, v_dim)

    return dq, dk, dv

@jit
def flash_attention_backward(res, do):
    q, k, v, key_mask, o, l = res

    q_len, dim = q.shape

    dk = jnp.zeros_like(k)
    dv = jnp.zeros_like(v)

    l = l.reshape(q_len, 1)

    def chunk_scanner(carries, _):
        chunk_idx, dk, dv = carries

        chunk_sizes = min(Q_CHUNK_SIZE, q_len)

        q_chunk = lax.dynamic_slice(q, (chunk_idx, 0), slice_sizes = (chunk_sizes, q.shape[-1]))
        l_chunk = lax.dynamic_slice(l, (chunk_idx, 0), slice_sizes = (chunk_sizes, 1))
        o_chunk = lax.dynamic_slice(o, (chunk_idx, 0), slice_sizes = (chunk_sizes, o.shape[-1]))
        do_chunk = lax.dynamic_slice(do, (chunk_idx, 0), slice_sizes = (chunk_sizes, do.shape[-1]))

        dq_chunk, dk_chunk, dv_chunk = _query_chunk_flash_attention_backward(q_chunk, k, v, key_mask, o_chunk, do_chunk, l_chunk)
        return (chunk_idx + chunk_sizes, dk + dk_chunk, dv + dv_chunk), dq_chunk

    (_, dk, dv), dq = lax.scan(chunk_scanner, init = (0, dk, dv), xs = None, length = math.ceil(q_len / Q_CHUNK_SIZE))

    dq = dq.reshape(q_len, dim)

    return dq, dk, dv, None

cosine_sim_flash_attention_after_l2norm.defvjp(flash_attention_forward, flash_attention_backward)
