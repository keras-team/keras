import jax
from jax import nn
from jax import jit, numpy as jnp
from jax.numpy import einsum

from einops import rearrange

EPSILON = 1e-10
MASK_VALUE = -1e10
COSINE_SIM_SCALE = 10

@jit
def attention(q, k, v, key_mask):
    dim, k_len = q.shape[-1], k.shape[-2]
    scale = 1 / jnp.sqrt(dim)

    q = q * scale
    sim = einsum('... i d, ... j d -> ... i j', q, k)

    key_mask = rearrange(key_mask, 'b j -> b 1 1 j')
    sim = jnp.where(key_mask, sim, MASK_VALUE)

    attn = nn.softmax(sim, axis = -1)
    return attn @ v

@jit
def causal_attention(q, k, v):
    q_len, dim, k_len = *q.shape[-2:], k.shape[-2]
    scale = 1 / jnp.sqrt(dim)

    q = q * scale
    sim = einsum('... i d, ... j d -> ... i j', q, k)

    causal_mask = jnp.triu(jnp.ones((q_len, k_len)), k_len - q_len + 1)
    sim = jnp.where(causal_mask, MASK_VALUE, sim)

    attn = nn.softmax(sim, axis = -1)
    return einsum('... i j, ... j d -> ... i d', attn, v)

# cosine sim attention

@jit
def l2norm(t):
    return t / (jnp.linalg.norm(t) + EPSILON)

@jit
def cosine_sim_attention(q, k, v, key_mask):
    dim, k_len = q.shape[-1], k.shape[-2]
    q, k = map(l2norm, (q, k))

    sim = einsum('... i d, ... j d -> ... i j', q, k) * COSINE_SIM_SCALE

    key_mask = rearrange(key_mask, 'b j -> b 1 1 j')
    sim = jnp.where(key_mask, sim, MASK_VALUE)

    attn = nn.softmax(sim, axis = -1)
    return einsum('... i j, ... j d -> ... i d', attn, v)
