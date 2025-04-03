# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module containing decode attention."""
from __future__ import annotations
import math

import functools
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import jax.numpy as jnp

def attn_forward_kernel(
    # inputs
    q_ref,           # [num_heads, head_dim]
    k_ref,           # [k_seq_len, head_dim]
    v_ref,           # [k_seq_len, head_dim]
    start_idx_ref,   # [] (i.e., scalar)
    kv_seq_len_ref,  # [] (i.e., scalar)
    # outputs
    o_ref: Any,      # [num_heads, head_dim]
    *residual_refs: Any,  # Residual outputs: [num_heads,], [num_heads,]
    sm_scale: float,
    block_k: int,
    block_h: int,
    num_heads: int,
):
  _, head_dim = q_ref.shape
  split_k_seq_len, _ = k_ref.shape
  prog_i, prog_j = pl.program_id(0), pl.program_id(1)
  q_slice = pl.ds(0, block_h)
  q_mask = (jnp.arange(block_h) < num_heads - block_h * prog_i)[:, None]

  def _compute(start_idx, kv_seq_len, o, m_i, l_i):
    # Load q: it will stay in L1 throughout. Indices form a matrix because we
    # read, compute, and write all in 2d chunks. 1 element ~= 1 CUDA thread index.
    # q tile has shape [block_h, head_dim].
    q = pl.load(q_ref, (q_slice, pl.ds(None)), mask=q_mask)

    def _dot(a, b):
      # if a.shape[0] == 1:
      #   # Use matrix vector product
      #   return (a.T * b).sum(axis=0, keepdims=True)
      return pl.dot(a, b)

    mask_indices = jnp.arange(block_k)

    # Loop over blocks of kv to process entire kv seq_len.
    # Grid loops over q blocks over num_heads.
    def body(start_k, carry):
      o_prev, m_prev, l_prev = carry
      curr_k_slice = pl.ds(start_k * block_k, block_k)

      k = pl.load(k_ref, (curr_k_slice, slice(None)))
      qk = _dot(q, k.T)  # [block_h, block_k]
      if sm_scale != 1.0:
        qk *= sm_scale  # [block_h, block_k]

      # apply mask if start or sequence length is specified
      if start_idx_ref is not None or kv_seq_len_ref is not None:
        indices = (prog_j * split_k_seq_len + start_k * block_k + mask_indices)
        mask = ((indices >= start_idx) & (indices < kv_seq_len))[None, :]
        qk += (~mask) * (0.7 * jnp.finfo(qk.dtype).min)

      m_curr = qk.max(axis=-1)
      m_next = jnp.maximum(m_prev, m_curr)
      correction = jnp.exp(m_prev - m_next)
      l_prev_corr = correction * l_prev
      s_curr = jnp.exp(
          qk - m_next[:, None]
      )  # Use m_next instead of m_curr to avoid a correction on l_curr
      l_curr = s_curr.sum(axis=-1)
      l_next = l_prev_corr + l_curr
      v = pl.load(v_ref, (curr_k_slice, slice(None)))
      o_curr = _dot(s_curr.astype(v.dtype), v)

      # flash2 unscaled_o
      o_next = correction[:, None] * o_prev + o_curr
      return o_next, m_next, l_next

    max_it = jnp.minimum(pl.cdiv((kv_seq_len - prog_j * split_k_seq_len),
                                 block_k), split_k_seq_len // block_k)
    (o, m_i, l_i) = lax.fori_loop(0, max_it, body, (o, m_i, l_i))
    return o, m_i, l_i

  # o is the buffer where we accumulate the output on sram.
  # m_i and l_i (see FlashAttention2 paper) are updated during the k,v loop.
  m_i = jnp.zeros(block_h, dtype=jnp.float32) + jnp.finfo(jnp.float32).min
  l_i = jnp.zeros(block_h, dtype=jnp.float32)
  o = jnp.zeros((block_h, head_dim), dtype=jnp.float32)

  start_idx = split_k_seq_len * prog_j
  if start_idx_ref is not None:
    start_idx = jnp.maximum(start_idx, pl.load(start_idx_ref, ()))
  kv_seq_len = (prog_j + 1) * split_k_seq_len  # lower bound on actual k_seq_len
  if kv_seq_len_ref is not None:
    kv_seq_len = jnp.minimum(kv_seq_len, pl.load(kv_seq_len_ref, ()))

  if start_idx_ref is None and kv_seq_len is None:
    o, m_i, l_i = _compute(start_idx, kv_seq_len, o, m_i, l_i)
  else:
    o, m_i, l_i = jax.lax.cond(
      start_idx >= kv_seq_len, lambda: (o, m_i, l_i),
      lambda: _compute(start_idx, kv_seq_len, o, m_i, l_i))

  # Write output to dram.
  if residual_refs:
    l_ref, m_ref = residual_refs
    vec_q_mask = q_mask.reshape(-1) if q_mask is not None else None
    pl.store(l_ref, q_slice, l_i, mask=vec_q_mask)
    pl.store(m_ref, q_slice, m_i, mask=vec_q_mask)
  o = o.astype(o_ref.dtype)
  pl.store(o_ref, (q_slice, pl.ds(None)), o, mask=q_mask)


def decode_attn_unbatched(
    q,           # [num_heads, head_dim]
    k,           # [k_seq_len, head_dim]
    v,           # [k_seq_len, head_dim]
    start_idx,   # []
    kv_seq_len,  # []
    sm_scale: float,
    block_h: int,
    block_k: int,
    k_splits: int,
    num_warps: int | None,
    num_stages: int,
    grid: tuple[int, ...] | None,
    interpret: bool,
    debug: bool,
    return_residuals: bool
):
  num_heads, head_dim = q.shape
  k_seq_len, _ = k.shape
  # Pad num query heads to 16 if needed, and slice output at the end.
  head_splits = pl.cdiv(num_heads, block_h)
  grid_ = grid
  if grid_ is None:
    grid_ = (head_splits, k_splits)

  assert (
      k_seq_len % k_splits == 0
  ), f"{k_seq_len=} must be divisible by {k_splits=}"
  assert k_seq_len // k_splits >= 16, (
    f"{k_seq_len=} divided by {k_splits=} must be >= 16.")
  assert block_k >= 16, "block_k must be >= 16"
  k = k.reshape(k_splits, k_seq_len // k_splits, head_dim)
  v = v.reshape(k_splits, k_seq_len // k_splits, head_dim)
  split_k_seq_len = k_seq_len // k_splits
  block_k = min(block_k, split_k_seq_len)
  assert split_k_seq_len % block_k == 0, (
    f"Sequence length ({k_seq_len=}) split by {k_splits=} must by divisible by"
    f" {block_k=}")
  num_warps_ = num_warps
  if num_warps_ is None:
    num_warps_ = 4
  kernel = functools.partial(
      attn_forward_kernel,
      sm_scale=sm_scale,
      block_k=block_k,
      block_h=block_h,
      num_heads=num_heads,
  )

  o, l, m = pl.pallas_call(
    kernel,
    grid=grid_,
    in_specs=[
      pl.BlockSpec((block_h, head_dim), lambda i, j: (i, 0)),
      pl.BlockSpec((None, split_k_seq_len, head_dim), lambda i, j: (j, 0, 0)),
      pl.BlockSpec((None, split_k_seq_len, head_dim), lambda i, j: (j, 0, 0)),
    ]
    + [None if start_idx is None else pl.BlockSpec((), lambda i, j: ())]
    + [None if kv_seq_len is None else pl.BlockSpec((), lambda i, j: ())],
    out_specs=[
      pl.BlockSpec((None, block_h, head_dim), lambda i, j: (j, i, 0)),  # o
      pl.BlockSpec((None, block_h), lambda i, j: (j, i)),  # l
      pl.BlockSpec((None, block_h), lambda i, j: (j, i)),  # m
    ],
    compiler_params=plgpu.TritonCompilerParams(
      num_warps=num_warps_, num_stages=num_stages
    ),
    out_shape=[
      jax.ShapeDtypeStruct(shape=(k_splits, *q.shape), dtype=q.dtype),  # o
      jax.ShapeDtypeStruct(
        shape=(k_splits, num_heads), dtype=jnp.float32
      ),  # l
      jax.ShapeDtypeStruct(
        shape=(k_splits, num_heads), dtype=jnp.float32
      ),  # m
    ],
    debug=debug,
    interpret=interpret,
    name="mha_forward",
  )(q, k, v, start_idx, kv_seq_len)

  # final round of flash
  m_next = m.max(axis=0)
  # TODO(b/389925439): This barrier is necessary to prevent NaNs/invalid
  # values appearing after JIT compilation.
  m_next = lax.optimization_barrier(m_next)
  correction = jnp.exp(m - m_next[None])
  o = o * correction[:, :, None].astype(o.dtype)
  l_next = (l * correction).sum(axis=0)
  eps = jnp.finfo(l_next.dtype).eps
  o = o.sum(axis=0) / (l_next[:, None].astype(o.dtype) + eps)
  if return_residuals:
    return o, (l_next, m_next)
  else:
    return o


@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "block_h",
        "block_k",
        "k_splits",
        "num_warps",
        "num_stages",
        "grid",
        "interpret",
        "debug",
        "return_residuals"
    ],
)
def mqa(
    q,                # [batch_size, num_heads, head_dim]
    k,                # [batch_size, k_seq_len, head_dim]
    v,                # [batch_size, k_seq_len, head_dim]
    start_idx=None,   # [batch_size]
    kv_seq_len=None,  # [batch_size]
    sm_scale: float | None = None,
    block_h: int = 16,
    block_k: int = 256,
    k_splits: int = 16,
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
    return_residuals: bool = False
):
  sm_scale = sm_scale if sm_scale is not None else (1 / math.sqrt(q.shape[-1]))
  bs = q.shape[0]
  if start_idx is not None:
    start_idx = jnp.broadcast_to(start_idx, (bs,))
  if kv_seq_len is not None:
    kv_seq_len = jnp.broadcast_to(kv_seq_len, (bs,))
  inner = functools.partial(
      decode_attn_unbatched,
      sm_scale=sm_scale,
      block_h=block_h,
      block_k=block_k,
      k_splits=k_splits,
      num_warps=num_warps,
      num_stages=num_stages,
      grid=grid,
      interpret=interpret,
      debug=debug,
      return_residuals=return_residuals
  )
  return jax.vmap(inner)(q, k, v, start_idx, kv_seq_len)


@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "block_h",
        "block_k",
        "k_splits",
        "num_warps",
        "num_stages",
        "grid",
        "interpret",
        "debug",
        "return_residuals"
    ],
)
def gqa(
    q,                # [batch_size, num_q_heads, head_dim]
    k,                # [batch_size, k_seq_len, num_kv_heads, head_dim]
    v,                # [batch_size, k_seq_len, num_kv_heads, head_dim]
    start_idx=None,   # [batch_size]
    kv_seq_len=None,  # [batch_size]
    sm_scale: float | None = None,
    block_h: int = 16,
    block_k: int = 128,
    k_splits: int = 16,
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
    return_residuals: bool = False,
):
  sm_scale = sm_scale if sm_scale is not None else (1 / math.sqrt(q.shape[-1]))
  batch_size, q_heads, head_dim = q.shape
  k_seq_len, kv_heads = k.shape[1], k.shape[2]
  assert kv_heads == v.shape[2]
  assert q_heads % kv_heads == 0
  if start_idx is not None:
    assert start_idx.ndim in (0, 1)
    start_idx = jnp.broadcast_to(jnp.asarray(start_idx)[..., None],
                                 (batch_size, kv_heads))
  if kv_seq_len is not None:
    assert kv_seq_len.ndim in (0, 1)
    kv_seq_len = jnp.broadcast_to(jnp.asarray(kv_seq_len)[..., None],
                                  (batch_size, kv_heads))
  q_heads_per_kv_head = q_heads // kv_heads
  q_reshaped = q.reshape(batch_size, kv_heads, q_heads_per_kv_head, head_dim)
  k_transposed = jnp.swapaxes(
      k, 1, 2
  )  # [batch_size, num_kv_heads, k_seq_len, head_dim]
  v_transposed = jnp.swapaxes(
      v, 1, 2
  )  # [batch_size, num_kv_heads, k_seq_len, head_dim]
  inner = functools.partial(
      decode_attn_unbatched,
      sm_scale=sm_scale,
      block_h=block_h,
      block_k=block_k,
      k_splits=k_splits,
      num_warps=num_warps,
      num_stages=num_stages,
      grid=grid,
      interpret=interpret,
      debug=debug,
      return_residuals=return_residuals,
  )
  with_kv_heads = jax.vmap(inner)
  o, *res = jax.vmap(with_kv_heads)(
      q_reshaped, k_transposed, v_transposed, start_idx, kv_seq_len
  )
  o = o.reshape(batch_size, q_heads, head_dim)
  if return_residuals:
    l, m = res[0]
    l = l.reshape(batch_size, q_heads)
    m = m.reshape(batch_size, q_heads)
    return o, (l, m)
  else:
    return o


@functools.partial(jax.jit, static_argnames=["sm_scale", "return_residuals"])
def mqa_reference(
    q,                # [bs, num_q_heads, head_dim]
    k,                # [bs, k_seq_len, head_dim]
    v,                # [bs, k_seq_len, head_dim]
    start_idx=None,   # [bs]
    kv_seq_len=None,  # [bs]
    sm_scale=None,
    return_residuals=False
):
  original_dtype = q.dtype
  q = q.astype(jnp.float32)
  k = k.astype(jnp.float32)
  bs = q.shape[0]
  sm_scale = sm_scale if sm_scale is not None else (1 / math.sqrt(q.shape[-1]))
  logits = jnp.einsum("bnd,bsd->bns", q, k).astype(jnp.float32)
  if sm_scale is not None and sm_scale != 1.0:
    logits = logits * sm_scale
  if start_idx is not None or kv_seq_len is not None:
    start_idx = jnp.broadcast_to(0 if start_idx is None else start_idx, (bs,))
    kv_seq_len = jnp.broadcast_to(k.shape[1] if kv_seq_len is None
                                  else kv_seq_len, (bs,))
    mask = ((jnp.arange(k.shape[1])[None, :] >= start_idx[:, None])
            & (jnp.arange(k.shape[1])[None, :] < kv_seq_len[:, None]))
    mask = mask[:, None, :]
    logits = logits + (~mask) * (0.7 * jnp.finfo(logits.dtype).min)

  m = logits.max(axis=-1)
  s = jnp.exp(logits - m[..., None])
  l = s.sum(axis=-1)
  s = s / l[..., None]
  o = jnp.einsum("bns,bsd->bnd", s, v).astype(original_dtype)

  if return_residuals:
    return o, (l, m)
  else:
    return o


@functools.partial(jax.jit, static_argnames=["sm_scale"])
def mha_reference(
    q,                # [bs, num_q_heads, head_dim]
    k,                # [bs, k_seq_len, num_k_heads, head_dim]
    v,                # [bs, k_seq_len, num_v_heads, head_dim]
    start_idx=None,   # [bs]
    kv_seq_len=None,  # [bs]
    sm_scale=None,
):
  bs = q.shape[0]
  sm_scale = sm_scale if sm_scale is not None else (1 / math.sqrt(q.shape[-1]))
  assert q.shape[1] == k.shape[2]
  logits = jnp.einsum("bnd,bsnd->bns", q, k).astype(jnp.float32)
  if start_idx is not None or kv_seq_len is not None:
    start_idx = jnp.broadcast_to(0 if start_idx is None else start_idx, (bs,))
    kv_seq_len = jnp.broadcast_to(k.shape[1] if kv_seq_len is None
                                  else kv_seq_len, (bs,))
    mask = ((jnp.arange(k.shape[1])[None, :] >= start_idx[:, None])
            & (jnp.arange(k.shape[1])[None, :] < kv_seq_len[:, None]))
    mask = mask[:, None, :]
    logits = logits + (~mask) * (0.7 * jnp.finfo(logits.dtype).min)
  weights = jax.nn.softmax(logits * sm_scale).astype(q.dtype)
  return jnp.einsum("bns,bsnd->bnd", weights, v)


@functools.partial(jax.jit, static_argnames=["sm_scale", "return_residuals"])
def gqa_reference(
    q,                # [bs, num_q_heads, head_dim]
    k,                # [bs, k_seq_len, num_k_heads, head_dim]
    v,                # [bs, k_seq_len, num_v_heads, head_dim]
    start_idx=None,   # [bs]
    kv_seq_len=None,  # [bs]
    sm_scale=None,
    return_residuals=False
):
  original_dtype = q.dtype
  q = q.astype(jnp.float32)
  k = k.astype(jnp.float32)
  sm_scale = sm_scale if sm_scale is not None else (1 / math.sqrt(q.shape[-1]))
  bs, num_q_heads, head_dim = q.shape
  num_kv_heads = k.shape[2]
  assert num_q_heads % num_kv_heads == 0
  q_reshaped = q.reshape(
      bs, num_kv_heads, num_q_heads // num_kv_heads, head_dim
  )
  k_transposed = jnp.swapaxes(
      k, 1, 2
  )  # [batch_size, num_kv_heads, k_seq_len, head_dim]
  v_transposed = jnp.swapaxes(
      v, 1, 2
  )  # [batch_size, num_kv_heads, k_seq_len, head_dim]
  logits = jnp.einsum("bkgd,bksd->bkgs", q_reshaped, k_transposed).astype(
      jnp.float32
  )
  if sm_scale is not None and sm_scale != 1.0:
    logits = logits * sm_scale
  if start_idx is not None or kv_seq_len is not None:
    start_idx = jnp.broadcast_to(0 if start_idx is None else start_idx, (bs,))
    kv_seq_len = jnp.broadcast_to(k.shape[1] if kv_seq_len is None
                                  else kv_seq_len, (bs,))
    mask = ((jnp.arange(k.shape[1])[None, :] >= start_idx[:, None])
            & (jnp.arange(k.shape[1])[None, :] < kv_seq_len[:, None]))
    mask = mask[:, None, None, :]
    logits = logits + (~mask) * (0.7 * jnp.finfo(logits.dtype).min)

  m = logits.max(axis=-1)
  s = jnp.exp(logits - m[..., None])
  l = s.sum(axis=-1)
  s = s / l[..., None]
  o = jnp.einsum("bkgs,bksd->bkgd", s, v_transposed).astype(original_dtype)
  o = o.reshape(bs, num_q_heads, head_dim)

  if return_residuals:
    l = l.reshape(bs, num_q_heads)
    m = m.reshape(bs, num_q_heads)
    return o, (l, m)
  else:
    return o
