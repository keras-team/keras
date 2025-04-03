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

"""Module containing fused attention forward and backward pass."""
from __future__ import annotations

import functools
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import jax.numpy as jnp
import numpy as np

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


def mha_forward_kernel(
    q_ref,
    k_ref,
    v_ref,  # Input arrays
    segment_ids_ref: jax.Array | None,  # segment_id arrays
    o_ref: Any,  # Output
    *residual_refs: Any,  # Residual outputs
    num_heads: int,
    sm_scale: float,
    causal: bool,
    block_q: int,
    block_d: int,
    block_k: int,
):
  seq_len = k_ref.shape[0]
  start_q = pl.program_id(0)

  # o is the buffer where we accumulate the output on sram.
  # m_i and l_i (see FlashAttention paper) are updated during the k,v loop.
  m_i = jnp.zeros(block_q, dtype=jnp.float32) - float('inf')
  l_i = jnp.zeros(block_q, dtype=jnp.float32)
  # acc is the buffer where we accumulate the output on sram.
  o = jnp.zeros((block_q, block_d), dtype=jnp.float32)

  # Load q: it will stay in L1 throughout. Indices form a matrix because we
  # read, compute, and write all in 2d chunks. 1 element ~= 1 CUDA thread index.
  # q tile has shape [block_q, block_d], block_d == head_dim.
  curr_q_slice = pl.dslice(start_q * block_q, block_q)
  q = q_ref[...]
  q_segment_ids = (
      None
      if segment_ids_ref is None
      else pl.load(segment_ids_ref, (curr_q_slice,))
  )
  # In FlashAttention algorithm 1 there are 2 loops: slow over tiles of kv (size
  # (Bc == block_k here), and fast over blocks of q (size Br == block_q here).
  # Here we only loop over blocks of kv to process entire seq_len, the loop over
  # blocks of q is carried out by the grid.
  def body(start_k, carry):
    o_prev, m_prev, l_prev = carry
    curr_k_slice = pl.dslice(start_k * block_k, block_k)

    k = pl.load(k_ref, (curr_k_slice, slice(None)))
    qk = pl.dot(q, k.T)   # [block_q, block_k]
    if sm_scale != 1.:
      qk *= sm_scale  # [block_q, block_k]

    # Avoids Triton crash.
    # if num_heads > 2:
    #   qk = qk.astype(q_ref.dtype)
    #   qk = qk.astype(jnp.float32)

    if causal or segment_ids_ref is not None:
      mask = None
      if segment_ids_ref is not None:
        kv_segment_ids = pl.load(segment_ids_ref, (curr_k_slice,))
        mask = segment_mask(q_segment_ids, kv_segment_ids)
      if causal:
        span_q = start_q * block_q + jnp.arange(block_q)
        span_k = start_k * block_k + jnp.arange(block_k)
        causal_mask = span_q[:, None] >= span_k[None, :]
        mask = (
            causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        )
      # Apply mask to qk.
      qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

    m_curr = qk.max(axis=-1)
    m_next = jnp.maximum(m_prev, m_curr)
    correction = jnp.exp(m_prev - m_next)
    l_prev_corr = correction * l_prev
    s_curr = jnp.exp(
        qk - m_next[:, None]
    )  # Use m_next instead of m_curr to avoid a correction on l_curr
    l_curr = s_curr.sum(axis=-1)
    l_next = l_prev_corr + l_curr
    o_prev_corr = correction[:, None] * o_prev
    v = pl.load(v_ref, (curr_k_slice, pl.dslice(block_d)))
    o_curr = pl.dot(s_curr.astype(v.dtype), v)

    o_next = o_prev_corr + o_curr
    return o_next, m_next, l_next
  if causal:
    # Ceildiv (`pl.cdiv` and `//` do not work due to type of start_q)
    upper_bound = lax.div(block_q * (start_q + 1) + block_k - 1, block_k)
  else:
    upper_bound = pl.cdiv(seq_len, block_k)
  o, m_i, l_i = lax.fori_loop(0, upper_bound, body, (o, m_i, l_i))

  # We keep an unscaled version of o during the scan over seq_len. Scaling it
  # by the last l_i gives us the correct final output. See section 3.1.1 in the
  # FlashAttention-2 paper: https://arxiv.org/pdf/2307.08691.
  o /= l_i[:, None]

  if residual_refs:
    lse_ref = residual_refs[0]
    lse_ref[...] = m_i + jnp.log(l_i)
  # Write output to dram.
  o_ref[...] = o.astype(o_ref.dtype)

def segment_mask(
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
):
  # [B, T, 1] or [T, 1]
  q_segment_ids = jnp.expand_dims(q_segment_ids, axis=-1)
  # [B, 1, S] or [1, S]
  if kv_segment_ids.ndim == 1:
    kv_segment_ids = jnp.expand_dims(kv_segment_ids, axis=0)
  else:
    kv_segment_ids = jnp.expand_dims(kv_segment_ids, axis=1)
  return jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)


@functools.partial(
    jax.custom_vjp, nondiff_argnums=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
)
@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "causal",
        "block_q",
        "block_k",
        "backward_pass_impl",
        "num_warps",
        "num_stages",
        "grid",
        "interpret",
        "debug",
    ],
)
def mha(
    q,
    k,
    v,
    segment_ids: jnp.ndarray | None,
    sm_scale: float = 1.0,
    causal: bool = False,
    block_q: int = 128,
    block_k: int = 128,
    backward_pass_impl: str = "triton",
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
):
  del backward_pass_impl
  batch_size, q_seq_len, num_heads, head_dim = q.shape
  kv_seq_len = k.shape[1]
  block_q = min(block_q, q_seq_len)
  block_k = min(block_k, kv_seq_len)
  # Heuristics.
  grid_ = grid
  if grid_ is None:
    grid_ = (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)

  num_warps_ = num_warps
  if num_warps_ is None:
    num_warps_ = 4 if head_dim <= 64 else 8
  kernel = functools.partial(mha_forward_kernel, num_heads=num_heads,
                             sm_scale=sm_scale, block_q=block_q,
                             block_k=block_k, block_d=head_dim,
                             causal=causal)

  in_specs = [
      pl.BlockSpec(
          (None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)
      ),
      pl.BlockSpec(
          (None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)
      ),
      pl.BlockSpec(
          (None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)
      ),
  ]
  in_specs.append(
      None  # type: ignore[arg-type]
      if segment_ids is None
      else pl.BlockSpec((None, kv_seq_len), lambda _, j, k: (j, 0))
  )
  out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  return pl.pallas_call(
      kernel,
      grid=grid_,
      in_specs=in_specs,
      out_specs=pl.BlockSpec(
          (None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)
      ),
      compiler_params=plgpu.TritonCompilerParams(
          num_warps=num_warps_, num_stages=num_stages),
      out_shape=out_shape,
      debug=debug,
      interpret=interpret,
      name="mha_forward",
  )(q, k, v, segment_ids)


def _mha_forward(
    q,
    k,
    v,
    segment_ids: jax.Array | None,
    sm_scale: float,
    causal: bool,
    block_q: int,
    block_k: int,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
):
  del backward_pass_impl
  batch_size, q_seq_len, num_heads, head_dim = q.shape
  kv_seq_len = k.shape[1]
  block_q = min(block_q, q_seq_len)
  block_k = min(block_k, kv_seq_len)
  # Heuristics.
  grid_ = grid
  if grid_ is None:
    grid_ = (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)

  num_warps_ = num_warps
  if num_warps_ is None:
    num_warps_ = 4 if head_dim <= 64 else 8
  kernel = functools.partial(mha_forward_kernel, num_heads=num_heads,
                             sm_scale=sm_scale, causal=causal, block_q=block_q,
                             block_k=block_k, block_d=head_dim)
  out_shape = [
      jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),  # out
      jax.ShapeDtypeStruct(
          shape=(batch_size, num_heads, q_seq_len), dtype=jnp.float32  # lse
      ),
  ]
  in_specs = [
      pl.BlockSpec(
          (None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)
      ),
      pl.BlockSpec(
          (None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)
      ),
      pl.BlockSpec(
          (None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)
      ),
  ]
  in_specs.append(
      None  # type: ignore[arg-type]
      if segment_ids is None
      else pl.BlockSpec((None, kv_seq_len), lambda _, j, k: (j, 0))
  )
  out, lse = pl.pallas_call(
      kernel,
      grid=grid_,
      in_specs=in_specs,
      out_specs=[
          pl.BlockSpec(
              (None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)
          ),
          pl.BlockSpec((None, None, block_q), lambda i, j, k: (j, k, i)),
      ],
      compiler_params=dict(
          triton=dict(num_warps=num_warps_, num_stages=num_stages)
      ),
      out_shape=out_shape,
      debug=debug,
      interpret=interpret,
      name="mha_forward",
  )(q, k, v, segment_ids)
  return out, (q, k, v, segment_ids, out, lse)


def _preprocess_backward_kernel(out_ref, dout_ref, delta_ref):
  # load
  o = out_ref[...].astype(jnp.float32)
  do = dout_ref[...].astype(jnp.float32)
  # compute
  delta = jnp.sum(o * do, axis=1)
  # write-back
  delta_ref[...] = delta.astype(delta_ref.dtype)

@jax.named_scope("preprocess_backward")
def _preprocess_backward(out, do, lse, block_q: int,
                         debug: bool, interpret: bool):
  batch_size, seq_len, num_heads, head_dim = out.shape
  out_shape = jax.ShapeDtypeStruct(lse.shape, lse.dtype)
  delta = pl.pallas_call(
      _preprocess_backward_kernel,
      grid=(pl.cdiv(seq_len, block_q), batch_size, num_heads),
      in_specs=[
          pl.BlockSpec(
              (None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)
          ),
          pl.BlockSpec(
              (None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)
          ),
      ],
      out_specs=pl.BlockSpec((None, None, block_q), lambda i, j, k: (j, k, i)),
      compiler_params=dict(triton=dict(num_warps=4, num_stages=3)),
      out_shape=out_shape,
      debug=debug,
      interpret=interpret,
      name="mha_preprocess_backward",
  )(out, do)
  return delta


# This kernel computes dK_i, dV_i and dQ_i in parallel across the sequence
# length.
# Inspired by the triton tutorial: https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py
def mha_backward_kernel(
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    segment_ids_ref: jax.Array | None,
    out_ref,
    do_scaled_ref,
    lse_ref,
    delta_ref,
    # Outputs
    dq_ref,
    dk_ref,
    dv_ref,
    *,
    sm_scale: float,
    causal: bool,
    block_q1: int,
    block_k1: int,
    block_q2: int,
    block_k2: int,
    block_d: int,
):
  del out_ref  # Not needed
  q_seq_len = q_ref.shape[0]
  kv_seq_len = k_ref.shape[0]

  # Scan #1: dK and dV
  #   1. Load a block of K and V of size (block_k1, head_dim) in SMEM.
  #   2. Iterate through Q in chunks of (block_q1, head_dim) to accumulate
  #      dK and dV.
  start_k = pl.program_id(2)
  curr_k_slice = pl.dslice(start_k * block_k1, block_k1)

  dv = jnp.zeros([block_k1, block_d], dtype=jnp.float32)
  dk = jnp.zeros([block_k1, block_d], dtype=jnp.float32)

  v = pl.load(v_ref, (curr_k_slice, slice(None)))
  k = pl.load(k_ref, (curr_k_slice, slice(None)))
  span_k = start_k * block_k1 + jnp.arange(block_k1)
  kv_segment_ids = (
      None
      if segment_ids_ref is None
      else pl.load(segment_ids_ref, (curr_k_slice,))
  )

  def inner_loop_dkdv(start_q, carry):
    dv, dk = carry
    curr_q_slice = pl.dslice(start_q * block_q1, block_q1)

    q = pl.load(q_ref, (curr_q_slice, slice(None)))
    qk = pl.dot(q, k.T)
    if sm_scale != 1.0:
      qk *= sm_scale

    if causal or segment_ids_ref is not None:
      mask = None
      if segment_ids_ref is not None:
        q_segment_ids = pl.load(segment_ids_ref, (curr_q_slice,))
        mask = segment_mask(q_segment_ids, kv_segment_ids)

      if causal:
        span_q = start_q * block_q1 + jnp.arange(block_q1)
        causal_mask = span_q[:, None] >= span_k[None, :]
        mask = (
            causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        )
      qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

    lse = pl.load(lse_ref, (curr_q_slice,))
    di = pl.load(delta_ref, (curr_q_slice,))
    do = pl.load(do_scaled_ref, (curr_q_slice, slice(None)))

    p = jnp.exp(qk - lse[:, None])
    dv = dv + pl.dot(p.astype(do.dtype).T, do)
    dp = jnp.zeros((block_q1, block_k1), dtype=jnp.float32) - di[:, None]
    dp = dp + pl.dot(do, v.T)
    ds = p * dp
    if sm_scale != 1.0:
      ds = ds * sm_scale
    dk = dk + pl.dot(ds.astype(q_ref.dtype).T, q)

    return dv, dk

  lower_bound = lax.div(start_k * block_k1, block_q1) if causal else 0
  dv, dk = lax.fori_loop(
      lower_bound, pl.cdiv(q_seq_len, block_q1), inner_loop_dkdv, (dv, dk)
  )
  dv_ref[...] = dv.astype(dv_ref.dtype)
  dk_ref[...] = dk.astype(dk_ref.dtype)

  del dv, dk

  # Scan #2: dQ
  #   1. Load a block of Q of size (block_q2, head_dim) in SMEM.
  #   2. Iterate through K and V in chunks of (block_k2, head_dim) to
  #     accumulate dQ.
  start_q = pl.program_id(2)
  curr_q_slice = pl.ds(start_q * block_q2, block_q2)
  span_q = start_q * block_q2 + jnp.arange(block_q2)
  dq = jnp.zeros([block_q2, block_d], dtype=jnp.float32)

  q = pl.load(q_ref, (curr_q_slice, slice(None)))
  q_segment_ids = (
      None
      if segment_ids_ref is None
      else pl.load(segment_ids_ref, (curr_q_slice,))
  )
  lse = pl.load(lse_ref, (curr_q_slice,))
  do = pl.load(do_scaled_ref, (curr_q_slice, slice(None)))
  di = pl.load(delta_ref, (curr_q_slice,))

  def inner_loop_dq(start_k, dq):
    curr_k_slice = pl.dslice(start_k * block_k2, block_k2)
    k = pl.load(k_ref, (curr_k_slice, slice(None)))
    v = pl.load(v_ref, (curr_k_slice, slice(None)))

    qk = pl.dot(q, k.T)
    if sm_scale != 1.0:
      qk *= sm_scale

    if causal or segment_ids_ref is not None:
      mask = None
      if segment_ids_ref is not None:
        kv_segment_ids = pl.load(segment_ids_ref, (curr_k_slice,))
        mask = segment_mask(q_segment_ids, kv_segment_ids)

      if causal:
        span_k = start_k * block_k2 + jnp.arange(block_k2)
        causal_mask = span_q[:, None] >= span_k[None, :]
        mask = (
            causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        )
      qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

    p = jnp.exp(qk - lse[:, None])
    dp = jnp.zeros((block_q2, block_k2), dtype=jnp.float32) - di[:, None]
    dp = dp + pl.dot(do, v.T)
    ds = p * dp
    if sm_scale != 1.0:
      ds = ds * sm_scale

    dq = dq + pl.dot(ds.astype(k.dtype), k).astype(dq.dtype)

    return dq

  if causal:
    upper_bound = lax.div((start_q + 1) * block_q2, block_k2)
  else:
    upper_bound = pl.cdiv(kv_seq_len, block_k2)

  dq = lax.fori_loop(0, upper_bound, inner_loop_dq, (dq))
  dq_ref[...] = dq.astype(dq_ref.dtype)


def _mha_backward(sm_scale: float, causal: bool, block_q: int, block_k: int,
                  backward_pass_impl: str, num_warps: int | None,
                  num_stages: int, grid: Any, interpret: bool,
                  debug: bool, res, do):
  del num_warps, num_stages, grid
  q, k, v, segment_ids, out, lse = res

  if backward_pass_impl == "xla":
    return jax.vjp(
        functools.partial(mha_reference, sm_scale=sm_scale, causal=causal),
        q,
        k,
        v,
        segment_ids,
    )[1](do)
  elif backward_pass_impl == "triton":
    batch_size, q_seq_len, num_heads, head_dim = q.shape
    kv_seq_len = k.shape[1]
    block_q = min(block_q, q_seq_len)
    block_k = min(block_k, kv_seq_len)
    delta = _preprocess_backward(out, do, lse, block_q, debug, interpret)
    out_shapes = [
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(v.shape, v.dtype),
    ]

    in_specs = [
        pl.BlockSpec(
            (None, q_seq_len, None, head_dim), lambda i, j, _: (i, 0, j, 0)
        ),
        pl.BlockSpec(
            (None, kv_seq_len, None, head_dim), lambda i, j, _: (i, 0, j, 0)
        ),
        pl.BlockSpec(
            (None, kv_seq_len, None, head_dim), lambda i, j, _: (i, 0, j, 0)
        ),
        pl.BlockSpec(
            (None, q_seq_len, None, head_dim), lambda i, j, _: (i, 0, j, 0)
        ),
        pl.BlockSpec(
            (None, q_seq_len, None, head_dim), lambda i, j, _: (i, 0, j, 0)
        ),
        pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),
        pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),
    ]
    if segment_ids is None:
      in_specs.insert(3, None)  # type: ignore[arg-type]
    else:
      in_specs.insert(3, pl.BlockSpec((None, kv_seq_len), lambda i, j, _: (i, 0)))

    grid = (batch_size, num_heads, pl.cdiv(kv_seq_len, block_k))
    num_warps = 8
    dq, dk, dv = pl.pallas_call(
        functools.partial(
            mha_backward_kernel,
            sm_scale=sm_scale,
            causal=causal,
            block_q1=block_q,
            block_k1=block_k,
            block_q2=block_q,
            block_k2=block_k,
            block_d=head_dim,
        ),
        out_shape=out_shapes,
        in_specs=in_specs,
        grid=grid,
        out_specs=[
            pl.BlockSpec(
                (None, block_q, None, head_dim),
                lambda i, j, k: (i, k, j, 0),  # dq
            ),
            pl.BlockSpec(
                (None, block_k, None, head_dim),
                lambda i, j, k: (i, k, j, 0),  # dk
            ),
            pl.BlockSpec(
                (None, block_k, None, head_dim),
                lambda i, j, k: (i, k, j, 0),  # dv
            ),
        ],
        name="mha_backward",
        debug=debug,
        interpret=interpret,
        compiler_params=dict(triton=dict(num_warps=num_warps, num_stages=2)),
    )(q, k, v, segment_ids, out, do, lse, delta)
  else:
    raise ValueError(f"Invalid backward pass implementation: {backward_pass_impl}")
  return dq.astype(q.dtype), dk, dv, None
mha.defvjp(_mha_forward, _mha_backward)


@functools.partial(jax.jit, static_argnames=['sm_scale', 'causal'])
def mha_reference(
    q,
    k,
    v,
    segment_ids: jnp.ndarray | None,
    sm_scale=1.0,
    causal: bool = False,
):
  q_seq_len = q.shape[1]
  kv_seq_len = k.shape[1]
  logits = jnp.einsum('bqhc,bkhc->bhqk', q, k).astype(jnp.float32)
  mask = None
  if segment_ids is not None:
    mask = jnp.expand_dims(segment_mask(segment_ids, segment_ids), 1)
    mask = jnp.broadcast_to(mask, logits.shape)
  if causal:
    causal_mask = jnp.tril(jnp.ones((1, 1, q_seq_len, kv_seq_len), dtype=bool))
    causal_mask = jnp.broadcast_to(causal_mask, logits.shape)
    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
  logits = logits if mask is None else jnp.where(mask, logits, float("-inf"))
  weights = jax.nn.softmax(logits * sm_scale).astype(q.dtype)
  return jnp.einsum('bhqk,bkhc->bqhc', weights, v)
