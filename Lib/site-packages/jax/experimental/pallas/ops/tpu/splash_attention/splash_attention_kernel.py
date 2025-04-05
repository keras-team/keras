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

"""Implementation of Sparse Flash Attention, a.k.a. "Splash" attention."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
import enum
import functools
from typing import Any, Literal, NamedTuple, Optional, Union, overload

import jax
from jax import ad_checkpoint
from jax import lax
from jax import tree_util
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask_info as mask_info_lib
import jax.numpy as jnp
import numpy as np

partial = functools.partial
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8
# We predefine some useful dimension numbers for dot_general
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))  # standard matmul
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))  # RHS transposed

# mypy: ignore-errors

class SegmentIds(NamedTuple):
  """SegmentIds for Q and KV sequences.

  SegmentIds are a mechanims to ensure that there is no cross-attention between
  segments (fraction of a sequence) that have been concatenated together into a
  sequence. Each array is a list of ids (integers). Only tokens with the same
  id are allowed to attend to each other.

  The static mask (e.g. causal) is "and-ed" with the segment id mask to form
  the actual attention mask. It is important that the latter does not have any
  all-zero rows (along dimension kv). Otherwise it would result in a invalid
  softmax (the denominator would be 0).
  This condition holds for causal self-attention because in this case segment
  ids form a block diagonal matrix so at least one element in each row is set.
  It is easy to break this condition with non-self-attention configurations.
  Attributes:
    q: segment ids along the Q sequence
    kv: segment ids along the KV sequence
  """

  q: jax.Array  # [q_seq_len]
  kv: jax.Array  # [kv_seq_len]


# Return type of SplashAttention function that implements the custom vjp rule.
SplashCustomReturnType = Union[
    # out, no residuals
    jax.Array,
    # out, residuals:
    tuple[jax.Array, tuple[jax.Array,]]
]

SplashResidualsType = tuple[
    jax.Array,  # q
    jax.Array,  # k
    jax.Array,  # v
    Optional[SegmentIds],  # segment_ids
    jax.Array,  # out
    jax.Array,  # logsumexp
    Optional[mask_info_lib.MaskInfo],  # dq_mask_info
    Optional[mask_info_lib.MaskInfo],  # dkv_mask_info
]

MaskFunctionType = Callable[..., jax.Array]


def get_kernel_name(
    block_metadata: Mapping[str, Any],
    is_mqa: bool,
    save_residuals: bool,
    is_segmented: bool,
    phase: str,
) -> str:
  """Returns a unique name for all SplashAttention kernel variants."""
  assert phase == "dq" or phase == "dkv" or phase == "fwd"
  # Saving residuals is supported only for the fwd phase.
  assert not save_residuals or phase == "fwd"
  residuals = ""
  if save_residuals:
    residuals = "_residuals"
  elif phase == "fwd":
    residuals = "_no_residuals"
  attention_type = "mqa" if is_mqa else "mha"
  segments = "_segmented" if is_segmented else ""
  return f"splash_{attention_type}_{phase}{segments}{residuals}_" + "_".join(
      f"{k}={v}" for k, v in sorted(block_metadata.items())
  )


# Reference attention implementations


@overload
def _attention_reference(
    mask: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    save_residuals: Literal[False],
    mask_value: float,
    custom_type: str,
    attn_logits_soft_cap: float | None,
) -> jax.Array:
  ...


@overload
def _attention_reference(
    mask: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    save_residuals: Literal[True],
    mask_value: float,
    custom_type: str,
    attn_logits_soft_cap: float | None,
) -> tuple[jax.Array, tuple[jax.Array]]:
  ...


def _attention_reference(
    mask: jax.Array,  # [q_seq_len, kv_seq_len]
    q: jax.Array,  # [q_seq_len, head_dim]
    k: jax.Array,  # [kv_seq_len, head_dim]
    v: jax.Array,  # [kv_seq_len, head_dim]
    segment_ids: SegmentIds | None,
    mask_value: float,
    save_residuals: bool,
    custom_type: str,
    attn_logits_soft_cap: float | None,
):
  return _attention_reference_default(  # pytype: disable=bad-return-type
      mask,
      q,
      k,
      v,
      segment_ids,
      mask_value,
      save_residuals,
      custom_type,
      attn_logits_soft_cap,
  )


def _attention_reference_default(
    mask: jax.Array,  # [q_seq_len, kv_seq_len]
    q: jax.Array,  # [q_seq_len, head_dim]
    k: jax.Array,  # [kv_seq_len, head_dim]
    v: jax.Array,  # [kv_seq_len, head_dim]
    segment_ids: SegmentIds | None,
    mask_value: float,
    save_residuals: bool,
    custom_type: str,
    attn_logits_soft_cap: float | None,
):
  del custom_type
  logits = jnp.einsum("sd,td->st", q.astype(jnp.float32), k.astype(jnp.float32))

  if segment_ids is not None:
    mask = jnp.logical_and(
        mask, segment_ids.q[:, None] == segment_ids.kv[None, :]
    )

  if attn_logits_soft_cap is not None:
    logits = jnp.tanh(logits / attn_logits_soft_cap)
    logits = logits * attn_logits_soft_cap

  logits = jnp.where(mask, logits, mask_value)
  m = logits.max(axis=-1)
  s = jnp.exp(logits - m[..., None])
  l = s.sum(axis=-1)
  s = s / l[..., None]

  o = jnp.einsum("st,td->sd", s, v.astype(jnp.float32))

  logsumexp = m + jnp.log(l)
  if save_residuals:
    return o, (logsumexp,)
  return o


def attention_reference(
    mask: jax.Array,  # [q_seq_len, kv_seq_len]
    q: jax.Array,  # [q_seq_len, head_dim]
    k: jax.Array,  # [kv_seq_len, head_dim]
    v: jax.Array,  # [kv_seq_len, head_dim]
    segment_ids: SegmentIds | None,
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
    save_residuals: bool = False,
    custom_type: str = "flash",
    attn_logits_soft_cap: float | None = None,
) -> SplashCustomReturnType:
  return _attention_reference(  # pytype: disable=wrong-arg-types
      mask,
      q,
      k,
      v,
      segment_ids,
      mask_value=mask_value,
      save_residuals=save_residuals,
      custom_type=custom_type,
      attn_logits_soft_cap=attn_logits_soft_cap,
  )


def _attention_reference_custom_fwd(
    mask: jax.Array,  # [q_seq_len, kv_seq_len]
    q: jax.Array,  # [q_seq_len, head_dim]
    k: jax.Array,  # [kv_seq_len, head_dim]
    v: jax.Array,  # [kv_seq_len, head_dim]
    segment_ids: SegmentIds | None,
    mask_value: float,
    save_residuals: bool,
    custom_type: str,
    attn_logits_soft_cap: float | None,
):
  if save_residuals:
    raise NotImplementedError("Higher-order AD not supported.")

  o, (logsumexp,) = _attention_reference(
      mask,
      q,
      k,
      v,
      segment_ids,
      mask_value=mask_value,
      save_residuals=True,
      custom_type=custom_type,
      attn_logits_soft_cap=attn_logits_soft_cap,
  )
  return o, (mask, q, k, v, segment_ids, o, logsumexp)


def _attention_reference_custom_bwd(
    mask_value: float,
    save_residuals: bool,
    custom_type: str,
    attn_logits_soft_cap: float | None,
    res,
    do: jax.Array,
) -> tuple[None, jax.Array, jax.Array, jax.Array, None]:
  del save_residuals
  mask, q, k, v, segment_ids, o, logsumexp = res

  uncapped_logits = jnp.einsum(
      "qc,kc->qk", q, k, preferred_element_type=jnp.float32)

  if attn_logits_soft_cap is not None:
    logits = jnp.tanh(uncapped_logits / attn_logits_soft_cap)
    logits = logits * attn_logits_soft_cap
  else:
    logits = uncapped_logits

  if segment_ids is not None:
    mask = jnp.logical_and(
        mask, segment_ids.q[:, None] == segment_ids.kv[None, :]
    )
  logits = jnp.where(mask, logits, mask_value)

  p = jnp.exp(logits - logsumexp[..., None])
  do = do.astype(jnp.float32)  # pytype: disable=attribute-error
  dv = jnp.einsum("pt,pd->td", p, do).astype(v.dtype)
  dp = jnp.einsum("pd,td->pt", do, v.astype(jnp.float32))

  # These two ways of computing ds are mathematically equivalent. The first
  # involves reducing over the head_dim dimension and the second involves
  # reducing over a sequence dimension. They tend to produce slightly different
  # numerics.
  if custom_type == "flash":
    di = jnp.sum(o.astype(jnp.float32) * do, axis=-1)[..., None]
  else:
    di = jnp.einsum("st,st->s", dp, p)[:, None]
  ds = (dp - di) * p
  if attn_logits_soft_cap is not None:
    normalized = uncapped_logits / attn_logits_soft_cap
    d = jnp.tanh(normalized)
    g = ds * (1 - d)
    ds = g + g * d
  dk = jnp.einsum("sd,st->td", q.astype(jnp.float32), ds).astype(k.dtype)
  dq = jnp.einsum("st,td->sd", ds, k.astype(jnp.float32)).astype(q.dtype)
  return None, dq, dk, dv, None


_attention_reference_custom = jax.custom_vjp(
    _attention_reference, nondiff_argnums=(5, 6, 7, 8)
)
_attention_reference_custom.defvjp(_attention_reference_custom_fwd,
                                   _attention_reference_custom_bwd)


def attention_reference_custom(
    mask: jax.Array,  # [q_seq_len, kv_seq_len]
    q: jax.Array,  # [q_seq_len, head_dim]
    k: jax.Array,  # [kv_seq_len, head_dim]
    v: jax.Array,  # [kv_seq_len, head_dim]
    segment_ids: SegmentIds | None,
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
    save_residuals: bool = False,
    custom_type: str = "flash",
    attn_logits_soft_cap: float | None = None,
):
  return _attention_reference_custom(
      mask,
      q,
      k,
      v,
      segment_ids,
      mask_value,
      save_residuals,
      custom_type=custom_type,
      attn_logits_soft_cap=attn_logits_soft_cap,
  )


def make_attention_reference(
    mask: mask_lib.Mask | np.ndarray,
    is_mqa: bool,
    backward_impl: str = "vanilla",
    **params: Any,
) -> Callable:
  @partial(
      jax.jit,
      static_argnames=[
          "mask_value",
          "save_residuals",
          "attn_logits_soft_cap",
      ],
  )
  def _wrapped(
      mask: jax.Array,
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      segment_ids: SegmentIds | None = None,
      *,
      mask_value: float = DEFAULT_MASK_VALUE,
      save_residuals: bool = False,
      attn_logits_soft_cap: float | None = None,
  ):
    if backward_impl == "custom":
      attn_impl = partial(
          attention_reference_custom, custom_type="flash",
      )
    elif backward_impl == "custom_vanilla":
      attn_impl = partial(
          attention_reference_custom, custom_type="vanilla",
      )
    else:
      attn_impl = attention_reference
    func = partial(
        attn_impl,
        mask_value=mask_value,
        save_residuals=save_residuals,
        attn_logits_soft_cap=attn_logits_soft_cap,
        **params,
    )

    if is_mqa:
      func = jax.vmap(func, in_axes=(0, 0, None, None, None))
      is_grouped = False
    else:
      # In grouped attention (1 < num_kv_heads && num_kv_heads < num_q_heads).
      # We interleave the KV heads across the Q heads.
      # For example: for 8 Q heads and 4 KV heads:
      # Q head [0, 1] see KV head 0
      # Q head [2, 3] see KV head 1
      # Q head [4, 5] see KV head 2
      # Q head [6, 7] see KV head 3
      #
      # The following implementation reshapes Q to expose KV heads and vmaps
      # Across the Q heads so it is similar to MQA.
      # Alternatively we can replicate K/V to match Q like so:
      # k = jnp.repeat(k, q_heads_per_kv_head, axis=0)
      # v = jnp.repeat(v, q_heads_per_kv_head, axis=0)

      kv_heads = k.shape[0]
      assert kv_heads == v.shape[0]
      q_heads, q_seq_len, head_dim = q.shape
      is_grouped = kv_heads < q_heads
      if is_grouped:
        assert q_heads % kv_heads == 0
        assert mask.shape[0] == q_heads
        q_heads_per_kv_head = q_heads // kv_heads
        q = q.reshape((kv_heads, q_heads_per_kv_head, q_seq_len, head_dim))
        mask = mask.reshape((kv_heads, q_heads_per_kv_head, *mask.shape[1:]))

        # Inner-most vmap: iterate over the q heads.
        func = jax.vmap(func, in_axes=(0, 0, None, None, None))

      # Outer-most vmap: iterate over the kv heads.
      func = jax.vmap(func, in_axes=(0, 0, 0, 0, None))

    out = func(mask, q, k, v, segment_ids)

    if is_grouped:

      def reshape_activations(activations):
        if activations.ndim == 4:  # pytype: disable=attribute-error
          kv_heads, q_heads_per_kv_head, q_seq_len, head_dim = activations.shape  # pytype: disable=attribute-error
          return activations.reshape(  # pytype: disable=attribute-error
              kv_heads * q_heads_per_kv_head, q_seq_len, head_dim
          )
        return activations

      def reshape_residuals(residuals):
        if residuals.ndim == 3:
          kv_heads, q_heads_per_kv_head, q_seq_len = residuals.shape
          return residuals.reshape(kv_heads * q_heads_per_kv_head, q_seq_len)
        return residuals

      if save_residuals:
        assert isinstance(out, tuple)
        assert isinstance(out[1], tuple)

        return (reshape_activations(out[0]), (reshape_residuals(out[1][0]),))
      else:
        return reshape_activations(out)
    else:
      return out

  return functools.partial(_wrapped, jnp.array(mask[:, :, :]))


make_masked_mha_reference = partial(make_attention_reference, is_mqa=False)
make_masked_mqa_reference = partial(make_attention_reference, is_mqa=True)


# Splash attention implementation

# We use an IntEnum to make it JSON serializable as regen metadata.
class QKVLayout(enum.IntEnum):
  HEAD_DIM_MINOR = enum.auto()  # [..., seq_len, head_dim]
  SEQ_MINOR = enum.auto()  # [..., head_dim, seq_len]


def from_head_minor(vals: tuple[Any, ...], layout: QKVLayout):
  if layout == QKVLayout.HEAD_DIM_MINOR:
    return vals
  return (*vals[:-2], vals[-1], vals[-2])


@dataclasses.dataclass(frozen=True, slots=True)
class BlockSizes:
  """Tile sizes parameterizing SplashAttention kernels.

  Those parameters have negligible effect on numerics, but affect performance
  greatly.

  Note that changing the layouts only influences the physical layout that the
  kernel will enforce. The logical interface to splash attention always takes
  the head dimension as the minormost one.
  """
  block_q: int
  block_kv: int
  block_kv_compute: int | None = None

  block_q_dkv: int | None = None
  block_kv_dkv: int | None = None
  block_kv_dkv_compute: int | None = None

  block_q_dq: int | None = None
  block_kv_dq: int | None = None

  use_fused_bwd_kernel: bool = False

  q_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  k_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  v_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR

  def __post_init__(self):
    if self.block_kv_compute is None:
      object.__setattr__(self, "block_kv_compute", self.block_kv)
    if self.block_kv_dkv_compute is None:
      object.__setattr__(self, "block_kv_dkv_compute", self.block_kv_dkv)
    if self.use_fused_bwd_kernel:
      if self.block_q_dq is not None or self.block_kv_dq is not None:
        raise ValueError(
            "Block sizes for dq kernel are not needed with a fused kernel."
        )

  @property
  def has_backward_blocks(self) -> bool:
    backward_blocks = (
        self.block_q_dkv, self.block_kv_dkv, self.block_kv_dkv_compute,
    )
    if not self.use_fused_bwd_kernel:
      backward_blocks += (self.block_q_dq, self.block_kv_dq)
    return all(b is not None for b in backward_blocks)

  @classmethod
  def get_default(cls):
    # TODO(apaszke,sharadmv): Select better parameters based on a heuristic.
    return BlockSizes(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        block_q_dkv=128,
        block_kv_dkv=128,
        block_kv_dkv_compute=128,
        block_q_dq=128,
        block_kv_dq=128,
    )


def _next_nonzero(
    h,
    i,
    j,
    data_next_ref,
    block_mask_ref,
    m_next_ref,
    next_i=False,
):
  assert (data_next_ref is None) == (block_mask_ref is None)

  if data_next_ref is None and block_mask_ref is None:
    # Handle the case in which we have no masking nor next data information.
    # Simply fetch the next data and apply the mask for every block.
    assert m_next_ref is None
    next_data = i if next_i else j
    return (
        next_data,
        None,  # next mask
        True,  # should run
        False,  # should not mask
    )

  assert data_next_ref.shape == block_mask_ref.shape
  assert m_next_ref is None or data_next_ref.shape[0] == m_next_ref.shape[0]

  # We are working with one head only. Force the head index to 0.
  if data_next_ref.shape[0] == 1:
    h = 0

  # When scalar-memory data is of types smaller than int32, then we have to
  # upcast it back to use it in the kernel.

  to_i32 = lambda x: x.astype(jnp.int32)

  is_nonzero = to_i32(block_mask_ref[h, i, j]) > 0
  if m_next_ref is None:
    should_not_mask = True
    next_m = None
  else:
    should_not_mask = to_i32(block_mask_ref[h, i, j]) != 1
    next_m = to_i32(m_next_ref[h, i, j])
  next_j = to_i32(data_next_ref[h, i, j])
  return next_j, next_m, is_nonzero, should_not_mask


def _apply_mask_and_soft_cap(
    qk: jax.Array,
    mask_value: float,
    should_not_mask,
    mask_ref,
    q_sequence_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    *,
    attn_logits_soft_cap: float,
    k_slice: pl.Slice,
    k_offset: int | jax.Array,
    bq: int,
    k_in_lanes=True,
    mask_function=None,
) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  assert mask_ref is None or q_sequence_ref is None
  assert (q_sequence_ref is None) == (mask_function is None)

  masks = []
  if mask_ref is not None:
    if k_in_lanes:
      mask = pl.load(mask_ref, (slice(None), k_slice))
    else:
      mask = pl.load(mask_ref, (k_slice, slice(None)))

    masks.append(
        jnp.bitwise_or(mask, jnp.broadcast_to(should_not_mask, mask.shape))
    )
  if mask_function is not None:
    # Compute the mask using the given q_sequence indices.
    # KV indices are computed on the fly. This works because we only support Q
    # sequence sharding. If we wanted to compute Q indices too, then we would
    # need to keep into account the current shard along Q sequence.

    if k_in_lanes:
      assert q_sequence_ref.shape == (bq, NUM_LANES)

      k_sequence = k_offset + jax.lax.broadcasted_iota(
          jnp.int32, (bq, k_slice.size), 1
      )

      repeats, rem = divmod(k_slice.size, NUM_LANES)
      assert rem == 0
      q_sequence = pltpu.repeat(
          q_sequence_ref[...], repeats, axis=1
      )  # [bq, k_slice.size]
    else:
      assert q_sequence_ref.shape == (NUM_SUBLANES, bq)

      k_sequence = k_offset + jax.lax.broadcasted_iota(
          jnp.int32, (k_slice.size, bq), 0
      )
      q_sequence = pl.load(q_sequence_ref, (pl.ds(1), slice(None)))  # [1, bq]
      q_sequence = jnp.broadcast_to(q_sequence, (k_slice.size, bq))

    assert q_sequence.shape == k_sequence.shape
    computed_mask = mask_function(q_sequence, k_sequence)  # pytype: disable=wrong-arg-count
    if computed_mask.dtype != jnp.dtype(jnp.bool_):
      raise ValueError(
          "Mask function must return a boolean-valued array, but got:"
          f" {computed_mask.dtype}"
      )
    masks.append(computed_mask)

  if q_segment_ids_ref is not None:
    if k_in_lanes:
      kv_ids = pl.load(kv_segment_ids_ref, (pl.ds(1), k_slice))  # [1, k_slice]
      repeats, rem = divmod(kv_ids.shape[1], NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_kv must be a multiple of {NUM_LANES}")
      q_ids = pltpu.repeat(q_segment_ids_ref[:], repeats, axis=1)  # [bq, bkv]
    else:
      assert bq == q_segment_ids_ref.shape[-1]
      repeats, rem = divmod(bq, NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_q must be a multiple of {NUM_LANES}")
      kv_ids = pltpu.repeat(
          pl.load(kv_segment_ids_ref, (k_slice, slice(None))), repeats, axis=1
      )  # [k_slice, bq]
      q_ids = pl.load(q_segment_ids_ref, (pl.ds(1), slice(None)))  # [1, bq]
    masks.append(q_ids == kv_ids)

  def cap_logits(logits):
    if attn_logits_soft_cap is not None:
      logits = jnp.tanh(qk / attn_logits_soft_cap)
      return logits * attn_logits_soft_cap
    else:
      return logits

  if masks:
    mask = functools.reduce(jnp.logical_and, masks)
    qk = cap_logits(qk)
    qk = jnp.where(mask, qk, mask_value)
  else:
    qk = cap_logits(qk)
  return qk


def flash_attention_kernel(
    # Prefetched inputs
    data_next_ref,
    block_mask_ref,
    mask_next_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    mask_ref,
    q_sequence_ref,
    # Outputs
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    o_ref,
    logsumexp_ref=None,
    *,
    mask_value: float,
    grid_width: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    head_dim: int,
    q_layout: QKVLayout,
    k_layout: QKVLayout,
    v_layout: QKVLayout,
    attn_logits_soft_cap: float | None,
    mask_function: MaskFunctionType | None,
):
  float32 = jnp.float32
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR

  head_dim_repeats, rem = divmod(head_dim, NUM_LANES)
  if rem != 0:
    raise NotImplementedError(
        f"{head_dim=} should be a multiple of {NUM_LANES}"
    )

  h, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)

  @pl.when(j == 0)
  def init():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
    m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)

  global_kv_index, _, should_run, should_not_mask = _next_nonzero(
      h,
      i,
      j,
      data_next_ref,
      block_mask_ref,
      mask_next_ref,
  )

  def body(kv_compute_index, _):
    slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
    m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
    assert m_prev.shape == (bq, NUM_LANES)
    assert l_prev.shape == (bq, NUM_LANES)

    q = q_ref[...] if q_layout == HEAD_DIM_MINOR else q_ref[...].T
    qk_dims = NT_DIM_NUMBERS if k_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    if k_layout == HEAD_DIM_MINOR:
      k = pl.load(k_ref, (slice_k, slice(None)))
    else:
      k = pl.load(k_ref, (slice(None), slice_k))
    qk = lax.dot_general(q, k, qk_dims, preferred_element_type=float32)

    assert qk.shape == (bq, bkv_compute)
    apply_mask_and_soft_cap = functools.partial(
        _apply_mask_and_soft_cap,
        qk,
        mask_value,
        should_not_mask,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=slice_k,
        # When the iteration space is shrunk (for local attention for example),
        # the kv_index program_id does not correspond to the actual coordinates
        # of the KV data. Make sure to use the 'unshrunk' index (coming from the
        # data_next array) when computing the mask.
        k_offset=global_kv_index * bkv + kv_compute_index * bkv_compute,
        bq=bq,
        mask_function=mask_function,
    )

    qk = apply_mask_and_soft_cap()

    m_curr = qk.max(axis=-1)[:, None]  # pytype: disable=attribute-error
    assert m_curr.shape == (bq, 1)
    m_next = jnp.maximum(m_prev, m_curr)
    assert m_next.shape == (bq, NUM_LANES)

    bkv_repeats, rem = divmod(bkv_compute, NUM_LANES)
    if rem != 0:
      raise NotImplementedError(
          f"{bkv_compute=} should be a multiple of {NUM_LANES}"
      )

    s_curr = jnp.exp(qk - pltpu.repeat(m_next, bkv_repeats, axis=1))
    assert s_curr.shape == (bq, bkv_compute)

    l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
    assert l_curr.shape == (bq, NUM_LANES)

    alpha = jnp.exp(m_prev - m_next)
    l_next = l_curr + alpha * l_prev
    m_scratch_ref[...], l_scratch_ref[...] = m_next, l_next

    sv_dims = NN_DIM_NUMBERS if v_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
    if v_layout == HEAD_DIM_MINOR:
      v = pl.load(v_ref, (slice_k, slice(None)))
    else:
      v = pl.load(v_ref, (slice(None), slice_k))
    v = v.astype(float32)
    o_curr = lax.dot_general(s_curr, v, sv_dims)

    alpha_o = pltpu.repeat(alpha, head_dim_repeats, axis=1)
    o_scratch_ref[:] = alpha_o * o_scratch_ref[:] + o_curr

  @pl.when(should_run)
  def run():
    assert bkv % bkv_compute == 0
    num_iters = (
        k_ref.shape[0 if k_layout == HEAD_DIM_MINOR else 1] // bkv_compute
    )
    lax.fori_loop(0, num_iters, body, None, unroll=True)

  @pl.when(j == grid_width - 1)
  def end():
    l = l_scratch_ref[...]
    l_inv = pltpu.repeat(1.0 / l, head_dim_repeats, axis=1)
    o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)
    if logsumexp_ref is not None:
      assert logsumexp_ref.shape == (bq, NUM_LANES)
      logsumexp_ref[...] = (jnp.log(l) + m_scratch_ref[...]).astype(
          logsumexp_ref.dtype
      )

    m_scratch_ref[...] = jnp.zeros_like(m_scratch_ref)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)


@overload
def _splash_attention_forward(
    fwd_mask_info: mask_info_lib.MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    residual_checkpoint_name: str | None,
    mask_function: MaskFunctionType | None,
    save_residuals: Literal[False] = False,
    attn_logits_soft_cap: float | None = None,
) -> jax.Array:
  ...


@overload
def _splash_attention_forward(
    fwd_mask_info: mask_info_lib.MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    residual_checkpoint_name: str | None,
    mask_function: MaskFunctionType | None,
    save_residuals: Literal[True],
    attn_logits_soft_cap: float | None = None,
) -> SplashCustomReturnType:
  ...


def _div(dividend: int, divisor: int):
  if divisor == 1:
    return dividend

  return lax.div(dividend, divisor)


def _splash_attention_forward(
    fwd_mask_info: mask_info_lib.MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    residual_checkpoint_name: str | None,
    save_residuals: bool,
    mask_function: MaskFunctionType | None,
    attn_logits_soft_cap: float | None = None,
    interpret: bool = False
) -> SplashCustomReturnType:
  num_q_heads, q_seq_len, head_dim = q.shape
  bq, bkv = block_sizes.block_q, block_sizes.block_kv
  bkv_compute = block_sizes.block_kv_compute

  if is_mqa:
    expected_kv_rank = 2
    kv_head_dimension = 1
    kv_seq_len_dimension = 0
    num_kv_heads = 1
  else:
    expected_kv_rank = 3
    kv_head_dimension = 2
    kv_seq_len_dimension = 1
    num_kv_heads = k.shape[0]

  partial_mask_blocks = fwd_mask_info.partial_mask_blocks
  if (
      partial_mask_blocks is not None
      and jnp.dtype(partial_mask_blocks.dtype) != np.bool_
  ):
    raise ValueError(
        "partial_mask_blocks must be of type np.bool_ but got"
        f" {partial_mask_blocks.dtype}"
    )

  if len(k.shape) != expected_kv_rank:
    raise ValueError(
        f"Expected {expected_kv_rank}-dim 'key' tensor for MQA. Instead got a"
        f" {len(k.shape)}-dim one."
    )

  if k.shape[kv_head_dimension] != head_dim:
    raise ValueError(
        f"Expected 'key' head dimension to be: {head_dim}. Instead got:"
        f" {k.shape[kv_head_dimension]}."
    )

  if not is_mqa and num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
        f" multiple of the number of 'query' heads ({num_q_heads})"
    )

  if k.shape != v.shape:
    raise ValueError(
        f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same"
        " shape."
    )

  if bkv % bkv_compute:
    raise ValueError(f"{bkv=} must be a multiple of {bkv_compute=}.")
  if bkv_compute % NUM_LANES:
    raise ValueError(f"{bkv_compute=} must be a multiple of {NUM_LANES}.")

  kv_seq_len = k.shape[kv_seq_len_dimension]

  q_heads_per_kv_head = num_q_heads // num_kv_heads

  if segment_ids is not None:
    if segment_ids.q.shape != (q_seq_len,):
      raise ValueError(
          "Invalid shape for q segment_ids: "
          f"{segment_ids.q.shape}. Expected: {(q_seq_len,)}"
      )
    if segment_ids.kv.shape != (kv_seq_len,):
      raise ValueError(
          "Invalid shape for kv segment_ids: "
          f"{segment_ids.kv.shape}. Expected: {(kv_seq_len,)}"
      )

  q_layout = block_sizes.q_layout
  def q_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
    del j, data_next_ref, mask_next_ref, block_mask_ref
    return from_head_minor((h, i, 0), q_layout)
  def out_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
    del j, data_next_ref, mask_next_ref, block_mask_ref
    return h, i, 0

  k_layout = block_sizes.k_layout
  def k_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
    next_j, *_ = _next_nonzero(
        h, i, j, data_next_ref, block_mask_ref, mask_next_ref
    )
    prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
    return from_head_minor((*prefix, next_j, 0), k_layout)

  v_layout = block_sizes.v_layout
  def v_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
    next_j, *_ = _next_nonzero(
        h, i, j, data_next_ref, block_mask_ref, mask_next_ref
    )
    prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
    return from_head_minor((*prefix, next_j, 0), v_layout)

  def mask_index_map(h, i, j, data_next_ref, block_mask_ref,
                     mask_next_ref=None):
    _, next_m, *_ = _next_nonzero(
        h, i, j, data_next_ref, block_mask_ref, mask_next_ref
    )
    return next_m, 0, 0

  def q_segment_ids_index_map(h, i, j, *_):
    del h, j  # Unused.
    return i, 0

  def kv_segment_ids_index_map(h, i, j, data_next_ref, block_mask_ref,
                               mask_next_ref=None):
    next_j, *_ = _next_nonzero(
        h, i, j, data_next_ref, block_mask_ref, mask_next_ref
    )
    return 0, next_j

  # Convert the logical shape from head-minor to sequence-minor.
  in_specs = [
      pl.BlockSpec(
          from_head_minor((None, bq, head_dim), q_layout), q_index_map
      ),
      pl.BlockSpec(
          from_head_minor(
              (bkv, head_dim) if is_mqa else (None, bkv, head_dim), k_layout
          ),
          k_index_map,
      ),
      pl.BlockSpec(
          from_head_minor(
              (bkv, head_dim) if is_mqa else (None, bkv, head_dim), v_layout
          ),
          v_index_map,
      ),
  ]
  if segment_ids is not None:
    in_specs += [
        pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map),
        pl.BlockSpec((NUM_SUBLANES, bkv), kv_segment_ids_index_map),
    ]
    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q, (q_seq_len, NUM_LANES), (0,)
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv, (NUM_SUBLANES, kv_seq_len), (1,)
    )
  else:
    in_specs += [None, None]
    q_segment_ids = kv_segment_ids = None

  if fwd_mask_info.partial_mask_blocks is not None:
    in_specs.append(pl.BlockSpec((None, bq, bkv), mask_index_map))
  else:
    in_specs.append(None)

  assert (
      fwd_mask_info.partial_mask_blocks is None
      or fwd_mask_info.q_sequence is None
  )

  if fwd_mask_info.q_sequence is not None:
    q_sequence = jax.lax.broadcast_in_dim(
        fwd_mask_info.q_sequence, (q_seq_len, NUM_LANES), (0,)
    )
    in_specs.append(pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map))
  else:
    q_sequence = None
    in_specs.append(None)

  num_scalar_prefetch = 3

  out_shapes = [
      jax.ShapeDtypeStruct((bq, NUM_LANES), jnp.float32),  # m_scratch
      jax.ShapeDtypeStruct((bq, NUM_LANES), jnp.float32),  # l_scratch
      jax.ShapeDtypeStruct((bq, head_dim), jnp.float32),  # o_scratch
      jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim), q.dtype),
  ]
  out_specs = [
      # TODO(sharadmv): convert m/l to be scratch
      pl.BlockSpec((bq, NUM_LANES), lambda h, i, j, *_: (0, 0)),
      pl.BlockSpec((bq, NUM_LANES), lambda h, i, j, *_: (0, 0)),
      pl.BlockSpec((bq, head_dim), lambda h, i, j, *_: (0, 0)),
      pl.BlockSpec((None, bq, head_dim), out_index_map),
  ]
  if save_residuals:
    out_shapes += [
        jax.ShapeDtypeStruct(
            (num_q_heads, q_seq_len, NUM_LANES), jnp.float32
        ),  # logsumexp
    ]

    def logsumexp_index_map(h, i, *_):
      return h, i, 0

    out_specs += [
        pl.BlockSpec((None, bq, NUM_LANES), logsumexp_index_map),
    ]
  else:
    out_shapes += [None]
    out_specs += [None]

  kernel_name = get_kernel_name(
      dataclasses.asdict(block_sizes),
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      is_segmented=segment_ids is not None,
      phase="fwd",
  )

  if fwd_mask_info.data_next is not None:
    grid_width = fwd_mask_info.data_next.shape[-1]
  else:
    grid_width = kv_seq_len // bkv

  grid = (num_q_heads, q_seq_len // bq, grid_width)
  with jax.named_scope(kernel_name):
    all_out = pl.pallas_call(
        partial(
            flash_attention_kernel,
            mask_value=mask_value,
            grid_width=grid_width,
            bq=bq,
            bkv=bkv,
            bkv_compute=bkv_compute,
            head_dim=head_dim,
            q_layout=q_layout,
            k_layout=k_layout,
            v_layout=v_layout,
            attn_logits_soft_cap=attn_logits_soft_cap,
            mask_function=mask_function,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_scalar_prefetch,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
        ),
        compiler_params=pltpu.TPUCompilerParams(
          dimension_semantics=("parallel", "arbitrary", "arbitrary"),
        ),
        out_shape=out_shapes,
        name=kernel_name,
        interpret=interpret,
    )(
        fwd_mask_info.data_next,
        fwd_mask_info.block_mask,
        fwd_mask_info.mask_next,
        q if q_layout == QKVLayout.HEAD_DIM_MINOR else q.swapaxes(-1, -2),
        k if k_layout == QKVLayout.HEAD_DIM_MINOR else k.swapaxes(-1, -2),
        v if v_layout == QKVLayout.HEAD_DIM_MINOR else v.swapaxes(-1, -2),
        q_segment_ids,
        kv_segment_ids,
        fwd_mask_info.partial_mask_blocks,
        q_sequence,
    )

  (
      _,
      _,
      _,
      out,
      logsumexp,
  ) = all_out

  if save_residuals:
    assert logsumexp is not None
    logsumexp = logsumexp[..., 0]

  if residual_checkpoint_name is not None:
    out = ad_checkpoint.checkpoint_name(out, name=residual_checkpoint_name)
    if logsumexp is not None:
      logsumexp = ad_checkpoint.checkpoint_name(
          logsumexp, name=residual_checkpoint_name
      )
  if save_residuals:
    return out, (logsumexp,)
  return out


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13, 14))
def _splash_attention_custom(
    fwd_mask_info: mask_info_lib.MaskInfo,
    dq_mask_info: mask_info_lib.MaskInfo | None,
    dkv_mask_info: mask_info_lib.MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    residual_checkpoint_name: str | None,
    mask_function: MaskFunctionType | None,
    attn_logits_soft_cap: float | None = None,
    interpret: bool = False,
) -> SplashCustomReturnType:
  # The forward function does not use the dq and dkv MaskInfos, it just forwards
  # them to the backward function as residuals. This is a way to communicate
  # arbitrary Arrays to the backward function. Since the three MaskInfos are
  # constants there is no overhead in passing them to the backward function as
  # residuals. When sharding computation MaskInfos are partitioned so both the
  # forward and the backward kernels need to work on the relevant slice. If we
  # recomputed the backward MaskInfos in the backward function from the numpy
  # mask then we would not work with the MaskInfo slice relevant to the current
  # device.
  del dq_mask_info, dkv_mask_info

  return _splash_attention_forward(  # pytype: disable=wrong-arg-types
      fwd_mask_info,
      q,
      k,
      v,
      segment_ids,
      mask_value=mask_value,
      is_mqa=is_mqa,
      block_sizes=block_sizes,
      residual_checkpoint_name=residual_checkpoint_name,
      save_residuals=save_residuals,
      mask_function=mask_function,
      attn_logits_soft_cap=attn_logits_soft_cap,
      interpret=interpret,
  )


def _splash_attention_fwd(
    fwd_mask_info: mask_info_lib.MaskInfo,
    dq_mask_info: mask_info_lib.MaskInfo | None,
    dkv_mask_info: mask_info_lib.MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    residual_checkpoint_name: str | None,
    mask_function: MaskFunctionType | None,
    attn_logits_soft_cap: float | None = None,
    interpret: bool = False,
) -> tuple[
    tuple[jax.Array],
    SplashResidualsType,
]:
  if save_residuals:
    raise NotImplementedError("Higher-order AD not supported")

  out, (logsumexp,) = _splash_attention_forward(  # pytype: disable=wrong-arg-types
      fwd_mask_info,
      q,
      k,
      v,
      segment_ids,
      mask_value=mask_value,
      is_mqa=is_mqa,
      block_sizes=block_sizes,
      residual_checkpoint_name=residual_checkpoint_name,
      save_residuals=True,
      mask_function=mask_function,
      attn_logits_soft_cap=attn_logits_soft_cap,
      interpret=interpret,
  )
  return out, (
      q,
      k,
      v,
      segment_ids,
      out,
      logsumexp,
      dq_mask_info,
      dkv_mask_info,
  )


def _flash_attention_dq_kernel(
    # Prefetched inputs
    data_next_ref,
    block_mask_ref,
    mask_next_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    logsumexp_ref,
    do_ref,
    di_ref,
    mask_ref,
    q_sequence_ref,
    # Outputs
    dq_scratch_ref,
    dq_ref,
    *,
    mask_value: float,
    grid_width: int,
    bq: int,
    bkv: int,
    attn_logits_soft_cap: float | None = None,
    q_layout: QKVLayout,
    k_layout: QKVLayout,
    v_layout: QKVLayout,
    mask_function: MaskFunctionType | None,
):
  float32 = jnp.float32
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR

  h, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  @pl.when(j == 0)
  def init():
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)

  global_kv_index, _, should_run, should_not_mask = _next_nonzero(
      h, i, j, data_next_ref, block_mask_ref, mask_next_ref
  )
  @pl.when(should_run)
  def run():
    q = q_ref[...] if q_layout == HEAD_DIM_MINOR else q_ref[...].T
    # We keep k and v possibly transposed, since they are RHS of dots.
    k = k_ref[...]
    v = v_ref[...]
    logsumexp = jnp.expand_dims(logsumexp_ref[0], -1)
    do = do_ref[...]
    di = jnp.expand_dims(di_ref[0], -1)

    qk_dims = NT_DIM_NUMBERS if k_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    qk_uncapped = lax.dot_general(q, k, qk_dims, preferred_element_type=float32)

    qk = _apply_mask_and_soft_cap(
        qk_uncapped,
        mask_value,
        should_not_mask,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=pl.ds(0, bkv),
        # When the iteration space is shrunk (for local attention for example),
        # the kv_index program_id does not correspond to the actual coordinates
        # of the KV data. Make sure to use the 'unshrunk' index (coming from the
        # data_next array) when computing the mask.
        k_offset=global_kv_index * bkv,
        bq=bq,
        mask_function=mask_function,
    )
    p = jnp.exp(qk - logsumexp)
    dp_dims = NT_DIM_NUMBERS if v_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    dp = lax.dot_general(
        do.astype(v.dtype), v, dp_dims, preferred_element_type=jnp.float32,
    )
    ds = (dp - di) * p
    if attn_logits_soft_cap is not None:
      normalized = qk_uncapped / attn_logits_soft_cap
      d = jnp.tanh(normalized)
      g = ds * (1 - d)
      ds = g + g * d

    dq_dims = NN_DIM_NUMBERS if k_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
    dq_scratch_ref[...] += lax.dot_general(
        ds.astype(k.dtype), k, dq_dims,
        preferred_element_type=jnp.float32,
    )

  @pl.when(j == grid_width - 1)
  def end():
    dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)


def _splash_attention_bwd_dq(
    q,
    k,
    v,
    segment_ids,
    logsumexp,
    do,
    di,
    *,
    bq: int,
    bkv: int,
    is_mqa: bool,
    mask_info: mask_info_lib.MaskInfo,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    q_layout: QKVLayout,
    k_layout: QKVLayout,
    v_layout: QKVLayout,
    mask_function: MaskFunctionType | None,
    interpret: bool,
):
  num_q_heads, q_seq_len, head_dim = q.shape
  if is_mqa:
    kv_seq_len = k.shape[0]
    num_kv_heads = 1
  else:
    kv_seq_len = k.shape[1]
    num_kv_heads = k.shape[0]

  if bq > q_seq_len:
    raise ValueError(
        f"{bq=} should not be greater than {q_seq_len=}")
  if bkv > kv_seq_len:
    raise ValueError(
        f"{bkv=} should not be greater than {kv_seq_len=}")

  if not is_mqa and num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
        f" multiple of the number of 'query' heads ({num_q_heads})"
    )

  if k.shape != v.shape:
    raise ValueError(
        f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same"
        " shape."
    )

  if bkv % NUM_LANES:
    raise ValueError(f"{bkv=} must be a multiple of {NUM_LANES}.")

  # TODO(amagni/sharadmv): when adding block_compute, make sure that is a
  # multiple of NUM_LANES.

  q_heads_per_kv_head = num_q_heads // num_kv_heads

  if mask_info.data_next is not None:
    grid_width = mask_info.data_next.shape[-1]
  else:
    grid_width = kv_seq_len // bkv

  grid = (num_q_heads, q_seq_len // bq, grid_width)

  def o_index_map(h, i, *_):
    return h, i, 0

  o_spec = pl.BlockSpec((None, bq, head_dim), o_index_map)

  def q_index_map(h, i, *_):
    return from_head_minor((h, i, 0), q_layout)

  q_spec = pl.BlockSpec(
      from_head_minor((None, bq, head_dim), q_layout), q_index_map
  )

  def k_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_):
    next_j, *_ = _next_nonzero(
        h, i, j, data_next_ref, block_mask_ref, mask_next_ref
    )
    prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
    return from_head_minor((*prefix, next_j, 0), k_layout)

  k_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim) if is_mqa else (None, bkv, head_dim), k_layout
      ),
      k_index_map,
  )

  def v_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_):
    next_j, *_ = _next_nonzero(
        h, i, j, data_next_ref, block_mask_ref, mask_next_ref
    )
    prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
    return from_head_minor((*prefix, next_j, 0), v_layout)

  v_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim) if is_mqa else (None, bkv, head_dim), v_layout
      ),
      v_index_map,
  )

  def mask_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_):
    _, next_m, *_ = _next_nonzero(
        h, i, j, data_next_ref, block_mask_ref, mask_next_ref
    )
    return next_m, 0, 0

  mask_spec = pl.BlockSpec((None, bq, bkv), mask_index_map)

  def q_segment_ids_index_map(h, i, j, *_):
    del h, j  # Unused.
    return i, 0

  if segment_ids is not None:

    def kv_segment_ids_index_map(
        h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_
    ):
      next_j, *_ = _next_nonzero(
          h, i, j, data_next_ref, block_mask_ref, mask_next_ref
      )
      return 0, next_j

    q_segment_spec = pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map)
    kv_segment_spec = pl.BlockSpec(
        (NUM_SUBLANES, bkv), kv_segment_ids_index_map
    )
    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q, (q_seq_len, NUM_LANES), (0,)
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv, (NUM_SUBLANES, kv_seq_len), (1,)
    )
  else:
    q_segment_spec = kv_segment_spec = None
    q_segment_ids = kv_segment_ids = None

  do_spec = dq_spec = o_spec

  def logsumexp_index_map(h, i, *_):
    return h, 0, i

  logsumexp = jnp.expand_dims(logsumexp, axis=-2)
  logsumexp_spec = pl.BlockSpec((None, 1, bq), logsumexp_index_map)
  assert logsumexp.ndim == len(logsumexp_spec.block_shape)

  di = jnp.expand_dims(di, axis=-2)
  di_spec = pl.BlockSpec((None, 1, bq), logsumexp_index_map)
  assert di.ndim == len(di_spec.block_shape)

  in_specs = [
      q_spec,
      k_spec,
      v_spec,
      q_segment_spec,
      kv_segment_spec,
      logsumexp_spec,
      do_spec,
      di_spec,
  ]
  if mask_info.partial_mask_blocks is not None:
    in_specs.append(mask_spec)
  else:
    in_specs.append(None)

  assert mask_info.partial_mask_blocks is None or mask_info.q_sequence is None

  if mask_info.q_sequence is not None:
    q_sequence = jax.lax.broadcast_in_dim(
        mask_info.q_sequence, (q_seq_len, NUM_LANES), (0,)
    )
    in_specs.append(pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map))
  else:
    q_sequence = None
    in_specs.append(None)

  out_shapes = [
      jax.ShapeDtypeStruct((bq, head_dim), jnp.float32),
      jax.ShapeDtypeStruct(q.shape, q.dtype),
  ]
  out_specs = [
      pl.BlockSpec((bq, head_dim), lambda *_: (0, 0)),
      dq_spec,
  ]

  kernel = functools.partial(
      _flash_attention_dq_kernel,
      grid_width=grid_width,
      mask_value=mask_value,
      bq=bq,
      bkv=bkv,
      attn_logits_soft_cap=attn_logits_soft_cap,
      q_layout=q_layout,
      k_layout=k_layout,
      v_layout=v_layout,
      mask_function=mask_function,
  )
  num_scalar_prefetch = 3

  kernel_name = get_kernel_name(
      dict(
          block_q_dq=bq,
          block_kv_dq=bkv,
          q_layout=q_layout,
          k_layout=k_layout,
          v_layout=v_layout,
      ),
      is_mqa=is_mqa,
      save_residuals=False,
      is_segmented=segment_ids is not None,
      phase="dq",
  )
  with jax.named_scope(kernel_name):
    _, dq = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_scalar_prefetch,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
        ),
        out_shape=out_shapes,
        compiler_params=pltpu.TPUCompilerParams(
          dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
        ),
        name=kernel_name,
        interpret=interpret,
    )(
        mask_info.data_next,
        mask_info.block_mask,
        mask_info.mask_next,
        q if q_layout == QKVLayout.HEAD_DIM_MINOR else q.swapaxes(-1, -2),
        k if k_layout == QKVLayout.HEAD_DIM_MINOR else k.swapaxes(-1, -2),
        v if v_layout == QKVLayout.HEAD_DIM_MINOR else v.swapaxes(-1, -2),
        q_segment_ids,
        kv_segment_ids,
        logsumexp,
        do,
        di,
        mask_info.partial_mask_blocks,
        q_sequence,
    )
  return dq


def _flash_attention_dkv_kernel(
    # Prefetched inputs
    data_next_ref,
    block_mask_ref,
    mask_next_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    logsumexp_ref,
    do_ref,
    di_ref,
    mask_ref,
    q_sequence_ref,
    # Outputs
    dq_scratch_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    mask_value: float,
    grid_width: int,
    bq: int,
    bkv_compute: int,
    is_mqa: bool,
    attn_logits_soft_cap: float | None,
    q_layout: QKVLayout,
    k_layout: QKVLayout,
    v_layout: QKVLayout,
    bkv: int,
    mask_function: MaskFunctionType | None,
):
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  kv_index, q_head_index, q_index = (
      pl.program_id(0),
      pl.program_id(1),
      pl.program_id(2),
  )
  should_initialize = q_index == 0

  q_heads_per_kv_heads = None
  q_head_index_per_kv_head = None

  # Consider this situation:
  # Q_heads:   0, 1, 2, 3, 4, 5, 6, 7
  # KV_heads:  0,    1,    2,    3
  # The gradient scratch buffers should be initialized for Q_heads 0, 2, 4, 6
  # (first Q_heads to 'see' a new KV_head).
  # The gradient output buffers should be written for Q_heads 1, 3, 5, 7 (last
  # Q_heads to 'see' the current KV_head).

  # We can use the same logic for both MQA and GA (grouped attention).
  # But for MQA there is no need for the rem instruction, so we skip it.
  if is_mqa:
    should_initialize = jnp.logical_and(should_initialize, q_head_index == 0)
  elif num_kv_heads < num_q_heads:
    q_heads_per_kv_heads = num_q_heads // num_kv_heads
    q_head_index_per_kv_head = lax.rem(q_head_index, q_heads_per_kv_heads)
    should_initialize = jnp.logical_and(
        should_initialize, q_head_index_per_kv_head == 0
    )
  @pl.when(should_initialize)
  def init():
    dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
    dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)

  _, _, should_run, should_not_mask = _next_nonzero(
      q_head_index,
      q_index,
      kv_index,
      data_next_ref,
      block_mask_ref,
      mask_next_ref,
      next_i=True,
  )

  def body(i, _):

    slice_k = pl.ds(i * bkv_compute, bkv_compute)
    q = q_ref[...]  # We keep q potentially transposed, since it's always RHS
    def _load_kv(ref, layout):
      if layout == HEAD_DIM_MINOR:
        return pl.load(ref, (slice_k, slice(None)))
      return pl.load(ref, (slice(None), slice_k)).T
    k = _load_kv(k_ref, k_layout)
    v = _load_kv(v_ref, v_layout)
    logsumexp = pl.load(logsumexp_ref, (pl.ds(1), slice(None)))
    do = do_ref[...]
    di = pl.load(di_ref, (pl.ds(1), slice(None)))

    qk_dims = NT_DIM_NUMBERS if q_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    qk_uncapped = lax.dot_general(
        k, q, qk_dims, preferred_element_type=jnp.float32
    )

    qk = _apply_mask_and_soft_cap(
        qk_uncapped,
        mask_value,
        should_not_mask,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=slice_k,
        k_offset=kv_index * bkv + i * bkv_compute,
        bq=bq,
        k_in_lanes=False,
        mask_function=mask_function,
    )
    p = jnp.exp(qk - logsumexp)
    dv = lax.dot(p.astype(do.dtype), do, preferred_element_type=jnp.float32)
    dv = dv.astype(dv_scratch_ref.dtype) + pl.load(
        dv_scratch_ref, (slice_k, slice(None))
    )
    pl.store(dv_scratch_ref, (slice_k, slice(None)), dv)

    dp = lax.dot_general(
        v, do, NT_DIM_NUMBERS,
        preferred_element_type=jnp.float32,
    )
    ds = (dp - di) * p
    if attn_logits_soft_cap is not None:
      normalized = qk_uncapped / attn_logits_soft_cap
      d = jnp.tanh(normalized)
      g = ds * (1 - d)
      ds = g + g * d
    dk_dims = NN_DIM_NUMBERS if q_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
    dk = lax.dot_general(
        ds.astype(do.dtype), q, dk_dims, preferred_element_type=jnp.float32
    )
    dk = dk.astype(dk_scratch_ref.dtype) + pl.load(
        dk_scratch_ref, (slice_k, slice(None))
    )
    pl.store(dk_scratch_ref, (slice_k, slice(None)), dk)
    if dq_scratch_ref is not None or dq_ref is not None:
      dq = lax.dot_general(
          ds.T.astype(k.dtype), k, NN_DIM_NUMBERS,
          preferred_element_type=jnp.float32,
      )
      if dq_scratch_ref is not None:
        # Compute block size != memory block size
        dq_scratch_ref[...] += dq
      else:
        # Compute block size == memory block size
        assert dq_ref is not None
        dq_ref[...] = dq.astype(dq_ref.dtype)

  if dq_scratch_ref is not None:
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)
  elif dq_scratch_ref is None and dq_ref is not None:
    dq_ref[...] = jnp.zeros_like(dq_ref)

  @pl.when(should_run)
  def run():
    num_iters = (
        k_ref.shape[0 if k_layout is HEAD_DIM_MINOR else 1] // bkv_compute
    )
    lax.fori_loop(0, num_iters, body, None, unroll=True)
  if dq_scratch_ref is not None:
    assert dq_ref is not None
    dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)

  should_write = q_index == grid_width - 1
  if is_mqa:
    should_write = jnp.logical_and(
        should_write, q_head_index == num_q_heads - 1
    )
  elif num_kv_heads < num_q_heads:
    should_write = jnp.logical_and(
        should_write, q_head_index_per_kv_head == q_heads_per_kv_heads - 1
    )

  @pl.when(should_write)
  def end():
    dk_ref[...] = dk_scratch_ref[...].astype(dk_ref.dtype)
    dv_ref[...] = dv_scratch_ref[...].astype(dv_ref.dtype)
    if dq_scratch_ref is not None:
      dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)

    dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
    dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)


def _splash_attention_bwd_dkv(
    q,
    k,
    v,
    segment_ids,
    logsumexp,
    do,
    di,
    *,
    bq: int,
    bkv: int,
    bkv_compute: int,
    is_mqa: bool,
    mask_info: mask_info_lib.MaskInfo,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    use_fused_bwd_kernel: bool,
    q_layout: QKVLayout,
    k_layout: QKVLayout,
    v_layout: QKVLayout,
    mask_function: MaskFunctionType | None,
    interpret: bool,
):
  num_q_heads, q_seq_len, head_dim = q.shape
  if is_mqa:
    num_kv_heads, kv_seq_len = 1, k.shape[0]
  else:
    num_kv_heads, kv_seq_len, _ = k.shape

  if bq > q_seq_len:
    raise ValueError(
        f"{bq=} should not be greater than {q_seq_len=}")
  if bkv > kv_seq_len:
    raise ValueError(
        f"{bkv=} should not be greater than {kv_seq_len=}")
  if bkv_compute > bkv:
    raise ValueError(
        f"{bkv_compute=} should not be greater than {bkv=}")
  if bkv % bkv_compute:
    raise ValueError(
        f"{bkv=} should be a multiple of {bkv_compute=}")

  if not is_mqa and num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
        f" multiple of the number of 'query' heads ({num_q_heads})"
    )

  if k.shape != v.shape:
    raise ValueError(
        f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same"
        " shape."
    )

  q_heads_per_kv_head = num_q_heads // num_kv_heads

  if mask_info.data_next is not None:
    grid_width = mask_info.data_next.shape[-2]
  else:
    grid_width = q_seq_len // bq

  grid = (
      kv_seq_len // bkv,
      num_q_heads,
      grid_width,
  )

  def o_index_map(
      kv_index,
      head_index,
      q_index,
      data_next_ref,
      block_mask_ref,
      mask_next_ref=None,
  ):
    next_i, *_ = _next_nonzero(
        head_index,
        q_index,
        kv_index,
        data_next_ref,
        block_mask_ref,
        mask_next_ref,
        next_i=True,
    )
    return head_index, next_i, 0

  o_spec = pl.BlockSpec((None, bq, head_dim), o_index_map)

  def q_index_map(
      kv_index,
      head_index,
      q_index,
      data_next_ref,
      block_mask_ref,
      mask_next_ref=None,
  ):
    next_i, *_ = _next_nonzero(
        head_index,
        q_index,
        kv_index,
        data_next_ref,
        block_mask_ref,
        mask_next_ref,
        next_i=True,
    )
    return from_head_minor((head_index, next_i, 0), q_layout)

  q_spec = pl.BlockSpec(
      from_head_minor((None, bq, head_dim), q_layout), q_index_map)

  def k_index_map(kv_index, head_index, *_):
    prefix = () if is_mqa else (_div(head_index, q_heads_per_kv_head),)
    return from_head_minor((*prefix, kv_index, 0), k_layout)

  k_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim) if is_mqa else (None, bkv, head_dim),
          k_layout,
      ),
      k_index_map,
  )

  def v_index_map(kv_index, head_index, *_):
    prefix = () if is_mqa else (_div(head_index, q_heads_per_kv_head),)
    return from_head_minor((*prefix, kv_index, 0), v_layout)

  v_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim) if is_mqa else (None, bkv, head_dim),
          v_layout,
      ),
      v_index_map,
  )

  if use_fused_bwd_kernel:
    def dq_index_map(kv_index, head_index, q_index, *_):
      return (kv_index, head_index, q_index, 0)
    dq_spec = pl.BlockSpec((None, None, bq, head_dim), dq_index_map)
    dq_shape = jax.ShapeDtypeStruct((kv_seq_len // bkv, *q.shape), q.dtype)
    if bkv == bkv_compute:
      dq_scratch_spec = dq_scratch_shape = None
    else:
      dq_scratch_spec = pl.BlockSpec((bq, head_dim), lambda *_: (0, 0))
      dq_scratch_shape = jax.ShapeDtypeStruct((bq, head_dim), jnp.float32)
  else:
    dq_spec = dq_shape = dq_scratch_spec = dq_scratch_shape = None

  def dkv_index_map(kv_index, head_index, *_):
    prefix = () if is_mqa else (_div(head_index, q_heads_per_kv_head),)
    return (*prefix, kv_index, 0)

  dk_spec = dv_spec = pl.BlockSpec(
      (bkv, head_dim) if is_mqa else (None, bkv, head_dim),
      dkv_index_map,
  )

  def mask_index_map(
      kv_index,
      head_index,
      q_index,
      data_next_ref,
      block_mask_ref,
      mask_next_ref,
  ):
    _, next_m, *_ = _next_nonzero(
        head_index,
        q_index,
        kv_index,
        data_next_ref,
        block_mask_ref,
        mask_next_ref,
        next_i=True,
    )
    return next_m, 0, 0

  mask_spec = pl.BlockSpec((None, bkv, bq), mask_index_map)

  def q_segment_ids_index_map(
      kv_index,
      head_index,
      q_index,
      data_next_ref,
      block_mask_ref,
      mask_next_ref=None,
  ):
    next_i, *_ = _next_nonzero(
        head_index,
        q_index,
        kv_index,
        data_next_ref,
        block_mask_ref,
        mask_next_ref,
        next_i=True,
    )
    return 0, next_i

  if segment_ids is not None:
    def kv_segment_ids_index_map(kv_index, *_):
      return kv_index, 0

    q_segment_spec = pl.BlockSpec((NUM_SUBLANES, bq), q_segment_ids_index_map)
    kv_segment_spec = pl.BlockSpec((bkv, NUM_LANES), kv_segment_ids_index_map)
    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q, (NUM_SUBLANES, q_seq_len), (1,)
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv, (kv_seq_len, NUM_LANES), (0,)
    )
  else:
    q_segment_spec = kv_segment_spec = None
    q_segment_ids = kv_segment_ids = None

  do_spec = o_spec

  def logsumexp_index_map(
      kv_index,
      head_index,
      q_index,
      data_next_ref,
      block_mask_ref,
      mask_next_ref=None,
  ):
    next_i, *_ = _next_nonzero(
        head_index,
        q_index,
        kv_index,
        data_next_ref,
        block_mask_ref,
        mask_next_ref,
        next_i=True,
    )
    return head_index, 0, next_i

  assert logsumexp.shape == di.shape == (num_q_heads, q_seq_len)
  # TODO(apaszke): Remove the sublane expansion once Mosaic has all retilings
  logsumexp_shape = (num_q_heads, NUM_SUBLANES, q_seq_len)
  logsumexp = jnp.broadcast_to(jnp.expand_dims(logsumexp, -2), logsumexp_shape)
  logsumexp_spec = pl.BlockSpec((None, NUM_SUBLANES, bq), logsumexp_index_map)
  assert logsumexp.ndim == len(logsumexp_spec.block_shape)

  # TODO(apaszke): Remove the sublane expansion once Mosaic has all retilings
  di = jnp.broadcast_to(jnp.expand_dims(di, -2), logsumexp_shape)
  di_spec = pl.BlockSpec((None, NUM_SUBLANES, bq), logsumexp_index_map)
  assert di.ndim == len(di_spec.block_shape)

  in_specs = [
      q_spec,
      k_spec,
      v_spec,
      q_segment_spec,
      kv_segment_spec,
      logsumexp_spec,
      do_spec,
      di_spec,
  ]
  if mask_info.partial_mask_blocks is not None:
    in_specs.append(mask_spec)
  else:
    in_specs.append(None)

  if mask_info.q_sequence is not None:
    in_specs.append(pl.BlockSpec((NUM_SUBLANES, bq), q_segment_ids_index_map))
    q_sequence = jax.lax.broadcast_in_dim(
        mask_info.q_sequence, (NUM_SUBLANES, q_seq_len), (1,)
    )
  else:
    q_sequence = None
    in_specs.append(None)

  out_shapes = [
      dq_scratch_shape,
      jax.ShapeDtypeStruct((bkv, head_dim), jnp.float32),
      jax.ShapeDtypeStruct((bkv, head_dim), jnp.float32),
      dq_shape,
      jax.ShapeDtypeStruct(k.shape, k.dtype),
      jax.ShapeDtypeStruct(v.shape, v.dtype),
  ]
  out_specs = [
      dq_scratch_spec,
      pl.BlockSpec((bkv, head_dim), lambda *_: (0, 0)),
      pl.BlockSpec((bkv, head_dim), lambda *_: (0, 0)),
      dq_spec,
      dk_spec,
      dv_spec,
  ]

  kernel = functools.partial(
      _flash_attention_dkv_kernel,
      mask_value=mask_value,
      num_q_heads=num_q_heads,
      num_kv_heads=num_kv_heads,
      is_mqa=is_mqa,
      grid_width=grid_width,
      bq=bq,
      bkv_compute=bkv_compute,
      attn_logits_soft_cap=attn_logits_soft_cap,
      q_layout=q_layout,
      k_layout=k_layout,
      v_layout=v_layout,
      bkv=bkv,
      mask_function=mask_function,
  )
  num_scalar_prefetch = 3

  kernel_name = get_kernel_name(
      dict(
          block_q_dkv=bq,
          block_kv_dkv=bkv,
          block_kv_dkv_compute=bkv_compute,
          q_layout=q_layout,
          k_layout=k_layout,
          v_layout=v_layout,
      ),
      is_mqa=is_mqa,
      save_residuals=False,
      is_segmented=segment_ids is not None,
      phase="dkv",
  )
  with jax.named_scope(kernel_name):
    _, _, _, dq_unreduced, dk, dv = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_scalar_prefetch,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
        ),
        out_shape=out_shapes,
        # We set all dimensions to arbitrary because:
        # 1) for kv_seq_len, the splash attention prefetch schedule assumes no
        #    megacore
        # 2) for heads, we are reducing over heads
        # 3) for q_seq_len, we are reducing over it to compute dkv
        compiler_params=pltpu.TPUCompilerParams(
          dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
        ),
        name=kernel_name,
        interpret=interpret,
    )(
        mask_info.data_next,
        mask_info.block_mask,
        mask_info.mask_next,
        q if q_layout == QKVLayout.HEAD_DIM_MINOR else q.swapaxes(-1, -2),
        k if k_layout == QKVLayout.HEAD_DIM_MINOR else k.swapaxes(-1, -2),
        v if v_layout == QKVLayout.HEAD_DIM_MINOR else v.swapaxes(-1, -2),
        q_segment_ids,
        kv_segment_ids,
        logsumexp,
        do,
        di,
        mask_info.partial_mask_blocks,
        q_sequence,
    )
  if use_fused_bwd_kernel:
    assert dq_unreduced is not None
    dq = dq_unreduced.sum(axis=0)
  else:
    assert dq_unreduced is None
    dq = None
  return dq, dk, dv


def _splash_attention_bwd(
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    residual_checkpoint_name: str | None,
    mask_function: MaskFunctionType | None,
    attn_logits_soft_cap: float | None,
    interpret: bool,
    res: SplashResidualsType,
    do: jax.Array,
) -> tuple[
    mask_info_lib.MaskInfo | None,  # fwd_mask_info
    mask_info_lib.MaskInfo | None,  # dq_mask_info
    mask_info_lib.MaskInfo | None,  # dvk_mask_info
    jax.Array,  # q
    jax.Array,  # k
    jax.Array,  # v
    SegmentIds | None,  # segmend_ids
]:
  del save_residuals, residual_checkpoint_name
  if not block_sizes.has_backward_blocks:
    raise ValueError("Need to specify backward blocks.")
  bq_dq, bkv_dq = block_sizes.block_q_dq, block_sizes.block_kv_dq
  bq_dkv, bkv_dkv_memory, bkv_dkv_compute = (
      block_sizes.block_q_dkv,
      block_sizes.block_kv_dkv,
      block_sizes.block_kv_dkv_compute,
  )
  use_fused_bwd_kernel = block_sizes.use_fused_bwd_kernel
  (
      q,
      k,
      v,
      segment_ids,
      o,
      logsumexp,
      dq_mask_info,
      dkv_mask_info,
  ) = res

  # di: [num_heads, q_seq_len]
  di = jnp.einsum("hsd,hsd->hs", o.astype(jnp.float32), do.astype(jnp.float32))  # pytype: disable=attribute-error
  dq, dk, dv = _splash_attention_bwd_dkv(
      q,
      k,
      v,
      segment_ids,
      logsumexp,
      do,
      di,
      bq=bq_dkv,
      bkv=bkv_dkv_memory,
      bkv_compute=bkv_dkv_compute,
      is_mqa=is_mqa,
      mask_info=dkv_mask_info,
      mask_value=mask_value,
      attn_logits_soft_cap=attn_logits_soft_cap,
      use_fused_bwd_kernel=use_fused_bwd_kernel,
      q_layout=block_sizes.q_layout,
      k_layout=block_sizes.k_layout,
      v_layout=block_sizes.v_layout,
      mask_function=mask_function,
      interpret=interpret,
  )
  if not use_fused_bwd_kernel:
    assert dq is None
    dq = _splash_attention_bwd_dq(
        q,
        k,
        v,
        segment_ids,
        logsumexp,
        do,
        di,
        bq=bq_dq,
        bkv=bkv_dq,
        is_mqa=is_mqa,
        mask_info=dq_mask_info,
        mask_value=mask_value,
        attn_logits_soft_cap=attn_logits_soft_cap,
        q_layout=block_sizes.q_layout,
        k_layout=block_sizes.k_layout,
        v_layout=block_sizes.v_layout,
        mask_function=mask_function,
        interpret=interpret,
    )
  # Match the signature of the fwd function.
  assert dq is not None
  return (
      None,  # fwd_mask_info
      None,  # dq_mask_info
      None,  # dvk_mak_info
      dq,  # q
      dk,  # k
      dv,  # v
      None,  # segment_ids
  )


_splash_attention_custom.defvjp(_splash_attention_fwd, _splash_attention_bwd)


@partial(
    jax.jit,
    static_argnames=[
        "is_mqa",
        "block_sizes",
        "save_residuals",
        "mask_value",
        "attn_logits_soft_cap",
        "residual_checkpoint_name",
        "mask_function",
        "interpret",
    ],
)
def _splash_attention(
    fwd_mask_info: mask_info_lib.MaskInfo,
    dq_mask_info: mask_info_lib.MaskInfo | None,
    dkv_mask_info: mask_info_lib.MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None = None,
    *,
    is_mqa: bool,
    block_sizes: BlockSizes | None,
    save_residuals: bool,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    residual_checkpoint_name: str | None,
    mask_function: MaskFunctionType | None,
    interpret: bool,
) -> SplashCustomReturnType:
  return _splash_attention_custom(
      fwd_mask_info,
      dq_mask_info,
      dkv_mask_info,
      q,
      k,
      v,
      segment_ids,
      mask_value=mask_value,
      is_mqa=is_mqa,
      block_sizes=block_sizes,
      save_residuals=save_residuals,
      attn_logits_soft_cap=attn_logits_soft_cap,
      residual_checkpoint_name=residual_checkpoint_name,
      mask_function=mask_function,
      interpret=interpret,
  )


@jax.tree_util.register_pytree_node_class
class SplashAttentionKernel:

  def __init__(
      self,
      fwd_mask_info: mask_info_lib.MaskInfo,
      dq_mask_info: mask_info_lib.MaskInfo | None,
      dkv_mask_info: mask_info_lib.MaskInfo | None,
      **kwargs,
  ):
    self.kwargs = kwargs
    self.fwd_mask_info = fwd_mask_info
    self.dq_mask_info = dq_mask_info
    self.dkv_mask_info = dkv_mask_info

  def __call__(self, *args, **kwargs) -> SplashCustomReturnType:
    return _splash_attention(
        self.fwd_mask_info,
        self.dq_mask_info,
        self.dkv_mask_info,
        *args,
        **kwargs,
        **self.kwargs,
    )

  def manual_sharding_spec(self, sharding: jax.sharding.NamedSharding):
    """Returns a value that can be used as a shard_map partition spec for the kernel."""
    if self.fwd_mask_info.data_next is not None:
      block_mask_shape = self.fwd_mask_info.data_next.shape
      try:
        shard_shape = sharding.shard_shape(block_mask_shape)
      except ValueError as exc:
        raise ValueError(
            "The sharding must divide the mask blocks evenly between devices"
        ) from exc
      if block_mask_shape[-1] != shard_shape[-1]:
        raise ValueError("Sharding the kv sequence dimension is not supported")
    spec = sharding.spec
    assert len(spec) == 2
    replicated = jax.sharding.PartitionSpec()
    # Shard q_sequence over the sequence dimension only.
    q_sequence_spec = jax.sharding.PartitionSpec(spec[1])
    mask_info_specs = mask_info_lib.MaskInfo(  # pytype: disable=wrong-arg-types
        data_next=spec if self.fwd_mask_info.data_next is not None else None,
        mask_next=spec if self.fwd_mask_info.mask_next is not None else None,
        block_mask=spec if self.fwd_mask_info.block_mask is not None else None,
        partial_mask_blocks=replicated
        if self.fwd_mask_info.partial_mask_blocks is not None
        else None,
        q_sequence=q_sequence_spec
        if self.fwd_mask_info.q_sequence is not None
        else None,
    )
    return SplashAttentionKernel(
        mask_info_specs,
        mask_info_specs if self.dq_mask_info is not None else None,
        mask_info_specs if self.dkv_mask_info is not None else None,
        **self.kwargs,
    )

  def tree_flatten(self):
    return (
        (self.fwd_mask_info, self.dq_mask_info, self.dkv_mask_info),
        self.kwargs,
    )

  @classmethod
  def tree_unflatten(cls, kwargs, values):
    fwd_mask_info, dq_mask_info, dkv_mask_info = values
    # NamedTuples are not preserved during pytree serialization.
    dq_mask_info = (
        mask_info_lib.MaskInfo(*dq_mask_info)
        if dq_mask_info is not None
        else None
    )
    dkv_mask_info = (
        mask_info_lib.MaskInfo(*dkv_mask_info)
        if dkv_mask_info is not None
        else None
    )
    return SplashAttentionKernel(
        mask_info_lib.MaskInfo(*fwd_mask_info),
        dq_mask_info,
        dkv_mask_info,
        **kwargs,
    )


def _make_splash_attention(
    mask: np.ndarray | mask_lib.MultiHeadMask,
    *,
    block_sizes: BlockSizes | None = None,
    is_mqa: bool,
    save_residuals: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    downcast_smem_data: bool = True,
    head_shards: int,
    q_seq_shards: int,
    residual_checkpoint_name: str | None = None,
    interpret: bool = False,
):
  if len(mask.shape) != 3:
    raise ValueError(f'Unexpected mask shape: {mask.shape}')

  if isinstance(mask, np.ndarray):
    mask = mask_lib.MultiHeadMask(
        [mask_lib.NumpyMask(head_mask) for head_mask in mask]
    )

  if block_sizes is None:
    block_sizes = BlockSizes.get_default()
  fwd_mask_info, mask_function_fwd = mask_info_lib.process_mask(
      mask,
      (block_sizes.block_q, block_sizes.block_kv),
      downcast_smem_data=downcast_smem_data,
      head_shards=head_shards,
      q_seq_shards=q_seq_shards,
  )

  fwd_mask_info = tree_util.tree_map(jnp.array, fwd_mask_info)

  dq_mask_info = None
  dkv_mask_info = None
  if block_sizes.has_backward_blocks:
    if block_sizes.use_fused_bwd_kernel:
      dq_mask_info = None
    else:
      bq_dq, bkv_dq = block_sizes.block_q_dq, block_sizes.block_kv_dq
      dq_mask_info, mask_function_dq = mask_info_lib.process_mask(
          mask,
          (bq_dq, bkv_dq),
          downcast_smem_data=downcast_smem_data,
          head_shards=head_shards,
          q_seq_shards=q_seq_shards,
      )
      assert (mask_function_fwd is None) == (mask_function_dq is None)
      dq_mask_info = tree_util.tree_map(jnp.array, dq_mask_info)
    bq_dkv, bkv_dkv = block_sizes.block_q_dkv, block_sizes.block_kv_dkv
    dkv_mask_info, mask_function_dkv = mask_info_lib.process_mask_dkv(
        mask,
        (bq_dkv, bkv_dkv),
        downcast_smem_data=downcast_smem_data,
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
        shrink_grid=not block_sizes.use_fused_bwd_kernel,
    )
    assert (mask_function_fwd is None) == (mask_function_dkv is None)

    dkv_mask_info = tree_util.tree_map(jnp.array, dkv_mask_info)

  return SplashAttentionKernel(
      fwd_mask_info,
      dq_mask_info,
      dkv_mask_info,
      block_sizes=block_sizes,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      mask_value=mask_value,
      attn_logits_soft_cap=attn_logits_soft_cap,
      residual_checkpoint_name=residual_checkpoint_name,
      mask_function=mask_function_fwd,
      interpret=interpret,
  )


make_splash_mha = partial(_make_splash_attention, is_mqa=False)
make_splash_mqa = partial(_make_splash_attention, is_mqa=True)

make_splash_mha_single_device = partial(
    make_splash_mha, is_mqa=False, head_shards=1, q_seq_shards=1
)

make_splash_mqa_single_device = partial(
    make_splash_mha, is_mqa=True, head_shards=1, q_seq_shards=1
)
