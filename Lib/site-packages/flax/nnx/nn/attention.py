# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attention core modules for Flax."""

from __future__ import annotations

import functools
from typing import Any
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax, random

from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import Module, first_from
from flax.nnx.nn import initializers
from flax.nnx.nn.dtypes import promote_dtype
from flax.nnx.nn.linear import (
  LinearGeneral,
  default_kernel_init,
)
from flax.nnx.nn.normalization import LayerNorm
from flax.typing import (
  Dtype,
  Shape,
  Initializer,
  PrecisionLike,
  DotGeneralT,
)

Array = jax.Array


def dot_product_attention_weights(
  query: Array,
  key: Array,
  bias: Array | None = None,
  mask: Array | None = None,
  broadcast_dropout: bool = True,
  dropout_rng: Array | None = None,
  dropout_rate: float = 0.0,
  deterministic: bool = False,
  dtype: Dtype | None = None,
  precision: PrecisionLike = None,
  module: Module | None = None,
):
  """Computes dot-product attention weights given query and key.

  Used by :func:`dot_product_attention`, which is what you'll most likely use.
  But if you want access to the attention weights for introspection, then
  you can directly call this function and call einsum yourself.

  Args:
    query: queries for calculating attention with shape of `[batch..., q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch..., kv_length,
      num_heads, qk_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks. Attention weights are masked out if their
      corresponding mask value is `False`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs and params)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    module: the Module that will sow the attention weights into the
      ``nnx.Intermediate`` collection. If ``module`` is None, the attention
      weights will not be sowed.

  Returns:
    Output of shape `[batch..., num_heads, q_length, kv_length]`.
  """
  query, key = promote_dtype((query, key), dtype=dtype)  # type: ignore[bad-unpacking]
  dtype = query.dtype

  assert query.ndim == key.ndim, 'q, k must have same rank.'
  assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
  assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # calculate attention matrix
  depth = query.shape[-1]
  query = query / jnp.sqrt(depth).astype(dtype)
  # attn weight shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = jnp.einsum(
    '...qhd,...khd->...hqk', query, key, precision=precision
  )

  # apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias
  # apply attention mask
  if mask is not None:
    big_neg = jnp.finfo(dtype).min
    attn_weights = jnp.where(mask, attn_weights, big_neg)

  # normalize the attention weights
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  if module:
    module.sow(nnx.Intermediate, 'attention_weights', attn_weights)

  # apply attention dropout
  if not deterministic and dropout_rate > 0.0:
    keep_prob = 1.0 - dropout_rate
    if broadcast_dropout:
      # dropout is broadcast across the batch + head dimensions
      dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
      keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
    else:
      keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)  # type: ignore
    multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
    attn_weights = attn_weights * multiplier

  return attn_weights


def dot_product_attention(
  query: Array,
  key: Array,
  value: Array,
  bias: Array | None = None,
  mask: Array | None = None,
  broadcast_dropout: bool = True,
  dropout_rng: Array | None = None,
  dropout_rate: float = 0.0,
  deterministic: bool = False,
  dtype: Dtype | None = None,
  precision: PrecisionLike = None,
  module: Module | None = None,
):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  .. note::
    ``query``, ``key``, ``value`` needn't have any batch dimensions.

  Args:
    query: queries for calculating attention with shape of ``[batch..., q_length,
      num_heads, qk_depth_per_head]``.
    key: keys for calculating attention with shape of ``[batch..., kv_length,
      num_heads, qk_depth_per_head]``.
    value: values to be used in attention with shape of ``[batch..., kv_length,
      num_heads, v_depth_per_head]``.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape `[batch..., num_heads, q_length, kv_length]`. This can be used for
      incorporating causal masks. Attention weights are masked out if their
      corresponding mask value is `False`.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    module: the Module that will sow the attention weights into the
      ``nnx.Intermediate`` collection. If ``module`` is None, the attention
      weights will not be sowed.

  Returns:
    Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
  """
  query, key, value = promote_dtype((query, key, value), dtype=dtype)  # type: ignore[bad-unpacking]
  dtype = query.dtype
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert (
    query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
  ), 'q, k, v batch dims must match.'
  assert (
    query.shape[-2] == key.shape[-2] == value.shape[-2]
  ), 'q, k, v num_heads must match.'
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'

  # compute attention weights
  attn_weights = dot_product_attention_weights(
    query,
    key,
    bias,
    mask,
    broadcast_dropout,
    dropout_rng,
    dropout_rate,
    deterministic,
    dtype,
    precision,
    module,
  )

  # return weighted sum over values for each query position
  return jnp.einsum(
    '...hqk,...khd->...qhd', attn_weights, value, precision=precision
  )


class MultiHeadAttention(Module):
  """Multi-head attention.

  Example usage::

    >>> from flax import nnx
    >>> import jax

    >>> layer = nnx.MultiHeadAttention(num_heads=8, in_features=5, qkv_features=16,
    ...                                decode=False, rngs=nnx.Rngs(0))
    >>> key1, key2, key3 = jax.random.split(jax.random.key(0), 3)
    >>> shape = (4, 3, 2, 5)
    >>> q, k, v = (
    ...   jax.random.uniform(key1, shape),
    ...   jax.random.uniform(key2, shape),
    ...   jax.random.uniform(key3, shape),
    ... )

    >>> # different inputs for inputs_q, inputs_k and inputs_v
    >>> out = layer(q, k, v)
    >>> # equivalent output when inferring v
    >>> assert (layer(q, k) == layer(q, k, k)).all()
    >>> # equivalent output when inferring k and v
    >>> assert (layer(q) == layer(q, q)).all()
    >>> assert (layer(q) == layer(q, q, q)).all()

  Args:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    in_features: int or tuple with number of input features.
    qkv_features: dimension of the key, query, and value.
    out_features: dimension of the last projection
    dtype: the dtype of the computation (default: infer from inputs and params)
    param_dtype: the dtype passed to parameter initializers (default: float32)
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rate: dropout rate
    deterministic: if false, the attention weight is masked randomly using
      dropout, whereas if true, the attention weights are deterministic.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the kernel of the Dense layers.
    out_kernel_init: optional initializer for the kernel of the output Dense layer,
      if None, the kernel_init is used.
    bias_init: initializer for the bias of the Dense layers.
    out_bias_init: optional initializer for the bias of the output Dense layer,
      if None, the bias_init is used.
    use_bias: bool: whether pointwise QKVO dense transforms use bias.
    attention_fn: dot_product_attention or compatible function. Accepts query,
      key, value, and returns output of shape `[bs, dim1, dim2, ..., dimN,,
      num_heads, value_channels]``
    decode: whether to prepare and use an autoregressive cache.
    normalize_qk: should QK normalization be applied (arxiv.org/abs/2302.05442).
    rngs: rng key.
  """

  def __init__(
    self,
    num_heads: int,
    in_features: int,
    qkv_features: int | None = None,
    out_features: int | None = None,
    *,
    dtype: Dtype | None = None,
    param_dtype: Dtype = jnp.float32,
    broadcast_dropout: bool = True,
    dropout_rate: float = 0.0,
    deterministic: bool | None = None,
    precision: PrecisionLike = None,
    kernel_init: Initializer = default_kernel_init,
    out_kernel_init: Initializer | None = None,
    bias_init: Initializer = initializers.zeros_init(),
    out_bias_init: Initializer | None = None,
    use_bias: bool = True,
    attention_fn: Callable[..., Array] = dot_product_attention,
    decode: bool | None = None,
    normalize_qk: bool = False,
    # Deprecated, will be removed.
    qkv_dot_general: DotGeneralT | None = None,
    out_dot_general: DotGeneralT | None = None,
    qkv_dot_general_cls: Any = None,
    out_dot_general_cls: Any = None,
    rngs: rnglib.Rngs,
  ):
    self.num_heads = num_heads
    self.in_features = in_features
    self.qkv_features = (
      qkv_features if qkv_features is not None else in_features
    )
    self.out_features = (
      out_features if out_features is not None else in_features
    )
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.broadcast_dropout = broadcast_dropout
    self.dropout_rate = dropout_rate
    self.deterministic = deterministic
    self.precision = precision
    self.kernel_init = kernel_init
    self.out_kernel_init = out_kernel_init
    self.bias_init = bias_init
    self.out_bias_init = out_bias_init
    self.use_bias = use_bias
    self.attention_fn = attention_fn
    self.decode = decode
    self.normalize_qk = normalize_qk
    self.qkv_dot_general = qkv_dot_general
    self.out_dot_general = out_dot_general
    self.qkv_dot_general_cls = qkv_dot_general_cls
    self.out_dot_general_cls = out_dot_general_cls

    if self.qkv_features % self.num_heads != 0:
      raise ValueError(
        f'Memory dimension ({self.qkv_features}) must be divisible by '
        f"'num_heads' heads ({self.num_heads})."
      )

    self.head_dim = self.qkv_features // self.num_heads

    linear_general = functools.partial(
      LinearGeneral,
      in_features=self.in_features,
      out_features=(self.num_heads, self.head_dim),
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      use_bias=self.use_bias,
      precision=self.precision,
      dot_general=self.qkv_dot_general,
      dot_general_cls=self.qkv_dot_general_cls,
    )
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    self.query = linear_general(rngs=rngs)
    self.key = linear_general(rngs=rngs)
    self.value = linear_general(rngs=rngs)

    self.query_ln: LayerNorm | None
    self.key_ln: LayerNorm | None
    if self.normalize_qk:
      # Normalizing query and key projections stabilizes training with higher
      # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
      self.query_ln = LayerNorm(
        self.head_dim,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        rngs=rngs,
      )
      self.key_ln = LayerNorm(
        self.head_dim,
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        rngs=rngs,
      )
    else:
      self.query_ln = None
      self.key_ln = None

    self.out = LinearGeneral(
      in_features=(self.num_heads, self.head_dim),
      out_features=self.out_features,
      axis=(-2, -1),
      kernel_init=self.out_kernel_init or self.kernel_init,
      bias_init=self.out_bias_init or self.bias_init,
      use_bias=self.use_bias,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      precision=self.precision,
      dot_general=self.out_dot_general,
      dot_general_cls=self.out_dot_general_cls,
      rngs=rngs,
    )
    self.rngs = rngs if dropout_rate > 0.0 else None

    self.cached_key: nnx.Cache[Array] | None = None
    self.cached_value: nnx.Cache[Array] | None = None
    self.cache_index: nnx.Cache[Array] | None = None

  def __call__(
    self,
    inputs_q: Array,
    inputs_k: Array | None = None,
    inputs_v: Array | None = None,
    *,
    mask: Array | None = None,
    deterministic: bool | None = None,
    rngs: rnglib.Rngs | None = None,
    sow_weights: bool = False,
    decode: bool | None = None,
  ):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    If both inputs_k and inputs_v are None, they will both copy the value of
    inputs_q (self attention).
    If only inputs_v is None, it will copy the value of inputs_k.

    Args:
      inputs_q: input queries of shape `[batch_sizes..., length, features]`.
      inputs_k: key of shape `[batch_sizes..., length, features]`. If None,
        inputs_k will copy the value of inputs_q.
      inputs_v: values of shape `[batch_sizes..., length, features]`. If None,
        inputs_v will copy the value of inputs_k.
      mask: attention mask of shape `[batch_sizes..., num_heads, query_length,
        key/value_length]`. Attention weights are masked out if their
        corresponding mask value is `False`.
      deterministic: if false, the attention weight is masked randomly using
        dropout, whereas if true, the attention weights are deterministic. The
        ``deterministic`` flag passed into the call method will take precedence
        over the ``deterministic`` flag passed into the constructor.
      rngs: rng key. The rng key passed into the call method will take
        precedence over the rng key passed into the constructor.
      sow_weights: if ``True``, the attention weights are sowed into the
        'intermediates' collection.
      decode: whether to prepare and use an autoregressive cache. The ``decode``
        flag passed into the call method will take precedence over the ``decode``
        flag passed into the constructor.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    if rngs is None:
      rngs = self.rngs

    if inputs_k is None:
      if inputs_v is not None:
        raise ValueError(
          '`inputs_k` cannot be None if `inputs_v` is not None. '
          'To have both `inputs_k` and `inputs_v` be the same value, pass in the '
          'value to `inputs_k` and leave `inputs_v` as None.'
        )
      inputs_k = inputs_q
    if inputs_v is None:
      inputs_v = inputs_k

    if inputs_q.shape[-1] != self.in_features:
      raise ValueError(
        f'Incompatible input dimension, got {inputs_q.shape[-1]} '
        f'but module expects {self.in_features}.'
      )

    query = self.query(inputs_q)
    key = self.key(inputs_k)
    value = self.value(inputs_v)

    if self.normalize_qk:
      assert self.query_ln is not None and self.key_ln is not None
      # Normalizing query and key projections stabilizes training with higher
      # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
      query = self.query_ln(query)
      key = self.key_ln(key)

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    decode = first_from(
      decode,
      self.decode,
      error_msg="""No `decode` argument was provided to MultiHeadAttention
        as either a __call__ argument, class attribute, or nnx.flag.""",
    )

    if decode:
      if (
        self.cached_key is None
        or self.cached_value is None
        or self.cache_index is None
      ):
        raise ValueError(
          'Autoregressive cache not initialized, call ``init_cache`` first.'
        )
      (
        *batch_dims,
        max_length,
        num_heads,
        depth_per_head,
      ) = self.cached_key.value.shape
      # shape check of cached keys against query input
      expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
      if expected_shape != query.shape:
        raise ValueError(
          'Autoregressive cache shape error, '
          'expected query shape %s instead got %s.'
          % (expected_shape, query.shape)
        )
      # update key, value caches with our new 1d spatial slices
      cur_index = self.cache_index.value
      zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
      indices = (zero,) * len(batch_dims) + (cur_index, zero, zero)
      key = lax.dynamic_update_slice(self.cached_key.value, key, indices)
      value = lax.dynamic_update_slice(self.cached_value.value, value, indices)
      self.cached_key.value = key
      self.cached_value.value = value
      self.cache_index.value += 1
      # causal mask for cached decoder self-attention:
      # our single query position should only attend to those key
      # positions that have already been generated and cached,
      # not the remaining zero elements.
      mask = combine_masks(
        mask,
        jnp.broadcast_to(
          jnp.arange(max_length) <= cur_index,
          tuple(batch_dims) + (1, 1, max_length),
        ),
      )

    if (
      self.dropout_rate > 0.0
    ):  # Require `deterministic` only if using dropout.
      deterministic = first_from(
        deterministic,
        self.deterministic,
        error_msg="""No `deterministic` argument was provided to MultiHeadAttention
          as either a __call__ argument, class attribute, or nnx.flag.""",
      )
      if not deterministic:
        if rngs is None:
          raise ValueError(
            "'rngs' must be provided if 'dropout_rng' is not given."
          )
        dropout_rng = rngs.dropout()
      else:
        dropout_rng = None
    else:
      deterministic = True
      dropout_rng = None

    # apply attention
    x = self.attention_fn(
      query,
      key,
      value,
      mask=mask,
      dropout_rng=dropout_rng,
      dropout_rate=self.dropout_rate,
      broadcast_dropout=self.broadcast_dropout,
      deterministic=deterministic,
      dtype=self.dtype,
      precision=self.precision,
      module=self if sow_weights else None,
    )
    # back to the original inputs dimensions
    out = self.out(x)
    return out

  def init_cache(self, input_shape: Shape, dtype: Dtype = jnp.float32):
    """Initializes cache for fast autoregressive decoding. When
    ``decode=True``, this method must be called first before performing
    forward inference. When in decode mode, only one token must be passed
    at a time.

    Example usage::

      >>> from flax import nnx
      >>> import jax.numpy as jnp
      ...
      >>> batch_size = 5
      >>> embed_dim = 3
      >>> x = jnp.ones((batch_size, 1, embed_dim)) # single token
      ...
      >>> model_nnx = nnx.MultiHeadAttention(
      ...   num_heads=2,
      ...   in_features=3,
      ...   qkv_features=6,
      ...   out_features=6,
      ...   decode=True,
      ...   rngs=nnx.Rngs(42),
      ... )
      ...
      >>> # out_nnx = model_nnx(x)  <-- throws an error because cache isn't initialized
      ...
      >>> model_nnx.init_cache(x.shape)
      >>> out_nnx = model_nnx(x)
    """
    cache_shape = (*input_shape[:-1], self.num_heads, self.head_dim)
    self.cached_key = nnx.Cache(jnp.zeros(cache_shape, dtype))
    self.cached_value = nnx.Cache(jnp.zeros(cache_shape, dtype))
    self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.int32))


# mask-making utility functions


def make_attention_mask(
  query_input: Array,
  key_input: Array,
  pairwise_fn: Callable[..., Any] = jnp.multiply,
  extra_batch_dims: int = 0,
  dtype: Dtype = jnp.float32,
):
  """Mask-making helper for attention weights.

  In case of 1d inputs (i.e., `[batch..., len_q]`, `[batch..., len_kv]`, the
  attention weights will be `[batch..., heads, len_q, len_kv]` and this
  function will produce `[batch..., 1, len_q, len_kv]`.

  Args:
    query_input: a batched, flat input of query_length size
    key_input: a batched, flat input of key_length size
    pairwise_fn: broadcasting elementwise comparison function
    extra_batch_dims: number of extra batch dims to add singleton axes for, none
      by default
    dtype: mask return dtype

  Returns:
    A `[batch..., 1, len_q, len_kv]` shaped mask for 1d attention.
  """
  mask = pairwise_fn(
    jnp.expand_dims(query_input, axis=-1), jnp.expand_dims(key_input, axis=-2)
  )
  mask = jnp.expand_dims(mask, axis=-3)
  mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
  return mask.astype(dtype)


def make_causal_mask(
  x: Array, extra_batch_dims: int = 0, dtype: Dtype = jnp.float32
) -> Array:
  """Make a causal mask for self-attention.

  In case of 1d inputs (i.e., `[batch..., len]`, the self-attention weights
  will be `[batch..., heads, len, len]` and this function will produce a
  causal mask of shape `[batch..., 1, len, len]`.

  Args:
    x: input array of shape `[batch..., len]`
    extra_batch_dims: number of batch dims to add singleton axes for, none by
      default
    dtype: mask return dtype

  Returns:
    A `[batch..., 1, len, len]` shaped causal mask for 1d attention.
  """
  idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
  return make_attention_mask(
    idxs,
    idxs,
    jnp.greater_equal,
    extra_batch_dims=extra_batch_dims,
    dtype=dtype,
  )


def combine_masks(
  *masks: Array | None, dtype: Dtype = jnp.float32
) -> Array | None:
  """Combine attention masks.

  Args:
    *masks: set of attention mask arguments to combine, some can be None.
    dtype: dtype for the returned mask.

  Returns:
    Combined mask, reduced by logical and, returns None if no masks given.
  """
  masks_list = [m for m in masks if m is not None]
  if not masks_list:
    return None
  assert all(
    map(lambda x: x.ndim == masks_list[0].ndim, masks_list)
  ), f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks_list))}'
  mask, *other_masks = masks_list
  for other_mask in other_masks:
    mask = jnp.logical_and(mask, other_mask)
  return mask.astype(dtype)
