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
import inspect
import warnings
from typing import Any, overload
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax, random

from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import (
  DenseGeneral,
  default_kernel_init,
)
from flax.linen.module import Module, compact, merge_param
from flax.linen.normalization import LayerNorm
from flax.typing import (
  Array,
  PRNGKey,
  Dtype,
  Shape as Shape,
  Initializer,
  PrecisionLike,
  DotGeneralT,
)


def dot_product_attention_weights(
    query: Array,
    key: Array,
    bias: Array | None = None,
    mask: Array | None = None,
    broadcast_dropout: bool = True,
    dropout_rng: PRNGKey | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Dtype | None = None,
    precision: PrecisionLike = None,
    module: Module | None = None,
    force_fp32_for_softmax: bool = False,
    einsum_dot_general: Callable[..., Array] | None = None,
    einsum: Callable[..., Array] | None = None,
):
  """Computes dot-product attention weights given query and key.

  Used by :func:`dot_product_attention`, which is what you'll most likely use.
  But if you want access to the attention weights for introspection, then
  you can directly call this function and call einsum yourself.

  Args:
    query: queries for calculating attention with shape of ``[batch...,
      q_length, num_heads, qk_depth_per_head]``.
    key: keys for calculating attention with shape of ``[batch..., kv_length,
      num_heads, qk_depth_per_head]``.
    bias: bias for the attention weights. This should be broadcastable to the
      shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
      incorporating causal masks. Attention weights are masked out if their
      corresponding mask value is ``False``.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs and params)
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    module: the Module that will sow the attention weights into the
      'intermediates' collection. Remember to mark 'intermediates' as mutable
      via ``mutable=['intermediates']`` in order to have that collection
      returned. If ``module`` is None, the attention weights will not be sowed.
    force_fp32_for_softmax: bool, whether to force the softmax to be computed in
      fp32. This is useful for mixed-precision training where higher precision
      is desired for numerical stability.
    einsum_dot_general: the dot_general to use in einsum.
    einsum: If unspecified, default `jnp.einsum` will be used. This argument is
      mutually exclusive with `precision` and `einsum_dot_general`.

  Raises:
    ValueError: if both `precision`/`einsum_dot_general` and `einsum` are
      specified.

  Returns:
    Output of shape ``[batch..., num_heads, q_length, kv_length]``.
  """
  if (precision or einsum_dot_general) and einsum:
    raise ValueError(
        'precision/einsum_dot_general and einsum are mutually exclusive. Please'
        ' specify only one of them.'
    )
  if not einsum:
    einsum = functools.partial(
        jnp.einsum,
        precision=precision,
        _dot_general=einsum_dot_general
        if einsum_dot_general
        else jax.lax.dot_general,
    )

  query, key = promote_dtype(query, key, dtype=dtype)
  dtype = query.dtype

  assert query.ndim == key.ndim, 'q, k must have same rank.'
  assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
  assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # calculate attention matrix
  depth = query.shape[-1]
  query = query / jnp.sqrt(depth).astype(dtype)
  # attn weight shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = einsum('...qhd,...khd->...hqk', query, key)

  # apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias
  # apply attention mask
  if mask is not None:
    big_neg = jnp.finfo(dtype).min
    attn_weights = jnp.where(mask, attn_weights, big_neg)

  # normalize the attention weights
  if force_fp32_for_softmax and dtype != jnp.float32:
    attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32))
  else:
    attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  if module:
    module.sow('intermediates', 'attention_weights', attn_weights)

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
    dropout_rng: PRNGKey | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Dtype | None = None,
    precision: PrecisionLike = None,
    module: Module | None = None,
    force_fp32_for_softmax: bool = False,
    einsum_dot_general: Callable[..., Array] | None = None,
    qk_attn_weights_einsum: Callable[..., Array] | None = None,
    attn_weights_value_einsum: Callable[..., Array] | None = None,
):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  .. note::
    ``query``, ``key``, ``value`` needn't have any batch dimensions.

  Args:
    query: queries for calculating attention with shape of ``[batch...,
      q_length, num_heads, qk_depth_per_head]``.
    key: keys for calculating attention with shape of ``[batch..., kv_length,
      num_heads, qk_depth_per_head]``.
    value: values to be used in attention with shape of ``[batch..., kv_length,
      num_heads, v_depth_per_head]``.
    bias: bias for the attention weights. This should be broadcastable to the
      shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    mask: mask for the attention weights. This should be broadcastable to the
      shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
      incorporating causal masks. Attention weights are masked out if their
      corresponding mask value is ``False``.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: infer from inputs)
    precision: numerical precision of the computation see ``jax.lax.Precision`
      for details.
    module: the Module that will sow the attention weights into the
      'intermediates' collection. Remember to mark 'intermediates' as mutable
      via ``mutable=['intermediates']`` in order to have that collection
      returned. If ``module`` is None, the attention weights will not be sowed.
    force_fp32_for_softmax: bool, whether to force the softmax to be computed in
      fp32. This is useful for mixed-precision training where higher precision
      is desired for numerical stability.
    einsum_dot_general: the dot_general to use in `jnp.einsum`.
    qk_attn_weights_einsum: the einsum for computing the attention weights. When
      unspecified, the default `jnp.einsum` will be used. This argument is
      mutually exclusive with `precision` and `einsum_dot_general`.
    attn_weights_value_einsum: the einsum for computing the product of the
      attention weights and the values. When unspecified, the default
      `jnp.einsum` will be used. This argument is mutually exclusive with
      `precision` and `einsum_dot_general`.

  Returns:
    Output of shape ``[batch..., q_length, num_heads, v_depth_per_head]``.

  Raises:
    ValueError: if both `precision`/`einsum_dot_general` and
    `qk_attn_weights_einsum`/`attn_weights_value_einsum` are
      specified.
  """
  if (qk_attn_weights_einsum and not attn_weights_value_einsum) or (
      not qk_attn_weights_einsum and attn_weights_value_einsum
  ):
    raise ValueError(
        'qk_attn_weights_einsum and attn_weights_value_einsum must be specified'
        ' together.'
    )
  if (precision or einsum_dot_general) and (
      qk_attn_weights_einsum or attn_weights_value_einsum
  ):
    raise ValueError(
        'precision/einsum_dot_general and'
        ' qk_attn_weights_einsum/attn_weights_value_einsum are mutually'
        ' exclusive. Please specify only one of them.'
    )

  query, key, value = promote_dtype(query, key, value, dtype=dtype)
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
      force_fp32_for_softmax,
      einsum_dot_general=einsum_dot_general,
      einsum=qk_attn_weights_einsum,
  )
  if not attn_weights_value_einsum:
    attn_weights_value_einsum = functools.partial(
        jnp.einsum,
        precision=precision,
        _dot_general=einsum_dot_general
        if einsum_dot_general
        else jax.lax.dot_general,
    )
  # return weighted sum over values for each query position
  return attn_weights_value_einsum(
      '...hqk,...khd->...qhd',
      attn_weights,
      value,
  )


class MultiHeadDotProductAttention(Module):
  """Multi-head dot-product attention.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax

    >>> layer = nn.MultiHeadDotProductAttention(num_heads=8, qkv_features=16)
    >>> key1, key2, key3, key4, key5, key6 = jax.random.split(jax.random.key(0), 6)
    >>> shape = (4, 3, 2, 5)
    >>> q, k, v = jax.random.uniform(key1, shape), jax.random.uniform(key2, shape), jax.random.uniform(key3, shape)
    >>> variables = layer.init(jax.random.key(0), q)

    >>> # different inputs for inputs_q, inputs_k and inputs_v
    >>> out = layer.apply(variables, q, k, v)
    >>> # equivalent to layer.apply(variables, inputs_q=q, inputs_k=k, inputs_v=k)
    >>> out = layer.apply(variables, q, k)
    >>> # equivalent to layer.apply(variables, inputs_q=q, inputs_k=q) and layer.apply(variables, inputs_q=q, inputs_k=q, inputs_v=q)
    >>> out = layer.apply(variables, q)

    >>> attention_kwargs = dict(
    ...     num_heads=8,
    ...     qkv_features=16,
    ...     kernel_init=nn.initializers.ones,
    ...     bias_init=nn.initializers.zeros,
    ...     dropout_rate=0.5,
    ...     deterministic=False,
    ...     )
    >>> class Module(nn.Module):
    ...   attention_kwargs: dict
    ...
    ...   @nn.compact
    ...   def __call__(self, x, dropout_rng=None):
    ...     out1 = nn.MultiHeadDotProductAttention(**self.attention_kwargs)(x, dropout_rng=dropout_rng)
    ...     out2 = nn.MultiHeadDotProductAttention(**self.attention_kwargs)(x, dropout_rng=dropout_rng)
    ...     return out1, out2
    >>> module = Module(attention_kwargs)
    >>> variables = module.init({'params': key1, 'dropout': key2}, q)

    >>> # out1 and out2 are different.
    >>> out1, out2 = module.apply(variables, q, rngs={'dropout': key3})
    >>> # out3 and out4 are different.
    >>> # out1 and out3 are different. out2 and out4 are different.
    >>> out3, out4 = module.apply(variables, q, rngs={'dropout': key4})
    >>> # out1 and out2 are the same.
    >>> out1, out2 = module.apply(variables, q, dropout_rng=key5)
    >>> # out1 and out2 are the same as out3 and out4.
    >>> # providing a `dropout_rng` arg will take precedence over the `rngs` arg in `.apply`
    >>> out3, out4 = module.apply(variables, q, rngs={'dropout': key6}, dropout_rng=key5)

  Attributes:
    num_heads: Number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    dtype: The dtype of the computation (default: infer from inputs and params)
    param_dtype: The dtype passed to parameter initializers (default: float32)
    qkv_features: Dimension of the key, query, and value.
    out_features: Dimension of the last projection
    broadcast_dropout: Use a broadcasted dropout along batch dims.
    dropout_rate: Dropout rate.
    deterministic: If False, the attention weight is masked randomly using
      dropout, whereas if True, the attention weights are deterministic.
    precision: Numerical precision of the computation see ``jax.lax.Precision``
      for details.
    kernel_init: Initializer for the kernel of the Dense layers.
    out_kernel_init: Optional Initializer for the kernel of the output Dense layer,
      if None, ``kernel_init`` will be used.
    bias_init: Initializer for the bias of the Dense layers.
    out_bias_init: Optional Initializer for the bias of the output Dense layer,
      if None, ``bias_init`` will be used.
    use_bias: Whether pointwise QKVO dense transforms use bias.
    attention_fn: dot_product_attention or compatible function. Accepts query,
      key, value, and returns output of shape ``[bs, dim1, dim2, ..., dimN,,
      num_heads, value_channels]``
    decode: Whether to prepare and use an autoregressive cache.
    normalize_qk: Should QK normalization be applied (arxiv.org/abs/2302.05442).
    qk_attn_weights_einsum_cls: factory function to create the einsum for
      computing the attention weights.
    attn_weights_value_einsum_cls: factory function to create the einsum for
      computing the product of the attention weights and the values.
  """

  num_heads: int
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  qkv_features: int | None = None
  out_features: int | None = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.0
  deterministic: bool | None = None
  precision: PrecisionLike = None
  kernel_init: Initializer = default_kernel_init
  out_kernel_init: Initializer | None = None
  bias_init: Initializer = initializers.zeros_init()
  out_bias_init: Initializer | None = None
  use_bias: bool = True
  attention_fn: Callable[..., Array] = dot_product_attention
  decode: bool = False
  normalize_qk: bool = False
  force_fp32_for_softmax: bool = False
  # Deprecated, will be removed.
  qkv_dot_general: DotGeneralT | None = None
  out_dot_general: DotGeneralT | None = None
  qkv_dot_general_cls: Any = None
  out_dot_general_cls: Any = None
  qk_attn_weights_einsum_cls: Callable[..., Callable[..., Array]] | None = None
  attn_weights_value_einsum_cls: Callable[..., Callable[..., Array]] | None = (
      None
  )

  @overload
  def __call__(
    self,
    inputs_q: Array,
    inputs_k: Array | None = None,
    inputs_v: Array | None = None,
    *,
    mask: Array | None = None,
    deterministic: bool | None = None,
    dropout_rng: PRNGKey | None = None,
    sow_weights: bool = False,
  ):
    ...

  @overload
  def __call__(
    self,
    inputs_q: Array,
    *,
    inputs_kv: Array | None = None,
    mask: Array | None = None,
    deterministic: bool | None = None,
    dropout_rng: PRNGKey | None = None,
    sow_weights: bool = False,
  ):
    ...

  @compact
  def __call__(
    self,
    inputs_q: Array,
    inputs_k: Array | None = None,
    inputs_v: Array | None = None,
    *,
    inputs_kv: Array | None = None,
    mask: Array | None = None,
    deterministic: bool | None = None,
    dropout_rng: PRNGKey | None = None,
    sow_weights: bool = False,
  ):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    If both inputs_k and inputs_v are None, they will both copy the value of
    inputs_q (self attention).
    If only inputs_v is None, it will copy the value of inputs_k.

    Args:
      inputs_q: input queries of shape ``[batch_sizes..., length, features]``.
      inputs_k: key of shape ``[batch_sizes..., length, features]``. If None,
        inputs_k will copy the value of inputs_q.
      inputs_v: values of shape ``[batch_sizes..., length, features]``. If None,
        inputs_v will copy the value of inputs_k.
      inputs_kv: key/values of shape ``[batch_sizes..., length, features]``. If
        None, inputs_kv will copy the value of inputs_q. This arg will be
        deprecated soon. Use inputs_k and inputs_v instead.
      mask: attention mask of shape ``[batch_sizes..., num_heads, query_length,
        key/value_length]``. Attention weights are masked out if their
        corresponding mask value is ``False``.
      deterministic: if false, the attention weight is masked randomly using
        dropout, whereas if true, the attention weights are deterministic.
      dropout_rng: optional rng key to pass to the attention layer's dropout
        mask. Otherwise, self.make_rng('dropout') is used instead.
      sow_weights: if ``True``, the attention weights are sowed into the
        'intermediates' collection. Remember to mark 'intermediates' as
        mutable via ``mutable=['intermediates']`` in order to have that
        collection returned.

    Returns:
      output of shape ``[batch_sizes..., length, features]``.
    """
    if inputs_kv is not None:
      if inputs_k is not None or inputs_v is not None:
        raise ValueError(
          'If either `inputs_k` or `inputs_v` is not None, '
          '`inputs_kv` must be None. If `inputs_kv` is not None, both `inputs_k` '
          'and `inputs_v` must be None. We recommend using `inputs_k` and '
          '`inputs_v` args, since `inputs_kv` will be deprecated soon. See '
          'https://github.com/google/flax/discussions/3389 for more '
          'information.'
        )
      inputs_k = inputs_v = inputs_kv
      warnings.warn(
        'The inputs_kv arg will be deprecated soon. '
        'Use inputs_k and inputs_v instead. See '
        'https://github.com/google/flax/discussions/3389 '
        'for more information.',
        DeprecationWarning,
      )
    else:
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
      elif inputs_v.shape[-1] == inputs_v.shape[-2]:
        warnings.warn(
          f'You are passing an array of shape {inputs_v.shape} '
          'to the `inputs_v` arg, when you may have intended '
          'to pass it to the `mask` arg. As of Flax version '
          '0.7.4, the function signature of '
          "MultiHeadDotProductAttention's `__call__` method "
          'has changed to `__call__(inputs_q, inputs_k=None, '
          'inputs_v=None, *, inputs_kv=None, mask=None, '
          'deterministic=None)`. Use the kwarg `mask` instead. '
          'See https://github.com/google/flax/discussions/3389 '
          'and read the docstring for more information.',
          DeprecationWarning,
        )

    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
      f'Memory dimension ({qkv_features}) must be divisible by number of'
      f' heads ({self.num_heads}).'
    )
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(
      DenseGeneral,
      axis=-1,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      features=(self.num_heads, head_dim),
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      use_bias=self.use_bias,
      precision=self.precision,
      dot_general=self.qkv_dot_general,
      dot_general_cls=self.qkv_dot_general_cls,
    )
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (
      dense(name='query')(inputs_q),
      dense(name='key')(inputs_k),
      dense(name='value')(inputs_v),
    )

    if self.normalize_qk:
      # Normalizing query and key projections stabilizes training with higher
      # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
      query = LayerNorm(
        name='query_ln',
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
      )(query)  # type: ignore[call-arg]
      key = LayerNorm(
        name='key_ln',
        use_bias=False,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
      )(key)  # type: ignore[call-arg]

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if self.decode:
      # detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      cached_key = self.variable(
        'cache', 'cached_key', jnp.zeros, key.shape, key.dtype
      )
      cached_value = self.variable(
        'cache', 'cached_value', jnp.zeros, value.shape, value.dtype
      )
      cache_index = self.variable(
        'cache', 'cache_index', lambda: jnp.array(0, dtype=jnp.int32)
      )
      if is_initialized:
        (
          *batch_dims,
          max_length,
          num_heads,
          depth_per_head,
        ) = cached_key.value.shape
        # shape check of cached keys against query input
        expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        if expected_shape != query.shape:
          raise ValueError(
            'Autoregressive cache shape error, '
            'expected query shape %s instead got %s.'
            % (expected_shape, query.shape)
          )
        # update key, value caches with our new 1d spatial slices
        cur_index = cache_index.value
        zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
        indices: tuple[int | jax.Array, ...] = (zero,) * len(
          batch_dims
        ) + (
          cur_index,
          zero,
          zero,
        )
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
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
      m_deterministic = merge_param(
        'deterministic', self.deterministic, deterministic
      )
      if not m_deterministic and dropout_rng is None:
        dropout_rng = self.make_rng('dropout')
    else:
      m_deterministic = True

    # `qk_attn_weights_einsum` and `attn_weights_value_einsum` are optional
    # arguments that can be used to override the default `jnp.einsum`. They
    # exist for quantized einsum support in AQT.
    qk_attn_weights_einsum = (
        self.qk_attn_weights_einsum_cls()
        if self.qk_attn_weights_einsum_cls
        else None
    )
    attn_weights_value_einsum = (
        self.attn_weights_value_einsum_cls()
        if self.attn_weights_value_einsum_cls
        else None
    )
    # apply attention
    attn_args = (query, key, value)
    # This kwargs list match the default nn.dot_product_attention.
    # For custom `attention_fn`s, invalid kwargs will be filtered.
    attn_kwargs = dict(
      mask=mask,
      dropout_rng=dropout_rng,
      dropout_rate=self.dropout_rate,
      broadcast_dropout=self.broadcast_dropout,
      deterministic=m_deterministic,
      dtype=self.dtype,
      precision=self.precision,
      force_fp32_for_softmax=self.force_fp32_for_softmax,
      qk_attn_weights_einsum=qk_attn_weights_einsum,
      attn_weights_value_einsum=attn_weights_value_einsum,
    )
    attn_kwargs = {
        k: v
        for k, v in attn_kwargs.items()
        if k in inspect.signature(self.attention_fn).parameters
    }
    if sow_weights:
      x = self.attention_fn(*attn_args, **attn_kwargs, module=self)
    else:
      x = self.attention_fn(*attn_args, **attn_kwargs)
    # back to the original inputs dimensions
    out = DenseGeneral(
      features=features,
      axis=(-2, -1),
      kernel_init=self.out_kernel_init or self.kernel_init,
      bias_init=self.out_bias_init or self.bias_init,
      use_bias=self.use_bias,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      precision=self.precision,
      dot_general=self.out_dot_general,
      dot_general_cls=self.out_dot_general_cls,
      name='out',  # type: ignore[call-arg]
    )(x)
    return out


class MultiHeadAttention(MultiHeadDotProductAttention):
  """Multi-head dot-product attention.
  Alias for ``MultiHeadDotProductAttention``.

  **NOTE**: ``MultiHeadAttention`` is a wrapper of ``MultiHeadDotProductAttention``,
  and so their implementations are identical. However ``MultiHeadAttention`` layers
  will, by default, be named ``MultiHeadAttention_{index}``, whereas ``MultiHeadDotProductAttention``
  will be named ``MultiHeadDotProductAttention_{index}``. Therefore, this could affect
  checkpointing, param collection names and RNG threading (since the layer name is
  used when generating new RNG's) within the module.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax

    >>> layer = nn.MultiHeadAttention(num_heads=8, qkv_features=16)
    >>> key1, key2, key3, key4, key5, key6 = jax.random.split(jax.random.key(0), 6)
    >>> shape = (4, 3, 2, 5)
    >>> q, k, v = jax.random.uniform(key1, shape), jax.random.uniform(key2, shape), jax.random.uniform(key3, shape)
    >>> variables = layer.init(jax.random.key(0), q)

    >>> # different inputs for inputs_q, inputs_k and inputs_v
    >>> out = layer.apply(variables, q, k, v)
    >>> # equivalent to layer.apply(variables, inputs_q=q, inputs_k=k, inputs_v=k)
    >>> out = layer.apply(variables, q, k)
    >>> # equivalent to layer.apply(variables, inputs_q=q, inputs_k=q) and layer.apply(variables, inputs_q=q, inputs_k=q, inputs_v=q)
    >>> out = layer.apply(variables, q)

    >>> attention_kwargs = dict(
    ...     num_heads=8,
    ...     qkv_features=16,
    ...     kernel_init=nn.initializers.ones,
    ...     bias_init=nn.initializers.zeros,
    ...     dropout_rate=0.5,
    ...     deterministic=False,
    ...     )
    >>> class Module(nn.Module):
    ...   attention_kwargs: dict
    ...
    ...   @nn.compact
    ...   def __call__(self, x, dropout_rng=None):
    ...     out1 = nn.MultiHeadAttention(**self.attention_kwargs)(x, dropout_rng=dropout_rng)
    ...     out2 = nn.MultiHeadAttention(**self.attention_kwargs)(x, dropout_rng=dropout_rng)
    ...     return out1, out2
    >>> module = Module(attention_kwargs)
    >>> variables = module.init({'params': key1, 'dropout': key2}, q)

    >>> # out1 and out2 are different.
    >>> out1, out2 = module.apply(variables, q, rngs={'dropout': key3})
    >>> # out3 and out4 are different.
    >>> # out1 and out3 are different. out2 and out4 are different.
    >>> out3, out4 = module.apply(variables, q, rngs={'dropout': key4})
    >>> # out1 and out2 are the same.
    >>> out1, out2 = module.apply(variables, q, dropout_rng=key5)
    >>> # out1 and out2 are the same as out3 and out4.
    >>> # providing a `dropout_rng` arg will take precedence over the `rngs` arg in `.apply`
    >>> out3, out4 = module.apply(variables, q, rngs={'dropout': key6}, dropout_rng=key5)

  Attributes:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    dtype: the dtype of the computation (default: infer from inputs and params)
    param_dtype: the dtype passed to parameter initializers (default: float32)
    qkv_features: dimension of the key, query, and value.
    out_features: dimension of the last projection
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rate: dropout rate
    deterministic: if false, the attention weight is masked randomly using
      dropout, whereas if true, the attention weights are deterministic.
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    kernel_init: initializer for the kernel of the Dense layers.
    bias_init: initializer for the bias of the Dense layers.
    use_bias: bool: whether pointwise QKVO dense transforms use bias.
    attention_fn: dot_product_attention or compatible function. Accepts query,
      key, value, and returns output of shape ``[bs, dim1, dim2, ..., dimN,,
      num_heads, value_channels]``
    decode: whether to prepare and use an autoregressive cache.
    normalize_qk: should QK normalization be applied (arxiv.org/abs/2302.05442).
  """


class SelfAttention(MultiHeadDotProductAttention):
  """Self-attention special case of multi-head dot-product attention.
  This layer is deprecated in favor of ``MultiHeadDotProductAttention``.

  Example usage::
    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp
    >>> layer = nn.MultiHeadDotProductAttention(num_heads=8, qkv_features=16)
    >>> variables = layer.init(jax.random.key(0), jnp.ones((4, 3, 2, 5)))
  """

  @compact
  def __call__(  # type: ignore
    self,
    inputs_q: Array,
    mask: Array | None = None,
    deterministic: bool | None = None,
    dropout_rng: PRNGKey | None = None,
    sow_weights: bool = False,
  ):
    """Applies multi-head dot product self-attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: input queries of shape ``[batch_sizes..., length, features]``.
      mask: attention mask of shape ``[batch_sizes..., num_heads, query_length,
        key/value_length]``. Attention weights are masked out if their
        corresponding mask value is ``False``.
      deterministic: if false, the attention weight is masked randomly using
        dropout, whereas if true, the attention weights are deterministic.

    Returns:
      output of shape ``[batch_sizes..., length, features]``.
    """
    warnings.warn(
      'SelfAttention will be deprecated soon. Use '
      '`MultiHeadDotProductAttention.__call__(inputs_q)` instead. '
      'See https://github.com/google/flax/discussions/3389 '
      'for more information.',
      DeprecationWarning,
    )
    return super().__call__(
      inputs_q,
      mask=mask,
      deterministic=deterministic,
      dropout_rng=dropout_rng,
      sow_weights=sow_weights,
    )


# mask-making utility functions


def make_attention_mask(
  query_input: Array,
  key_input: Array,
  pairwise_fn: Callable[..., Any] = jnp.multiply,
  extra_batch_dims: int = 0,
  dtype: Dtype = jnp.float32,
):
  """Mask-making helper for attention weights.

  In case of 1d inputs (i.e., ``[batch..., len_q]``, ``[batch..., len_kv]``, the
  attention weights will be ``[batch..., heads, len_q, len_kv]`` and this
  function will produce ``[batch..., 1, len_q, len_kv]``.

  Args:
    query_input: a batched, flat input of query_length size
    key_input: a batched, flat input of key_length size
    pairwise_fn: broadcasting elementwise comparison function
    extra_batch_dims: number of extra batch dims to add singleton axes for, none
      by default
    dtype: mask return dtype

  Returns:
    A ``[batch..., 1, len_q, len_kv]`` shaped mask for 1d attention.
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

  In case of 1d inputs (i.e., ``[batch..., len]``, the self-attention weights
  will be ``[batch..., heads, len, len]`` and this function will produce a
  causal mask of shape ``[batch..., 1, len, len]``.

  Args:
    x: input array of shape ``[batch..., len]``
    extra_batch_dims: number of batch dims to add singleton axes for, none by
      default
    dtype: mask return dtype

  Returns:
    A ``[batch..., 1, len, len]`` shaped causal mask for 1d attention.
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
