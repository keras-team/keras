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

import functools
from collections.abc import Iterable  # pylint: disable=g-importing-member
from typing import Any
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random

from flax import struct
from flax.core import Scope
from flax.linen import initializers

from .linear import default_kernel_init, dense_general


def dot_product_attention(
  scope,
  query,
  key,
  value,
  dtype=jnp.float32,
  bias=None,
  axis=None,
  broadcast_dropout=True,
  dropout_rng=None,
  dropout_rate=0.0,
  deterministic=False,
  precision=None,
):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights. This
  function supports multi-dimensional inputs.


  Args:
    query: queries for calculating attention with shape of `[batch_size, dim1,
      dim2, ..., dimN, num_heads, mem_channels]`.
    key: keys for calculating attention with shape of `[batch_size, dim1, dim2,
      ..., dimN, num_heads, mem_channels]`.
    value: values to be used in attention with shape of `[batch_size, dim1,
      dim2,..., dimN, num_heads, value_channels]`.
    dtype: the dtype of the computation (default: float32)
    bias: bias for the attention weights. This can be used for incorporating
      autoregressive mask, padding mask, proximity bias.
    axis: axises over which the attention is applied.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]`.
  """
  assert key.shape[:-1] == value.shape[:-1]
  assert query.shape[0:1] == key.shape[0:1] and query.shape[-1] == key.shape[-1]

  if axis is None:
    axis = tuple(range(1, key.ndim - 2))
  if not isinstance(axis, Iterable):
    axis = (axis,)
  assert key.ndim == query.ndim
  assert key.ndim == value.ndim
  for ax in axis:
    if not (query.ndim >= 3 and 1 <= ax < query.ndim - 2):
      raise ValueError(
        'Attention axis must be between the batch axis and the last-two axes.'
      )
  depth = query.shape[-1]
  n = key.ndim
  # batch_dims is  <bs, <non-attention dims>, num_heads>
  batch_dims = tuple(np.delete(range(n), axis + (n - 1,)))
  # q & k -> (bs, <non-attention dims>, num_heads, <attention dims>, channels)
  qk_perm = batch_dims + axis + (n - 1,)
  key = key.transpose(qk_perm)
  query = query.transpose(qk_perm)
  # v -> (bs, <non-attention dims>, num_heads, channels, <attention dims>)
  v_perm = batch_dims + (n - 1,) + axis
  value = value.transpose(v_perm)

  query = query / jnp.sqrt(depth).astype(dtype)
  batch_dims_t = tuple(range(len(batch_dims)))
  attn_weights = lax.dot_general(
    query,
    key,
    (((n - 1,), (n - 1,)), (batch_dims_t, batch_dims_t)),
    precision=precision,
  )

  # apply attention bias: masking, droput, proximity bias, ect.
  if bias is not None:
    attn_weights = attn_weights + bias

  # normalize the attention weights
  norm_dims = tuple(range(attn_weights.ndim - len(axis), attn_weights.ndim))
  attn_weights = lax.exp(
    attn_weights
    - jax.scipy.special.logsumexp(attn_weights, axis=norm_dims, keepdims=True)
  )
  attn_weights = attn_weights.astype(dtype)

  # apply dropout
  if not deterministic and dropout_rate > 0.0:
    if dropout_rng is None:
      dropout_rng = scope.make_rng('dropout')
    keep_prob = 1.0 - dropout_rate
    if broadcast_dropout:
      # dropout is broadcast across the batch+head+non-attention dimension
      dropout_dims = attn_weights.shape[-(2 * len(axis)) :]
      dropout_shape = tuple([1] * len(batch_dims_t)) + dropout_dims
      keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    else:
      keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
    multiplier = keep.astype(attn_weights.dtype) / jnp.asarray(
      keep_prob, dtype=dtype
    )
    attn_weights = attn_weights * multiplier

  # compute the new values given the attention weights
  wv_contracting_dims = (norm_dims, range(value.ndim - len(axis), value.ndim))
  y = lax.dot_general(
    attn_weights,
    value,
    (wv_contracting_dims, (batch_dims_t, batch_dims_t)),
    precision=precision,
  )

  # back to (bs, dim1, dim2, ..., dimN, num_heads, channels)
  perm_inv = _invert_perm(qk_perm)
  y = y.transpose(perm_inv)
  return y


def _invert_perm(perm):
  perm_inv = [0] * len(perm)
  for i, j in enumerate(perm):
    perm_inv[j] = i
  return tuple(perm_inv)


class CacheEntry(struct.PyTreeNode):
  key: np.ndarray
  value: np.ndarray
  i: np.ndarray


def multi_head_dot_product_attention(
  scope: Scope,
  inputs_q,
  inputs_kv,
  num_heads,
  dtype=jnp.float32,
  qkv_features=None,
  out_features=None,
  attention_axis=None,
  causal_mask=False,
  padding_mask=None,
  key_padding_mask=None,
  segmentation=None,
  key_segmentation=None,
  cache=False,
  broadcast_dropout=True,
  dropout_rng=None,
  dropout_rate=0.0,
  deterministic=False,
  precision=None,
  kernel_init=default_kernel_init,
  bias_init=initializers.zeros_init(),
  bias=True,
  attention_fn=dot_product_attention,
):
  """Applies multi-head dot product attention on the input data.

  Projects the inputs into multi-headed query, key, and value vectors,
  applies dot-product attention and project the results to an output vector.

  This can be used for encoder-decoder attention by specifying both `inputs_q`
  and `inputs_kv` orfor self-attention by only specifying `inputs_q` and
  setting `inputs_kv` to None.

  Args:
    inputs_q: input queries of shape `[bs, dim1, dim2, ..., dimN, features]`.
    inputs_kv: key/values of shape `[bs, dim1, dim2, ..., dimN, features]`
      or None for self-attention, inn which case key/values will be derived
      from inputs_q.
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    dtype: the dtype of the computation (default: float32)
    qkv_features: dimension of the key, query, and value.
    out_features: dimension of the last projection
    attention_axis: axes over which the attention is applied ( 'None' means
      attention over all axes, but batch, heads, and features).
    causal_mask: boolean specifying whether to apply a causal mask on the
      attention weights. If True, the output at timestep `t` will not depend
      on inputs at timesteps strictly greater than `t`.
    padding_mask: boolean specifying query tokens that are pad token.
    key_padding_mask: boolean specifying key-value tokens that are pad token.
    segmentation: segment indices for packed inputs_q data.
    key_segmentation: segment indices for packed inputs_kv data.
    cache: an instance of `flax.deprecated.nn.attention.Cache` used for
      efficient autoregressive decoding.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the kernel of the Dense layers.
    bias_init: initializer for the bias of the Dense layers.
    bias: bool: whether pointwise QKVO dense transforms use bias.
    attention_fn: dot_product_attention or compatible function. Accepts
    query, key, value, and returns output of shape
    `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``

  Returns:
    output of shape `[bs, dim1, dim2, ..., dimN, features]`.
  """

  assert (
    causal_mask or not cache
  ), 'Caching is only support for causal attention.'

  if inputs_kv is None:
    inputs_kv = inputs_q

  if attention_axis is None:
    attention_axis = tuple(range(1, inputs_q.ndim - 1))

  features = out_features or inputs_q.shape[-1]
  qkv_features = qkv_features or inputs_q.shape[-1]

  assert (
    qkv_features % num_heads == 0
  ), 'Memory dimension must be divisible by number of heads.'
  head_dim = qkv_features // num_heads

  dense = functools.partial(
    dense_general,
    axis=-1,
    dtype=dtype,
    features=(num_heads, head_dim),
    kernel_init=kernel_init,
    bias_init=bias_init,
    bias=bias,
    precision=precision,
  )
  # project inputs_q to multi-headed q/k/v
  # dimensions are then [bs, dims..., n_heads, n_features_per_head]
  query = scope.child(dense, 'query')(inputs_q)
  key = scope.child(dense, 'key')(inputs_kv)
  value = scope.child(dense, 'value')(inputs_kv)

  if cache:
    cache_entry: Callable[[Any], CacheEntry] | CacheEntry
    if not scope.has_variable('cache', 'entry'):
      ndim, tail_shape = (key.ndim, key.shape[-2:])

      def init_fn(shape, dtype=jnp.float32):
        full_shape = shape + tail_shape
        if len(full_shape) != ndim:
          raise ValueError(
            'Shape should be a tuple with the shape of the batch'
            'and attention dims.'
          )
        return CacheEntry(
          key=jnp.zeros(full_shape, dtype),
          value=jnp.zeros(full_shape, dtype),
          i=jnp.zeros((), jnp.uint32),
        )

      cache_entry = init_fn
    else:
      cache_entry = scope.get_variable('cache', 'entry')
      if not isinstance(cache_entry, CacheEntry):
        raise ValueError('Cache is not initialized.')

      expected_shape = list(cache_entry.key.shape[:-2])
      for attn_dim in attention_axis:
        expected_shape[attn_dim] = 1
      expected_shape = tuple(expected_shape) + inputs_q.shape[-1:]
      if expected_shape != inputs_q.shape:
        raise ValueError(
          'Invalid shape provided, expected shape %s instead got %s.'
          % (expected_shape, inputs_q.shape)
        )

      cshape = cache_entry.key.shape
      indices = [0] * len(cshape)
      i = cache_entry.i
      attn_size = np.prod(np.take(cshape, attention_axis))
      for attn_dim in attention_axis:
        attn_size //= cshape[attn_dim]
        indices[attn_dim] = i // attn_size
        i = i % attn_size

      key = lax.dynamic_update_slice(cache_entry.key, key, indices)  # type: ignore
      value = lax.dynamic_update_slice(cache_entry.value, value, indices)  # type: ignore
      one = jnp.array(1, jnp.uint32)
      cache_entry = cache_entry.replace(
        i=cache_entry.i + one, key=key, value=value
      )

      # TODO(levskaya): verify this is still needed in translation decoding.
      key_padding_mask = jnp.broadcast_to(
        (jnp.arange(cshape[1]) < cache_entry.i), cshape[:2]
      )
      key_padding_mask = key_padding_mask.astype(jnp.float32)[..., None]
    scope.put_variable('cache', 'entry', cache_entry)

  # create attention masks
  mask_components = []

  if causal_mask:
    if cache and isinstance(cache_entry, CacheEntry):
      bias_pre_shape = (1,) * (key.ndim - 1)
      attn_shape = tuple(np.take(key.shape, attention_axis))
      attn_size = np.prod(attn_shape)
      ii = jnp.arange(attn_size, dtype=jnp.uint32)
      mask = ii < cache_entry.i
      mask_components.append(mask.reshape(bias_pre_shape + attn_shape))
    else:
      mask_components.append(_make_causal_mask(key, attention_axis))

  if padding_mask is not None:
    if key_padding_mask is None:
      key_padding_mask = padding_mask
    padding_mask = make_padding_mask(
      padding_mask_query=padding_mask,
      padding_mask_key=key_padding_mask,
      query_shape=query.shape,
      key_shape=key.shape,
      attention_axis=attention_axis,
    )
    mask_components.append(padding_mask)

  if segmentation is not None:
    if key_segmentation is None:
      key_segmentation = segmentation
    segmentation_mask = make_padding_mask(
      padding_mask_query=segmentation,
      padding_mask_key=key_segmentation,
      query_shape=query.shape,
      key_shape=key.shape,
      attention_axis=attention_axis,
      segmentation_mask=True,
    )
    mask_components.append(segmentation_mask)

  if mask_components:
    attention_mask = mask_components[0]
    for component in mask_components[1:]:
      attention_mask = jnp.logical_and(attention_mask, component)

    # attention mask in the form of attention bias
    attention_bias = lax.select(
      attention_mask > 0,
      jnp.full(attention_mask.shape, 0.0).astype(dtype),
      jnp.full(attention_mask.shape, -1e10).astype(dtype),
    )
  else:
    attention_bias = None

  # apply attention
  x = scope.child(attention_fn)(
    query,
    key,
    value,
    dtype=dtype,
    axis=attention_axis,
    bias=attention_bias,
    precision=precision,
    dropout_rng=dropout_rng,
    dropout_rate=dropout_rate,
    broadcast_dropout=broadcast_dropout,
    deterministic=deterministic,
  )

  # back to the original inputs dimensions
  out = scope.child(dense_general, name='out')(
    x,
    features=features,
    axis=(-2, -1),
    kernel_init=kernel_init,
    bias_init=bias_init,
    bias=bias,
    dtype=dtype,
    precision=precision,
  )

  return out


# TODO(flax-dev): Consider refactoring MultiHeadDotProductAttention and moving
# causal_mask and cache support into this class instead.
# SelfAttention = MultiHeadDotProductAttention.partial(inputs_kv=None)


def make_padding_mask(
  padding_mask_query,
  padding_mask_key,
  query_shape,
  key_shape,
  attention_axis=None,
  segmentation_mask=False,
):
  """Makes padding mask for attention weights.

  In case of 1d inputs (i.e., `[bs, len, features]`, the attention weights will
  be `[bs, len, len]` and this function makes a square matrix [len, len].

  Args:
    padding_mask_query: padding mask of query <bs, qdim1,.., qdimn>
    padding_mask_key: padding mask of query <bs, key1,.., keyn>
    query_shape: shape of the query
    key_shape: shape of the key, which is equal to the shape of value.
    attention_axis: axis over which attention is applied.
    segmentation_mask: bool: if true use equality on cartesian product rather
      than outer product for constructing segmentation masks.
  Returns:
    The padding mask for attention weights.
  """
  assert query_shape[0] == key_shape[0]
  assert len(query_shape) == len(key_shape)

  ndim = len(key_shape)
  if attention_axis is None:
    attention_axis = tuple(range(1, ndim - 2))
  assert isinstance(attention_axis, tuple)
  for ax in attention_axis:
    if not (ndim >= 3 and 1 <= ax < ndim - 2):
      raise ValueError(
        'Attention axis must be between the batch axis and the last-two axes.'
      )

  mask_shape_final = (query_shape[0], 1)  #  batch_size, 1 (for all heads)s
  for ax in attention_axis:
    mask_shape_final += (query_shape[ax],)
  for ax in attention_axis:
    mask_shape_final += (key_shape[ax],)

  padding_mask_query = padding_mask_query[..., None]
  padding_mask_key = padding_mask_key[..., None]
  perm = (0,) + tuple(np.flip(np.arange(padding_mask_key.ndim)))[:-1]
  if segmentation_mask:
    mask = jnp.equal(padding_mask_query, padding_mask_key.transpose(perm))
  else:
    mask = jnp.multiply(padding_mask_query, padding_mask_key.transpose(perm))

  mask = mask.reshape(mask_shape_final)
  mask = jax.lax.convert_element_type(mask, jnp.float32)
  return mask


def _make_causal_mask(key, attention_axis=None, self_mask=False):
  """Makes a causal mask, to be used for masking out the future for attention.

  In case of 1d inputs (i.e., `[bs, len, features]`, the attention weights will
  be `[bs, len, len]` and this function makes a square matrix [len, len] with
  zeros in upper triangle and ones in lower triangle.

  Args:
    key: shape of the key, which is equal to the shape of value and is
      assumed to be equal to the shape of the query (since this is used in
      self-attention when decoding).
    attention_axis: axis over which attention is applied.
    self_mask: if mask out the diagonal or not.

  Returns:
    A causal mask to be used to mask out future positions.
  """
  if attention_axis is None:
    attention_axis = tuple(range(1, key.ndim - 2))
  assert isinstance(attention_axis, tuple)
  for ax in attention_axis:
    if not (key.ndim >= 3 and 1 <= ax < key.ndim - 2):
      raise ValueError(
        'Attention axis must be between the batch axis and the last-two axes.'
      )

  mask_shape = tuple([1] * (key.ndim - len(attention_axis) - 1))
  mask_shape_final = mask_shape
  for _ in range(2):
    flatten_dim = 1
    for ax in attention_axis:
      mask_shape_final += (key.shape[ax],)
      flatten_dim *= key.shape[ax]
    mask_shape += (flatten_dim,)

  def tri(n, m, k=0):
    # Tie in the key to avoid the mask becoming a constant.
    # This way XLA can construct the mask during computation and fuse it
    # with the attention ops.
    x = jnp.arange(n, dtype=jnp.int32)
    y = jnp.arange(m, dtype=jnp.int32)
    mask = lax.ge(
      (lax.broadcast_in_dim(x, shape=(n, m), broadcast_dimensions=(0,))) + k,
      lax.broadcast(y, [n]),
    )
    return mask

  k = -1 if self_mask else 0
  mask = tri(*mask_shape[-2:], k=k).reshape(mask_shape_final)
  return mask
