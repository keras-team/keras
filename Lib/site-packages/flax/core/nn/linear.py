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

"""Linear modules."""

from collections.abc import Iterable  # pylint: disable=g-importing-member

import jax.numpy as jnp
import numpy as np
from jax import lax

from flax import struct
from flax.core import Scope
from flax.linen import initializers

default_kernel_init = initializers.lecun_normal()


def _normalize_axes(axes, ndim):
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def dense_general(
  scope,
  inputs,
  features,
  axis=-1,
  batch_dims=(),
  bias=True,
  dtype=jnp.float32,
  kernel_init=default_kernel_init,
  bias_init=initializers.zeros_init(),
  precision=None,
):
  """Applies a linear transformation to the inputs along multiple dimensions.

  Args:
    inputs: The nd-array to be transformed.
    features: tuple with numbers of output features.
    axis: tuple with axes to apply the transformation on.
    batch_dims: tuple with batch axes.
    bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
  Returns:
    The transformed input.
  """
  inputs = jnp.asarray(inputs, dtype)

  if not isinstance(features, Iterable):
    features = (features,)
  if not isinstance(axis, Iterable):
    axis = (axis,)
  if not isinstance(batch_dims, Iterable):
    batch_dims = (batch_dims,)
  features, axis, batch_dims = tuple(features), tuple(axis), tuple(batch_dims)

  if batch_dims:
    max_dim = np.max(batch_dims)
    if set(batch_dims) != set(range(max_dim + 1)):
      raise ValueError(
        'batch_dims %s must be consecutive leading '
        'dimensions starting from 0.' % str(batch_dims)
      )

  ndim = inputs.ndim
  n_batch_dims = len(batch_dims)
  axis = _normalize_axes(axis, ndim)
  batch_dims = _normalize_axes(batch_dims, ndim)
  n_axis, n_features = len(axis), len(features)

  def kernel_init_wrap(rng, shape, dtype=jnp.float32):
    size_batch_dims = np.prod(shape[:n_batch_dims], dtype=np.int32)
    flat_shape = (
      np.prod(shape[n_batch_dims : n_axis + n_batch_dims]),
      np.prod(shape[-n_features:]),
    )
    kernel = jnp.concatenate(
      [kernel_init(rng, flat_shape, dtype) for _ in range(size_batch_dims)],
      axis=0,
    )
    return jnp.reshape(kernel, shape)

  batch_shape = tuple(inputs.shape[ax] for ax in batch_dims)
  kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
  kernel = scope.param('kernel', kernel_init_wrap, batch_shape + kernel_shape)
  kernel = jnp.asarray(kernel, dtype)

  batch_ind = tuple(range(n_batch_dims))
  contract_ind = tuple(range(n_batch_dims, n_axis + n_batch_dims))
  out = lax.dot_general(
    inputs,
    kernel,
    ((axis, contract_ind), (batch_dims, batch_ind)),
    precision=precision,
  )
  if bias:

    def bias_init_wrap(rng, shape, dtype=jnp.float32):
      size_batch_dims = np.prod(shape[:n_batch_dims], dtype=np.int32)
      flat_shape = (np.prod(shape[-n_features:]),)
      bias = jnp.concatenate(
        [bias_init(rng, flat_shape, dtype) for _ in range(size_batch_dims)],
        axis=0,
      )
      return jnp.reshape(bias, shape)

    bias = scope.param('bias', bias_init_wrap, batch_shape + features)

    # Reshape bias for broadcast.
    expand_dims = sorted(set(range(inputs.ndim)) - set(axis) - set(batch_dims))
    for ax in expand_dims:
      bias = jnp.expand_dims(bias, ax)
    bias = jnp.asarray(bias, dtype)
    out = out + bias
  return out


def dense(
  scope,
  inputs,
  features,
  bias=True,
  dtype=jnp.float32,
  precision=None,
  kernel_init=default_kernel_init,
  bias_init=initializers.zeros_init(),
):
  """Applies a linear transformation to the inputs along the last dimension.

  Args:
    inputs: The nd-array to be transformed.
    features: the number of output features.
    bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  Returns:
    The transformed input.
  """
  inputs = jnp.asarray(inputs, dtype)
  kernel = scope.param('kernel', kernel_init, (inputs.shape[-1], features))
  kernel = jnp.asarray(kernel, dtype)
  y = lax.dot_general(
    inputs,
    kernel,
    (((inputs.ndim - 1,), (0,)), ((), ())),
    precision=precision,
  )
  if bias:
    bias = scope.param('bias', bias_init, (features,))
    bias = jnp.asarray(bias, dtype)
    y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
  return y


def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


def conv(
  scope,
  inputs,
  features,
  kernel_size,
  strides=None,
  padding='SAME',
  input_dilation=None,
  kernel_dilation=None,
  feature_group_count=1,
  bias=True,
  dtype=jnp.float32,
  precision=None,
  kernel_init=default_kernel_init,
  bias_init=initializers.zeros_init(),
):
  """Applies a convolution to the inputs.

  Args:
    inputs: input data with dimensions (batch, spatial_dims..., features).
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel.
    strides: a sequence of `n` integers, representing the inter-window
      strides.
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
      of `n` `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension.
    input_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `inputs`.
      Convolution with input dilation `d` is equivalent to transposed
      convolution with stride `d`.
    kernel_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel. Convolution with kernel dilation is also known as 'atrous
      convolution'.
    feature_group_count: integer, default 1. If specified divides the input
      features into groups.
    bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
  Returns:
    The convolved data.
  """

  inputs = jnp.asarray(inputs, dtype)

  if strides is None:
    strides = (1,) * (inputs.ndim - 2)

  in_features = inputs.shape[-1]
  assert in_features % feature_group_count == 0
  kernel_shape = kernel_size + (in_features // feature_group_count, features)
  kernel = scope.param('kernel', kernel_init, kernel_shape)
  kernel = jnp.asarray(kernel, dtype)

  dimension_numbers = _conv_dimension_numbers(inputs.shape)
  y = lax.conv_general_dilated(
    inputs,
    kernel,
    strides,
    padding,
    lhs_dilation=input_dilation,
    rhs_dilation=kernel_dilation,
    dimension_numbers=dimension_numbers,
    feature_group_count=feature_group_count,
    precision=precision,
  )

  if bias:
    bias = scope.param('bias', bias_init, (features,))
    bias = jnp.asarray(bias, dtype)
    y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
  return y


def conv_transpose(
  scope,
  inputs,
  features,
  kernel_size,
  strides=None,
  padding='SAME',
  kernel_dilation=None,
  bias=True,
  dtype=jnp.float32,
  precision=None,
  kernel_init=default_kernel_init,
  bias_init=initializers.zeros_init(),
):
  """Applies a transposed convolution to the inputs. Behaviour mirrors that of
  `jax.lax.conv_transpose`.

  Args:
    scope: functional scope.
    inputs: input data with dimensions (batch, spatial_dims..., features).
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel.
    strides: a sequence of `n` integers, representing the inter-window
      strides.
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
      of `n` `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension.
    kernel_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel. Convolution with kernel dilation is also known as 'atrous
      convolution'.
    bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
  Returns:
    The convolved data.
  """
  inputs = jnp.asarray(inputs, dtype)
  strides = strides or (1,) * (inputs.ndim - 2)

  in_features = inputs.shape[-1]
  kernel_shape = kernel_size + (in_features, features)
  kernel = scope.param('kernel', kernel_init, kernel_shape)
  kernel = jnp.asarray(kernel, dtype)

  y = lax.conv_transpose(
    inputs,
    kernel,
    strides,
    padding,
    rhs_dilation=kernel_dilation,
    precision=precision,
  )

  if bias:
    bias = scope.param('bias', bias_init, (features,))
    bias = jnp.asarray(bias, dtype)
    y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
  return y


default_embed_init = initializers.variance_scaling(
  1.0, 'fan_in', 'normal', out_axis=0
)


@struct.dataclass
class Embedding:
  table: np.ndarray

  def lookup(self, indices):
    """Embeds the inputs along the last dimension.

    Args:
      indices: input data, all dimensions are considered batch dimensions.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `features` dimension appended.
    """
    if indices.dtype not in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]:
      raise ValueError('Input type must be an integer or unsigned integer.')
    return self.table[indices]

  def attend(self, query):
    """Attend over the embedding using a query array.

    Args:
      query: array with last dimension equal the feature depth `features` of the
        embedding.

    Returns:
      An array with final dim `num_embeddings` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    return jnp.dot(query, self.table.T)


def embedding(
  scope: Scope, num_embeddings: int, features: int, init_fn=default_embed_init
) -> Embedding:
  """Creates embedding dataclass.

  Args:
    num_embeddings: number of embeddings.
    features: Number of feature dimensions for each embedding.
    embedding_init: embedding initializer.

  Returns:
    Embedding dataclass with lookup and attend methods.
  """
  table = scope.param('table', init_fn, (num_embeddings, features))
  return Embedding(table)  # type: ignore
