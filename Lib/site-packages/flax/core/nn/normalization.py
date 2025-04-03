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

"""Normalization modules for Flax."""

import jax.numpy as jnp
from jax import lax

from flax.core import Scope
from flax.linen import initializers


def _absolute_dims(ndim, dims):
  return tuple(ndim + dim if dim < 0 else dim for dim in dims)


def batch_norm(
  scope: Scope,
  x,
  use_running_average=False,
  axis=-1,
  momentum=0.99,
  epsilon=1e-5,
  dtype=jnp.float32,
  bias=True,
  scale=True,
  bias_init=initializers.zeros_init(),
  scale_init=initializers.ones_init(),
  axis_name=None,
  axis_index_groups=None,
  kind='batch_stats',
):
  x = jnp.asarray(x, jnp.float32)
  axis = axis if isinstance(axis, tuple) else (axis,)
  axis = _absolute_dims(x.ndim, axis)
  redux = tuple(i for i in range(x.ndim) if i not in axis)

  def pmean(x):
    m = jnp.mean(x, redux, keepdims=True)
    if axis_name is not None:
      m = lax.pmean(m, axis_name=axis_name, axis_index_groups=axis_index_groups)
    return m

  mean = pmean(x)
  squeeze_shape = jnp.squeeze(mean).shape
  mean2 = pmean(jnp.square(x))
  var = mean2 - jnp.square(mean)

  is_init = not scope.has_variable(kind, 'mean')
  ra_mean = scope.variable(kind, 'mean', jnp.zeros, squeeze_shape)
  ra_var = scope.variable(kind, 'var', jnp.ones, squeeze_shape)

  if use_running_average:
    # if ra_mean is not None:
    #   raise ValueError('batch_stats should be provided if use_running_averages=True')
    mean = jnp.reshape(ra_mean.value, mean.shape)
    var = jnp.reshape(ra_var.value, var.shape)
  else:
    if not is_init:
      beta = 1.0 - momentum
      ra_mean.value += beta * (jnp.squeeze(mean) - ra_mean.value)
      ra_var.value += beta * (jnp.squeeze(var) - ra_var.value)
  y = x - mean
  mul = lax.rsqrt(var + epsilon)
  if scale:
    mul = mul * scope.param('scale', scale_init, squeeze_shape).reshape(
      mean.shape
    )
  y = y * mul
  if bias:
    y = y + scope.param('bias', bias_init, squeeze_shape).reshape(mean.shape)
  return jnp.asarray(y, dtype)


def layer_norm(
  scope: Scope,
  x,
  epsilon=1e-6,
  dtype=jnp.float32,
  bias=True,
  scale=True,
  bias_init=initializers.zeros_init(),
  scale_init=initializers.ones_init(),
):
  """Applies layer normalization on the input.
  It normalizes the activations of the layer for each given example in a
  batch independently, rather than across a batch like Batch Normalization.
  i.e. applies a transformation that maintains the mean activation within
  each example close to 0 and the activation standard deviation close to 1.
  Args:
    x: the inputs
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    bias:  If True, bias (beta) is added.
    scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
  Returns:
    Normalized inputs (the same shape as inputs).
  """
  features = x.shape[-1]
  mean = jnp.mean(x, axis=-1, keepdims=True)
  mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
  var = mean2 - lax.square(mean)
  mul = lax.rsqrt(var + epsilon)
  if scale:
    mul = mul * jnp.asarray(
      scope.param('scale', scale_init, (features,)), dtype
    )
  y = (x - mean) * mul
  if bias:
    y = y + jnp.asarray(scope.param('bias', bias_init, (features,)), dtype)
  return y


def group_norm(
  scope,
  x,
  num_groups=32,
  group_size=None,
  epsilon=1e-6,
  dtype=jnp.float32,
  bias=True,
  scale=True,
  bias_init=initializers.zeros_init(),
  scale_init=initializers.ones_init(),
):
  """Applies group normalization to the input (arxiv.org/abs/1803.08494).
  This op is similar to batch normalization, but statistics are shared across
  equally-sized groups of channels and not shared across batch dimension.
  Thus, group normalization does not depend on the batch composition and does
  not require maintaining internal state for storing statistics.
  The user should either specify the total number of channel groups or the
  number of channels per group.
  Args:
    x: the input of shape N...C, where N is a batch dimension and C is a
      channels dimensions. `...` represents an arbitrary number of extra
      dimensions that are used to accumulate statistics over.
    num_groups: the total number of channel groups. The default value of 32 is
      proposed by the original group normalization paper.
    group_size: the number of channels in a group.
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    bias:  If True, bias (beta) is added.
    scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
  Returns:
    Normalized inputs (the same shape as inputs).
  """
  x = jnp.asarray(x, jnp.float32)
  if (num_groups is None and group_size is None) or (
    num_groups is not None and group_size is not None
  ):
    raise ValueError(
      'Either `num_groups` or `group_size` should be '
      'specified, but not both of them.'
    )

  if group_size is not None:
    channels = x.shape[-1]
    if channels % group_size != 0:
      raise ValueError(
        'Number of channels ({}) is not multiple of the '
        'group size ({}).'.format(channels, group_size)
      )
    num_groups = channels // group_size

  input_shape = x.shape
  group_shape = x.shape[:-1] + (num_groups, x.shape[-1] // num_groups)

  x = x.reshape(group_shape)

  reduction_axis = list(range(1, x.ndim - 2)) + [x.ndim - 1]

  mean = jnp.mean(x, axis=reduction_axis, keepdims=True)
  mean_of_squares = jnp.mean(jnp.square(x), axis=reduction_axis, keepdims=True)
  var = mean_of_squares - jnp.square(mean)

  x = (x - mean) * lax.rsqrt(var + epsilon)

  x = x.reshape(input_shape)

  feature_shape = tuple([1 for d in input_shape[:-1]] + [input_shape[-1]])
  if scale:
    x = x * scope.param('scale', scale_init, feature_shape)
  if bias:
    x = x + scope.param('bias', bias_init, feature_shape)

  return x.astype(dtype)
