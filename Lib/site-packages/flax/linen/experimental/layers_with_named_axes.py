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

"""Experimental layers with named axes for the partitioning API."""
import dataclasses
from typing import Any
from collections.abc import Callable, Iterable, Sequence

import jax.numpy as jnp
from jax import lax

from flax import linen as nn
from flax.linen import initializers
from flax.linen.partitioning import param_with_axes, with_sharding_constraint
from flax.typing import (
  Array,
  Dtype,
  Axes,
  Initializer,
  PrecisionLike,
  DotGeneralT,
)

# Type annotations
Activation = Callable[..., Array]


default_kernel_init = initializers.lecun_normal()
default_embed_init = initializers.variance_scaling(
  1.0, 'fan_in', 'normal', out_axis=0
)


class Dense(nn.Module):
  """A Dense layer with named axes for :meth:`jax.experimental.pjit.pjit`.

  .. warning:: This class is hightly EXPERIMENTAL and the API is likely to
      change. For regular (non-pjit) use, please use
      :class:`flax.linen.linear.Dense`.

  Attributes:
    features: the number of output features.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """

  features: int
  use_bias: bool = True
  dtype: Dtype = jnp.float32
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Initializer = default_kernel_init
  bias_init: Initializer = initializers.zeros_init()
  kernel_axes: tuple[str, ...] = ()
  # Deprecated. Will be removed.
  dot_general: DotGeneralT | None = None
  dot_general_cls: Any = None

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    kernel = param_with_axes(
      'kernel',
      self.kernel_init,
      (inputs.shape[-1], self.features),
      self.param_dtype,
      axes=self.kernel_axes,
    )
    kernel = jnp.asarray(kernel, self.dtype)

    if self.dot_general_cls is not None:
      dot_general = self.dot_general_cls()
    elif self.dot_general is not None:
      dot_general = self.dot_general
    else:
      dot_general = lax.dot_general
    y = dot_general(
      inputs,
      kernel,
      (((inputs.ndim - 1,), (0,)), ((), ())),
      precision=self.precision,
    )
    if self.use_bias:
      bias = param_with_axes(
        'bias',
        self.bias_init,
        (self.features,),
        self.param_dtype,
        axes=(self.kernel_axes[-1],),
      )
      bias = jnp.asarray(bias, self.dtype)
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y


class Embed(nn.Module):
  """An embedding layer with named axes for :meth:`jax.experimental.pjit.pjit`.

  .. warning:: This class is hightly EXPERIMENTAL and the API is likely to
      change. For regular (non-pjit) use, please use
      :class:`flax.linen.linear.Embed`.

  Attributes:
    num_embeddings: number of embeddings.
    features: number of feature dimensions for each embedding.
    dtype: the dtype of the embedding vectors (default: float32).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    embedding_init: embedding initializer.
    one_hot: performs the gather with a one-hot contraction rather than a true
      gather. This is currently needed for SPMD partitioning.
  """

  num_embeddings: int
  features: int
  cast_input_dtype: Dtype | None = None
  dtype: Dtype = jnp.float32
  param_dtype: Dtype = jnp.float32
  attend_dtype: Dtype | None = None
  embedding_init: Initializer = default_embed_init
  one_hot: bool = False
  embedding: Array = dataclasses.field(init=False)

  def setup(self):
    self.embedding = param_with_axes(
      'embedding',
      self.embedding_init,
      (self.num_embeddings, self.features),
      self.param_dtype,
      axes=('vocab', 'embed'),
    )

  def __call__(self, inputs: Array) -> Array:
    """Embeds the inputs along the last dimension.

    Args:
      inputs: input data, all dimensions are considered batch dimensions.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `features` dimension appended.
    """
    if self.cast_input_dtype:
      inputs = inputs.astype(self.cast_input_dtype)
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError('Input type must be an integer or unsigned integer.')
    if self.one_hot:
      iota = lax.iota(jnp.int32, self.num_embeddings)
      one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
      output = jnp.dot(one_hot, jnp.asarray(self.embedding, self.dtype))
    else:
      output = jnp.asarray(self.embedding, self.dtype)[inputs]
      output = with_sharding_constraint(output, ('batch', 'length', 'embed'))
    return output

  def attend(self, query: Array) -> Array:
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
    dtype = self.attend_dtype if self.attend_dtype is not None else self.dtype
    return jnp.dot(query, jnp.asarray(self.embedding, dtype).T)


def _canonicalize_axes(rank: int, axes: Axes) -> Sequence[int]:
  """Returns a tuple of deduplicated, sorted, and positive axes."""
  if not isinstance(axes, Iterable):
    axes = (axes,)
  return tuple({rank + axis if axis < 0 else axis for axis in axes})


def _abs_sq(x):
  """Computes the elementwise square of the absolute value |x|^2."""
  if jnp.iscomplexobj(x):
    return lax.square(lax.real(x)) + lax.square(lax.imag(x))
  else:
    return lax.square(x)


def _compute_stats(x: Array, axes: Axes):
  """Computes mean and variance statistics.

  This implementation takes care of a few important details:
  - Computes in float32 precision for half precision inputs
  -  mean and variance is computable in a single XLA fusion,
    by using Var = E[|x|^2] - |E[x]|^2 instead of Var = E[|x - E[x]|^2]).
  - Clips negative variances to zero which can happen due to
    roundoff errors. This avoids downstream NaNs.
  - Supports averaging across a parallel axis and subgroups of a parallel axis
    with a single `lax.pmean` call to avoid latency.
  """
  # promote x to at least float32, this avoids half precision computation
  # but preserves double or complex floating points
  x = jnp.asarray(x, jnp.promote_types(jnp.float32, jnp.result_type(x)))
  mean = jnp.mean(x, axes)
  mean2 = jnp.mean(_abs_sq(x), axes)
  # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
  # to floating point round-off errors.
  var = jnp.maximum(0.0, mean2 - _abs_sq(mean))
  return mean, var


def _normalize(
  mdl: nn.Module,
  x: Array,
  mean: Array,
  var: Array,
  reduction_axes: Axes,
  feature_axes: Axes,
  dtype: Dtype,
  param_dtype: Dtype,
  epsilon: float,
  use_bias: bool,
  use_scale: bool,
  bias_init: Initializer,
  scale_init: Initializer,
):
  """ "Normalizes the input of a normalization layer and optionally applies a learned scale and bias.

  A seperate bias and scale is learned for each feature as specified by
  feature_axes.
  """
  reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
  feature_axes = _canonicalize_axes(x.ndim, feature_axes)
  stats_shape = list(x.shape)
  for axis in reduction_axes:
    stats_shape[axis] = 1
  mean = mean.reshape(stats_shape)
  var = var.reshape(stats_shape)
  feature_shape = [1] * x.ndim
  reduced_feature_shape = []
  for ax in feature_axes:
    feature_shape[ax] = x.shape[ax]
    reduced_feature_shape.append(x.shape[ax])
  y = x - mean
  mul = lax.rsqrt(var + epsilon)
  if use_scale:
    scale = mdl.param_with_axes(
      'scale', scale_init, reduced_feature_shape, param_dtype, axes=('embed',)
    ).reshape(feature_shape)
    mul *= scale
  y *= mul
  if use_bias:
    bias = mdl.param_with_axes(
      'bias', bias_init, reduced_feature_shape, param_dtype, axes=('embed',)
    ).reshape(feature_shape)
    y += bias
  return jnp.asarray(y, dtype)


class LayerNorm(nn.Module):
  """Layer normalization (https://arxiv.org/abs/1607.06450) with named axes for :meth:`jax.experimental.pjit.pjit`.

  .. warning:: This class is hightly EXPERIMENTAL and the API is likely to
      change. For regular (non-pjit) use, please use
      :class:`flax.linen.normalization.LayerNorm`.

  Operates on the last axis of the input data.

  It normalizes the activations of the layer for each given example in a
  batch independently, rather than across a batch like Batch Normalization.
  i.e. applies a transformation that maintains the mean activation within
  each example close to 0 and the activation standard deviation close to 1.

  Attributes:
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  If True, bias (beta) is added.
    use_scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
  """

  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  param_dtype: Dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Initializer = initializers.zeros_init()
  scale_init: Initializer = initializers.ones_init()

  @nn.compact
  def __call__(self, x):
    """Applies layer normalization on the input.

    Args:
      x: the inputs

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    reduction_axes = (-1,)
    feature_axes = (-1,)

    mean, var = _compute_stats(x, reduction_axes)

    return _normalize(
      self,
      x,
      mean,
      var,
      reduction_axes,
      feature_axes,
      self.dtype,
      self.param_dtype,
      self.epsilon,
      self.use_bias,
      self.use_scale,
      self.bias_init,
      self.scale_init,
    )
