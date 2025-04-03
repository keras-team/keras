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

import dataclasses
import functools
from typing import Any
from collections.abc import Iterable

import jax
import jax.numpy as jnp
from jax import lax
from jax.nn import initializers

from flax.linen import dtypes, module, transforms
from flax.typing import (
  Array,
  PRNGKey as PRNGKey,
  Dtype,
  Shape as Shape,
  Initializer,
  Axes,
)

field = dataclasses.field
canonicalize_dtype = dtypes.canonicalize_dtype
compact = module.compact
Module = module.Module
merge_param = module.merge_param
map_variables = transforms.map_variables


def _canonicalize_axes(rank: int, axes: Axes) -> tuple[int, ...]:
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


def _compute_stats(
    x: Array,
    axes: Axes,
    dtype: Dtype | None,
    axis_name: str | None = None,
    axis_index_groups: Any = None,
    use_mean: bool = True,
    use_fast_variance: bool = True,
    mask: Array | None = None,
    force_float32_reductions=True,
):
  """Computes mean and variance statistics.

  This implementation takes care of a few important details:
  - By default, computes in float32 precision for stability
    in half precision training.
  - If `use_fast_variance` is `True`, mean and variance are computed using
    Var = E[|x|^2] - |E[x]|^2, instead of Var = E[|x - E[x]|^2]), in a single
    XLA fusion.
  - Clips negative variances to zero which can happen due to
    roundoff errors. This avoids downstream NaNs.
  - Supports averaging across a parallel axis and subgroups of a parallel axis
    with a single `lax.pmean` call to avoid latency.

  Arguments:
    x: Input array.
    axes: The axes in ``x`` to compute mean and variance statistics for.
    dtype: Optional dtype specifying the minimal precision. Statistics are
      always at least float32 for stability (default: dtype of x).
    axis_name: Optional name for the pmapped axis to compute mean over. Note,
      this is only used for pmap and shard map. For SPMD jit, you do not need to
      manually synchronize. Just make sure that the axes are correctly annotated
      and XLA:SPMD will insert the necessary collectives.
    axis_index_groups: Optional axis indices.
    use_mean: If true, calculate the mean from the input and use it when
      computing the variance. If false, set the mean to zero and compute the
      variance without subtracting the mean.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
    mask: Binary array of shape broadcastable to `inputs` tensor, indicating the
      positions for which the mean and variance should be computed.
    force_float32_reductions: If false, this will skip float32 promotion and use
      the input dtype or inherited dtype from ``x``.

  Returns:
    A pair ``(mean, var)``.
  """
  if dtype is None:
    dtype = jnp.result_type(x)
  # promote x to at least float32, this avoids half precision computation
  # but preserves double or complex floating points
  if force_float32_reductions:
    dtype = jnp.promote_types(dtype, jnp.float32)
  x = jnp.asarray(x, dtype)
  axes = _canonicalize_axes(x.ndim, axes)

  def maybe_distributed_mean(*xs, mask=None):
    mus = tuple(x.mean(axes, where=mask) for x in xs)
    if axis_name is None:
      return mus if len(xs) > 1 else mus[0]
    else:
      # In the distributed case we stack multiple arrays to speed comms.
      if len(xs) > 1:
        reduced_mus = lax.pmean(
          jnp.stack(mus, axis=0),
          axis_name,
          axis_index_groups=axis_index_groups,
        )
        return tuple(reduced_mus[i] for i in range(len(xs)))
      else:
        return lax.pmean(mus[0], axis_name, axis_index_groups=axis_index_groups)

  if use_mean:
    if use_fast_variance:
      mu, mu2 = maybe_distributed_mean(x, _abs_sq(x), mask=mask)
      # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
      # to floating point round-off errors.
      var = jnp.maximum(0.0, mu2 - _abs_sq(mu))
    else:
      mu = maybe_distributed_mean(x, mask=mask)
      var = maybe_distributed_mean(
        _abs_sq(x - jnp.expand_dims(mu, axes)), mask=mask
      )
  else:
    var = maybe_distributed_mean(_abs_sq(x), mask=mask)
    mu = jnp.zeros_like(var)
  return mu, var


def _normalize(
  mdl: Module,
  x: Array,
  mean: Array,
  var: Array,
  reduction_axes: Axes,
  feature_axes: Axes,
  dtype: Dtype | None,
  param_dtype: Dtype,
  epsilon: float,
  use_bias: bool,
  use_scale: bool,
  bias_init: Initializer,
  scale_init: Initializer,
  force_float32_reductions: bool = True
):
  """Normalizes the input of a normalization layer and optionally applies a learned scale and bias.

  Arguments:
    mdl: Module to apply the normalization in (normalization params will reside
      in this module).
    x: The input.
    mean: Mean to use for normalization.
    var: Variance to use for normalization.
    reduction_axes: The axes in ``x`` to reduce.
    feature_axes: Axes containing features. A separate bias and scale is learned
      for each specified feature.
    dtype: The dtype of the result (default: infer from input and params).
    param_dtype: The dtype of the parameters.
    epsilon: Normalization epsilon.
    use_bias: If true, add a bias term to the output.
    use_scale: If true, scale the output.
    bias_init: Initialization function for the bias term.
    scale_init: Initialization function for the scaling function.
    force_float32_reductions: If false, the scale and bias parameters use the
      param_dtype. Otherwise, they will have at least float32 precision due to
      the mean and var being promoted to float32.

  Returns:
    The normalized input.
  """
  reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
  feature_axes = _canonicalize_axes(x.ndim, feature_axes)
  feature_shape = [1] * x.ndim
  reduced_feature_shape = []
  for ax in feature_axes:
    feature_shape[ax] = x.shape[ax]
    reduced_feature_shape.append(x.shape[ax])

  mean = jnp.expand_dims(mean, reduction_axes)
  var = jnp.expand_dims(var, reduction_axes)
  y = x - mean
  mul = lax.rsqrt(var + epsilon)
  args = [x]
  if use_scale:
    scale = mdl.param(
      'scale', scale_init, reduced_feature_shape, param_dtype
    ).reshape(feature_shape)
    if not force_float32_reductions:
      scale = jnp.asarray(scale, param_dtype)
    mul *= scale
    args.append(scale)
  y *= mul
  if use_bias:
    bias = mdl.param(
      'bias', bias_init, reduced_feature_shape, param_dtype
    ).reshape(feature_shape)
    if not force_float32_reductions:
      bias = jnp.asarray(bias, param_dtype)
    y += bias
    args.append(bias)
  dtype = dtypes.canonicalize_dtype(*args, dtype=dtype)
  return jnp.asarray(y, dtype)


def _l2_normalize(x, axis=None, eps=1e-12):
  """Normalizes along dimension `axis` using an L2 norm.

  This specialized function exists for numerical stability reasons.

  Args:
    x: An input ndarray.
    axis: Dimension along which to normalize, e.g. `1` to separately normalize
      vectors in a batch. Passing `None` views `t` as a flattened vector when
      calculating the norm (equivalent to Frobenius norm).
    eps: Epsilon to avoid dividing by zero.

  Returns:
    An array of the same shape as 'x' L2-normalized along 'axis'.
  """
  return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


class BatchNorm(Module):
  """BatchNorm Module.

  Usage Note:
  If we define a model with BatchNorm, for example::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp
    >>> BN = nn.BatchNorm(momentum=0.9, epsilon=1e-5, dtype=jnp.float32)

  The initialized variables dict will contain, in addition to a 'params'
  collection, a separate 'batch_stats' collection that will contain all the
  running statistics for all the BatchNorm layers in a model::

    >>> x = jax.random.normal(jax.random.key(0), (5, 6))
    >>> variables = BN.init(jax.random.key(1), x, use_running_average=False)
    >>> jax.tree_util.tree_map(jnp.shape, variables)
    {'batch_stats': {'mean': (6,), 'var': (6,)}, 'params': {'bias': (6,), 'scale': (6,)}}

  We then update the batch_stats during training by specifying that the
  ``batch_stats`` collection is mutable in the ``apply`` method for our
  module.::

    >>> y, new_batch_stats = BN.apply(variables, x, mutable=['batch_stats'], use_running_average=False)

  During eval we would define BN with ``use_running_average=True`` and use the
  batch_stats collection from training to set the statistics.  In this case
  we are not mutating the batch statistics collection, and needn't mark it
  mutable::

    >>> y = BN.apply(variables, x, mutable=['batch_stats'], use_running_average=True)

  Attributes:
    use_running_average: if True, the statistics stored in batch_stats will be
      used instead of computing the batch statistics on the input.
    axis: the feature or non-batch axis of the input.
    momentum: decay rate for the exponential moving average of the batch
      statistics.
    epsilon: a small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  if True, bias (beta) is added.
    use_scale: if True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: initializer for bias, by default, zero.
    scale_init: initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See ``jax.pmap`` for a description of axis names (default: None).
      Note, this is only used for pmap and shard map. For SPMD jit, you do not
      need to manually synchronize. Just make sure that the axes are correctly
      annotated and XLA:SPMD will insert the necessary collectives.
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, ``[[0, 1], [2, 3]]`` would independently batch-normalize over the
      examples on the first two and last two devices. See ``jax.lax.psum`` for
      more details.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
  """

  use_running_average: bool | None = None
  axis: int = -1
  momentum: float = 0.99
  epsilon: float = 1e-5
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Initializer = initializers.zeros
  scale_init: Initializer = initializers.ones
  axis_name: str | None = None
  axis_index_groups: Any = None
  use_fast_variance: bool = True
  force_float32_reductions: bool = True

  @compact
  def __call__(
      self,
      x,
      use_running_average: bool | None = None,
      *,
      mask: jax.Array | None = None,
  ):
    """Normalizes the input using batch statistics.

    .. note::
      During initialization (when ``self.is_initializing()`` is ``True``) the running
      average of the batch statistics will not be updated. Therefore, the inputs
      fed during initialization don't need to match that of the actual input
      distribution and the reduction axis (set with ``axis_name``) does not have
      to exist.

    Args:
      x: the input to be normalized.
      use_running_average: if true, the statistics stored in batch_stats will be
        used instead of computing the batch statistics on the input.
      mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
        the positions for which the mean and variance should be computed.

    Returns:
      Normalized inputs (the same shape as inputs).
    """

    use_running_average = module.merge_param(
      'use_running_average', self.use_running_average, use_running_average
    )
    feature_axes = _canonicalize_axes(x.ndim, self.axis)
    reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
    feature_shape = [x.shape[ax] for ax in feature_axes]

    ra_mean = self.variable(
        'batch_stats',
        'mean',
        lambda s: jnp.zeros(
            s,
            jnp.float32 if self.force_float32_reductions else self.param_dtype,
        ),
        feature_shape,
    )
    ra_var = self.variable(
        'batch_stats',
        'var',
        lambda s: jnp.ones(
            s,
            jnp.float32 if self.force_float32_reductions else self.param_dtype,
        ),
        feature_shape,
    )

    if use_running_average:
      mean = (
          ra_mean.value
          if self.force_float32_reductions
          else jnp.asarray(ra_mean.value, self.param_dtype)
      )
      var = (
          ra_var.value
          if self.force_float32_reductions
          else jnp.asarray(ra_var.value, self.param_dtype)
      )
    else:
      mean, var = _compute_stats(
          x,
          reduction_axes,
          dtype=self.dtype,
          axis_name=self.axis_name if not self.is_initializing() else None,
          axis_index_groups=self.axis_index_groups,
          use_fast_variance=self.use_fast_variance,
          mask=mask,
          force_float32_reductions=self.force_float32_reductions,
      )

      if not self.is_initializing():
        ra_mean.value = (
          self.momentum * ra_mean.value + (1 - self.momentum) * mean
        )
        ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

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
      self.force_float32_reductions,
    )


class LayerNorm(Module):
  """Layer normalization (https://arxiv.org/abs/1607.06450).

  LayerNorm normalizes the activations of the layer for each given example in a
  batch independently, rather than across a batch like Batch Normalization.
  i.e. applies a transformation that maintains the mean activation within
  each example close to 0 and the activation standard deviation close to 1.

  .. note::
    This normalization operation is identical to InstanceNorm and GroupNorm;
    the difference is simply which axes are reduced and the shape of the feature
    axes (i.e. the shape of the learnable scale and bias parameters).

  Example usage::

    >>> import flax.linen as nn
    >>> import jax
    >>> import numpy as np

    >>> x = jax.random.normal(jax.random.key(0), (3, 4, 5, 6))
    >>> layer = nn.LayerNorm()
    >>> variables = layer.init(jax.random.key(1), x)
    >>> variables
    {'params': {'scale': Array([1., 1., 1., 1., 1., 1.], dtype=float32), 'bias': Array([0., 0., 0., 0., 0., 0.], dtype=float32)}}
    >>> y = layer.apply(variables, x)

    >>> y = nn.LayerNorm(reduction_axes=(1, 2, 3)).apply(variables, x)
    >>> y2 = nn.GroupNorm(num_groups=1).apply(variables, x)
    >>> np.testing.assert_allclose(y, y2)

    >>> y = nn.LayerNorm(reduction_axes=(1, 2), feature_axes=-1).apply(variables, x)
    >>> y2 = nn.InstanceNorm(feature_axes=-1).apply(variables, x)
    >>> np.testing.assert_allclose(y, y2)

  Attributes:
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  If True, bias (beta) is added.
    use_scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
    reduction_axes: Axes for computing normalization statistics.
    feature_axes: Feature axes for learned bias and scaling.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See ``jax.pmap`` for a description of axis names (default: None).
      This is only needed if the model is subdivided across devices, i.e. the
      array being normalized is sharded across devices within a pmap or shard
      map. For SPMD jit, you do not need to manually synchronize. Just make sure
      that the axes are correctly annotated and XLA:SPMD will insert the
      necessary collectives.
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, ``[[0, 1], [2, 3]]`` would independently batch-normalize over the
      examples on the first two and last two devices. See ``jax.lax.psum`` for
      more details.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
  """

  epsilon: float = 1e-6
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Initializer = initializers.zeros
  scale_init: Initializer = initializers.ones
  reduction_axes: Axes = -1
  feature_axes: Axes = -1
  axis_name: str | None = None
  axis_index_groups: Any = None
  use_fast_variance: bool = True
  force_float32_reductions: bool = True

  @compact
  def __call__(self, x, *, mask: jax.Array | None = None):
    """Applies layer normalization on the input.

    Args:
      x: the inputs
      mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
        the positions for which the mean and variance should be computed.

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    mean, var = _compute_stats(
        x,
        self.reduction_axes,
        self.dtype,
        self.axis_name,
        self.axis_index_groups,
        use_fast_variance=self.use_fast_variance,
        mask=mask,
        force_float32_reductions=self.force_float32_reductions,
    )

    return _normalize(
      self,
      x,
      mean,
      var,
      self.reduction_axes,
      self.feature_axes,
      self.dtype,
      self.param_dtype,
      self.epsilon,
      self.use_bias,
      self.use_scale,
      self.bias_init,
      self.scale_init,
      self.force_float32_reductions,
    )


class RMSNorm(Module):
  """RMS Layer normalization (https://arxiv.org/abs/1910.07467).

  RMSNorm normalizes the activations of the layer for each given example in a
  batch independently, rather than across a batch like Batch Normalization.
  Unlike LayerNorm which re-centers the mean to be 0 and normalizes by the
  standard deviation of the activations, RMSNorm does not re-center at all
  and instead normalizes by the root mean square of the activations.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax

    >>> x = jax.random.normal(jax.random.key(0), (5, 6))
    >>> layer = nn.RMSNorm()
    >>> variables = layer.init(jax.random.key(1), x)
    >>> variables
    {'params': {'scale': Array([1., 1., 1., 1., 1., 1.], dtype=float32)}}
    >>> y = layer.apply(variables, x)

  Attributes:
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    scale_init: Initializer for scale, by default, one.
    reduction_axes: Axes for computing normalization statistics.
    feature_axes: Feature axes for learned bias and scaling.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See ``jax.pmap`` for a description of axis names (default: None).
      This is only needed if the model is subdivided across devices, i.e. the
      array being normalized is sharded across devices within a pmap or shard
      map. For SPMD jit, you do not need to manually synchronize. Just make sure
      that the axes are correctly annotated and XLA:SPMD will insert the
      necessary collectives.
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, ``[[0, 1], [2, 3]]`` would independently batch-normalize over the
      examples on the first two and last two devices. See ``jax.lax.psum`` for
      more details.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
  """

  epsilon: float = 1e-6
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  use_scale: bool = True
  scale_init: Initializer = initializers.ones
  reduction_axes: Axes = -1
  feature_axes: Axes = -1
  axis_name: str | None = None
  axis_index_groups: Any = None
  use_fast_variance: bool = True
  force_float32_reductions: bool = True

  @compact
  def __call__(self, x, *, mask: jax.Array | None = None):
    """Applies RMS layer normalization on the input.

    Args:
      x: the inputs
      mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
        the positions for which the mean and variance should be computed.

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    mean, var = _compute_stats(
        x,
        self.reduction_axes,
        self.dtype,
        self.axis_name,
        self.axis_index_groups,
        use_mean=False,
        use_fast_variance=self.use_fast_variance,
        mask=mask,
        force_float32_reductions=self.force_float32_reductions,
    )

    return _normalize(
      self,
      x,
      mean,
      var,
      self.reduction_axes,
      self.feature_axes,
      self.dtype,
      self.param_dtype,
      self.epsilon,
      False,
      self.use_scale,
      initializers.zeros,
      self.scale_init,
      self.force_float32_reductions,
    )


class GroupNorm(Module):
  """Group normalization (arxiv.org/abs/1803.08494).

  This op is similar to batch normalization, but statistics are shared across
  equally-sized groups of channels and not shared across batch dimension.
  Thus, group normalization does not depend on the batch composition and does
  not require maintaining internal state for storing statistics.
  The user should either specify the total number of channel groups or the
  number of channels per group.

  .. note::
    LayerNorm is a special case of GroupNorm where ``num_groups=1``, and
    InstanceNorm is a special case of GroupNorm where ``group_size=1``.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax
    >>> import numpy as np

    >>> x = jax.random.normal(jax.random.key(0), (3, 4, 5, 6))
    >>> layer = nn.GroupNorm(num_groups=3)
    >>> variables = layer.init(jax.random.key(1), x)
    >>> variables
    {'params': {'scale': Array([1., 1., 1., 1., 1., 1.], dtype=float32), 'bias': Array([0., 0., 0., 0., 0., 0.], dtype=float32)}}
    >>> y = layer.apply(variables, x)

    >>> y = nn.GroupNorm(num_groups=1).apply(variables, x)
    >>> y2 = nn.LayerNorm(reduction_axes=(1, 2, 3)).apply(variables, x)
    >>> np.testing.assert_allclose(y, y2)

    >>> y = nn.GroupNorm(num_groups=None, group_size=1).apply(variables, x)
    >>> y2 = nn.InstanceNorm(feature_axes=-1).apply(variables, x)
    >>> np.testing.assert_allclose(y, y2)

  Attributes:
    num_groups: the total number of channel groups. The default value of 32 is
      proposed by the original group normalization paper.
    group_size: the number of channels in a group.
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  If True, bias (beta) is added.
    use_scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
    reduction_axes: List of axes used for computing normalization statistics.
      This list must include the final dimension, which is assumed to be the
      feature axis. Furthermore, if the input used at call time has additional
      leading axes compared to the data used for initialisation, for example due
      to batching, then the reduction axes need to be defined explicitly.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See ``jax.pmap`` for a description of axis names (default: None).
      This is only needed if the model is subdivided across devices, i.e. the
      array being normalized is sharded across devices within a pmap or shard
      map. For SPMD jit, you do not need to manually synchronize. Just make sure
      that the axes are correctly annotated and XLA:SPMD will insert the
      necessary collectives.
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, ``[[0, 1], [2, 3]]`` would independently batch-normalize over the
      examples on the first two and last two devices. See ``jax.lax.psum`` for
      more details.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
  """

  num_groups: int | None = 32
  group_size: int | None = None
  epsilon: float = 1e-6
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Initializer = initializers.zeros
  scale_init: Initializer = initializers.ones
  reduction_axes: Axes | None = None
  axis_name: str | None = None
  axis_index_groups: Any = None
  use_fast_variance: bool = True
  force_float32_reductions: bool = True

  @compact
  def __call__(self, x, *, mask: jax.Array | None = None):
    """Applies group normalization to the input (arxiv.org/abs/1803.08494).

    Args:
      x: the input of shape ``...C`` where ``C`` is a channels dimension and ``...``
        represents an arbitrary number of extra dimensions that can be used to
        accumulate statistics over. If no reduction axes have been specified
        then all additional dimensions ``...`` will be used to accumulate
        statistics apart from the leading dimension which is assumed to
        represent the batch.
      mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
        the positions for which the mean and variance should be computed.

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    if self.reduction_axes is not None:
      reduction_axes = self.reduction_axes
    else:
      reduction_axes = list(range(1, x.ndim - 1)) + [-1]
    feature_axis = -1

    reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)

    if reduction_axes[-1] != (feature_axis % x.ndim):
      raise ValueError(
          'The reduction axes must include the final dimension '
          'as this is assumed to be the feature axis.'
      )

    if (self.num_groups is None and self.group_size is None) or (
      self.num_groups is not None and self.group_size is not None
    ):
      raise ValueError(
        'Either `num_groups` or `group_size` should be '
        'specified. If `group_size` is to be specified, '
        'pass `num_groups=None` as argument to override '
        'the default `num_groups` value of 32.'
      )

    channels = x.shape[-1]
    if self.group_size is not None:
      if channels % self.group_size != 0:
        raise ValueError(
          'Number of channels ({}) is not multiple of the '
          'group size ({}).'.format(channels, self.group_size)
        )
      num_groups = channels // self.group_size
    else:
      num_groups = self.num_groups
      assert isinstance(num_groups, int)

    if num_groups <= 0 or channels % num_groups != 0:
      raise ValueError(
        'Number of groups ({}) does not divide the number'
        ' of channels ({}).'.format(num_groups, channels)
      )

    group_size = x.shape[-1] // num_groups
    group_shape = x.shape[:-1] + (num_groups, group_size)

    if mask is not None:
      mask = mask.reshape(mask.shape[:-1] + (num_groups, group_size))

    mean, var = _compute_stats(
        x.reshape(group_shape),
        list(reduction_axes[:-1]) + [-1],
        self.dtype,
        self.axis_name,
        self.axis_index_groups,
        use_fast_variance=self.use_fast_variance,
        mask=mask,
        force_float32_reductions=self.force_float32_reductions,
    )
    mean = jnp.repeat(mean, group_size, axis=-1)
    var = jnp.repeat(var, group_size, axis=-1)

    return _normalize(
      self,
      x,
      mean,
      var,
      reduction_axes[:-1],
      (feature_axis,),
      self.dtype,
      self.param_dtype,
      self.epsilon,
      self.use_bias,
      self.use_scale,
      self.bias_init,
      self.scale_init,
      self.force_float32_reductions,
    )


class InstanceNorm(Module):
  """Instance normalization (https://arxiv.org/abs/1607.08022v3).

  InstanceNorm normalizes the activations of the layer for each channel (rather
  than across all channels like Layer Normalization), and for each given example
  in a batch independently (rather than across an entire batch like Batch
  Normalization). i.e. applies a transformation that maintains the mean activation
  within each channel within each example close to 0 and the activation standard
  deviation close to 1.

  .. note::
    This normalization operation is identical to LayerNorm and GroupNorm; the
    difference is simply which axes are reduced and the shape of the feature axes
    (i.e. the shape of the learnable scale and bias parameters).

  Example usage::

    >>> import flax.linen as nn
    >>> import jax
    >>> import numpy as np

    >>> # dimensions: (batch, height, width, channel)
    >>> x = jax.random.normal(jax.random.key(0), (2, 3, 4, 5))
    >>> layer = nn.InstanceNorm()
    >>> variables = layer.init(jax.random.key(1), x)
    >>> variables
    {'params': {'scale': Array([1., 1., 1., 1., 1.], dtype=float32), 'bias': Array([0., 0., 0., 0., 0.], dtype=float32)}}
    >>> y = layer.apply(variables, x)

    >>> # having a channel_axis of -1 in InstanceNorm is identical to reducing all non-batch,
    >>> # non-channel axes and using the feature_axes as the feature_axes in LayerNorm
    >>> y2 = nn.LayerNorm(reduction_axes=[1, 2], feature_axes=-1).apply(variables, x)
    >>> np.testing.assert_allclose(y, y2, atol=1e-7)
    >>> y3 = nn.GroupNorm(num_groups=x.shape[-1]).apply(variables, x)
    >>> np.testing.assert_allclose(y, y3, atol=1e-7)

  Attributes:
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  If True, bias (beta) is added.
    use_scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
    feature_axes: Axes for features. The learned bias and scaling parameters will
      be in the shape defined by the feature axes. All other axes except the batch
      axes (which is assumed to be the leading axis) will be reduced.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See ``jax.pmap`` for a description of axis names (default: None).
      This is only needed if the model is subdivided across devices, i.e. the
      array being normalized is sharded across devices within a pmap or shard
      map. For SPMD jit, you do not need to manually synchronize. Just make sure
      that the axes are correctly annotated and XLA:SPMD will insert the
      necessary collectives.
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, ``[[0, 1], [2, 3]]`` would independently batch-normalize over the
      examples on the first two and last two devices. See ``jax.lax.psum`` for
      more details.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
  """

  epsilon: float = 1e-6
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Initializer = initializers.zeros
  scale_init: Initializer = initializers.ones
  feature_axes: Axes = -1
  axis_name: str | None = None
  axis_index_groups: Any = None
  use_fast_variance: bool = True
  force_float32_reductions: bool = True

  @compact
  def __call__(self, x, *, mask: jax.Array | None = None):
    """Applies instance normalization on the input.

    Args:
      x: the inputs
      mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
        the positions for which the mean and variance should be computed.

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    feature_axes = _canonicalize_axes(x.ndim, self.feature_axes)
    if 0 in feature_axes:
      raise ValueError('The channel axes cannot include the leading dimension '
                       'as this is assumed to be the batch axis.')
    reduction_axes = [i for i in range(1, x.ndim) if i not in feature_axes]

    mean, var = _compute_stats(
        x,
        reduction_axes,
        self.dtype,
        self.axis_name,
        self.axis_index_groups,
        use_fast_variance=self.use_fast_variance,
        mask=mask,
        force_float32_reductions=self.force_float32_reductions,
    )

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
      self.force_float32_reductions,
    )


class SpectralNorm(Module):
  """Spectral normalization.

  See:

  - https://arxiv.org/abs/1802.05957
  - https://arxiv.org/abs/1805.08318
  - https://arxiv.org/abs/1809.11096

  Spectral normalization normalizes the weight params so that the spectral
  norm of the matrix is equal to 1. This is implemented as a layer wrapper
  where each wrapped layer will have its params spectral normalized before
  computing its ``__call__`` output.

  .. note::
    The initialized variables dict will contain, in addition to a 'params'
    collection, a separate 'batch_stats' collection that will contain a
    ``u`` vector and ``sigma`` value, which are intermediate values used
    when performing spectral normalization. During training, we pass in
    ``update_stats=True`` and ``mutable=['batch_stats']`` so that ``u``
    and ``sigma`` are updated with the most recently computed values using
    power iteration. This will help the power iteration method approximate
    the true singular value more accurately over time. During eval, we pass
    in ``update_stats=False`` to ensure we get deterministic behavior from
    the model.

  Example usage::

    >>> import flax, flax.linen as nn
    >>> import jax, jax.numpy as jnp
    >>> import optax

    >>> class Foo(nn.Module):
    ...   @nn.compact
    ...   def __call__(self, x, train):
    ...     x = nn.Dense(3)(x)
    ...     # only spectral normalize the params of the second Dense layer
    ...     x = nn.SpectralNorm(nn.Dense(4))(x, update_stats=train)
    ...     x = nn.Dense(5)(x)
    ...     return x

    >>> # init
    >>> x = jnp.ones((1, 2))
    >>> y = jnp.ones((1, 5))
    >>> model = Foo()
    >>> variables = model.init(jax.random.PRNGKey(0), x, train=False)
    >>> flax.core.freeze(jax.tree_util.tree_map(jnp.shape, variables))
    FrozenDict({
        batch_stats: {
            SpectralNorm_0: {
                Dense_1/kernel/sigma: (),
                Dense_1/kernel/u: (1, 4),
            },
        },
        params: {
            Dense_0: {
                bias: (3,),
                kernel: (2, 3),
            },
            Dense_1: {
                bias: (4,),
                kernel: (3, 4),
            },
            Dense_2: {
                bias: (5,),
                kernel: (4, 5),
            },
        },
    })

    >>> # train
    >>> def train_step(variables, x, y):
    ...   def loss_fn(params):
    ...     logits, updates = model.apply(
    ...         {'params': params, 'batch_stats': variables['batch_stats']},
    ...         x,
    ...         train=True,
    ...         mutable=['batch_stats'],
    ...     )
    ...     loss = jnp.mean(optax.l2_loss(predictions=logits, targets=y))
    ...     return loss, updates
    ...
    ...   (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(
    ...       variables['params']
    ...   )
    ...   return {
    ...       'params': jax.tree_util.tree_map(
    ...           lambda p, g: p - 0.1 * g, variables['params'], grads
    ...       ),
    ...       'batch_stats': updates['batch_stats'],
    ...   }, loss
    >>> for _ in range(10):
    ...   variables, loss = train_step(variables, x, y)

    >>> # inference / eval
    >>> out = model.apply(variables, x, train=False)

  Attributes:
    layer_instance: Module instance that is wrapped with SpectralNorm
    n_steps: How many steps of power iteration to perform to approximate the
      singular value of the weight params.
    epsilon: A small float added to l2-normalization to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    error_on_non_matrix: Spectral normalization is only defined on matrices. By
      default, this module will return scalars unchanged and flatten
      higher-order tensors in their leading dimensions. Setting this flag to
      True will instead throw an error if a weight tensor with dimension greater
      than 2 is used by the layer.
    collection_name: Name of the collection to store intermediate values used
      when performing spectral normalization.
  """

  layer_instance: Module
  n_steps: int = 1
  epsilon: float = 1e-12
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  error_on_non_matrix: bool = False
  collection_name: str = 'batch_stats'

  @compact
  def __call__(self, *args, update_stats: bool, **kwargs):
    """Compute the largest singular value of the weights in ``self.layer_instance``
    using power iteration and normalize the weights using this value before
    computing the ``__call__`` output.

    Args:
      *args: positional arguments to be passed into the call method of the
        underlying layer instance in ``self.layer_instance``.
      update_stats: if True, update the internal ``u`` vector and ``sigma``
        value after computing their updated values using power iteration. This
        will help the power iteration method approximate the true singular value
        more accurately over time.
      **kwargs: keyword arguments to be passed into the call method of the
        underlying layer instance in ``self.layer_instance``.

    Returns:
      Output of the layer using spectral normalized weights.
    """

    def layer_forward(layer_instance):
      return layer_instance(*args, **kwargs)

    return transforms.map_variables(
      layer_forward,
      trans_in_fn=lambda vs: jax.tree_util.tree_map_with_path(
        functools.partial(
          self._spectral_normalize,
          update_stats=update_stats,
        ),
        vs,
      ),
      init=self.is_initializing(),
      mutable=True,
    )(self.layer_instance)

  def _spectral_normalize(self, path, vs, update_stats):
    """Compute the largest singular value using power iteration and normalize
    the variables ``vs`` using this value. This is intended to be a helper
    function used in this Module's ``__call__`` method in conjunction with
    ``nn.transforms.map_variables`` and ``jax.tree_util.tree_map_with_path``.

    Args:
      path: dict key path, used for naming the ``u`` and ``sigma`` variables
      vs: variables to be spectral normalized
      update_stats: if True, update the ``u`` vector and ``sigma`` variables
        after computing their updated values using power iteration. This will
        help the power iteration method approximate the true singular value
        more accurately over time.
    """
    value = jnp.asarray(vs)
    value_shape = value.shape

    # Skip and return value if input is scalar, vector or if number of power
    # iterations is less than 1
    if value.ndim <= 1 or self.n_steps < 1:
      return value
    # Handle higher-order tensors.
    elif value.ndim > 2:
      if self.error_on_non_matrix:
        raise ValueError(
          f'Input is {value.ndim}D but error_on_non_matrix is True'
        )
      else:
        value = jnp.reshape(value, (-1, value.shape[-1]))

    u_var_name = (
      self.layer_instance.name
      + '/'
      + '/'.join(dict_key.key for dict_key in path[1:])
      + '/u'
    )
    u_var = self.variable(
      self.collection_name,
      u_var_name,
      jax.random.normal,
      self.make_rng('params')
      if not self.has_variable(self.collection_name, u_var_name)
      else None,
      (1, value.shape[-1]),
      self.param_dtype,
    )
    u0 = u_var.value
    sigma_var_name = (
      self.layer_instance.name
      + '/'
      + '/'.join(dict_key.key for dict_key in path[1:])
      + '/sigma'
    )
    sigma_var = self.variable(
      self.collection_name, sigma_var_name, jnp.ones, (), self.param_dtype
    )

    # Power iteration for the weight's singular value.
    for _ in range(self.n_steps):
      v0 = _l2_normalize(
        jnp.matmul(u0, value.transpose([1, 0])), eps=self.epsilon
      )
      u0 = _l2_normalize(jnp.matmul(v0, value), eps=self.epsilon)

    u0 = jax.lax.stop_gradient(u0)
    v0 = jax.lax.stop_gradient(v0)

    sigma = jnp.matmul(jnp.matmul(v0, value), jnp.transpose(u0))[0, 0]

    value /= jnp.where(sigma != 0, sigma, 1)
    value_bar = value.reshape(value_shape)

    if update_stats:
      u_var.value = u0
      sigma_var.value = sigma

    dtype = dtypes.canonicalize_dtype(vs, u0, v0, sigma, dtype=self.dtype)
    return jnp.asarray(value_bar, dtype)


class WeightNorm(Module):
  """L2 weight normalization (https://arxiv.org/abs/1602.07868).

  Weight normalization normalizes the weight params so that the l2-norm of
  the matrix is equal to 1. This is implemented as a layer wrapper where
  each wrapped layer will have its params l2-normalized before computing
  its ``__call__`` output.

  Example usage::

    >>> import flax, flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> class Baz(nn.Module):
    ...   @nn.compact
    ...   def __call__(self, x):
    ...     return nn.Dense(2)(x)

    >>> class Bar(nn.Module):
    ...   @nn.compact
    ...   def __call__(self, x):
    ...     x = Baz()(x)
    ...     x = nn.Dense(3)(x)
    ...     x = Baz()(x)
    ...     x = nn.Dense(3)(x)
    ...     return x

    >>> class Foo(nn.Module):
    ...   @nn.compact
    ...   def __call__(self, x):
    ...     x = nn.Dense(3)(x)
    ...     # l2-normalize all params of the second Dense layer
    ...     x = nn.WeightNorm(nn.Dense(4), variable_filter=None)(x)
    ...     x = nn.Dense(5)(x)
    ...     # l2-normalize all kernels in the Bar submodule and all params in
    ...     # the Baz submodule
    ...     x = nn.WeightNorm(Bar(), variable_filter={'kernel', 'Baz'})(x)
    ...     return x

    >>> # init
    >>> x = jnp.ones((1, 2))
    >>> model = Foo()
    >>> variables = model.init(jax.random.key(0), x)
    >>> flax.core.freeze(jax.tree_util.tree_map(jnp.shape, variables))
    FrozenDict({
        params: {
            Bar_0: {
                Baz_0: {
                    Dense_0: {
                        bias: (2,),
                        kernel: (5, 2),
                    },
                },
                Baz_1: {
                    Dense_0: {
                        bias: (2,),
                        kernel: (3, 2),
                    },
                },
                Dense_0: {
                    bias: (3,),
                    kernel: (2, 3),
                },
                Dense_1: {
                    bias: (3,),
                    kernel: (2, 3),
                },
            },
            Dense_0: {
                bias: (3,),
                kernel: (2, 3),
            },
            Dense_1: {
                bias: (4,),
                kernel: (3, 4),
            },
            Dense_2: {
                bias: (5,),
                kernel: (4, 5),
            },
            WeightNorm_0: {
                Dense_1/bias/scale: (4,),
                Dense_1/kernel/scale: (4,),
            },
            WeightNorm_1: {
                Bar_0/Baz_0/Dense_0/bias/scale: (2,),
                Bar_0/Baz_0/Dense_0/kernel/scale: (2,),
                Bar_0/Baz_1/Dense_0/bias/scale: (2,),
                Bar_0/Baz_1/Dense_0/kernel/scale: (2,),
                Bar_0/Dense_0/kernel/scale: (3,),
                Bar_0/Dense_1/kernel/scale: (3,),
            },
        },
    })

  Attributes:
    layer_instance: Module instance that is wrapped with WeightNorm
    epsilon: A small float added to l2-normalization to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_scale: If True, creates a learnable variable ``scale`` that is
      multiplied to the ``layer_instance`` variables after l2-normalization.
    scale_init: Initialization function for the scaling function.
    feature_axes: The feature axes dimension(s). The l2-norm is calculated by
      reducing the ``layer_instance`` variables over the remaining (non-feature)
      axes. Therefore a separate l2-norm value is calculated and a separate
      scale (if ``use_scale=True``) is learned for each specified feature. By
      default, the trailing dimension is treated as the feature axis.
    variable_filter: An optional iterable that contains string items. The
      WeightNorm layer will selectively apply l2-normalization to the
      ``layer_instance`` variables whose key path (delimited by '/') has a match
      with ``variable_filter``. For example, ``variable_filter={'kernel'}`` will
      only apply l2-normalization to variables whose key path contains 'kernel'.
      By default, ``variable_filter={'kernel'}``.
  """

  layer_instance: Module
  epsilon: float = 1e-12
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  use_scale: bool = True
  scale_init: Initializer = initializers.ones
  feature_axes: Axes | None = -1
  variable_filter: Iterable | None = dataclasses.field(
    default_factory=lambda: {'kernel'}
  )

  @compact
  def __call__(self, *args, **kwargs):
    """Compute the l2-norm of the weights in ``self.layer_instance``
    and normalize the weights using this value before computing the
    ``__call__`` output.

    Args:
      *args: positional arguments to be passed into the call method of the
        underlying layer instance in ``self.layer_instance``.
      **kwargs: keyword arguments to be passed into the call method of the
        underlying layer instance in ``self.layer_instance``.

    Returns:
      Output of the layer using l2-normalized weights.
    """

    def layer_forward(layer_instance):
      return layer_instance(*args, **kwargs)

    return transforms.map_variables(
      layer_forward,
      trans_in_fn=lambda vs: jax.tree_util.tree_map_with_path(
        self._l2_normalize,
        vs,
      ),
      init=self.is_initializing(),
    )(self.layer_instance)

  def _l2_normalize(self, path, vs):
    """Compute the l2-norm and normalize the variables ``vs`` using this
    value. This is intended to be a helper function used in this Module's
    ``__call__`` method in conjunction with ``nn.transforms.map_variables``
    and ``jax.tree_util.tree_map_with_path``.

    Args:
      path: dict key path, used for naming the ``scale`` variable
      vs: variables to be l2-normalized
    """
    value = jnp.asarray(vs)
    str_path = (
      self.layer_instance.name
      + '/'
      + '/'.join(dict_key.key for dict_key in path[1:])
    )
    if self.variable_filter:
      for variable_name in self.variable_filter:
        if variable_name in str_path:
          break
      else:
        return value

    if self.feature_axes is None:
      feature_axes = ()
      reduction_axes = tuple(i for i in range(value.ndim))
    else:
      feature_axes = _canonicalize_axes(value.ndim, self.feature_axes)
      reduction_axes = tuple(
        i for i in range(value.ndim) if i not in feature_axes
      )

    feature_shape = [1] * value.ndim
    reduced_feature_shape = []
    for ax in feature_axes:
      feature_shape[ax] = value.shape[ax]
      reduced_feature_shape.append(value.shape[ax])

    value_bar = _l2_normalize(value, axis=reduction_axes, eps=self.epsilon)

    args = [vs]
    if self.use_scale:
      scale = self.param(
        str_path + '/scale',
        self.scale_init,
        reduced_feature_shape,
        self.param_dtype,
      ).reshape(feature_shape)
      value_bar *= scale
      args.append(scale)

    dtype = dtypes.canonicalize_dtype(*args, dtype=self.dtype)
    return jnp.asarray(value_bar, dtype)
