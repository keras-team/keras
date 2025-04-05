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

import typing as tp

import jax
import jax.numpy as jnp
from jax import lax

from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import Module, first_from
from flax.nnx.nn import dtypes, initializers
from flax.typing import (
  Array,
  Dtype,
  Initializer,
  Axes,
)


def _canonicalize_axes(rank: int, axes: Axes) -> tp.Tuple[int, ...]:
  """Returns a tuple of deduplicated, sorted, and positive axes."""
  if not isinstance(axes, tp.Iterable):
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
  dtype: tp.Optional[Dtype],
  axis_name: tp.Optional[str] = None,
  axis_index_groups: tp.Any = None,
  use_mean: bool = True,
  use_fast_variance: bool = True,
  mask: tp.Optional[Array] = None,
):
  """Computes mean and variance statistics.

  This implementation takes care of a few important details:
  - Computes in float32 precision for stability in half precision training.
  - If ``use_fast_variance`` is ``True``, mean and variance are computed using
    Var = E[|x|^2] - |E[x]|^2, instead of Var = E[|x - E[x]|^2]), in a single
    XLA fusion.
  - Clips negative variances to zero which can happen due to
    roundoff errors. This avoids downstream NaNs.
  - Supports averaging across a parallel axis and subgroups of a parallel axis
    with a single ``lax.pmean`` call to avoid latency.

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
    mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
      the positions for which the mean and variance should be computed.

  Returns:
    A pair ``(mean, var)``.
  """
  if dtype is None:
    dtype = jnp.result_type(x)
  # promote x to at least float32, this avoids half precision computation
  # but preserves double or complex floating points
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
  x: Array,
  mean: Array,
  var: Array,
  scale: tp.Optional[Array],
  bias: tp.Optional[Array],
  reduction_axes: Axes,
  feature_axes: Axes,
  dtype: tp.Optional[Dtype],
  epsilon: float,
):
  """ "Normalizes the input of a normalization layer and optionally applies a learned scale and bias.

  Arguments:
    x: The input.
    mean: Mean to use for normalization.
    var: Variance to use for normalization.
    reduction_axes: The axes in ``x`` to reduce.
    feature_axes: Axes containing features. A separate bias and scale is learned
      for each specified feature.
    dtype: The dtype of the result (default: infer from input and params).
    epsilon: Normalization epsilon.

  Returns:
    The normalized input.
  """
  reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
  feature_axes = _canonicalize_axes(x.ndim, feature_axes)
  stats_shape = list(x.shape)
  for axis in reduction_axes:
    stats_shape[axis] = 1
  mean = mean.reshape(stats_shape)
  var = var.reshape(stats_shape)
  feature_shape = [1] * x.ndim
  for ax in feature_axes:
    feature_shape[ax] = x.shape[ax]
  y = x - mean
  mul = lax.rsqrt(var + epsilon)
  args = [x]
  if scale is not None:
    scale = scale.reshape(feature_shape)
    mul *= scale
    args.append(scale)
  y *= mul
  if bias is not None:
    bias = bias.reshape(feature_shape)
    y += bias
    args.append(bias)
  dtype = dtypes.canonicalize_dtype(*args, dtype=dtype)
  return jnp.asarray(y, dtype)


class BatchNorm(Module):
  """BatchNorm Module.

  To calculate the batch norm on the input and update the batch statistics,
  call the :func:`train` method (or pass in ``use_running_average=False`` in
  the constructor or during call time).

  To use the stored batch statistics' running average, call the :func:`eval`
  method (or pass in ``use_running_average=True`` in the constructor or
  during call time).

  Example usage::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> x = jax.random.normal(jax.random.key(0), (5, 6))
    >>> layer = nnx.BatchNorm(num_features=6, momentum=0.9, epsilon=1e-5,
    ...                       dtype=jnp.float32, rngs=nnx.Rngs(0))
    >>> jax.tree.map(jnp.shape, nnx.state(layer))
    State({
      'bias': VariableState(
        type=Param,
        value=(6,)
      ),
      'mean': VariableState(
        type=BatchStat,
        value=(6,)
      ),
      'scale': VariableState(
        type=Param,
        value=(6,)
      ),
      'var': VariableState(
        type=BatchStat,
        value=(6,)
      )
    })

    >>> # calculate batch norm on input and update batch statistics
    >>> layer.train()
    >>> y = layer(x)
    >>> batch_stats1 = nnx.state(layer, nnx.BatchStat)
    >>> y = layer(x)
    >>> batch_stats2 = nnx.state(layer, nnx.BatchStat)
    >>> assert (batch_stats1['mean'].value != batch_stats2['mean'].value).all()
    >>> assert (batch_stats1['var'].value != batch_stats2['var'].value).all()

    >>> # use stored batch statistics' running average
    >>> layer.eval()
    >>> y = layer(x)
    >>> batch_stats3 = nnx.state(layer, nnx.BatchStat)
    >>> assert (batch_stats2['mean'].value == batch_stats3['mean'].value).all()
    >>> assert (batch_stats2['var'].value == batch_stats3['var'].value).all()

  Args:
    num_features: the number of input features.
    use_running_average: if True, the stored batch statistics will be
      used instead of computing the batch statistics on the input.
    axis: the feature or non-batch axis of the input.
    momentum: decay rate for the exponential moving average of
      the batch statistics.
    epsilon: a small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  if True, bias (beta) is added.
    use_scale: if True, multiply by scale (gamma).
      When the next layer is linear (also e.g. nn.relu), this can be disabled
      since the scaling will be done by the next layer.
    bias_init: initializer for bias, by default, zero.
    scale_init: initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See ``jax.pmap`` for a description of axis names (default: None).
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, ``[[0, 1], [2, 3]]`` would independently batch-normalize over
      the examples on the first two and last two devices. See ``jax.lax.psum``
      for more details.
    use_fast_variance: If true, use a faster, but less numerically stable,
      calculation for the variance.
    rngs: rng key.
  """

  def __init__(
    self,
    num_features: int,
    *,
    use_running_average: bool = False,
    axis: int = -1,
    momentum: float = 0.99,
    epsilon: float = 1e-5,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    use_bias: bool = True,
    use_scale: bool = True,
    bias_init: Initializer = initializers.zeros_init(),
    scale_init: Initializer = initializers.ones_init(),
    axis_name: tp.Optional[str] = None,
    axis_index_groups: tp.Any = None,
    use_fast_variance: bool = True,
    rngs: rnglib.Rngs,
  ):
    feature_shape = (num_features,)
    self.mean = nnx.BatchStat(jnp.zeros(feature_shape, jnp.float32))
    self.var = nnx.BatchStat(jnp.ones(feature_shape, jnp.float32))

    self.scale: nnx.Param[jax.Array] | None
    if use_scale:
      key = rngs.params()
      self.scale = nnx.Param(scale_init(key, feature_shape, param_dtype))
    else:
      self.scale = None

    self.bias: nnx.Param[jax.Array] | None
    if use_bias:
      key = rngs.params()
      self.bias = nnx.Param(bias_init(key, feature_shape, param_dtype))
    else:
      self.bias = None

    self.num_features = num_features
    self.use_running_average = use_running_average
    self.axis = axis
    self.momentum = momentum
    self.epsilon = epsilon
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.use_bias = use_bias
    self.use_scale = use_scale
    self.bias_init = bias_init
    self.scale_init = scale_init
    self.axis_name = axis_name
    self.axis_index_groups = axis_index_groups
    self.use_fast_variance = use_fast_variance

  def __call__(
    self,
    x,
    use_running_average: tp.Optional[bool] = None,
    *,
    mask: tp.Optional[jax.Array] = None,
  ):
    """Normalizes the input using batch statistics.

    Args:
      x: the input to be normalized.
      use_running_average: if true, the stored batch statistics will be
        used instead of computing the batch statistics on the input. The
        ``use_running_average`` flag passed into the call method will take
        precedence over the ``use_running_average`` flag passed into the
        constructor.

    Returns:
      Normalized inputs (the same shape as inputs).
    """

    use_running_average = first_from(
      use_running_average,
      self.use_running_average,
      error_msg="""No `use_running_average` argument was provided to BatchNorm
        as either a __call__ argument, class attribute, or nnx.flag.""",
    )
    feature_axes = _canonicalize_axes(x.ndim, self.axis)
    reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)

    if use_running_average:
      mean, var = self.mean.value, self.var.value
    else:
      mean, var = _compute_stats(
        x,
        reduction_axes,
        dtype=self.dtype,
        axis_name=self.axis_name,
        axis_index_groups=self.axis_index_groups,
        use_fast_variance=self.use_fast_variance,
        mask=mask,
      )

      self.mean.value = (
        self.momentum * self.mean.value + (1 - self.momentum) * mean
      )
      self.var.value = (
        self.momentum * self.var.value + (1 - self.momentum) * var
      )

    return _normalize(
      x,
      mean,
      var,
      self.scale.value if self.scale else None,
      self.bias.value if self.bias else None,
      reduction_axes,
      feature_axes,
      self.dtype,
      self.epsilon,
    )


class LayerNorm(Module):
  """Layer normalization (https://arxiv.org/abs/1607.06450).

  LayerNorm normalizes the activations of the layer for each given example in a
  batch independently, rather than across a batch like Batch Normalization.
  i.e. applies a transformation that maintains the mean activation within
  each example close to 0 and the activation standard deviation close to 1.

  Example usage::

    >>> from flax import nnx
    >>> import jax

    >>> x = jax.random.normal(jax.random.key(0), (3, 4, 5, 6))
    >>> layer = nnx.LayerNorm(num_features=6, rngs=nnx.Rngs(0))

    >>> nnx.state(layer)
    State({
      'bias': VariableState( # 6 (24 B)
        type=Param,
        value=Array([0., 0., 0., 0., 0., 0.], dtype=float32)
      ),
      'scale': VariableState( # 6 (24 B)
        type=Param,
        value=Array([1., 1., 1., 1., 1., 1.], dtype=float32)
      )
    })

    >>> y = layer(x)

  Args:
    num_features: the number of input features.
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the result (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  If True, bias (beta) is added.
    use_scale: If True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nnx.relu), this can be disabled since the scaling will be done
        by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
    reduction_axes: Axes for computing normalization statistics.
    feature_axes: Feature axes for learned bias and scaling.
    axis_name: the axis name used to combine batch statistics from multiple
        devices. See ``jax.pmap`` for a description of axis names (default: None).
        This is only needed if the model is subdivided across devices, i.e. the
        array being normalized is sharded across devices within a pmap.
    axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For
        example, ``[[0, 1], [2, 3]]`` would independently batch-normalize over
        the examples on the first two and last two devices. See ``jax.lax.psum``
        for more details.
    use_fast_variance: If true, use a faster, but less numerically stable,
        calculation for the variance.
    rngs: rng key.
  """

  def __init__(
    self,
    num_features: int,
    *,
    epsilon: float = 1e-6,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    use_bias: bool = True,
    use_scale: bool = True,
    bias_init: Initializer = initializers.zeros_init(),
    scale_init: Initializer = initializers.ones_init(),
    reduction_axes: Axes = -1,
    feature_axes: Axes = -1,
    axis_name: tp.Optional[str] = None,
    axis_index_groups: tp.Any = None,
    use_fast_variance: bool = True,
    rngs: rnglib.Rngs,
  ):
    feature_shape = (num_features,)

    self.scale: nnx.Param[jax.Array] | None
    if use_scale:
      key = rngs.params()
      self.scale = nnx.Param(scale_init(key, feature_shape, param_dtype))
    else:
      self.scale = None

    self.bias: nnx.Param[jax.Array] | None
    if use_bias:
      key = rngs.params()
      self.bias = nnx.Param(bias_init(key, feature_shape, param_dtype))
    else:
      self.bias = None

    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.use_bias = use_bias
    self.use_scale = use_scale
    self.bias_init = bias_init
    self.scale_init = scale_init
    self.reduction_axes = reduction_axes
    self.feature_axes = feature_axes
    self.axis_name = axis_name
    self.axis_index_groups = axis_index_groups
    self.use_fast_variance = use_fast_variance

  def __call__(self, x, *, mask: tp.Optional[jax.Array] = None):
    """Applies layer normalization on the input.

    Args:
      x: the inputs

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
    )

    return _normalize(
      x,
      mean,
      var,
      self.scale.value if self.scale else None,
      self.bias.value if self.bias else None,
      self.reduction_axes,
      self.feature_axes,
      self.dtype,
      self.epsilon,
    )


class RMSNorm(Module):
  """RMS Layer normalization (https://arxiv.org/abs/1910.07467).

  RMSNorm normalizes the activations of the layer for each given example in a
  batch independently, rather than across a batch like Batch Normalization.
  Unlike LayerNorm which re-centers the mean to be 0 and normalizes by the
  standard deviation of the activations, RMSNorm does not re-center at all
  and instead normalizes by the root mean square of the activations.

  Example usage::

    >>> from flax import nnx
    >>> import jax

    >>> x = jax.random.normal(jax.random.key(0), (5, 6))
    >>> layer = nnx.RMSNorm(num_features=6, rngs=nnx.Rngs(0))

    >>> nnx.state(layer)
    State({
      'scale': VariableState( # 6 (24 B)
        type=Param,
        value=Array([1., 1., 1., 1., 1., 1.], dtype=float32)
      )
    })

    >>> y = layer(x)

  Args:
    num_features: the number of input features.
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
        array being normalized is sharded across devices within a pmap.
    axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For
        example, ``[[0, 1], [2, 3]]`` would independently batch-normalize over
        the examples on the first two and last two devices. See ``jax.lax.psum``
        for more details.
    use_fast_variance: If true, use a faster, but less numerically stable,
        calculation for the variance.
    rngs: rng key.
  """

  def __init__(
    self,
    num_features: int,
    *,
    epsilon: float = 1e-6,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    use_scale: bool = True,
    scale_init: Initializer = initializers.ones,
    reduction_axes: Axes = -1,
    feature_axes: Axes = -1,
    axis_name: tp.Optional[str] = None,
    axis_index_groups: tp.Any = None,
    use_fast_variance: bool = True,
    rngs: rnglib.Rngs,
  ):
    feature_shape = (num_features,)

    self.scale: nnx.Param[jax.Array] | None
    if use_scale:
      key = rngs.params()
      self.scale = nnx.Param(scale_init(key, feature_shape, param_dtype))
    else:
      self.scale = None

    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.use_scale = use_scale
    self.scale_init = scale_init
    self.reduction_axes = reduction_axes
    self.feature_axes = feature_axes
    self.axis_name = axis_name
    self.axis_index_groups = axis_index_groups
    self.use_fast_variance = use_fast_variance

  def __call__(self, x, mask: tp.Optional[jax.Array] = None):
    """Applies layer normalization on the input.

    Args:
      x: the inputs

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
    )

    return _normalize(
      x,
      mean,
      var,
      self.scale.value if self.scale else None,
      None,
      self.reduction_axes,
      self.feature_axes,
      self.dtype,
      self.epsilon,
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
    LayerNorm is a special case of GroupNorm where ``num_groups=1``.

  Example usage::

    >>> from flax import nnx
    >>> import jax
    >>> import numpy as np
    ...
    >>> x = jax.random.normal(jax.random.key(0), (3, 4, 5, 6))
    >>> layer = nnx.GroupNorm(num_features=6, num_groups=3, rngs=nnx.Rngs(0))
    >>> nnx.state(layer)
    State({
      'bias': VariableState( # 6 (24 B)
        type=Param,
        value=Array([0., 0., 0., 0., 0., 0.], dtype=float32)
      ),
      'scale': VariableState( # 6 (24 B)
        type=Param,
        value=Array([1., 1., 1., 1., 1., 1.], dtype=float32)
      )
    })
    >>> y = layer(x)
    ...
    >>> y = nnx.GroupNorm(num_features=6, num_groups=1, rngs=nnx.Rngs(0))(x)
    >>> y2 = nnx.LayerNorm(num_features=6, reduction_axes=(1, 2, 3), rngs=nnx.Rngs(0))(x)
    >>> np.testing.assert_allclose(y, y2)

  Args:
    num_features: the number of input features/channels.
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
    rngs: rng key.
  """

  def __init__(
    self,
    num_features: int,
    num_groups: tp.Optional[int] = 32,
    group_size: tp.Optional[int] = None,
    *,
    epsilon: float = 1e-6,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    use_bias: bool = True,
    use_scale: bool = True,
    bias_init: Initializer = initializers.zeros_init(),
    scale_init: Initializer = initializers.ones_init(),
    reduction_axes: tp.Optional[Axes] = None,
    axis_name: tp.Optional[str] = None,
    axis_index_groups: tp.Any = None,
    use_fast_variance: bool = True,
    rngs: rnglib.Rngs,
  ):
    self.feature_axis = -1

    if (num_groups is None and group_size is None) or (
      num_groups is not None and group_size is not None
    ):
      raise ValueError(
        'Either `num_groups` or `group_size` should be '
        'specified. If `group_size` is to be specified, '
        'pass `num_groups=None` as argument to override '
        'the default `num_groups` value of 32.'
      )

    if group_size is not None:
      if num_features % group_size != 0:
        raise ValueError(
          'Number of features ({}) is not multiple of the '
          'group size ({}).'.format(num_features, group_size)
        )
      self.num_groups = num_features // group_size
      self.group_size = group_size
    else:
      if not isinstance(num_groups, int) or num_groups <= 0 or (
        num_features % num_groups != 0
      ):
        raise ValueError(
          'Number of groups ({}) does not divide the number'
          ' of channels ({}).'.format(num_groups, num_features)
        )
      self.num_groups = num_groups
      self.group_size = num_features // num_groups

    feature_shape = (num_features,)
    self.scale: nnx.Param[jax.Array] | None
    if use_scale:
      key = rngs.params()
      self.scale = nnx.Param(scale_init(key, feature_shape, param_dtype))
    else:
      self.scale = None

    self.bias: nnx.Param[jax.Array] | None
    if use_bias:
      key = rngs.params()
      self.bias = nnx.Param(bias_init(key, feature_shape, param_dtype))
    else:
      self.bias = None

    self.epsilon = epsilon
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.use_bias = use_bias
    self.use_scale = use_scale
    self.bias_init = bias_init
    self.scale_init = scale_init
    self.reduction_axes = reduction_axes
    self.axis_name = axis_name
    self.axis_index_groups = axis_index_groups
    self.use_fast_variance = use_fast_variance

  def __call__(self, x, *, mask: tp.Optional[jax.Array] = None):
    """Applies group normalization to the input (arxiv.org/abs/1803.08494).

    Args:
      x: the input of shape ``...self.num_features`` where ``self.num_features``
        is a channels dimension and ``...`` represents an arbitrary number of
        extra dimensions that can be used to accumulate statistics over. If no
        reduction axes have been specified then all additional dimensions ``...``
        will be used to accumulate statistics apart from the leading dimension
        which is assumed to represent the batch.
      mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
        the positions for which the mean and variance should be computed.

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    if self.reduction_axes is not None:
      reduction_axes = self.reduction_axes
    else:
      reduction_axes = list(range(1, x.ndim - 1)) + [-1]
    reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)

    group_shape = x.shape[:-1] + (self.num_groups, self.group_size)
    if mask is not None:
      mask = mask.reshape(mask.shape[:-1] + (self.num_groups, self.group_size))

    mean, var = _compute_stats(
      x.reshape(group_shape),
      list(reduction_axes[:-1]) + [-1],
      self.dtype,
      self.axis_name,
      self.axis_index_groups,
      use_fast_variance=self.use_fast_variance,
      mask=mask,
    )
    mean = jnp.repeat(mean, self.group_size, axis=1)
    var = jnp.repeat(var, self.group_size, axis=1)
    return _normalize(
      x,
      mean,
      var,
      self.scale.value if self.scale else None,
      self.bias.value if self.bias else None,
      reduction_axes[:-1],
      (self.feature_axis,),
      self.dtype,
      self.epsilon,
    )