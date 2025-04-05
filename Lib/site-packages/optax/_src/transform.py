# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Gradient transformations."""

import functools
from typing import NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics
from optax._src import utils
from optax.transforms import _accumulation
from optax.transforms import _adding


abs_sq = numerics.abs_sq


def _reject_complex(params):
  if any(jnp.iscomplexobj(x) for x in jax.tree.leaves(params)):
    raise ValueError('This transformation does not support complex parameters.')


class ScaleByRssState(NamedTuple):
  """State holding the sum of gradient squares to date."""

  sum_of_squares: base.Updates


def scale_by_rss(
    initial_accumulator_value: float = 0.1, eps: float = 1e-7
) -> base.GradientTransformation:
  """Rescale updates by the root of the sum of all squared gradients to date.

  See :func:`optax.adagrad` for more details.

  Args:
    initial_accumulator_value: Starting value for accumulators, must be >= 0.
    eps: A small floating point value to avoid zero denominator.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def init_fn(params):
    return ScaleByRssState(
        sum_of_squares=otu.tree_full_like(params, initial_accumulator_value)
    )

  def update_fn(updates, state, params=None):
    del params
    sum_of_squares = jax.tree.map(
        lambda g, t: abs_sq(g) + t, updates, state.sum_of_squares
    )
    inv_sqrt_g_square = jax.tree.map(
        lambda t: jnp.where(t > 0, jax.lax.rsqrt(t + eps), 0.0), sum_of_squares
    )
    updates = otu.tree_mul(inv_sqrt_g_square, updates)
    return updates, ScaleByRssState(sum_of_squares=sum_of_squares)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByRmsState(NamedTuple):
  """State for exponential root mean-squared (RMS)-normalized updates."""

  # Kept for backward compatibility, even though ScaleByRmsWithCountState
  # encompasses this state.
  nu: base.Updates


class ScaleByRmsWithCountState(NamedTuple):
  """State for exponential root mean-squared (RMS)-normalized updates."""

  count: chex.Array  # shape=(), dtype=jnp.int32.
  nu: base.Updates


def scale_by_rms(
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.0,
    eps_in_sqrt: bool = True,
    bias_correction: bool = False,
) -> base.GradientTransformation:
  r"""Rescale updates by the root of the exp. moving avg of the square.

  See :func:`optax.rmsprop` for more details.

  Args:
    decay: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    initial_scale: Initial value for second moment.
    eps_in_sqrt: Whether to add ``eps`` in the square root of the denominator or
      outside the square root.
    bias_correction: Whether to apply bias correction to the exponentially
      weighted average of squared grads.

  Returns:
    A :class:`optax.GradientTransformation` object.

  .. note::
    Using `scale_by_rms(decay=b2, eps_in_sqrt=False, bias_correction=True)`
    will match the behavior of `scale_by_adam(b1=0, b2=b2)`, while sparing the
    memory cost of storing the first moment.
  """

  def init_fn(params):
    nu = otu.tree_full_like(params, initial_scale)  # second moment
    if bias_correction:
      return ScaleByRmsWithCountState(count=jnp.zeros([], jnp.int32), nu=nu)
    else:
      return ScaleByRmsState(nu=nu)

  def update_fn(updates, state, params=None):
    del params
    nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, decay, 2)
    if bias_correction:
      count_inc = numerics.safe_increment(state.count)
      nu_hat = otu.tree_bias_correction(nu, decay, count_inc)
    else:
      count_inc = jnp.asarray(0)
      nu_hat = nu
    if eps_in_sqrt:
      scaling = jax.tree.map(lambda n: jax.lax.rsqrt(n + eps), nu_hat)
    else:
      scaling = jax.tree.map(lambda n: 1 / (jnp.sqrt(n) + eps), nu_hat)
    updates = jax.tree.map(lambda s, g: s * g, scaling, updates)
    if bias_correction:
      new_state = ScaleByRmsWithCountState(count=count_inc, nu=nu)
    else:
      new_state = ScaleByRmsState(nu=nu)
    return updates, new_state

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByRStdDevState(NamedTuple):
  """State for centered exponential moving average of squares of updates."""

  # Kept for backward compatibility, even though ScaleByRStdDevWithCountState
  # encompasses this state.
  mu: base.Updates
  nu: base.Updates


class ScaleByRStdDevWithCountState(NamedTuple):
  """State for centered exponential moving average of squares of updates."""

  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  nu: base.Updates


def scale_by_stddev(
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.0,
    eps_in_sqrt: bool = True,
    bias_correction: bool = False,
) -> base.GradientTransformation:
  """Rescale updates by the root of the centered exp. moving average of squares.

  See :func:`optax.rmsprop` for more details.

  Args:
    decay: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    initial_scale: Initial value for second moment.
    eps_in_sqrt: Whether to add ``eps`` in the square root of the denominator or
      outside the square root.
    bias_correction: Whether to apply bias correction to the first and second
      moment.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def init_fn(params):
    mu = otu.tree_zeros_like(params)  # First moment
    nu = otu.tree_full_like(params, initial_scale)  # second moment
    if bias_correction:
      return ScaleByRStdDevWithCountState(
          count=jnp.zeros([], jnp.int32), mu=mu, nu=nu
      )
    else:
      return ScaleByRStdDevState(mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = otu.tree_update_moment(updates, state.mu, decay, 1)
    nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, decay, 2)
    if bias_correction:
      count_inc = numerics.safe_increment(state.count)
      mu_hat = otu.tree_bias_correction(mu, decay, count_inc)
      nu_hat = otu.tree_bias_correction(nu, decay, count_inc)
    else:
      count_inc = jnp.asarray(0)
      mu_hat = mu
      nu_hat = nu

    if eps_in_sqrt:
      scaling = jax.tree.map(
          lambda m, n: jax.lax.rsqrt(n - abs_sq(m) + eps),
          mu_hat,
          nu_hat,
      )
    else:
      scaling = jax.tree.map(
          lambda m, n: 1 / (jnp.sqrt(n - abs_sq(m)) + eps),
          mu_hat,
          nu_hat,
      )
    updates = jax.tree.map(lambda s, g: s * g, scaling, updates)
    if bias_correction:
      new_state = ScaleByRStdDevWithCountState(count=count_inc, mu=mu, nu=nu)
    else:
      new_state = ScaleByRStdDevState(mu=mu, nu=nu)
    return updates, new_state

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByAdamState(NamedTuple):
  """State for the Adam algorithm."""

  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  nu: base.Updates


def scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:
  r"""Rescale updates according to the Adam algorithm.

  See :func:`optax.adam` for more details.

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    nesterov: Whether to use Nesterov momentum. The variant of Adam with
      Nesterov momentum is described in [Dozat 2016]

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = otu.tree_zeros_like(params, dtype=mu_dtype)  # First moment
    nu = otu.tree_zeros_like(params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = otu.tree_update_moment(updates, state.mu, b1, 1)
    nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_increment(state.count)
    if nesterov:
      mu_hat = jax.tree.map(
          lambda m, g: b1 * m + (1 - b1) * g,
          otu.tree_bias_correction(mu, b1, numerics.safe_increment(count_inc)),
          otu.tree_bias_correction(updates, b1, count_inc),
      )
    else:
      mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
    # Dozat 2016 https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
    # Algorithm 2 further multiplies Adam's standard nu_hat by b2. It is
    # unclear why. Other Nadam implementations also omit the extra b2 factor.
    nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
    updates = jax.tree.map(
        lambda m, v: None if m is None else m / (jnp.sqrt(v + eps_root) + eps),
        mu_hat,
        nu_hat,
        is_leaf=lambda x: x is None,
    )
    mu = otu.tree_cast(mu, mu_dtype)
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByAmsgradState(NamedTuple):
  """State for the AMSGrad algorithm."""

  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  nu: base.Updates
  nu_max: base.Updates


def scale_by_amsgrad(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
  """Rescale updates according to the AMSGrad algorithm.

  See :func:`optax.amsgrad` for more details.

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = otu.tree_zeros_like(params, dtype=mu_dtype)  # First moment
    nu = otu.tree_zeros_like(params)  # Second moment
    nu_max = otu.tree_zeros_like(params)
    return ScaleByAmsgradState(
        count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, nu_max=nu_max
    )

  def update_fn(updates, state, params=None):
    del params
    mu = otu.tree_update_moment(updates, state.mu, b1, 1)
    nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_increment(state.count)
    mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
    nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
    nu_max = jax.tree.map(jnp.maximum, state.nu_max, nu_hat)
    updates = jax.tree.map(
        lambda m, v: None if m is None else m / (jnp.sqrt(v + eps_root) + eps),
        mu_hat,
        nu_max,
        is_leaf=lambda x: x is None,
    )
    mu = otu.tree_cast(mu, mu_dtype)
    return updates, ScaleByAmsgradState(
        count=count_inc, mu=mu, nu=nu, nu_max=nu_max
    )

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_adamax(
    b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8
) -> base.GradientTransformation:
  """Rescale updates according to the Adamax algorithm.

  See :func:`optax.adamax` for more details.

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted maximum of grads.
    eps: Term added to the denominator to improve numerical stability.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def init_fn(params):
    mu = otu.tree_zeros_like(params)  # First moment
    nu = otu.tree_zeros_like(params)  # Infinite moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    count_inc = numerics.safe_increment(state.count)
    mu = otu.tree_update_moment(updates, state.mu, b1, 1)
    nu = otu.tree_update_infinity_moment(updates, state.nu, b2, eps)
    # Bias correction for mean. No bias correction needed for infinity moment.
    mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
    updates = jax.tree.map(lambda m, v: m / v, mu_hat, nu)
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByLionState(NamedTuple):
  """State for the Lion algorithm."""

  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates


def scale_by_lion(
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
  """Rescale updates according to the Lion algorithm.

  See :func:`optax.lion` for more details.

  Args:
    b1: Rate for combining the momentum and the current grad.
    b2: Decay rate for the exponentially weighted average of grads.
    mu_dtype: Optional `dtype` to be used for the momentum; if `None` then the
      `dtype is inferred from `params` and `updates`.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = otu.tree_zeros_like(params, dtype=mu_dtype)  # moment
    return ScaleByLionState(count=jnp.zeros([], jnp.int32), mu=mu)

  def update_fn(updates, state, params=None):
    del params
    updates_new = jax.tree.map(
        lambda g, m: jnp.sign((1.0 - b1) * g + b1 * m), updates, state.mu
    )
    mu = otu.tree_update_moment(updates, state.mu, b2, 1)
    mu = otu.tree_cast(mu, mu_dtype)
    count_inc = numerics.safe_increment(state.count)
    return updates_new, ScaleByLionState(count=count_inc, mu=mu)

  return base.GradientTransformation(init_fn, update_fn)


def scale(step_size: float) -> base.GradientTransformation:
  """Scale updates by some fixed scalar `step_size`.

  Args:
    step_size: A scalar corresponding to a fixed scaling factor for updates.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree.map(lambda g: step_size * g, updates)
    return updates, state

  return base.GradientTransformation(base.init_empty_state, update_fn)


def scale_by_param_block_norm(
    min_scale: float = 1e-3,
) -> base.GradientTransformation:
  """Scale updates for each param block by the norm of that block's parameters.

  A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
  (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.

  Args:
    min_scale: Minimum scaling factor.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    updates = jax.tree.map(
        lambda u, p: u * numerics.safe_norm(p, min_scale), updates, params
    )
    return updates, state

  return base.GradientTransformation(base.init_empty_state, update_fn)


def scale_by_param_block_rms(
    min_scale: float = 1e-3,
) -> base.GradientTransformation:
  """Scale updates by rms of the gradient for each param vector or matrix.

  A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
  (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.

  Args:
    min_scale: Minimum scaling factor.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    updates = jax.tree.map(
        lambda u, p: u * numerics.safe_root_mean_squares(p, min_scale),
        updates,
        params,
    )
    return updates, state

  return base.GradientTransformation(base.init_empty_state, update_fn)


class ScaleByAdaDeltaState(NamedTuple):
  """State for the rescaling by Adadelta algorithm."""

  e_g: base.Updates
  e_x: base.Updates


def scale_by_adadelta(
    rho: float = 0.9, eps: float = 1e-6
) -> base.GradientTransformation:
  """Rescale updates according to the Adadelta algorithm.

  See :func:`optax.adadelta` for more details.

  Args:
    rho: A coefficient used for computing a running average of squared
      gradients.
    eps: Term added to the denominator to improve numerical stability.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def init_fn(params):
    e_g = otu.tree_zeros_like(params)  # E[squared gradient]
    e_x = otu.tree_zeros_like(params)  # E[squared update]
    return ScaleByAdaDeltaState(e_g=e_g, e_x=e_x)

  def update_fn(updates, state, params=None):
    del params
    e_g = otu.tree_update_moment(updates, state.e_g, rho, 2)
    updates = jax.tree.map(
        lambda g, cur_e_g, prev_e_x: (
            jnp.sqrt(prev_e_x + eps) / jnp.sqrt(cur_e_g + eps)
        )
        * g,
        updates,
        e_g,
        state.e_x,
    )
    e_x = otu.tree_update_moment(updates, state.e_x, rho, 2)
    return updates, ScaleByAdaDeltaState(e_g=e_g, e_x=e_x)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByAdanState(NamedTuple):
  m: base.Updates
  v: base.Updates
  n: base.Updates
  g: base.Updates
  t: chex.Array


def scale_by_adan(
    b1: float = 0.98,
    b2: float = 0.92,
    b3: float = 0.99,
    eps: float = 1e-8,
    eps_root: float = 0.0,
) -> base.GradientTransformation:
  """Rescale updates according to the Adan algorithm.

  See :func:`optax.adan` for more details.

  Args:
    b1: Decay rate for the EWMA of gradients.
    b2: Decay rate for the EWMA of differences of gradients.
    b3: Decay rate for the EMWA of the algorithm's squared term.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.

  Returns:
    An :class:`optax.GradientTransformation` object.
  """

  def init_fn(params):
    return ScaleByAdanState(
        m=otu.tree_zeros_like(params),
        v=otu.tree_zeros_like(params),
        n=otu.tree_zeros_like(params),
        g=otu.tree_zeros_like(params),
        t=jnp.zeros([], jnp.int32),
    )

  def update_fn(updates, state, params=None):
    """Based on Algorithm 1 in https://arxiv.org/pdf/2208.06677v4#page=6."""
    del params
    g = updates

    diff = otu.tree_where(
        state.t == 0,
        otu.tree_zeros_like(g),
        otu.tree_sub(g, state.g),
    )
    m = otu.tree_update_moment(g, state.m, b1, 1)
    v = otu.tree_update_moment(diff, state.v, b2, 1)

    sq = otu.tree_add_scalar_mul(g, 1 - b2, diff)
    n = otu.tree_update_moment_per_elem_norm(sq, state.n, b3, 2)

    t = numerics.safe_increment(state.t)
    m_hat = otu.tree_bias_correction(m, b1, t)
    v_hat = otu.tree_bias_correction(v, b2, t)
    n_hat = otu.tree_bias_correction(n, b3, t)

    u = otu.tree_add_scalar_mul(m_hat, 1 - b2, v_hat)
    denom = jax.tree.map(lambda n_hat: jnp.sqrt(n_hat + eps_root) + eps, n_hat)
    u = otu.tree_div(u, denom)

    new_state = ScaleByAdanState(
        m=m,
        v=v,
        n=n,
        g=g,
        t=t,
    )

    return u, new_state

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByBeliefState(NamedTuple):
  """State for the rescaling by AdaBelief algorithm."""

  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  nu: base.Updates


def scale_by_belief(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-16,
    eps_root: float = 1e-16,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:
  """Rescale updates according to the AdaBelief algorithm.

  See :func:`optax.adabelief` for more details.

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of variance of grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the second moment of the prediction error to improve
      numerical stability. If backpropagating gradients through the gradient
      transformation (e.g. for meta-learning), this must be non-zero.
    nesterov: Whether to use Nesterov momentum.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def init_fn(params):
    mu = otu.tree_zeros_like(params)  # First moment
    s = otu.tree_zeros_like(params)  # Second Central moment
    return ScaleByBeliefState(count=jnp.zeros([], jnp.int32), mu=mu, nu=s)

  def update_fn(updates, state, params=None):
    del params
    mu = otu.tree_update_moment(updates, state.mu, b1, 1)
    prediction_error = otu.tree_sub(updates, mu)
    nu = otu.tree_update_moment_per_elem_norm(prediction_error, state.nu, b2, 2)
    nu = jax.tree.map(lambda v: v + eps_root, nu)
    count_inc = numerics.safe_increment(state.count)
    if nesterov:
      mu_hat = jax.tree.map(
          lambda m, g: b1 * m + (1 - b1) * g,
          otu.tree_bias_correction(
              mu, b1, numerics.safe_increment(count_inc)),
          otu.tree_bias_correction(updates, b1, count_inc))
    else:
      mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
    nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
    updates = jax.tree.map(
        lambda m, v: None if m is None else m / (jnp.sqrt(v) + eps),
        mu_hat,
        nu_hat,
        is_leaf=lambda x: x is None,
    )
    return updates, ScaleByBeliefState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_yogi(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-3,
    eps_root: float = 0.0,
    initial_accumulator_value: float = 1e-6,
) -> base.GradientTransformation:
  """Rescale updates according to the Yogi algorithm.

  See :func:`optax.yogi` for more details.

  Supports complex numbers, see
  https://gist.github.com/wdphy16/118aef6fb5f82c49790d7678cf87da29

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of variance of grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    initial_accumulator_value: The starting value for accumulators. Only
      positive values are allowed.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def init_fn(params):
    mu = otu.tree_full_like(params, initial_accumulator_value)  # First moment
    nu = otu.tree_full_like(params, initial_accumulator_value)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = otu.tree_update_moment(updates, state.mu, b1, 1)
    nu = jax.tree.map(
        lambda g, v: v - (1 - b2) * jnp.sign(v - abs_sq(g)) * abs_sq(g),
        updates,
        state.nu,
    )
    count_inc = numerics.safe_increment(state.count)
    mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
    nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
    updates = jax.tree.map(
        lambda m, v: None if m is None else m / (jnp.sqrt(v + eps_root) + eps),
        mu_hat,
        nu_hat,
        is_leaf=lambda x: x is None,
    )
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_radam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    threshold: float = 5.0,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:
  """Rescale updates according to the Rectified Adam algorithm.

  See :func:`optax.radam` for more details.

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    threshold: Threshold for variance tractability.
    nesterov: Whether to use Nesterov momentum.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  ro_inf = 2.0 / (1.0 - b2) - 1.0

  def _radam_update(ro, mu_hat, nu_hat):
    r = jnp.sqrt(
        (ro - 4.0)
        * (ro - 2.0)
        * ro_inf
        / ((ro_inf - 4.0) * (ro_inf - 2.0) * ro)
    )
    updates = jax.tree.map(
        lambda m, v: r.astype(m.dtype) * m / (jnp.sqrt(v + eps_root) + eps),
        mu_hat,
        nu_hat,
    )
    return updates

  def init_fn(params):
    mu = otu.tree_zeros_like(params)  # First moment
    nu = otu.tree_zeros_like(params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = otu.tree_update_moment(updates, state.mu, b1, 1)
    nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_increment(state.count)
    b2t = b2**count_inc
    ro = ro_inf - 2 * count_inc * b2t / (1 - b2t)
    if nesterov:
      mu_hat = jax.tree.map(
          lambda m, g: b1 * m + (1 - b1) * g,
          otu.tree_bias_correction(mu, b1, numerics.safe_increment(count_inc)),
          otu.tree_bias_correction(updates, b1, count_inc),
      )
    else:
      mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
    nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
    updates = jax.tree.map(
        lambda t, f: jnp.where(ro >= threshold, t, f),
        _radam_update(ro, mu_hat, nu_hat),
        mu_hat,
    )
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByRpropState(NamedTuple):
  step_sizes: base.Updates
  prev_updates: base.Updates


def scale_by_rprop(
    learning_rate: float,
    eta_minus: float = 0.5,
    eta_plus: float = 1.2,
    min_step_size: float = 1e-6,
    max_step_size: float = 50.0,
) -> base.GradientTransformation:
  """Scale with the Rprop optimizer.

  See :func:`optax.rprop` for more details.

  Args:
    learning_rate: The initial step size.
    eta_minus: Multiplicative factor for decreasing step size. This is applied
      when the gradient changes sign from one step to the next.
    eta_plus: Multiplicative factor for increasing step size. This is applied
      when the gradient has the same sign from one step to the next.
    min_step_size: Minimum allowed step size. Smaller steps will be clipped to
      this value.
    max_step_size: Maximum allowed step size. Larger steps will be clipped to
      this value.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.
  """

  def init_fn(params):
    step_sizes = otu.tree_full_like(params, learning_rate)
    prev_updates = otu.tree_zeros_like(params)
    return ScaleByRpropState(step_sizes, prev_updates)

  def update_fn(updates, state, params=None):
    del params
    sign = jax.tree.map(
        lambda g, prev_g: g * prev_g, updates, state.prev_updates
    )
    step_sizes = jax.tree.map(
        lambda s, step_size: jnp.where(
            s == 0,
            step_size,
            jnp.clip(
                step_size * jnp.where(s > 0, eta_plus, eta_minus),
                min=min_step_size,
                max=max_step_size,
            ),
        ),
        sign,
        state.step_sizes,
    )
    prev_updates = jax.tree.map(
        lambda s, g, step_size: jnp.where(
            s < 0, jnp.zeros_like(g), step_size * jnp.sign(g)
        ),
        sign,
        updates,
        step_sizes,
    )
    updates = jax.tree.map(
        lambda s, g, prev_g: jnp.where(s < 0, jnp.zeros_like(prev_g), prev_g),
        sign,
        prev_updates,
        state.prev_updates,
    )
    return updates, ScaleByRpropState(step_sizes, prev_updates)

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_sign() -> base.GradientTransformation:
  """Compute the signs of the gradient elements.

  Returns:
    An optax.GradientTransformation that contains the signs of the input
    gradient.
  """

  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree.map(jnp.sign, updates)
    return updates, state

  return base.GradientTransformation(base.init_empty_state, update_fn)


class ScaleByScheduleState(NamedTuple):
  """Maintains count for scale scheduling."""

  count: chex.Array  # shape=(), dtype=jnp.int32


def scale_by_learning_rate(
    learning_rate: base.ScalarOrSchedule,
    *,
    flip_sign: bool = True,
) -> base.GradientTransformation:
  """Scale by the (negative) learning rate (either as scalar or as schedule).

  Args:
    learning_rate: Can either be a scalar or a schedule (i.e. a callable that
      maps an (int) step to a float).
    flip_sign: When set to True (the default) this corresponds to scaling by the
      negative learning rate.

  Returns:
    An optax.GradientTransformation that corresponds to multiplying the gradient
    with `-learning_rate` (if flip_sign is True) or with `learning_rate` (if
    flip_sign is False).
  """
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return scale_by_schedule(lambda count: m * learning_rate(count))
  return scale(m * learning_rate)


def scale_by_schedule(
    step_size_fn: base.Schedule,
) -> base.GradientTransformation:
  """Scale updates using a custom schedule for the `step_size`.

  Args:
    step_size_fn: A function that takes an update count as input and proposes
      the step_size to multiply the updates by.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def init_fn(params):
    del params
    return ScaleByScheduleState(count=jnp.zeros([], jnp.int32))

  def update_fn(updates, state, params=None):
    del params
    step_size = step_size_fn(state.count)
    updates = jax.tree.map(
        lambda g: jnp.array(step_size, dtype=g.dtype) * g, updates
    )
    return updates, ScaleByScheduleState(
        count=numerics.safe_increment(state.count)
    )

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_trust_ratio(
    min_norm: float = 0.0,
    trust_coefficient: float = 1.0,
    eps: float = 0.0,
) -> base.GradientTransformation:
  """Scale updates by `trust ratio`.

  Used in :func:`optax.fromage`, :func:`optax.lars`, :func:`optax.lamb`.

  Args:
    min_norm: Minimum norm for params and gradient norms; by default is zero.
    trust_coefficient: A multiplier for the trust ratio.
    eps: Additive constant added to the denominator for numerical stability.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)

    def _scale_update(update, param):

      # Clip norms to minimum value, by default no clipping.
      param_norm = numerics.safe_norm(param, min_norm)
      update_norm = numerics.safe_norm(update, min_norm)
      trust_ratio = trust_coefficient * param_norm / (update_norm + eps)

      # If no minimum norm clipping is used
      # Set trust_ratio to 1 in case where parameters would never be updated.
      zero_norm = jnp.logical_or(param_norm == 0.0, update_norm == 0.0)
      safe_trust_ratio = jnp.where(
          zero_norm, jnp.array(1.0, dtype=param.dtype), trust_ratio
      )

      return update * safe_trust_ratio

    updates = jax.tree.map(_scale_update, updates, params)
    return updates, state

  return base.GradientTransformation(base.init_empty_state, update_fn)


class ApplyEvery(NamedTuple):
  """Contains a counter and a gradient accumulator."""

  count: chex.Array
  grad_acc: base.Updates


def apply_every(k: int = 1) -> base.GradientTransformation:
  """Accumulate gradients and apply them every k steps.

  Note that if this transformation is part of a chain, the states of the other
  transformations will still be updated at every step. In particular, using
  `apply_every` with a batch size of N/2 and k=2 is not necessarily equivalent
  to not using `apply_every` with a batch size of N. If this equivalence is
  important for you, consider using the `optax.MultiSteps`.

  Args:
    k: Emit non-zero gradients every k steps, otherwise accumulate them.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def init_fn(params):
    grad_acc = otu.tree_zeros_like(params)
    return ApplyEvery(count=jnp.zeros([], jnp.int32), grad_acc=grad_acc)

  def update_fn(updates, state, params=None):
    del params
    c = state.count % k
    acc = c != 0
    grad_acc = jax.tree.map(lambda g, ga: acc * ga + g, updates, state.grad_acc)
    emit = c == (k - 1)
    updates = jax.tree.map(lambda ga: emit * ga, grad_acc)
    count_inc = numerics.safe_increment(state.count)
    return updates, ApplyEvery(count=count_inc % k, grad_acc=grad_acc)

  return base.GradientTransformation(init_fn, update_fn)


def _subtract_mean(g):
  if len(g.shape) > 1:
    return g - g.mean(tuple(range(1, len(g.shape))), keepdims=True)
  else:
    return g


CentralState = base.EmptyState


def centralize() -> base.GradientTransformation:
  """Centralize gradients.

  Returns:
    A :class:`optax.GradientTransformation` object.

  References:
    Yong et al, `Gradient Centralization: A New Optimization Technique for Deep
    Neural Networks <https://arxiv.org/abs/2004.01461>`_, 2020.
  """

  def init_fn(params):
    del params
    return CentralState()

  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree.map(_subtract_mean, updates)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


class ScaleBySM3State(NamedTuple):
  """State for the SM3 algorithm."""

  mu: base.Updates
  nu: base.Updates


def scale_by_sm3(
    b1: float = 0.9, b2: float = 1.0, eps: float = 1e-8
) -> base.GradientTransformation:
  """Scale updates by `sm3`.

  See :func:`optax.sm3` for more details.

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def zeros_for_dim(p):
    return [jnp.zeros([s], dtype=p.dtype) for s in p.shape]

  def init_fn(params):
    _reject_complex(params)
    mu = jax.tree.map(zeros_for_dim, params)
    nu = otu.tree_zeros_like(params)
    return ScaleBySM3State(mu, nu)

  def _expanded_shape(shape, axis):
    # Replaces a `shape` of [M, N, K] with 1 in all dimensions except for i.
    # For eg: i = 1 returns [1, N, 1].
    rank = len(shape)
    return [1] * axis + [shape[axis]] + [1] * (rank - axis - 1)

  def _new_accum(g, v):
    coeffs = ((1.0 - b2) if b2 != 1.0 else 1.0, b2)
    if g.ndim < 2:
      return coeffs[0] * g**2 + coeffs[1] * v[0]
    else:
      return coeffs[0] * g**2 + coeffs[1] * functools.reduce(jnp.minimum, v)

  def _new_mu(g, i):
    if g.ndim < 2:
      return g
    else:
      return jnp.max(g, axis=other_axes(i, g.ndim))

  def other_axes(idx, ndim):
    return list(range(idx)) + list(range(idx + 1, ndim))

  def update_fn(updates, state, params=None):
    del params
    mu = jax.tree.map(
        lambda g, v: [  # pylint:disable=g-long-lambda
            jnp.reshape(v[i], _expanded_shape(g.shape, i))
            for i in range(g.ndim)
        ],
        updates,
        state.mu,
    )
    accum = jax.tree.map(_new_accum, updates, mu)
    accum_inv_sqrt = jax.tree.map(
        lambda t: jnp.where(t > 0, jax.lax.rsqrt(t + eps), 0.0), accum
    )
    up = jax.tree.map(lambda g, a: g * a, updates, accum_inv_sqrt)
    nu = otu.tree_update_moment(up, state.nu, b1, 1)
    mu = jax.tree.map(lambda g: [_new_mu(g, i) for i in range(g.ndim)], accum)

    return nu, ScaleBySM3State(mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByNovogradState(NamedTuple):
  """State for Novograd."""

  count: chex.Array
  mu: base.Updates
  nu: base.Updates


def scale_by_novograd(
    b1: float = 0.9,
    b2: float = 0.25,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
  """Computes NovoGrad updates.

  See :func:`optax.novograd` for more details.

  Args:
    b1: A decay rate for the exponentially weighted average of grads.
    b2: A decay rate for the exponentially weighted average of squared grads.
    eps: A term added to the denominator to improve numerical stability.
    eps_root: A term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    weight_decay: A scalar weight decay rate.
    mu_dtype: An optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = otu.tree_zeros_like(params, dtype=mu_dtype)
    nu = jax.tree.map(lambda p: jnp.asarray(0.0, dtype=p.dtype), params)
    return ScaleByNovogradState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def nu_addition(grads):
    return jnp.linalg.norm(grads) ** 2

  def mu_addition(grads, params, nu):
    return grads / (jnp.sqrt(nu + eps_root) + eps) + weight_decay * params

  def init_nu(grads, nu):
    return jax.tree.map(lambda g, n: nu_addition(g).astype(n.dtype), grads, nu)

  def update_nu(grads, nu):
    updates = jax.tree.map(nu_addition, grads)
    return otu.tree_update_moment(updates, nu, b2, 1)

  def init_mu(grads, params, mu, nu):
    del mu
    return jax.tree.map(mu_addition, grads, params, nu)

  def update_mu(grads, params, mu, nu):
    updates = jax.tree.map(mu_addition, grads, params, nu)
    return jax.tree.map(
        lambda m, u: None if m is None else b1 * m + u,
        mu,
        updates,
        is_leaf=lambda x: x is None,
    )

  def update_fn(updates, state, params):
    count_inc = numerics.safe_increment(state.count)

    nu = jax.lax.cond(count_inc == 1, init_nu, update_nu, updates, state.nu)
    mu = jax.lax.cond(
        count_inc == 1, init_mu, update_mu, updates, params, state.mu, nu
    )

    mu = otu.tree_cast(mu, mu_dtype)
    updates = mu
    return updates, ScaleByNovogradState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByOptimisticGradientState(NamedTuple):
  is_initial_step: chex.Array
  previous_gradient: base.Updates


def scale_by_optimistic_gradient(
    alpha: float = 1.0, beta: float = 1.0
) -> base.GradientTransformation:
  """Compute generalized optimistic gradients.

  See :func:`optax.optimistic_adam`, :func:`optax.optimistic_gradient_descent`
  for more details.

  Args:
    alpha: Coefficient for generalized optimistic gradient descent.
    beta: Coefficient for negative momentum.

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def init_fn(params):
    return ScaleByOptimisticGradientState(
        is_initial_step=jnp.array(True),
        previous_gradient=otu.tree_zeros_like(params),
    )

  def update_fn(updates, state, params=None):
    del params

    def f(grad, prev_grad):
      # At the initial step, the previous gradient doesn't exist, so we use the
      # current gradient instead.
      # https://github.com/google-deepmind/optax/issues/1082
      prev_grad = jnp.where(state.is_initial_step, grad, prev_grad)
      return (alpha + beta) * grad - beta * prev_grad

    new_updates = jax.tree.map(f, updates, state.previous_gradient)

    new_state = ScaleByOptimisticGradientState(
        is_initial_step=jnp.array(False),
        previous_gradient=updates,
    )

    return new_updates, new_state

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByDistanceOverGradientsState(NamedTuple):
  """State for scale_by_distance_over_gradients."""

  max_dist: base.OptState
  grad_sum_of_squares: base.OptState
  init_params: base.OptState


def scale_by_distance_over_gradients(
    reps_rel=1e-6, eps=1e-8, param_dtype=jnp.float32, global_scale=1.0
) -> base.GradientTransformation:
  """Distance-over-gradients learning rate-free optimizer.

  This implementation stores a single copy of the model parameters, plus two
  scalars per parameter array. It is equivalent to "Layer-wise DoG" (LDoG)
  in the paper.

  The authors recommend using model averaging with this optimizer.

  Args:
    reps_rel: Used to compute initial learning rate. Recommended values are 1e-4
      for models using batch norm, 1e-6 otherwise.
    eps: Small loading term to avoid divide-by-zero errors.
    param_dtype: dtype for storing initial parameters.
    global_scale: Global scale factor, typically 1.0 or -1.0

  Returns:
    A :class:`optax.GradientTransformation` object.

  References:
    Ivgi et al, `DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size
    Schedule <https://arxiv.org/pdf/2302.12022.pdf>`_, 2023
  """

  def _l2(x, y=0.0):
    return jnp.sqrt(jnp.square(x - y).sum())

  def init_fn(params):
    return ScaleByDistanceOverGradientsState(
        # Initial distance (needed to prevent zero step sizes).
        jax.tree.map(lambda x: reps_rel * (1 + _l2(x)), params),
        # Initial gradient sum-of-squares.
        jax.tree.map(lambda x: jnp.zeros(1), params),
        # Initial params, cast to preferred precision.
        otu.tree_cast(params, param_dtype),
    )

  def update_fn(updates, state: ScaleByDistanceOverGradientsState, params):
    # update max distance
    max_dist = jax.tree.map(
        lambda d, x, y: jnp.maximum(d, _l2(x, y)),
        state.max_dist,
        params,
        state.init_params,
    )

    # update gradient sum-of-squares
    g_sos = jax.tree.map(
        lambda x, y: x + jnp.square(y).sum(), state.grad_sum_of_squares, updates
    )

    def _tx(g, d, g_sos):
      """Apply the transformation."""
      eta = global_scale * (d / jnp.sqrt(g_sos + eps))
      return eta * g

    updates = jax.tree.map(_tx, max_dist, g_sos, updates)

    # new state
    state = ScaleByDistanceOverGradientsState(
        max_dist, g_sos, state.init_params
    )

    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_polyak(
    f_min: float = 0.0,
    max_learning_rate: float = 1.0,
    eps: float = 0.0,
) -> base.GradientTransformationExtraArgs:
  r"""Scales the update by Polyak's step-size.

  See :func:`optax.polyak_sgd` for more details.

  Args:
    f_min: a lower bound on the objective function (defaults to 0). Corresponds
      to :math:`f^\star` in the formula above.
    max_learning_rate: a maximum step size to use (defaults to 1).
    eps: a value to add in the denominator of the update (defaults to 0).

  Returns:
    A :class:`optax.GradientTransformationExtraArgs`, where the ``update``
    function takes an additional keyword argument ``value`` containing the
    current value of the objective function.
  """

  def update_fn(
      updates: base.Updates,
      state: base.EmptyState,
      params: Optional[base.Params] = None,
      *,
      value: float,
      **extra_args,
  ) -> tuple[base.Updates, base.EmptyState]:
    """Scales the update by the Polyak step-size.

    Args:
      updates: the updates to be scaled.
      state: the state of the transformation.
      params: the parameters of the model.
      value: the value of the loss function.
      **extra_args: additional keyword arguments. They are ignored by this
        transformation.

    Returns:
      The scaled updates and the state of the transformation.
    """
    del params, extra_args
    grad_sq_norm = otu.tree_l2_norm(updates, squared=True)
    # avoid division by zero
    step = jnp.where(
        grad_sq_norm + eps <= jnp.finfo(float).eps,
        jnp.array(0.0),
        jnp.minimum((value - f_min) / (grad_sq_norm + eps), max_learning_rate),
    )
    updates = otu.tree_scalar_mul(step, updates)
    return updates, state

  return base.GradientTransformationExtraArgs(base.init_empty_state, update_fn)


class ScaleByLBFGSState(NamedTuple):
  """State for LBFGS solver.

  Attributes:
    count: iteration of the algorithm.
    params: current parameters.
    updates: current updates.
    diff_params_memory: represents a list of past parameters' differences up to
      some predetermined ``memory_size`` fixed in :func:`optax.scale_by_lbfgs`.
    diff_updates_memory: represents a list of past gradients/updates'
      differences up to some predetermined ``memory_size`` fixed in
      :func:`optax.scale_by_lbfgs`.
    weights_memory: list of past weights multiplying the rank one matrices
      defining the inverse Hessian approximation, see
      :func:`optax.scale_by_lbfgs` for more details.
  """

  count: chex.Numeric
  params: base.Params
  updates: base.Params
  diff_params_memory: chex.ArrayTree
  diff_updates_memory: chex.ArrayTree
  weights_memory: chex.Array


def _precondition_by_lbfgs(
    updates: base.Updates,
    diff_params_memory: chex.ArrayTree,
    diff_updates_memory: chex.ArrayTree,
    weights_memory: chex.Array,
    identity_scale: Union[float, jax.Array],
    memory_idx: Union[int, jax.Array],
) -> base.Updates:
  r"""Multiplies updates by an approximation of the inverse Hessian.

  The approximation of the inverse Hessian is parameterized
  by rank one matrices defined by past differences of parameters and
  gradients/updates. See :func:`optax.scale_by_lbfgs` for the mathematical
  formulation.

  Args:
    updates: updates (gradients a priori) to be multiplied by approximate
      inverse Hessian.
    diff_params_memory: represents a list of past parameters' differences.
    diff_updates_memory: represents a list of past gradients/updates'
      differences.
    weights_memory: list of past weights multiplying the rank one matrices
      defining the inverse Hessian approximation, see
      :func:`optax.scale_by_lbfgs` for more details.
    identity_scale: scaling factor multiplying an identity matrix used as an
      initial approximation of the inverse Hessian (:math:`\gamma` in the
      formulation given in :func:`optax.scale_by_lbfgs`).
    memory_idx: current index between ``0`` and ``memory_size-1`` in the memory
      buffer.

  Returns:
    Preconditioned updates, that is, updates multiplied by an approximation of
    the inverse Hessian defined by past parameters and gradients/updates
    differences up to some predetermined memory buffer size.

  Reference:
    Algorithm 7.4 (page 178) in Nocedal et al, `Numerical Optimization
    <https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf>_`
    , 1999
  """
  rhos = weights_memory
  memory_size = weights_memory.shape[0]
  indices = (memory_idx + jnp.arange(memory_size)) % memory_size

  def right_product(vec, idx):
    dwi, dui = jax.tree.map(
        lambda x: x[idx], (diff_params_memory, diff_updates_memory)
    )
    alpha = rhos[idx] * otu.tree_vdot(dwi, vec)
    vec = otu.tree_add_scalar_mul(vec, -alpha, dui)
    return vec, alpha

  precond_updates, alphas = jax.lax.scan(
      right_product, updates, indices, reverse=True
  )

  precond_updates = otu.tree_scalar_mul(identity_scale, precond_updates)

  def left_product(vec, idx_alpha):
    idx, alpha = idx_alpha
    dwi, dui = jax.tree.map(
        lambda x: x[idx], (diff_params_memory, diff_updates_memory)
    )
    beta = rhos[idx] * otu.tree_vdot(dui, vec)
    vec = otu.tree_add_scalar_mul(vec, alpha - beta, dwi)
    return vec, beta

  precond_updates, _ = jax.lax.scan(
      left_product, precond_updates, (indices, alphas)
  )

  return precond_updates


def scale_by_lbfgs(
    memory_size: int = 10,
    scale_init_precond: bool = True,
) -> base.GradientTransformation:
  r"""Scales updates by L-BFGS.

  L-BFGS is a quasi-Newton method that multiplies the update (gradient)
  with an approximation of the inverse Hessian. This algorithm does not need
  access to the Hessian, as this approximation is constructed from the gradient
  evaluations seen during optimization. L-BFGS is a limited-memory variant of
  the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm. The BFGS algorithm
  requires storing a matrix of size :math:`p \times p` with :math:`p` the
  dimension of the parameters.
  The limited variant circuments this issue by computing the approximation of
  the inverse using only :math:`m` (``memory_size``) past differences of
  parameters/gradients. Namely, the approximation of the Hessian inverse is
  denoted :math:`P_k = P_{k, k}`, where

  .. math::

    \begin{align*}
      P_{k, j+1} & = V_j^\top P_{k, j} V_j + \rho_j \delta w_j \delta w_j^\top
      \quad \text{for} \ j \in \{k-m, \ldots, k-1\}\\
      P_{k, k-m} & = \gamma_k I \\
      V_k & = I - \rho_k \delta u_k \delta w_k^\top \\
      \rho_k & = 1/(\delta u_k^\top \delta w_k) \\
      \delta w_k & = w_{k+1} - w_k \\
      \delta u_k & = u_{k+1} - u_k \\
      \gamma_k & =
        \begin{cases}
          (\delta w_{k-1}^\top \delta u_{k-1}) /
          (\delta u_{k-1}^\top \delta u_{k-1})
          & \text{if} \ \texttt{scale\_init\_hess} \\
          1 & \text{otherwise}
        \end{cases},
    \end{align*}

  for
  :math:`u_k` the gradients/updates at iteration :math:`k`,
  :math:`w_k` the parameters at iteration :math:`k`.

  The formula for updating :math:`P_k` is obtained by computing the optimal
  preconditioning matrix subject to some secant condition, see references
  for more details. Computing :math:`P_k u_k` can be done by a sequence of 
  vector operations using past differences of parameters and gradients stored in
  a memory bufffer.

  The present function just outputs the LBFGS direction :math:`P_k u_k`.
  It can be chained with a linesearch ensuring sufficient decrease and low
  curvature, such as a zoom linesearch. The linesearch computes a stepsize
  :math:`\eta_k`, such that the updated parameters
  (using :func:`optax.apply_updates`) take the form
  :math:`w_{k+1} = w_k - \eta_k P_k u_k`.

  Args:
    memory_size: number of past parameters, gradients/updates to keep in memory
      to approximate the Hessian inverse.
    scale_init_precond: whether to use a scaled identity as the initial
      preconditioner, see formula of :math:`\gamma_k` above.

  Returns:
    A :class:`optax.GradientTransformation` object.

  References:
    Algorithms 7.4, 7.5 (page 199) of Nocedal et al, `Numerical Optimization
    <https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf>`__
    , 1999

    Liu et al., `On the limited memory BFGS method for large scale optimization
    <https://users.iems.northwestern.edu/~nocedal/PDFfiles/limited-memory.pdf>`_
    , 1989.

  .. note::
    We initialize the scaling of the identity as a capped reciprocal of the
    gradient norm. This avoids wasting linesearch iterations for the first step
    by taking into account the magnitude of the gradients. In other words, we
    constrain the trust-region of the first step to an Euclidean ball of radius
    1 at the first iteration. The choice of :math:`\gamma_0` is not detailed in
    the references above, so this is a heuristic choice.
  """
  if memory_size < 1:
    raise ValueError('memory_size must be >= 1')

  def init_fn(params: base.Params) -> ScaleByLBFGSState:
    # diff_params_memory and diff_updates_memory represent tuple/list of trees
    # Since we cannot access the element of a tuple using a traced index such
    # as memory_idx below, we instantiate them by stacking leaves.
    # We can then access the ith element of the underlying tuple/list
    # represented by e.g. diff_params_memory through the ith stacked
    # element in the leaves, see update_fn below for practical examples.
    stacked_zero_params = jax.tree.map(
        lambda leaf: jnp.zeros((memory_size,) + leaf.shape, dtype=leaf.dtype),
        params,
    )
    return ScaleByLBFGSState(
        count=jnp.asarray(0, dtype=jnp.int32),
        params=otu.tree_zeros_like(params),
        updates=otu.tree_zeros_like(params),
        diff_params_memory=stacked_zero_params,
        diff_updates_memory=stacked_zero_params,
        weights_memory=jnp.zeros(memory_size),
    )

  def update_fn(
      updates: base.Updates, state: ScaleByLBFGSState, params: base.Params
  ) -> tuple[base.Updates, ScaleByLBFGSState]:
    # Essentially memory_idx is the iteration k (modulo the memory size)
    # and prev_memory_idx is k-1 (modulo the memory size).
    memory_idx = state.count % memory_size
    prev_memory_idx = (state.count - 1) % memory_size

    # We first update the preconditioner and then preconditon the updates.
    # That way, we can chain this function with a linesearch to update the
    # preconditioner only once a valid stepsize has been found by the linesearch
    # and the step has been done.

    # 1. Updates the memory buffers given fresh params and gradients/updates
    diff_params = otu.tree_sub(params, state.params)
    diff_updates = otu.tree_sub(updates, state.updates)
    vdot_diff_params_updates = otu.tree_vdot(diff_updates, diff_params)
    weight = jnp.where(
        vdot_diff_params_updates == 0.0, 0.0, 1.0 / vdot_diff_params_updates
    )
    # params_diff, updates_diff, weight depend on differences of parameters
    # that are not defined at the first iteration. Hence we keep them at 0 if
    # state.count = 0.
    diff_params, diff_updates, weight = jax.tree.map(
        lambda x: jnp.where(state.count > 0, x, jnp.zeros_like(x)),
        (diff_params, diff_updates, weight),
    )
    diff_params_memory, diff_updates_memory, weights_memory = jax.tree.map(
        lambda x, y: x.at[prev_memory_idx].set(y),
        (
            state.diff_params_memory,
            state.diff_updates_memory,
            state.weights_memory,
        ),
        (diff_params, diff_updates, weight),
    )

    # 2. Compute scaling of the identity matrix (gamma_k in the formula above)
    # used to initialize the approximation of the inverse through the memory
    # buffer.
    if scale_init_precond:
      numerator = otu.tree_vdot(diff_updates, diff_params)
      denominator = otu.tree_l2_norm(diff_updates, squared=True)
      identity_scale = jnp.where(
          denominator > 0.0, numerator / denominator, 1.0
      )
      # For the very first step of the algorithm, we consider scaling by a
      # capped reciprocal of the gradient norm, see note in the docstring.
      capped_inv_norm = jnp.minimum(1.0, 1.0/otu.tree_l2_norm(updates))
      identity_scale = jnp.where(
          state.count > 0, identity_scale, capped_inv_norm
      )
    else:
      identity_scale = 1.0

    # 3. Computes the matrix vector product P_k u_k by decomposing P_k in the
    # associated rank one matrices and perform the associated vector operations
    precond_updates = _precondition_by_lbfgs(
        updates,
        diff_params_memory,
        diff_updates_memory,
        weights_memory,
        identity_scale,
        memory_idx,
    )
    return precond_updates, ScaleByLBFGSState(
        count=numerics.safe_increment(state.count),
        params=params,
        updates=updates,
        diff_params_memory=diff_params_memory,
        diff_updates_memory=diff_updates_memory,
        weights_memory=weights_memory,
    )

  return base.GradientTransformation(init_fn, update_fn)


def normalize_by_update_norm(
    scale_factor: float = 1.0, eps: float = 1e-6
) -> base.GradientTransformation:
  """Scale by the inverse of the update norm.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.normalize_by_update_norm(scale_factor=-1.0)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function:', f(params))
    Objective function: 14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 7.52E+00
    Objective function: 3.03E+00
    Objective function: 5.50E-01
    Objective function: 6.67E-02
    Objective function: 5.50E-01

  Args:
    scale_factor: factor by which the update will be multiplied (defaults to 1).
    eps: jitter term to avoid dividing by 0

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  def update_fn(
      updates: base.Updates,
      state: base.EmptyState,
      params: Optional[base.Params] = None,
  ) -> tuple[base.Updates, base.EmptyState]:
    del params
    g_norm = (otu.tree_l2_norm(updates) + eps) / scale_factor
    updates = jax.tree.map(lambda g: g / g_norm, updates)
    return updates, state

  return base.GradientTransformation(base.init_empty_state, update_fn)


### Legacy symbols to be removed. ###


@functools.partial(
    chex.warn_deprecated_function, replacement='optax.tree_utils.tree_cast'
)
def cast_tree(
    tree: chex.ArrayTree, dtype: Optional[chex.ArrayDType]
) -> chex.ArrayTree:
  return otu.tree_cast(tree, dtype)


trace = _accumulation.trace
TraceState = _accumulation.TraceState
ema = _accumulation.ema
EmaState = _accumulation.EmaState
add_noise = _adding.add_noise
AddNoiseState = _adding.AddNoiseState
add_decayed_weights = _adding.add_decayed_weights
AddDecayedWeightsState = base.EmptyState
ScaleState = base.EmptyState
ScaleByTrustRatioState = base.EmptyState
