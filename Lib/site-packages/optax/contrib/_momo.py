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
"""MoMo.

Implementation of
"MoMo: Momentum Models for Adaptive Learning Rates"
(https://arxiv.org/abs/2305.07583) by Fabian Schaipp, Ruben Ohana,
Michael Eickenberg, Aaron Defazio and Robert M. Gower.
"""

from typing import NamedTuple, Optional

import chex
import jax
from jax import lax
import jax.numpy as jnp
from optax._src import base
from optax._src import numerics
import optax.tree_utils as otu


class MomoState(NamedTuple):
  """State of the `GradientTransformation` returned by `momo`."""

  exp_avg: base.Updates
  barf: chex.Array  # shape=(), dtype=jnp.float32.
  gamma: chex.Array  # shape=(), dtype=jnp.float32.
  lb: chex.Array  # shape=(), dtype=jnp.float32.
  count: chex.Array  # shape=(), dtype=jnp.int32.


def momo(
    learning_rate: base.ScalarOrSchedule = 1.0,
    beta: float = 0.9,
    lower_bound: float = 0.0,
    weight_decay: float = 0.0,
    adapt_lower_bound: bool = False,
) -> base.GradientTransformationExtraArgs:
  """Adaptive Learning Rates for SGD with momentum.

  MoMo typically needs less tuning for value of ``learning_rate``,
  by exploting the fact that a lower bound of the loss (or the optimal value) is
  known. For most tasks, zero is a lower bound and an accurate estimate of the
  final loss.

  MoMo performs SGD with momentum with a Polyak-type learning rate. The
  effective step size is ``min(learning_rate, <adaptive term>)``, where the
  adaptive term is computed on the fly.

  Note that one needs to pass the latest (batch) loss value to the update
  function using the keyword argument ``value``.

  Args:
    learning_rate: User-specified learning rate. Recommended to be chosen rather
      large, by default 1.0.
    beta: Momentum coefficient (for EMA).
    lower_bound: Lower bound of the loss. Zero should be a good choice for many
      tasks.
    weight_decay: Weight-decay parameter.
    adapt_lower_bound: If no good guess for the lower bound is available, set
      this to true, in order to estimate the lower bound on the fly (see the
      paper for details).

  Returns:
    A :class:`optax.GradientTransformation` object.

  Examples:
    >>> from optax import contrib
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = contrib.momo()
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  value, grad = jax.value_and_grad(f)(params)
    ...  params, opt_state = solver.update(grad, opt_state, params, value=value)
    ...  print('Objective function: ', f(params))
    Objective function:  3.5
    Objective function:  0.0
    Objective function:  0.0
    Objective function:  0.0
    Objective function:  0.0

  References:
    Schaipp et al., `MoMo: Momentum Models for Adaptive Learning Rates
    <https://arxiv.org/abs/2305.07583>`_, 2023

  .. versionadded:: 0.2.3
  """

  def init_fn(params: base.Params) -> MomoState:
    # Define state parameters with the lowest dtype of the parameters to avoid
    # dtype promotion of parameters resulting in a dtype mismatch between
    # parameters and updates.
    params_dtype = otu.tree_dtype(params, 'lowest')
    exp_avg = otu.tree_zeros_like(params)
    barf = jnp.zeros([], dtype=params_dtype)
    gamma = jnp.zeros([], dtype=params_dtype)
    init_lb = jnp.array(lower_bound, dtype=params_dtype)
    count = jnp.zeros([], jnp.int32)
    return MomoState(exp_avg, barf, gamma, init_lb, count)

  def update_fn(
      updates: base.Updates,
      state: MomoState,
      params: Optional[base.Params],
      *,
      value: Optional[jax.Array] = None,
      **extra_args,
  ) -> tuple[base.Updates, MomoState]:
    del extra_args
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    if value is None:
      raise ValueError("""You need to pass the latest loss value to Momo.
                       Use ``jax.value_and_grad`` for this.""")
    count = state.count
    # initialize at first gradient, and loss
    bt = jnp.where(count == 0, 0.0, beta)
    barf = bt * state.barf + (1 - bt) * value
    exp_avg = jax.tree.map(
        lambda ea, g: bt * ea + (1 - bt) * g, state.exp_avg, updates
    )
    gamma = bt * state.gamma + (1 - bt) * otu.tree_vdot(updates, params)
    exp_avg_norm = otu.tree_l2_norm(exp_avg, squared=True)
    iprod = otu.tree_vdot(exp_avg, params)
    alpha = learning_rate(count) if callable(learning_rate) else learning_rate
    # Reset lower bound
    if adapt_lower_bound:
      cap = (1 + alpha * weight_decay) * (barf - gamma) + iprod
      this_lb = lax.cond(
          cap < (1 + alpha * weight_decay) * state.lb,
          lambda: jnp.maximum(
              cap / (2 * (1 + alpha * weight_decay)), lower_bound
          ),
          lambda: state.lb,
      )
    else:
      this_lb = state.lb
    t1 = jnp.maximum(
        (1 + alpha * weight_decay) * (barf - this_lb - gamma) + iprod, 0.0
    ) / (exp_avg_norm)
    # if denom is zero, take no step
    t1 = jnp.where(exp_avg_norm <= jnp.finfo(float).eps, 0.0, t1)
    tau = jnp.minimum(alpha, t1)
    p_update = jax.tree.map(
        lambda ea, p: -(alpha * weight_decay) / (1 + alpha * weight_decay) * p
        - tau * ea,
        exp_avg,
        params,
    )
    if adapt_lower_bound:
      new_lb = jnp.maximum(
          (barf + iprod - gamma) - (1 / 2) * tau * exp_avg_norm, lower_bound
      )
    else:
      new_lb = state.lb
    new_state = MomoState(
        exp_avg=exp_avg,
        barf=barf,
        gamma=gamma,
        lb=new_lb,
        count=numerics.safe_increment(count),
    )
    return p_update, new_state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)


class MomoAdamState(NamedTuple):
  """State of the ``GradientTransformation`` returned by ``momo_adam``."""

  exp_avg: base.Updates
  exp_avg_sq: base.Updates
  barf: chex.Array  # shape=(), dtype=jnp.float32.
  gamma: chex.Array  # shape=(), dtype=jnp.float32.
  lb: chex.Array  # shape=(), dtype=jnp.float32.
  count: chex.Array  # shape=(), dtype=jnp.int32.


def momo_adam(
    learning_rate: base.ScalarOrSchedule = 1e-2,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    lower_bound: float = 0.0,
    weight_decay: float = 0.0,
    adapt_lower_bound: bool = False,
) -> base.GradientTransformationExtraArgs:
  """Adaptive Learning Rates for Adam(W).

  MoMo-Adam typically needs less tuning for value of ``learning_rate``,
  by exploting the fact that a lower bound of the loss (or the optimal value) is
  known. For most tasks, zero is a lower bound and an accurate estimate of the
  final loss.

  MoMo performs Adam(W) with a Polyak-type learning rate. The
  effective step size is ``min(learning_rate, <adaptive term>)``, where the
  adaptive term is computed on the fly.

  Note that one needs to pass the latest (batch) loss value to the update
  function using the keyword argument ``value``.

  Args:
    learning_rate: User-specified learning rate. Recommended to be chosen rather
      large, by default 1.0.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps: eps for the underlying Adam Optimizer.
    lower_bound: Lower bound of the loss. Zero should be a good choice for many
      tasks.
    weight_decay: Weight-decay parameter. Momo-Adam performs weight decay in
      similar fashion to AdamW.
    adapt_lower_bound: If no good guess for the lower bound is available, set
      this to true, in order to estimate the lower bound on the fly (see the
      paper for details).

  Returns:
    A ``GradientTransformation`` object.

  Examples:
    >>> from optax import contrib
    >>> import jax
    >>> import jax.numpy as jnp
    >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = contrib.momo_adam()
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  value, grad = jax.value_and_grad(f)(params)
    ...  params, opt_state = solver.update(grad, opt_state, params, value=value)
    ...  print('Objective function: ', f(params))
    Objective function:  0.00029999594
    Objective function:  0.0
    Objective function:  0.0
    Objective function:  0.0
    Objective function:  0.0

  References:
    Schaipp et al., `MoMo: Momentum Models for Adaptive Learning Rates
    <https://arxiv.org/abs/2305.07583>`_, 2023

  .. versionadded:: 0.2.3
  """

  def init_fn(params: base.Params) -> MomoAdamState:
    # Define state parameters with the lowest dtype of the parameters to avoid
    # dtype promotion of parameters resulting in a dtype mismatch between
    # parameters and updates.
    params_dtype = otu.tree_dtype(params, 'lowest')
    exp_avg = otu.tree_zeros_like(params)
    exp_avg_sq = otu.tree_zeros_like(params)
    barf = jnp.zeros([], dtype=params_dtype)
    gamma = jnp.zeros([], dtype=params_dtype)
    init_lb = jnp.array(lower_bound, dtype=params_dtype)
    count = jnp.zeros([], jnp.int32)
    return MomoAdamState(exp_avg, exp_avg_sq, barf, gamma, init_lb, count)

  def update_fn(
      updates: base.Updates,
      state: MomoAdamState,
      params: Optional[base.Params],
      *,
      value: Optional[jax.Array],
      **extra_args,
  ) -> tuple[base.Updates, MomoAdamState]:
    del extra_args
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    if value is None:
      raise ValueError("""You need to pass the latest loss value to Momo.
                       Use ``jax.value_and_grad`` for this.""")
    count = state.count
    count_inc = numerics.safe_increment(count)
    barf = b1 * state.barf + (1 - b1) * value
    exp_avg = jax.tree.map(
        lambda ea, g: b1 * ea + (1 - b1) * g, state.exp_avg, updates
    )
    exp_avg_sq = jax.tree.map(
        lambda eas, g: b2 * eas + (1 - b2) * g * g,
        state.exp_avg_sq,
        updates,
    )
    bc2 = jnp.asarray(1 - b2**count_inc, dtype=barf.dtype)
    precond = jax.tree.map(lambda eas: eps + jnp.sqrt(eas / bc2), exp_avg_sq)
    exp_avg_weighted = jax.tree.map(
        lambda ea, prec: ea / prec, exp_avg, precond
    )
    exp_avg_norm = otu.tree_vdot(exp_avg, exp_avg_weighted)
    gamma = b1 * state.gamma + (1 - b1) * otu.tree_vdot(updates, params)
    iprod = otu.tree_vdot(exp_avg, params)
    alpha = learning_rate(count) if callable(learning_rate) else learning_rate
    bc1 = jnp.asarray(1 - b1**count_inc, dtype=barf.dtype)
    # Reset lower bound
    if adapt_lower_bound:
      cap = (1 + alpha * weight_decay) * (barf - gamma) + iprod
      this_lb = jnp.where(
          cap < (1 + alpha * weight_decay) * bc1 * state.lb,
          jnp.maximum(
              cap / (2 * bc1 * (1 + alpha * weight_decay)), lower_bound
          ),
          state.lb,
      )
    else:
      this_lb = state.lb
    t1 = jnp.maximum(
        (1 + alpha * weight_decay) * (barf - bc1 * this_lb - gamma) + iprod, 0.0
    ) / (exp_avg_norm)
    # if denom is zero, take no step
    t1 = jnp.where(exp_avg_norm <= jnp.finfo(float).eps, 0.0, t1)
    tau = jnp.minimum(alpha / bc1, t1)
    p_update = jax.tree.map(
        lambda ea, prec, p: -(alpha * weight_decay)
        / (1 + alpha * weight_decay)
        * p
        - tau * ea / prec,
        exp_avg,
        precond,
        params,
    )
    if adapt_lower_bound:
      new_lb = ((barf + iprod - gamma) - (1 / 2) * tau * exp_avg_norm) / bc1
      new_lb = jnp.maximum(new_lb, lower_bound)
    else:
      new_lb = state.lb
    new_state = MomoAdamState(
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        barf=barf,
        gamma=gamma,
        lb=new_lb,
        count=count_inc,
    )
    return p_update, new_state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)
