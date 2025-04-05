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
"""Prodigy Optimizer.

A contributed implementation of the method from "Prodigy: An Expeditiously
Adaptive Parameter-Free Learner" (https://arxiv.org/abs/2306.06101) by
Konstantin Mishchenko and Aaron Defazio. A new variant of D-Adapt Adam that
adapts the learning rate faster.
"""
from typing import NamedTuple, Optional
import chex
import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import numerics
import optax.tree_utils as otu


class ProdigyState(NamedTuple):
  """State of the `GradientTransformation` returned by `prodigy`."""

  exp_avg: base.Updates
  exp_avg_sq: base.Updates
  # Exponential moving average of the sum of gradients.
  grad_sum: base.Updates
  # Initial point.
  params0: base.Updates
  # Distance to solution estimate.
  estim_lr: chex.Array  # shape=(), dtype=jnp.float32.
  numerator_weighted: chex.Array  # shape=(), dtype=jnp.float32.
  count: chex.Array  # shape=(), dtype=int32.


def prodigy(
    learning_rate: base.ScalarOrSchedule = 1.0,
    betas: tuple[float, float] = (0.9, 0.999),
    beta3: Optional[float] = None,
    eps: float = 1e-8,
    estim_lr0: float = 1e-6,
    estim_lr_coef: float = 1.0,
    weight_decay: float = 0.0,
    safeguard_warmup: bool = False,
) -> base.GradientTransformation:
  """Learning rate free AdamW with Prodigy.

  Implementation of the Prodigy method from "Prodigy: An Expeditiously
  Adaptive Parameter-Free Learner", a version of D-Adapt AdamW that adapts the
  baseline learning rate faster by using a weighting of the gradients that
  places higher weights on more recent gradients.
  This method works best when combined with a learning rate schedule that
  treats 1.0 as the base (usually max) value.

  Args:
    learning_rate: Learning rate scheduling parameter. The recommended schedule
      is a linear_schedule with init_value=1.0 and end_value=0, combined with a
      0-20% learning rate warmup.
    betas: Betas for the underlying AdamW Optimizer.
    beta3: Optional momentum parameter for estimation of D.
    eps: eps for the underlying AdamW Optimizer.
    estim_lr0: Initial (under-)estimate of the learning rate.
    estim_lr_coef: LR estimates are multiplied by this parameter.
    weight_decay: AdamW style weight-decay. To use Regular Adam decay, chain
      with add_decayed_weights.
    safeguard_warmup: Remove lr from the denominator of D estimate to avoid
      issues during warm-up stage. Off by default.

  Returns:
    A :class:`optax.GradientTransformation` object.

  References:
    Mishchenko et al, `Prodigy: An Expeditiously Adaptive Parameter-Free Learner
    <https://arxiv.org/abs/2306.06101>`_, 2023
  """
  beta1, beta2 = betas
  if beta3 is None:
    beta3 = beta2**0.5

  def init_fn(params: base.Params) -> ProdigyState:
    # Define state parameters with the lowest dtype of the parameters to avoid
    # dtype promotion of parameters resulting in a dtype mismatch between
    # parameters and updates.
    params_dtype = otu.tree_dtype(params, 'lowest')
    exp_avg = otu.tree_zeros_like(params)
    exp_avg_sq = otu.tree_zeros_like(params)
    grad_sum = otu.tree_zeros_like(params)
    params0 = params
    estim_lr = jnp.asarray(estim_lr0, dtype=params_dtype)
    numerator_weighted = jnp.zeros((), dtype=params_dtype)
    count = jnp.zeros((), jnp.int32)
    return ProdigyState(
        exp_avg,
        exp_avg_sq,
        grad_sum,
        params0,
        estim_lr,
        numerator_weighted,
        count,
    )

  def update_fn(
      updates: base.Updates,
      state: ProdigyState,
      params: Optional[base.Params] = None,
  ) -> tuple[base.Updates, ProdigyState]:
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    count = state.count
    count_inc = numerics.safe_increment(count)
    sched = learning_rate(count) if callable(learning_rate) else learning_rate
    grad_sum = state.grad_sum
    params0 = state.params0
    estim_lr = state.estim_lr
    numerator_weighted = state.numerator_weighted
    bc = ((1 - beta2**count_inc) ** 0.5) / (1 - beta1**count_inc)
    dlr = jnp.asarray(estim_lr * sched * bc, dtype=estim_lr.dtype)
    dg = jax.tree.map(lambda g: estim_lr * g, updates)
    param_diff = jax.tree.map(lambda p0, p: p0 - p, params0, params)
    numerator_acum = otu.tree_vdot(updates, param_diff)
    exp_avg = jax.tree.map(
        lambda ea, dgk: beta1 * ea + (1 - beta1) * dgk, state.exp_avg, dg
    )
    exp_avg_sq = jax.tree.map(
        lambda eas, dgk: beta2 * eas + (1 - beta2) * dgk * dgk,
        state.exp_avg_sq,
        dg,
    )
    if safeguard_warmup:
      grad_sum = jax.tree.map(
          lambda sk, dgk: beta3 * sk + estim_lr * dgk / estim_lr0, grad_sum, dg
      )
    else:
      grad_sum = jax.tree.map(
          lambda sk, dgk: beta3 * sk + dlr * dgk / estim_lr0, grad_sum, dg
      )
    numerator_weighted = beta3 * numerator_weighted
    numerator_weighted += (estim_lr / estim_lr0) * dlr * numerator_acum
    denominator = otu.tree_sum(jax.tree.map(jnp.abs, grad_sum))
    lr_estimate = estim_lr_coef * numerator_weighted / denominator
    estim_lr = jnp.maximum(state.estim_lr, lr_estimate)
    p_update = jax.tree.map(
        lambda ea, eas, p: -weight_decay * dlr * p
        - dlr * ea / (jnp.sqrt(eas) + estim_lr * eps),
        exp_avg,
        exp_avg_sq,
        params,
    )
    new_state = ProdigyState(
        exp_avg,
        exp_avg_sq,
        grad_sum,
        params0,
        estim_lr,
        numerator_weighted,
        count_inc,
    )
    return p_update, new_state

  return base.GradientTransformation(init_fn, update_fn)
