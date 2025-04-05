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
"""D-Adatation (AdamW variant).

A contributed implementation of the method from "Learning-Rate-Free Learning by
D-Adaptation" (https://arxiv.org/abs/2301.07733) by Aaron Defazio and Konstantin
Mishchenko (ICML 2023 Outstanding Paper award).
"""
from typing import NamedTuple, Optional
import chex
import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import numerics
import optax.tree_utils as otu


class DAdaptAdamWState(NamedTuple):
  """State of the `GradientTransformation` returned by `dadapt_adamw`."""

  exp_avg: base.Updates
  exp_avg_sq: base.Updates
  # Exponential moving average of the sum of gradients.
  grad_sum: base.Updates  # shape=(), dtype=jnp.float32.
  # Distance to solution estimate.
  estim_lr: chex.Array  # shape=(), dtype=jnp.float32.
  numerator_weighted: chex.Array  # shape=(), dtype=jnp.float32.
  count: chex.Array  # shape=(), dtype=jnp.int32.


def dadapt_adamw(
    learning_rate: base.ScalarOrSchedule = 1.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    estim_lr0: float = 1e-6,
    weight_decay: float = 0.0,
) -> base.GradientTransformation:
  """Learning rate free AdamW by D-Adaptation.

  Adapts the baseline learning rate of AdamW automatically by estimating the
  initial distance to solution in the infinity norm.
  This method works best when combined with a learning rate schedule that
  treats 1.0 as the base (usually max) value.

  Args:
    learning_rate: Learning rate scheduling parameter. The recommended schedule
      is a linear_schedule with init_value=1.0 and end_value=0, combined with a
      0-20% learning rate warmup.
    betas: Betas for the underlying AdamW Optimizer.
    eps: eps for the underlying AdamW Optimizer.
    estim_lr0: Initial (under-)estimate of the learning rate.
    weight_decay: AdamW style weight-decay. To use Regular Adam decay, chain
      with add_decayed_weights.

  Returns:
    The corresponding :class:`optax.GradientTransformation`.

  References:
    Defazio et al, `Learning-Rate-Free Learning by D-Adaptation
    <https://arxiv.org/abs/2301.07733>`_, 2023
  """

  def init_fn(params: base.Params) -> DAdaptAdamWState:
    # Define state parameters with the lowest dtype of the parameters to avoid
    # dtype promotion of parameters resulting in a dtype mismatch between
    # parameters and updates.
    params_dtype = otu.tree_dtype(params, 'lowest')
    exp_avg = otu.tree_zeros_like(params)
    exp_avg_sq = otu.tree_zeros_like(params)
    grad_sum = otu.tree_zeros_like(params)
    estim_lr = jnp.asarray(estim_lr0, dtype=params_dtype)
    numerator_weighted = jnp.zeros([], dtype=params_dtype)
    count = jnp.zeros([], jnp.int32)
    return DAdaptAdamWState(
        exp_avg, exp_avg_sq, grad_sum, estim_lr, numerator_weighted, count
    )

  def update_fn(
      updates: base.Updates,
      state: DAdaptAdamWState,
      params: Optional[base.Params] = None,
  ) -> tuple[base.Updates, DAdaptAdamWState]:
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    count = state.count
    beta1, beta2 = betas
    sb2 = beta2 ** (0.5)
    sched = learning_rate(count) if callable(learning_rate) else learning_rate
    grad_sum = state.grad_sum
    numerator_weighted = state.numerator_weighted
    count_inc = numerics.safe_increment(count)
    bc = ((1 - beta2**count_inc) ** 0.5) / (1 - beta1**count_inc)
    dlr = state.estim_lr * sched * bc
    dlr = dlr.astype(numerator_weighted.dtype)
    s_weighted = jax.tree.map(
        lambda sk, eas: sk / (jnp.sqrt(eas) + eps), grad_sum, state.exp_avg_sq
    )
    numerator_acum = otu.tree_vdot(updates, s_weighted)
    exp_avg = jax.tree.map(
        lambda ea, g: beta1 * ea + (1 - beta1) * dlr * g, state.exp_avg, updates
    )
    exp_avg_sq = jax.tree.map(
        lambda eas, g: beta2 * eas + (1 - beta2) * g * g,
        state.exp_avg_sq,
        updates,
    )
    grad_sum = jax.tree.map(
        lambda sk, g: sb2 * sk + (1 - sb2) * dlr * g, grad_sum, updates
    )
    grad_sum_l1 = otu.tree_sum(jax.tree.map(jnp.abs, grad_sum))
    numerator_weighted = (
        sb2 * numerator_weighted + (1 - sb2) * dlr * numerator_acum
    )
    d_estimate = numerator_weighted / ((1 - sb2) * grad_sum_l1)
    estim_lr = jnp.maximum(state.estim_lr, d_estimate)
    p_update = jax.tree.map(
        lambda ea, eas, p: -weight_decay * dlr * p - ea / (jnp.sqrt(eas) + eps),
        exp_avg,
        exp_avg_sq,
        params,
    )
    new_state = DAdaptAdamWState(
        exp_avg,
        exp_avg_sq,
        grad_sum,
        estim_lr,
        numerator_weighted,
        count_inc,
    )
    return p_update, new_state

  return base.GradientTransformation(init_fn, update_fn)
