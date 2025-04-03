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
"""Mechanic wrapper for automatic black box learning rate tuning.

Mechanic is a contributed optimizer implemented from
https://arxiv.org/pdf/2306.00144.pdf.

This implementation matches the paper exactly and implemented by the original
authors. More specifically, mechanic is implemented to work well with other
optax optimizers that it can wrap to learn the learning rate.

Mechanic incurs an extra O(d) slot to store the initial weights and a handful
of O(d) computations. We largely expect the wall clock time with and without
using Mechanic to be the same for reasonably large batch sizes (>1k).
"""

from typing import NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import numerics
import optax.tree_utils as otu


class MechanicState(NamedTuple):
  """State of the `GradientTransformation` returned by `mechanize`."""

  base_optimizer_state: base.OptState
  count: chex.Array  # shape=(), dtype=jnp.int32.
  r: chex.Array
  m: chex.Array
  v: chex.Array
  s: chex.Array
  x0: base.Updates


def mechanize(
    base_optimizer: base.GradientTransformation,
    weight_decay: float = 1e-2,
    eps: float = 1e-8,
    s_init: float = 1e-6,
    num_betas: int = 6,
) -> base.GradientTransformation:
  """Mechanic - a black box learning rate tuner/optimizer.

  Accumulates updates returned by the base_optimizer and learns the scale of
  the updates (also know as learning rate or step size) to apply on a per
  iteration basis.

  Note that Mechanic does NOT eschew the need for a learning rate schedule,
  you are free to apply a learning rate schedule with base learning rate set to
  1.0 (or any other constant) and Mechanic will learn the right scale factor
  automatically.

  For example, change this::

    learning_rate_fn = optax.warmup_cosine_decay_schedule(peak_value=tuned_lr)
    optimizer = optax.adam(learning_rate_fn)

  To::

    learning_rate_fn = optax.warmup_cosine_decay_schedule(peak_value=1.0)
    optimizer = optax.adam(learning_rate_fn)
    optimizer = optax.contrib.mechanize(optimizer)

  As of June, 2023, Mechanic is tested with SGD, Momentum, Adam and Lion as
  inner optimizers but we expect it to work with almost any first-order
  optimizer (except for normalized gradient optimizer like LARS or LAMB).

  Args:
    base_optimizer: Base optimizer to compute updates from.
    weight_decay: A scalar weight decay rate. Note that this weight decay is not
      the same as the weight decay one would use for the base_optimizer. In
      addition to sometimes helping converge faster, this helps Mechanic reduce
      the variance between training runs using different seeds. You likely would
      not need to tune this, the default should work in most cases.
    eps: epsilon for mechanic.
    s_init: initial scale factor. Default should work almost all the time.
    num_betas: unlike traditional exp accumulators (like 1st or 2nd moment of
      adam), where one has to choose an explicit beta, mechanic has a clever way
      to automatically learn the right beta for all accumulators. We only
      provide the range of possible betas, and not the tuned value. For
      instance, if you set num_betas to 3, it will use betas = [0.9, 0.99,
      0.999].

  Returns:
    A :class:`optax.GradientTransformation`.

  References:
    Cutkosky et al, `Mechanic: A Learning Rate Tuner
    <https://arxiv.org/pdf/2306.00144.pdf>`_ 2023
  """

  def init_fn(params: base.Params) -> MechanicState:
    x0 = params
    # Define state parameters with the lowest dtype of the parameters to avoid
    # dtype promotion of parameters resulting in a dtype mismatch between
    # parameters and updates.
    params_dtype = otu.tree_dtype(params, 'lowest')
    r = jnp.zeros(
        [
            num_betas,
        ],
        dtype=params_dtype,
    )
    v = jnp.zeros(
        [
            num_betas,
        ],
        dtype=params_dtype,
    )
    m = jnp.zeros(
        [
            num_betas,
        ],
        dtype=params_dtype,
    )
    s = (
        jnp.ones(
            [
                num_betas,
            ],
            dtype=params_dtype,
        )
        * s_init
    )
    return MechanicState(
        base_optimizer_state=base_optimizer.init(params),
        count=jnp.zeros([], jnp.int32),
        r=r,
        m=m,
        v=v,
        s=s,
        x0=x0,
    )

  def update_fn(
      updates: base.Updates,
      state: MechanicState,
      params: Optional[base.Params] = None,
  ) -> tuple[base.Params, MechanicState]:
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)

    count_inc = numerics.safe_increment(state.count)
    new_neg_updates, base_optimizer_state = base_optimizer.update(
        updates, state.base_optimizer_state, params
    )
    # Since a lot of training loops unfreezes weights to replace it with
    # pre-trained weights, we want to make sure we start from actually used
    # weights instead of what they were initialized with.
    x0 = jax.lax.cond(state.count == 0, lambda: params, lambda: state.x0)

    # Add weight decay to raw gradients, note that this is orthogonal to any
    # weight decay applied to inner_optimizer updates.
    s_sum = jnp.sum(state.s)
    grad_norm = otu.tree_l2_norm(updates)
    param_norm = otu.tree_l2_norm(params)

    def add_weight_decay(gi, pi):
      return gi + weight_decay * s_sum * grad_norm / (param_norm + eps) * pi

    updates = jax.tree.map(
        add_weight_decay,
        updates,
        params,
    )

    # We use the memory efficient version of Mechanic where we re-compute
    # \Delta every iteration.
    delta_prev = jax.tree.map(
        lambda xti, x0i: (x0i - xti) / (s_sum + eps), params, x0
    )

    # We actually want to add the updates, but since optax by default flips
    # signs when applying the learning rate, we substract instead.
    delta = jax.tree.map(lambda si, ui: si - ui, delta_prev, new_neg_updates)

    # Now we are ready to run the actual Mechanic algorithm.
    h = otu.tree_vdot(updates, delta_prev)

    # This clipping was not part of the original paper but we introduced it
    # a little later.
    clipped_h = jax.lax.clamp(-state.m, jnp.ones_like(state.m) * h, state.m)
    betas = jnp.array(
        [1.0 - 0.1**betai for betai in range(1, num_betas + 1)],
        dtype=state.s.dtype,
    )

    m = jnp.maximum(betas * state.m, jnp.abs(h) + eps)
    v = (betas**2) * state.v + h**2
    r = betas * state.r + clipped_h * state.s
    rc = jnp.maximum(0.0, r)
    wealth = (s_init / jnp.size(betas)) * m + rc
    s = wealth / (jnp.sqrt(v) + eps)

    # Once we have the scale factor s, we produce new params with it.
    new_x0 = x0
    new_params = jax.tree.map(
        lambda x0, deltai: x0 - jnp.sum(s) * deltai, new_x0, delta
    )
    new_neg_updates = jax.tree.map(lambda np, op: np - op, new_params, params)

    return new_neg_updates, MechanicState(
        base_optimizer_state=base_optimizer_state,
        count=count_inc,
        r=r,
        m=m,
        v=v,
        s=s,
        x0=new_x0,
    )

  return base.GradientTransformation(init_fn, update_fn)
