# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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
"""Reduce Learning Rate on Plateau callback.

This callback monitors a quantity and if no improvement is seen for a 'patience'
number of epochs, the learning rate is reduced by a factor of 'reduce_factor'.
Optionally, a cooldown period can be specified during which the learning rate
will not be reduced.
"""
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import numerics
import optax.tree_utils as otu


class ReduceLROnPlateauState(NamedTuple):
  """State for the ReduceLROnPlateau callback."""

  scale: chex.Array
  best_value: chex.Array
  plateau_count: chex.Array  # shape=(), dtype=jnp.int32
  cooldown_count: chex.Array  # shape=(), dtype=jnp.int32
  count: chex.Array  # shape=(), dtype=jnp.int32
  avg_value: chex.Array


def reduce_on_plateau(
    factor: float = 0.1,
    patience: int = 10,
    rtol: float = 1e-4,
    atol: float = 0.0,
    cooldown: int = 0,
    accumulation_size: int = 1,
    min_scale: float = 0.0,
) -> base.GradientTransformationExtraArgs:
  """Reduce learning rate when a metric has stopped improving.

  Models often benefit from reducing the learning once learning stagnates.
  his scheduler reads a metrics quantity and if no improvement is seen for
  a ``patience`` number of epochs, the learning rate is reduced.

  Args:
    factor: Factor by which to reduce the learning rate. new_scale = scale *
      factor.
    patience: Number of iterations with no improvement after which learning rate
      will be reduced.
    rtol: Relative tolerance for measuring new optimum.
    atol: Absolute tolerance for measuring new optimum.
    cooldown: Number of iterations to wait before resuming normal operation
      after scale has been reduced.
    accumulation_size: Number of values to aggregate before applying the logic
      of reduce on plateau. If the value fed to the optimizer is a test value,
      simply take 1 (default). If the value fed to the optimizer is the loss on
      a the current minibatch, consider using a larger accumulation size.
    min_scale: Scale at which the learning rate decay stops.

  Returns:
    A :class:`optax.GradientTransformationExtraArgs` object.

  .. seealso::
    * :doc:`../../_collections/examples/contrib/reduce_on_plateau` example.
  """
  if factor <= 0.0 or factor >= 1.0:
    raise ValueError(
        f"Factor must be in the range (0, 1), got factor = {factor}."
    )

  if rtol < 0.0 or atol < 0.0:
    raise ValueError(
        "Both rtol and atol must be non-negative, got "
        f"rtol = {rtol} and atol = {atol}."
    )
  elif rtol == 0.0 and atol == 0.0:
    raise ValueError(
        "At least one of rtol or atol must be positive, got "
        f"rtol = {rtol} and atol = {atol}."
    )
  elif rtol > 1.0:
    raise ValueError(
        f"rtol must be less than or equal to 1.0, got rtol = {rtol}."
    )

  def init_fn(params) -> ReduceLROnPlateauState:
    # Define state parameters with the lowest dtype of the parameters to avoid
    # dtype promotion of parameters resulting in a dtype mismatch between
    # parameters and updates.
    params_dtype = otu.tree_dtype(params, "lowest")
    return ReduceLROnPlateauState(
        best_value=jnp.asarray(float("inf")),
        plateau_count=jnp.asarray(0, jnp.int32),
        scale=jnp.asarray(1.0, dtype=params_dtype),
        cooldown_count=jnp.asarray(0, jnp.int32),
        count=jnp.asarray(0, jnp.int32),
        avg_value=jnp.asarray(0.0),
    )

  def _update_scale(state):
    # Update plateau count and check if plateaued
    avg_value = state.avg_value
    has_improved = jnp.where(
        avg_value < (1 - rtol) * state.best_value - atol, 1, 0
    )
    new_best_value = jnp.where(has_improved, avg_value, state.best_value)
    curr_plateau_count = jnp.where(
        has_improved, 0, numerics.safe_increment(state.plateau_count)
    )

    # We're in cooldown, so reduce the counter and ignore any bad epochs
    def in_cooldown():
      new_plateau_count = jnp.asarray(0, jnp.int32)
      new_scale = state.scale
      new_cooldown_count = state.cooldown_count - 1
      return new_plateau_count, new_scale, new_cooldown_count

    # We're not in cooldown, so update the plateau count and scale as usual
    def not_in_cooldown():
      new_plateau_count = jnp.where(
          curr_plateau_count == patience, 0, curr_plateau_count
      )
      new_scale = jnp.maximum(
          jnp.where(
              curr_plateau_count == patience,
              state.scale * factor,
              state.scale,
          ),
          min_scale,
      )
      new_cooldown_count = jnp.where(
          curr_plateau_count == patience, cooldown, 0
      ).astype(jnp.int32)

      return new_plateau_count, new_scale, new_cooldown_count

    new_plateau_count, new_scale, new_cooldown_count = jax.lax.cond(
        state.cooldown_count > 0, in_cooldown, not_in_cooldown
    )
    new_state = ReduceLROnPlateauState(
        plateau_count=new_plateau_count,
        best_value=new_best_value,
        scale=new_scale,
        cooldown_count=new_cooldown_count,
        count=jnp.asarray(0, dtype=jnp.int32),
        avg_value=jnp.asarray(0.0),
    )
    return new_state

  def update_fn(
      updates: base.Updates,
      state: ReduceLROnPlateauState,
      params=None,
      *,
      value: float,
      **extra_args,
  ) -> tuple[base.Params, ReduceLROnPlateauState]:
    del params, extra_args

    count = state.count
    new_count = numerics.safe_increment(count)
    new_avg_value = (
        count * state.avg_value + jnp.astype(value, state.avg_value.dtype)
    ) / new_count
    new_state = state._replace(avg_value=new_avg_value, count=new_count)

    new_state = jax.lax.cond(
        new_count == accumulation_size, _update_scale, lambda x: x, new_state
    )

    updates = jax.tree.map(lambda g: new_state.scale * g, updates)

    return updates, new_state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)
