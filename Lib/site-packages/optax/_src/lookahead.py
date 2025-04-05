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
"""A lookahead optimization wrapper."""

from typing import NamedTuple, Union

from absl import logging
import jax
import jax.numpy as jnp
from optax._src import base


class LookaheadState(NamedTuple):
  """State of the `GradientTransformation` returned by `lookahead`.

  Attributes:
    fast_state: Optimizer state of the fast optimizer.
    steps_since_sync: Number of fast optimizer steps taken since slow and fast
      parameters were synchronized.
  """

  fast_state: base.OptState
  steps_since_sync: jnp.ndarray


class LookaheadParams(NamedTuple):
  """Holds a pair of slow and fast parameters for the lookahead optimizer.

  Gradients should always be calculated with the fast parameters. The slow
  parameters should be used for testing and inference as they generalize better.
  See the reference for a detailed discussion.

  Attributes:
    fast: Fast parameters.
    slow: Slow parameters.

  References:
    Zhang et al, `Lookahead Optimizer: k steps forward, 1 step back
    <https://arxiv.org/abs/1907.08610>`_, 2019
  """

  fast: base.Params
  slow: base.Params

  @classmethod
  def init_synced(cls, params: base.Params) -> 'LookaheadParams':
    """Initialize a pair of synchronized lookahead parameters."""
    return cls(slow=params, fast=params)


def lookahead(
    fast_optimizer: base.GradientTransformation,
    sync_period: int,
    slow_step_size: float,
    reset_state: bool = False,
) -> base.GradientTransformation:
  """Lookahead optimizer.

  Performs steps with a fast optimizer and periodically updates a set of slow
  parameters. Optionally resets the fast optimizer state after synchronization
  by calling the init function of the fast optimizer.

  Updates returned by the lookahead optimizer should not be modified before they
  are applied, otherwise fast and slow parameters are not synchronized
  correctly.

  Args:
    fast_optimizer: The optimizer to use in the inner loop of lookahead.
    sync_period: Number of fast optimizer steps to take before synchronizing
      parameters. Must be >= 1.
    slow_step_size: Step size of the slow parameter updates.
    reset_state: Whether to reset the optimizer state of the fast optimizer
      after each synchronization.

  Returns:
    A :class:`optax.GradientTransformation` with init and update functions. The
    updates passed to the update function should be calculated using the fast
    lookahead parameters only.

  References:
    Zhang et al, `Lookahead Optimizer: k steps forward, 1 step back
    <https://arxiv.org/abs/1907.08610>`_, 2019
  """
  if sync_period < 1:
    raise ValueError('Synchronization period must be >= 1.')

  def init_fn(params: base.Params) -> LookaheadState:
    fast_params = getattr(params, 'fast', None)
    if fast_params is None:
      # Allowing init_fn to be called with fast parameters reduces the
      # modifications necessary to adapt code to use lookahead in some cases.
      logging.warning(
          '`params` has no attribute `fast`. Continuing by assuming that '
          'only fast parameters were passed to lookahead init.'
      )
      fast_params = params

    return LookaheadState(
        fast_state=fast_optimizer.init(fast_params),
        steps_since_sync=jnp.zeros(shape=(), dtype=jnp.int32),
    )

  def update_fn(
      updates: base.Updates, state: LookaheadState, params: LookaheadParams
  ) -> tuple[LookaheadParams, LookaheadState]:
    updates, fast_state = fast_optimizer.update(
        updates, state.fast_state, params.fast
    )

    sync_next = state.steps_since_sync == (sync_period - 1)
    updates = _lookahead_update(updates, sync_next, params, slow_step_size)
    if reset_state:
      # Jittable way of resetting the fast optimizer state if parameters will be
      # synchronized after this update step.
      initial_state = fast_optimizer.init(params.fast)
      fast_state = jax.tree.map(
          lambda current, init: (1 - sync_next) * current + sync_next * init,
          fast_state,
          initial_state,
      )

    steps_since_sync = (state.steps_since_sync + 1) % sync_period
    return updates, LookaheadState(fast_state, steps_since_sync)

  return base.GradientTransformation(init_fn, update_fn)


def _lookahead_update(
    updates: base.Updates,
    sync_next: Union[bool, jax.Array],
    params: LookaheadParams,
    slow_step_size: float,
) -> LookaheadParams:
  """Returns the updates corresponding to one lookahead step.

  Args:
    updates: Updates returned by the fast optimizer.
    sync_next: Wether fast and slow parameters should be synchronized after the
      fast optimizer step.
    params: Current fast and slow parameters as `LookaheadParams` object.
    slow_step_size: Step size of the slow optimizer.

  Returns:
    The updates for the lookahead parameters.

  References:
    Zhang et al, `Lookahead Optimizer: k steps forward, 1 step back
    <https://arxiv.org/abs/1907.08610>`_, 2019
  """
  # In the paper, lookahead is presented as two nested loops. To write lookahead
  # as optax wrapper, these loops have to be broken into successive updates.
  # This leads to two types of update steps:
  #
  # Non-synchronization steps (sync_next == False):
  # The updates returned by the fast optimizer are used for the fast parameters
  # without change and the slow parameter updates are zero (i.e. fast_updates =
  # updates, slow_updates = 0).
  #
  # Synchronization step (sync_next == True):
  # This consists of two substeps: a last fast optimizer step and the
  # synchronization.
  #   Substep 1 (last fast optimizer step):
  #     last_fast_params = fast_params + updates
  #   Substep 2 (synchronization):
  #     new_slow_params = slow_params + slow_step_size * (
  #                       last_fast_params - slow_params)
  #     new_fast_params = new_slow_params
  #
  #   Merging into a single update step we get the update rules:
  #     slow_updates = slow_step_size * (fast_params + updates - slow_params)
  #     fast_updates = new_slow_params - fast_params = updates - (1 -
  #       slow_step_size) * (fast_params + updates - slow_params)
  #
  # To make the equations jittable, the two types of steps are merged. Defining
  # last_difference = fast_params + updates - slow_params, this yields the
  # following equations which are implemented below:
  #   slow_updates = slow_step_size * sync_next * last_difference
  #   fast_updates = updates - (
  #                  1 - slow_step_size) * sync_next * last_difference
  last_difference = jax.tree.map(
      lambda f, u, s: f + u - s, params.fast, updates, params.slow
  )
  slow_updates = jax.tree.map(
      lambda diff: slow_step_size * sync_next * diff, last_difference
  )
  fast_updates = jax.tree.map(
      lambda up, diff: up - sync_next * (1 - slow_step_size) * diff,
      updates,
      last_difference,
  )

  return LookaheadParams(fast=fast_updates, slow=slow_updates)
