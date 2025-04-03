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
"""Tests for the lookahead optimizer in `lookahead.py`."""

from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import alias
from optax._src import base
from optax._src import lookahead
from optax._src import update
from optax.tree_utils import _state_utils


def _build_sgd():
  return alias.sgd(1.0)


class OptimizerTestState(NamedTuple):
  """Fast optimizer state for the lookahead tests."""

  aggregate_grads: base.Params
  # Include a variable with non-zero initial value to check that it is reset
  # correctly by the lookahead optimizer.
  is_reset: bool = True


def _test_optimizer(step_size: float) -> base.GradientTransformation:
  """Fast optimizer for the lookahead tests."""

  # Use SGD for simplicity but add non-trivial optimizer state so that the
  # resetting behavior of lookahead can be tested.
  def init_fn(params):
    aggregate_grads = jax.tree.map(jnp.zeros_like, params)
    return OptimizerTestState(aggregate_grads, is_reset=True)

  def update_fn(updates, state, params):
    # The test optimizer does not use the parameters, but we check that they
    # have been passed correctly.
    chex.assert_trees_all_equal_shapes(updates, params)
    aggregate_grads = update.apply_updates(state.aggregate_grads, updates)
    updates = jax.tree.map(lambda u: step_size * u, updates)
    return updates, OptimizerTestState(aggregate_grads, is_reset=False)

  return base.GradientTransformation(init_fn, update_fn)


class LookaheadTest(chex.TestCase):
  """Tests for the lookahead optimizer."""

  def setUp(self):
    super().setUp()
    self.grads = {'x': np.array(2.0), 'y': np.array(-2.0)}
    self.initial_params = {'x': np.array(3.0), 'y': np.array(-3.0)}
    self.synced_initial_params = lookahead.LookaheadParams.init_synced(
        self.initial_params
    )

  def loop(self, optimizer, num_steps, params):
    """Performs a given number of optimizer steps."""
    init_fn, update_fn = optimizer
    # Use the chex variant to check various function versions (jit, pmap, etc).
    step = self.variant(update_fn)
    opt_state = self.variant(init_fn)(params)

    # A no-op change, to verify that tree map works.
    opt_state = _state_utils.tree_map_params(init_fn, lambda v: v, opt_state)

    for _ in range(num_steps):
      updates, opt_state = step(self.grads, opt_state, params)
      params = update.apply_updates(params, updates)

    return params, opt_state

  @chex.all_variants
  def test_lookahead(self):
    """Tests the lookahead optimizer in an analytically tractable setting."""
    sync_period = 3
    optimizer = lookahead.lookahead(
        _test_optimizer(-0.5), sync_period=sync_period, slow_step_size=1 / 3
    )

    final_params, _ = self.loop(
        optimizer, 2 * sync_period, self.synced_initial_params
    )
    # x steps must be: 3 -> 2 -> 1 -> 2 (sync) -> 1 -> 0 -> 1 (sync).
    # Similarly for y (with sign flipped).
    correct_final_params = {'x': 1, 'y': -1}
    chex.assert_trees_all_close(final_params.slow, correct_final_params)

  @chex.all_variants
  @parameterized.parameters([False], [True])
  def test_lookahead_state_reset(self, reset_state):
    """Checks that lookahead resets the fast optimizer state correctly."""
    num_steps = sync_period = 3
    fast_optimizer = _test_optimizer(-0.5)
    optimizer = lookahead.lookahead(
        fast_optimizer,
        sync_period=sync_period,
        slow_step_size=0.5,
        reset_state=reset_state,
    )

    _, opt_state = self.loop(optimizer, num_steps, self.synced_initial_params)

    # A no-op change, to verify that this does not break anything
    opt_state = _state_utils.tree_map_params(optimizer, lambda v: v, opt_state)

    fast_state = opt_state.fast_state
    if reset_state:
      correct_state = fast_optimizer.init(self.initial_params)
    else:
      _, correct_state = self.loop(
          fast_optimizer, num_steps, self.initial_params
      )

    chex.assert_trees_all_close(fast_state, correct_state)

  @chex.all_variants
  @parameterized.parameters(
      [1, 0.5, {'x': np.array(1.), 'y': np.array(-1.)}],
      [1, 0, {'x': np.array(3.), 'y': np.array(-3.)}],
      [1, 1, {'x': np.array(-1.), 'y': np.array(1.)}],
      [2, 1, {'x': np.array(-1.), 'y': np.array(1.)}])  # pyformat: disable
  def test_lookahead_edge_cases(
      self, sync_period, slow_step_size, correct_result
  ):
    """Checks special cases of the lookahed optimizer parameters."""
    # These edge cases are important to check since users might use them as
    # simple ways of disabling lookahead in experiments.
    optimizer = lookahead.lookahead(
        _test_optimizer(-1), sync_period, slow_step_size
    )
    final_params, _ = self.loop(
        optimizer, num_steps=2, params=self.synced_initial_params
    )
    chex.assert_trees_all_close(final_params.slow, correct_result)


if __name__ == '__main__':
  absltest.main()
