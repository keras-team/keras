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
"""Specific tests for `mechanic.py`, see `common_test.py` for usual tests."""

from typing import NamedTuple

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import base
from optax._src import update
from optax.contrib import _mechanic
from optax.tree_utils import _state_utils


class OptimizerTestState(NamedTuple):
  """Inner optimizer state for the Mechanic tests."""

  aggregate_grads: base.Params


def _test_optimizer(step_size: float) -> base.GradientTransformation:
  """Inner optimizer for the Mechanic tests."""

  # Use SGD for simplicity but add non-trivial optimizer state so that the
  # resetting behavior of lookahead can be tested.
  def init_fn(params):
    aggregate_grads = jax.tree.map(jnp.zeros_like, params)
    return OptimizerTestState(aggregate_grads)

  def update_fn(updates, state, params):
    # The test optimizer does not use the parameters, but we check that they
    # have been passed correctly.
    chex.assert_trees_all_equal_shapes(updates, params)
    aggregate_grads = update.apply_updates(state.aggregate_grads, updates)
    updates = jax.tree.map(lambda u: step_size * u, updates)
    return updates, OptimizerTestState(aggregate_grads)

  return base.GradientTransformation(init_fn, update_fn)


class MechanicTest(chex.TestCase):

  def setUp(self):
    super().setUp()
    self.grads = {'x': np.array(2.0), 'y': np.array(-2.0)}
    self.initial_params = {'x': np.array(3.0), 'y': np.array(-3.0)}

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
      print(updates)
      params = update.apply_updates(params, updates)

    return params, opt_state

  @chex.all_variants(with_pmap=False)
  def test_mechanized(self):
    params = self.initial_params
    num_betas = 6

    inner_optimizer = _test_optimizer(-0.1)
    optimizer = _mechanic.mechanize(
        inner_optimizer,
        weight_decay=1e-2,
        eps=1e-10,
        s_init=1e-8,
        num_betas=num_betas,
    )

    final_params, final_state = self.loop(
        optimizer=optimizer, num_steps=1, params=params
    )
    expected_m = np.array([1.0e-10] * num_betas)
    expected_v = np.array([0.0] * num_betas)
    expected_s = np.array([1.6666667e-09] * num_betas)

    chex.assert_trees_all_close(expected_m, final_state.m)
    chex.assert_trees_all_close(expected_v, final_state.v)
    chex.assert_trees_all_close(expected_s, final_state.s)
    chex.assert_trees_all_close(final_params, params)
    chex.assert_tree_all_finite((final_params, final_state))


if __name__ == '__main__':
  absltest.main()
