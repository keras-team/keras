# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for the schedule-free wrapper."""

import functools
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from optax._src import alias
from optax._src import numerics
from optax._src import update
from optax.contrib import _schedule_free


def _setup_parabola(dtype):
  """Quadratic function as an optimization target."""
  initial_params = jnp.array([-1.0, 10.0, 1.0], dtype=dtype)
  final_params = jnp.array([1.0, -1.0, 1.0], dtype=dtype)

  @jax.grad
  def get_updates(params):
    return jnp.sum(numerics.abs_sq(params - final_params))

  return initial_params, final_params, get_updates


class ScheduleFreeTest(chex.TestCase):

  def test_learning_rate_zero(self):
    base_opt = alias.sgd(learning_rate=0.0, momentum=0.0)
    opt = _schedule_free.schedule_free(base_opt, learning_rate=0.0)
    initial_params = jnp.array([1.0, 2.0])
    fun = lambda x: jnp.sum(x**2)

    @jax.jit
    def step(params, state):
      updates = jax.grad(fun)(params)
      updates, state = opt.update(updates, state, params)
      params = update.apply_updates(params, updates)
      return params, state

    params = initial_params
    state = opt.init(params)
    for _ in range(5):
      params, state = step(params, state)

    chex.assert_trees_all_close(
        _schedule_free.schedule_free_eval_params(state, params),
        initial_params,
    )

  def test_schedule_free_adamw(self):

    initial_params = jnp.array([1.0, 2.0])
    fun = lambda x: jnp.sum(x**2)

    def step(params, state, opt):
      updates = jax.grad(fun)(params)
      updates, state = opt.update(updates, state, params)
      params = update.apply_updates(params, updates)
      return params, state

    def run(opt):
      params = initial_params
      state = opt.init(params)

      for _ in range(5):
        params, state = step(params, state, opt)
      return params

    # Test with shortcut implementation
    opt_shortcut = _schedule_free.schedule_free_adamw(
        learning_rate=1.0,
        b1=0.9,
        weight_decay=1e-4,
    )
    params_shortcut = run(opt_shortcut)

    # Test with wrapper implementation
    opt_wrapper = _schedule_free.schedule_free(
        alias.adamw(learning_rate=1.0, b1=0.0, weight_decay=1e-4),
        learning_rate=1.0,
        b1=0.9,
    )
    params_wrapper = run(opt_wrapper)
    chex.assert_trees_all_close(
        params_shortcut, params_wrapper, atol=1e-6, rtol=1e-6
    )

  def test_scalar_preservance(self):
    # Test whether the scalar arrays of shape () are preserved through
    # _schedule_free.schedule_free_eval_params.
    base_opt = alias.sgd(learning_rate=1.0, momentum=0.0)
    opt = _schedule_free.schedule_free(base_opt, learning_rate=1.0)

    params = jnp.ones((), dtype=jnp.float32)
    state = opt.init(params)
    eval_params = _schedule_free.schedule_free_eval_params(state, params)
    chex.assert_equal_shape([params, eval_params])
    chex.assert_trees_all_equal_dtypes(params, eval_params)

  def test_buffer_donation(self):
    # Check that you can donate the params and optimizer state when doing a JIT
    # update.
    opt = _schedule_free.schedule_free_sgd()
    initial_params, _, get_updates = _setup_parabola(jnp.float32)

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def step(params, state):
      updates = get_updates(params)
      updates, state = opt.update(updates, state, params)
      params = update.apply_updates(params, updates)
      return params, state

    state = opt.init(initial_params)
    _, _ = step(initial_params, state)

  @parameterized.product(
      params_dtype=('bfloat16', 'float32', 'complex64', None),
      state_dtype=('bfloat16', 'float32', 'complex64', None),
  )
  def test_explicit_dtype(self, params_dtype, state_dtype):
    base_opt = alias.sgd(learning_rate=1.0, momentum=0.0)
    opt = _schedule_free.schedule_free(
        base_opt, learning_rate=1.0, state_dtype=state_dtype
    )

    params_dtype = jax.dtypes.canonicalize_dtype(params_dtype)
    params = jnp.array([0.0, 0.0], dtype=params_dtype)
    state = opt.init(params)

    with self.subTest('Test that attribute dtype is correct'):
      if state_dtype is None:
        expected_dtype = params_dtype
      else:
        expected_dtype = jax.dtypes.canonicalize_dtype(state_dtype)
      self.assertEqual(expected_dtype, getattr(state, 'z').dtype)


if __name__ == '__main__':
  absltest.main()
