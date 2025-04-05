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
"""Tests for methods in `update.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from optax._src import update


class UpdateTest(chex.TestCase):

  @chex.all_variants
  def test_apply_updates(self):
    params = ({'a': jnp.ones((3, 2))}, jnp.ones((1,)))
    grads = jax.tree.map(lambda t: 2 * t, params)
    exp_params = jax.tree.map(lambda t: 3 * t, params)
    new_params = self.variant(update.apply_updates)(params, grads)

    chex.assert_trees_all_close(exp_params, new_params, atol=1e-10, rtol=1e-5)

  @chex.all_variants
  def test_apply_updates_mixed_precision(self):
    params = (
        {'a': jnp.ones((3, 2), dtype=jnp.bfloat16)},
        jnp.ones((1,), dtype=jnp.bfloat16),
    )
    grads = jax.tree.map(lambda t: (2 * t).astype(jnp.float32), params)
    new_params = self.variant(update.apply_updates)(params, grads)

    for leaf in jax.tree.leaves(new_params):
      assert leaf.dtype == jnp.bfloat16

  @chex.all_variants
  def test_incremental_update(self):
    params_1 = ({'a': jnp.ones((3, 2))}, jnp.ones((1,)))
    params_2 = jax.tree.map(lambda t: 2 * t, params_1)
    exp_params = jax.tree.map(lambda t: 1.5 * t, params_1)
    new_params = self.variant(update.incremental_update)(
        params_2, params_1, 0.5
    )

    chex.assert_trees_all_close(exp_params, new_params, atol=1e-10, rtol=1e-5)

  @chex.all_variants
  def test_periodic_update(self):
    params_1 = ({'a': jnp.ones((3, 2))}, jnp.ones((1,)))
    params_2 = jax.tree.map(lambda t: 2 * t, params_1)

    update_period = 5
    update_fn = self.variant(update.periodic_update)

    for j in range(3):
      for i in range(1, update_period):
        new_params = update_fn(
            params_2, params_1, j * update_period + i, update_period
        )
        chex.assert_trees_all_close(params_1, new_params, atol=1e-10, rtol=1e-5)

      new_params = update_fn(
          params_2, params_1, (j + 1) * update_period, update_period
      )
      chex.assert_trees_all_close(params_2, new_params, atol=1e-10, rtol=1e-5)

  @parameterized.named_parameters(
      dict(testcase_name='apply_updates', operation=update.apply_updates),
      dict(
          testcase_name='incremental_update',
          operation=lambda x, y: update.incremental_update(x, y, 1),
      ),
  )
  def test_none_argument(self, operation):
    x = jnp.array([1.0, 2.0, 3.0])
    operation(None, x)


if __name__ == '__main__':
  absltest.main()
