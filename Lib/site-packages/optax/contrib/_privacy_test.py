# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for methods in `privacy.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from optax.contrib import _privacy


class DifferentiallyPrivateAggregateTest(chex.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 8
    self.params = {
        'key_a': (jnp.zeros((2, 3, 4)), jnp.zeros([])),
        'key_b': jnp.zeros((6, 7)),
    }
    # Example `i`'s grads are full of `i`s. Important to include 0 to ensure
    # there are no divisions by 0 (e.g. in norm clipping)
    a = jnp.arange(self.batch_size)
    self.per_eg_grads = jax.tree.map(
        lambda p: jnp.moveaxis(
            a * jnp.ones(p.shape + (self.batch_size,)), -1, 0
        ),
        self.params,
    )

  @chex.all_variants
  def test_no_privacy(self):
    """l2_norm_clip=MAX_FLOAT32 and noise_multiplier=0 should recover SGD."""
    dp_agg = _privacy.differentially_private_aggregate(
        l2_norm_clip=jnp.finfo(jnp.float32).max, noise_multiplier=0.0, seed=0
    )
    state = dp_agg.init(self.params)
    update_fn = self.variant(dp_agg.update)
    mean_grads = jax.tree.map(lambda g: g.mean(0), self.per_eg_grads)

    for _ in range(3):
      updates, state = update_fn(self.per_eg_grads, state)
      chex.assert_trees_all_close(updates, mean_grads)

  @chex.all_variants
  @parameterized.parameters(0.5, 10.0, 20.0, 40.0, 80.0)
  def test_clipping_norm(self, l2_norm_clip):
    dp_agg = _privacy.differentially_private_aggregate(
        l2_norm_clip=l2_norm_clip, noise_multiplier=0.0, seed=42
    )
    state = dp_agg.init(self.params)
    update_fn = self.variant(dp_agg.update)

    # Shape of the three arrays below is (self.batch_size, )
    norms = [
        jnp.linalg.norm(g.reshape(self.batch_size, -1), axis=1)
        for g in jax.tree.leaves(self.per_eg_grads)
    ]
    global_norms = jnp.linalg.norm(jnp.stack(norms), axis=0)
    divisors = jnp.maximum(global_norms / l2_norm_clip, 1.0)
    # Since the values of all the parameters are the same within each example,
    # we can easily compute what the values should be:
    expected_val = jnp.mean(jnp.arange(self.batch_size) / divisors)
    expected_tree = jax.tree.map(
        lambda p: jnp.broadcast_to(expected_val, p.shape), self.params
    )

    for _ in range(3):
      updates, state = update_fn(self.per_eg_grads, state, self.params)
      chex.assert_trees_all_close(updates, expected_tree, rtol=2e-7)

  @chex.all_variants
  @parameterized.parameters((3.0, 2.0), (1.0, 5.0), (100.0, 4.0), (1.0, 90.0))
  def test_noise_multiplier(self, l2_norm_clip, noise_multiplier):
    """Standard dev. of noise should be l2_norm_clip * noise_multiplier."""
    dp_agg = _privacy.differentially_private_aggregate(
        l2_norm_clip=l2_norm_clip, noise_multiplier=noise_multiplier, seed=1337
    )
    state = dp_agg.init(self.params)
    update_fn = self.variant(dp_agg.update)
    expected_std = l2_norm_clip * noise_multiplier

    grads = [jnp.ones((1, 100, 100))]  # batch size 1
    for _ in range(3):
      updates, state = update_fn(grads, state)
      chex.assert_trees_all_close(
          expected_std, jnp.std(updates[0]), atol=0.1 * expected_std
      )

  def test_aggregated_updates_as_input_fails(self):
    """Expect per-example gradients as input to this transform."""
    dp_agg = _privacy.differentially_private_aggregate(
        l2_norm_clip=0.1, noise_multiplier=1.1, seed=2021
    )
    state = dp_agg.init(self.params)
    mean_grads = jax.tree.map(lambda g: g.mean(0), self.per_eg_grads)
    with self.assertRaises(ValueError):
      dp_agg.update(mean_grads, state, self.params)


if __name__ == '__main__':
  absltest.main()
