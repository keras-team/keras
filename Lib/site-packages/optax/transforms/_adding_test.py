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
"""Tests for methods in `optax.transforms._adding.py`."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
from optax.transforms import _adding

STEPS = 50


class AddingTest(chex.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    self.per_step_updates = (jnp.array([500.0, 5.0]), jnp.array([300.0, 3.0]))

  @chex.all_variants
  def test_add_decayed_weights(self):
    # Define a transform that add decayed weights.
    # We can define a mask either as a pytree, or as a function that
    # returns the pytree. Below we define the pytree directly.
    mask = (True, dict(a=True, b=False))
    tx = _adding.add_decayed_weights(0.1, mask=mask)
    # Define input updates and weights.
    updates = (
        jnp.zeros((2,), dtype=jnp.float32),
        dict(
            a=jnp.zeros((2,), dtype=jnp.float32),
            b=jnp.zeros((2,), dtype=jnp.float32),
        ),
    )
    weights = (
        jnp.ones((2,), dtype=jnp.float32),
        dict(
            a=jnp.ones((2,), dtype=jnp.float32),
            b=jnp.ones((2,), dtype=jnp.float32),
        ),
    )
    # This mask means that we will add decayed weights to the first two
    # terms in the input updates, but not to the last element.
    expected_tx_updates = (
        0.1 * jnp.ones((2,), dtype=jnp.float32),
        dict(
            a=0.1 * jnp.ones((2,), dtype=jnp.float32),
            b=jnp.zeros((2,), dtype=jnp.float32),
        ),
    )
    # Apply transform
    state = tx.init(weights)
    transform_fn = self.variant(tx.update)
    new_updates, _ = transform_fn(updates, state, weights)
    # Assert output as expected.
    chex.assert_trees_all_close(new_updates, expected_tx_updates)

  @chex.all_variants
  def test_add_noise_has_correct_variance_scaling(self):
    # Prepare to compare noise with a rescaled unit-variance substitute.
    eta = 0.3
    gamma = 0.55
    seed = 314
    noise = _adding.add_noise(eta, gamma, seed)
    noise_unit = _adding.add_noise(1.0, 0.0, seed)

    params = self.init_params
    state = noise.init(params)
    state_unit = noise_unit.init(params)

    # Check the noise itself by adding it to zeros.
    updates = jax.tree.map(jnp.zeros_like, params)

    for i in range(1, STEPS + 1):
      updates_i, state = self.variant(noise.update)(updates, state)
      updates_i_unit, state_unit = noise_unit.update(updates, state_unit)

      scale = jnp.sqrt(eta / i**gamma)

      updates_i_rescaled = jax.tree.map(
          lambda g, s=scale: g * s, updates_i_unit
      )

      chex.assert_trees_all_close(updates_i, updates_i_rescaled, rtol=1e-4)

  def test_none_argument(self):
    weights = (
        jnp.ones((2,), dtype=jnp.float32),
        dict(
            a=jnp.ones((2,), dtype=jnp.float32),
            b=jnp.ones((2,), dtype=jnp.float32),
        ),
    )
    tf = _adding.add_decayed_weights(0.1, mask=None)
    tf.update(None, 0, weights)


if __name__ == "__main__":
  absltest.main()
