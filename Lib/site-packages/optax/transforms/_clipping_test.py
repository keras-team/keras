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
"""Tests for methods in `optax.transforms._clipping.py`."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax._src import linear_algebra
from optax.transforms import _clipping


STEPS = 50
LR = 1e-2


class ClippingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    self.per_step_updates = (jnp.array([500.0, 5.0]), jnp.array([300.0, 3.0]))

  def test_clip(self):
    updates = self.per_step_updates
    # For a sufficiently high delta the update should not be changed.
    clipper = _clipping.clip(1e6)
    clipped_updates, _ = clipper.update(updates, None)
    chex.assert_trees_all_close(clipped_updates, clipped_updates)
    # Clipping at delta=1 should make all updates exactly 1.
    clipper = _clipping.clip(1.0)
    clipped_updates, _ = clipper.update(updates, None)
    chex.assert_trees_all_close(
        clipped_updates, jax.tree.map(jnp.ones_like, updates)
    )

  def test_clip_by_block_rms(self):
    rmf_fn = lambda t: jnp.sqrt(jnp.mean(t**2))
    updates = self.per_step_updates
    for i in range(1, STEPS + 1):
      clipper = _clipping.clip_by_block_rms(1.0 / i)
      # Check that the clipper actually works and block rms is <= threshold
      updates, _ = clipper.update(updates, None)
      self.assertAlmostEqual(rmf_fn(updates[0]), 1.0 / i)
      self.assertAlmostEqual(rmf_fn(updates[1]), 1.0 / i)
      # Check that continuously clipping won't cause numerical issues.
      updates_step, _ = clipper.update(self.per_step_updates, None)
      chex.assert_trees_all_close(updates, updates_step)

  def test_clip_by_global_norm(self):
    updates = self.per_step_updates
    for i in range(1, STEPS + 1):
      clipper = _clipping.clip_by_global_norm(1.0 / i)
      # Check that the clipper actually works and global norm is <= max_norm
      updates, _ = clipper.update(updates, None)
      self.assertAlmostEqual(
          linear_algebra.global_norm(updates), 1.0 / i, places=6
      )
      # Check that continuously clipping won't cause numerical issues.
      updates_step, _ = clipper.update(self.per_step_updates, None)
      chex.assert_trees_all_close(updates, updates_step)

  def test_adaptive_grad_clip(self):
    updates = self.per_step_updates
    params = self.init_params
    for i in range(1, STEPS + 1):
      clip_r = 1.0 / i
      clipper = _clipping.adaptive_grad_clip(clip_r)

      # Check that the clipper actually works and upd_norm is < c * param_norm.
      updates, _ = clipper.update(updates, None, params)
      u_norm, p_norm = jax.tree.map(_clipping.unitwise_norm, (updates, params))
      cmp = jax.tree.map(
          lambda u, p, c=clip_r: u - c * p < 1e-6, u_norm, p_norm
      )
      for leaf in jax.tree.leaves(cmp):
        self.assertTrue(leaf.all())

      # Check that continuously clipping won't cause numerical issues.
      updates_step, _ = clipper.update(self.per_step_updates, None, params)
      chex.assert_trees_all_close(updates, updates_step)

  def test_per_example_global_norm_clip(self):
    grads = [  # 3 users, 2 components
        jnp.array([
            [0, -0.5],  # norm = sqrt(0^2 + 0.5^2 + 0^2)
            [3, 4],  # norm = sqrt(3^2 + 4^2 + 5^2)
            [5, 6],  # norm = sqrt(5^2 + 6^2 + 3^2)
            [0, 0],  # norm = 0
        ]),
        jnp.array([[0], [5], [-3], [0]]),
    ]
    answer = [
        jnp.array([0, -0.5])
        + jnp.array([3, 4]) / jnp.sqrt(50)
        + jnp.array([5, 6]) / jnp.sqrt(70),
        jnp.array([0])
        + jnp.array([5]) / jnp.sqrt(50)
        + jnp.array([-3]) / jnp.sqrt(70),
    ]
    sum_clipped_grads, num_clipped = _clipping.per_example_global_norm_clip(
        grads, l2_norm_clip=1.0
    )

    for actual, expected in zip(sum_clipped_grads, answer):
      np.testing.assert_allclose(actual, expected, atol=1e-6)
    self.assertEqual(num_clipped, 2)

  def test_per_example_layer_norm_clip(self):
    # Test data for a model with two layers and a batch size of 4. The
    # 0th layer has one parameter (shape (1)), and the 1st layer has shape
    # (3, 3, 2).
    grads_flat = [
        jnp.array([[0.5], [1.5], [-2.0], [3.0]]),
        jnp.ones([4, 3, 3, 2], dtype=jnp.float32),
    ]

    with self.subTest(name='Uniform Variant'):
      sum_clipped_grads, num_clipped = _clipping.per_example_layer_norm_clip(
          grads_flat, global_l2_norm_clip=jnp.sqrt(2), uniform=True
      )

      # For the uniform variant, with global_l2_norm_clip=sqrt(2), the per-layer
      # clip norm is 1.0. Thus the per-example per-layer clipped grads are
      # [[0.5], [1.0], [-1.0], [1.0]] and [1 / sqrt(18) ... ]. The sum of
      # these over the 4 input gradients are [1.5] and [4 / sqrt(18) ...].
      self.assertAlmostEqual(sum_clipped_grads[0], 1.5)
      for element in sum_clipped_grads[1].flatten():
        self.assertAlmostEqual(element, 4 / jnp.sqrt(18), places=4)

      # The three values in grads_flat[0] with magnitude > 1.0 are clipped, as
      # are all four values in grads_flat[1].
      self.assertEqual(num_clipped[0], 3)
      self.assertEqual(num_clipped[1], 4)

    with self.subTest(name='Scaled Variant'):
      sum_clipped_grads, num_clipped = _clipping.per_example_layer_norm_clip(
          grads_flat, global_l2_norm_clip=jnp.sqrt(19), uniform=False
      )

      # For the scaled variant, with global_l2_norm_clip=sqrt(19), the per-layer
      # clip norm for the 0th layer is 1.0, and the per-layer clip norm for
      # the 1st layer is sqrt(18). Thus the per-example per-layer clipped grads
      # are [[0.5], [1.0], [-1.0], [1.0]] and [[1.0)] ... ]. The sum of
      # these over the 4 input gradients are [1.5] and [4.0 ...].
      self.assertAlmostEqual(sum_clipped_grads[0], 1.5)
      for element in sum_clipped_grads[1].flatten():
        self.assertAlmostEqual(element, 4.0)

      # The three values in grads_flat[0] with magnitude > 1.0 are clipped. The
      # grad norms for grads_flat[1] are all equal to the per-layer clip norm,
      # so none of these grads are clipped.
      self.assertEqual(num_clipped[0], 3)
      self.assertEqual(num_clipped[1], 0)


if __name__ == '__main__':
  absltest.main()
