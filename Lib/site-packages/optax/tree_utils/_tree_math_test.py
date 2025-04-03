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
"""Tests for methods in `optax.tree_utils._tree_math.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax import flatten_util
import jax.numpy as jnp
import numpy as np
from optax import tree_utils as tu


class TreeUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    rng = np.random.RandomState(0)

    self.rng_jax = jax.random.PRNGKey(0)

    self.tree_a = (rng.randn(20, 10) + 1j * rng.randn(20, 10), rng.randn(20))
    self.tree_b = (rng.randn(20, 10), rng.randn(20))

    self.tree_a_dict = (1.0, {'k1': 1.0, 'k2': (1.0, 1.0)}, 1.0)
    self.tree_b_dict = (1.0, {'k1': 2.0, 'k2': (3.0, 4.0)}, 5.0)

    self.array_a = rng.randn(20) + 1j * rng.randn(20)
    self.array_b = rng.randn(20)

    self.tree_a_dict_jax = jax.tree.map(jnp.array, self.tree_a_dict)
    self.tree_b_dict_jax = jax.tree.map(jnp.array, self.tree_b_dict)

    self.data = dict(
        tree_a=self.tree_a,
        tree_b=self.tree_b,
        tree_a_dict=self.tree_a_dict,
        tree_b_dict=self.tree_b_dict,
        array_a=self.array_a,
        array_b=self.array_b,
    )

  def test_tree_add(self):
    expected = self.array_a + self.array_b
    got = tu.tree_add(self.array_a, self.array_b)
    np.testing.assert_array_almost_equal(expected, got)

    expected = (
        self.tree_a[0] + self.tree_b[0],
        self.tree_a[1] + self.tree_b[1],
    )
    got = tu.tree_add(self.tree_a, self.tree_b)
    chex.assert_trees_all_close(expected, got)

  def test_tree_sub(self):
    expected = self.array_a - self.array_b
    got = tu.tree_sub(self.array_a, self.array_b)
    np.testing.assert_array_almost_equal(expected, got)

    expected = (
        self.tree_a[0] - self.tree_b[0],
        self.tree_a[1] - self.tree_b[1],
    )
    got = tu.tree_sub(self.tree_a, self.tree_b)
    chex.assert_trees_all_close(expected, got)

  def test_tree_mul(self):
    expected = self.array_a * self.array_b
    got = tu.tree_mul(self.array_a, self.array_b)
    np.testing.assert_array_almost_equal(expected, got)

    expected = (
        self.tree_a[0] * self.tree_b[0],
        self.tree_a[1] * self.tree_b[1],
    )
    got = tu.tree_mul(self.tree_a, self.tree_b)
    chex.assert_trees_all_close(expected, got)

  def test_tree_div(self):
    expected = self.array_a / self.array_b
    got = tu.tree_div(self.array_a, self.array_b)
    np.testing.assert_array_almost_equal(expected, got)

    expected = (
        self.tree_a[0] / self.tree_b[0],
        self.tree_a[1] / self.tree_b[1],
    )
    got = tu.tree_div(self.tree_a, self.tree_b)
    chex.assert_trees_all_close(expected, got)

  def test_tree_scalar_mul(self):
    expected = 0.5 * self.array_a
    got = tu.tree_scalar_mul(0.5, self.array_a)
    np.testing.assert_array_almost_equal(expected, got)

    expected = (0.5 * self.tree_a[0], 0.5 * self.tree_a[1])
    got = tu.tree_scalar_mul(0.5, self.tree_a)
    chex.assert_trees_all_close(expected, got)

  def test_tree_add_scalar_mul(self):
    expected = (
        self.tree_a[0] + 0.5 * self.tree_b[0],
        self.tree_a[1] + 0.5 * self.tree_b[1],
    )
    got = tu.tree_add_scalar_mul(self.tree_a, 0.5, self.tree_b)
    chex.assert_trees_all_close(expected, got)

  def test_tree_vdot(self):
    expected = jnp.vdot(self.array_a, self.array_b)
    got = tu.tree_vdot(self.array_a, self.array_b)
    np.testing.assert_allclose(expected, got)

    expected = 15.0
    got = tu.tree_vdot(self.tree_a_dict, self.tree_b_dict)
    np.testing.assert_allclose(expected, got)

    expected = jnp.vdot(self.tree_a[0], self.tree_b[0]) + jnp.vdot(
        self.tree_a[1], self.tree_b[1]
    )
    got = tu.tree_vdot(self.tree_a, self.tree_b)
    chex.assert_trees_all_close(expected, got)

  def test_tree_sum(self):
    expected = jnp.sum(self.array_a)
    got = tu.tree_sum(self.array_a)
    np.testing.assert_allclose(expected, got)

    expected = jnp.sum(self.tree_a[0]) + jnp.sum(self.tree_a[1])
    got = tu.tree_sum(self.tree_a)
    np.testing.assert_allclose(expected, got)

  @parameterized.parameters(
      'array_a', 'tree_a', 'tree_a_dict', 'tree_b', 'tree_b_dict'
  )
  def test_tree_max(self, key):
    tree = self.data[key]
    values, _ = flatten_util.ravel_pytree(tree)
    expected = jnp.max(values)
    got = tu.tree_max(tree)
    np.testing.assert_allclose(expected, got)

  def test_tree_l2_norm(self):
    expected = jnp.sqrt(jnp.vdot(self.array_a, self.array_a).real)
    got = tu.tree_l2_norm(self.array_a)
    np.testing.assert_allclose(expected, got)

    expected = jnp.sqrt(
        jnp.vdot(self.tree_a[0], self.tree_a[0]).real
        + jnp.vdot(self.tree_a[1], self.tree_a[1]).real
    )
    got = tu.tree_l2_norm(self.tree_a)
    np.testing.assert_allclose(expected, got)

  @parameterized.parameters(
      'tree_a', 'tree_a_dict', 'tree_b', 'array_a', 'array_b', 'tree_b_dict'
  )
  def test_tree_l1_norm(self, key):
    tree = self.data[key]
    values, _ = flatten_util.ravel_pytree(tree)
    expected = jnp.sum(jnp.abs(values))
    got = tu.tree_l1_norm(tree)
    np.testing.assert_allclose(expected, got, atol=1e-4)

  @parameterized.parameters(
      'tree_a', 'tree_a_dict', 'tree_b', 'array_a', 'array_b', 'tree_b_dict'
  )
  def test_tree_linf_norm(self, key):
    tree = self.data[key]
    values, _ = flatten_util.ravel_pytree(tree)
    expected = jnp.max(jnp.abs(values))
    got = tu.tree_linf_norm(tree)
    np.testing.assert_allclose(expected, got, atol=1e-4)

  def test_tree_zeros_like(self):
    expected = jnp.zeros_like(self.array_a)
    got = tu.tree_zeros_like(self.array_a)
    np.testing.assert_array_almost_equal(expected, got)

    expected = (jnp.zeros_like(self.tree_a[0]), jnp.zeros_like(self.tree_a[1]))
    got = tu.tree_zeros_like(self.tree_a)
    chex.assert_trees_all_close(expected, got)

  def test_tree_ones_like(self):
    expected = jnp.ones_like(self.array_a)
    got = tu.tree_ones_like(self.array_a)
    np.testing.assert_array_almost_equal(expected, got)

    expected = (jnp.ones_like(self.tree_a[0]), jnp.ones_like(self.tree_a[1]))
    got = tu.tree_ones_like(self.tree_a)
    chex.assert_trees_all_close(expected, got)

  def test_add_multiple_trees(self):
    """Test adding more than 2 trees with tree_add."""
    trees = [self.tree_a_dict_jax, self.tree_a_dict_jax, self.tree_a_dict_jax]
    expected = tu.tree_scalar_mul(3.0, self.tree_a_dict_jax)
    got = tu.tree_add(*trees)
    chex.assert_trees_all_close(expected, got)

  def test_tree_clip(self):
    """Clip the tree to range [min_value, max_value]."""
    expected = tu.tree_scalar_mul(0.5, self.tree_a_dict_jax)
    got = tu.tree_clip(self.tree_a_dict_jax, min_value=0, max_value=0.5)
    chex.assert_trees_all_close(expected, got)
    expected = tu.tree_scalar_mul(0.5, self.tree_a_dict_jax)
    got = tu.tree_clip(self.tree_a_dict_jax, min_value=None, max_value=0.5)
    chex.assert_trees_all_close(expected, got)
    expected = tu.tree_scalar_mul(2.0, self.tree_a_dict_jax)
    got = tu.tree_clip(self.tree_a_dict_jax, min_value=2.0, max_value=None)
    chex.assert_trees_all_close(expected, got)

  def test_update_infinity_moment(self):
    values = jnp.array([5.0, 7.0])
    decay = 0.9
    d = decay

    transform_fn = tu.tree_update_infinity_moment

    # identity if updating with itself (and positive decay)
    np.testing.assert_allclose(
        transform_fn(values, values, decay=d, eps=0.0), values, atol=1e-4
    )
    # return (decayed) max when updating with zeros
    np.testing.assert_allclose(
        transform_fn(jnp.zeros_like(values), values, decay=d, eps=0.0),
        d * values,
        atol=1e-4,
    )
    # infinity norm takes absolute values
    np.testing.assert_allclose(
        transform_fn(-values, jnp.zeros_like(values), decay=d, eps=0.0),
        values,
        atol=1e-4,
    )
    # return at least `eps`
    np.testing.assert_allclose(
        transform_fn(
            jnp.zeros_like(values), jnp.zeros_like(values), decay=d, eps=1e-2
        ),
        jnp.ones_like(values) * 1e-2,
        atol=1e-4,
    )

  def test_bias_correction_bf16(self):
    m = jnp.logspace(-10, 10, num=21, dtype=jnp.bfloat16)  # 1e-10 ... 1e10
    for decay in (0.9, 0.99, 0.999, 0.9995):
      for count in (1, 10, 100, 1000):
        chex.assert_tree_all_finite(
            tu.tree_bias_correction(m, decay, count),
            custom_message=f'failed with decay={decay}, count={count}',
        )

  def test_empty_tree_reduce(self):
    for tree in [{}, (), [], None, {'key': [None, [None]]}]:
      self.assertEqual(tu.tree_sum(tree), 0)
      self.assertEqual(tu.tree_vdot(tree, tree), 0)

  @parameterized.named_parameters(
      dict(
          testcase_name='tree_add_scalar_mul',
          operation=lambda m: tu.tree_add_scalar_mul(None, 1, m),
      ),
      dict(
          testcase_name='tree_update_moment',
          operation=lambda m: tu.tree_update_moment(None, m, 1, 1),
      ),
      dict(
          testcase_name='tree_update_infinity_moment',
          operation=lambda m: tu.tree_update_infinity_moment(None, m, 1, 1),
      ),
      dict(
          testcase_name='tree_update_moment_per_elem_norm',
          operation=lambda m: tu.tree_update_moment_per_elem_norm(
              None, m, 1, 1
          ),
      ),
  )
  def test_none_arguments(self, operation):
    m = jnp.array([1.0, 2.0, 3.0])
    operation(m)


if __name__ == '__main__':
  absltest.main()
