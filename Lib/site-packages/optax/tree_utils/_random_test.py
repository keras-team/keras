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
"""Tests for methods defined in optax.tree_utils._random."""

from collections.abc import Callable

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
from optax import tree_utils as otu

# We consider samplers with varying input dtypes, we do not test all possible
# samplers from `jax.random`.
_SAMPLER_DTYPES = (
    dict(sampler=jrd.normal, dtype=None),
    dict(sampler=jrd.normal, dtype='bfloat16'),
    dict(sampler=jrd.normal, dtype='float32'),
    dict(sampler=jrd.rademacher, dtype='int32'),
    dict(sampler=jrd.bits, dtype='uint32'),
)


def get_variable(type_var: str):
  """Get a variable of various shape."""
  if type_var == 'real_array':
    return jnp.asarray([1.0, 2.0])
  if type_var == 'complex_array':
    return jnp.asarray([1.0 + 1j * 2.0, 3.0 + 4j * 5.0])
  if type_var == 'pytree':
    pytree = {'k1': 1.0, 'k2': (2.0, 3.0), 'k3': jnp.asarray([4.0, 5.0])}
    return jax.tree.map(jnp.asarray, pytree)


class RandomTest(chex.TestCase):

  def test_tree_split_key_like(self):
    rng_key = jrd.PRNGKey(0)
    tree = {'a': jnp.zeros(2), 'b': {'c': [jnp.ones(3), jnp.zeros([4, 5])]}}
    keys_tree = otu.tree_split_key_like(rng_key, tree)

    with self.subTest('Test structure matches'):
      self.assertEqual(jax.tree.structure(tree), jax.tree.structure(keys_tree))

    with self.subTest('Test random key split'):
      fst = jnp.stack(jax.tree.flatten(keys_tree)[0])
      snd = jrd.split(rng_key, jax.tree.structure(tree).num_leaves)
      np.testing.assert_array_equal(fst, snd)

  @parameterized.product(
      _SAMPLER_DTYPES,
      type_var=['real_array', 'complex_array', 'pytree'],
  )
  def test_tree_random_like(
      self,
      sampler: Callable[
          [chex.PRNGKey, chex.Shape, chex.ArrayDType], chex.Array
      ],
      dtype: str,
      type_var: str,
  ):
    """Test that tree_random_like matches its flat counterpart."""
    if dtype is not None:
      dtype = jnp.dtype(dtype)
    rng_key = jrd.PRNGKey(0)
    target_tree = get_variable(type_var)

    rand_tree = otu.tree_random_like(
        rng_key, target_tree, sampler=sampler, dtype=dtype
    )

    flat_tree, tree_def = jax.tree.flatten(target_tree)

    with self.subTest('Test structure matches'):
      self.assertEqual(tree_def, jax.tree.structure(rand_tree))

    with self.subTest('Test tree_random_like matches flat random like'):
      flat_rand_tree, _ = jax.tree.flatten(rand_tree)
      keys = jrd.split(rng_key, tree_def.num_leaves)
      expected_flat_rand_tree = [
          sampler(key, x.shape, dtype or x.dtype)
          for key, x in zip(keys, flat_tree)
      ]
      chex.assert_trees_all_close(flat_rand_tree, expected_flat_rand_tree)

    with self.subTest('Test dtype are as expected'):
      if dtype is not None:
        for x in jax.tree.leaves(rand_tree):
          self.assertEqual(x.dtype, dtype)
      else:
        chex.assert_trees_all_equal_dtypes(rand_tree, target_tree)


if __name__ == '__main__':
  absltest.main()
