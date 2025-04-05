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
"""Tests for tree utilities on data types."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from optax import tree_utils as otu


class CastingTest(parameterized.TestCase):

  @parameterized.parameters([
      (jnp.float32, [1.3, 2.001, 3.6], [-3.3], [1.3, 2.001, 3.6], [-3.3]),
      (jnp.float32, [1.3, 2.001, 3.6], [-3], [1.3, 2.001, 3.6], [-3.0]),
      (jnp.int32, [1.3, 2.001, 3.6], [-3.3], [1, 2, 3], [-3]),
      (jnp.int32, [1.3, 2.001, 3.6], [-3], [1, 2, 3], [-3]),
      (None, [1.123, 2.33], [0.0], [1.123, 2.33], [0.0]),
      (None, [1, 2, 3], [0.0], [1, 2, 3], [0.0]),
  ])
  def test_tree_cast(self, dtype, b, c, new_b, new_c):
    def _build_tree(val1, val2):
      dict_tree = {'a': {'b': jnp.array(val1)}, 'c': jnp.array(val2)}
      return jax.tree.map(lambda x: x, dict_tree)

    tree = _build_tree(b, c)
    tree = otu.tree_cast(tree, dtype=dtype)
    jax.tree.map(np.testing.assert_array_equal, tree, _build_tree(new_b, new_c))

  def test_tree_dtype(self):
    """Test fecthing data type of a tree."""

    with self.subTest('Check that it returns the right dtype'):
      tree = {
          'a': {'b': jnp.array(1.0, dtype=jnp.float32)},
          'c': jnp.array(2.0, dtype=jnp.float32),
      }
      dtype = otu.tree_dtype(tree)
      self.assertEqual(dtype, jnp.float32)

    with self.subTest('Check that it raises an error if dtypes differ'):
      tree = {
          'a': {'b': jnp.array(1.0, dtype=jnp.bfloat16)},
          'c': jnp.array(2.0, dtype=jnp.float32),
      }
      self.assertRaises(ValueError, otu.tree_dtype, tree)

    tree = {
        'a': {'b': jnp.array(1.0, dtype=jnp.bfloat16)},
        'c': jnp.array(2.0, dtype=jnp.float32),
    }

    with self.subTest('Check that it works with lowest common dtype'):
      dtype = otu.tree_dtype(tree, 'lowest')
      self.assertEqual(dtype, jnp.bfloat16)

    with self.subTest('Check that it works with highest common dtype'):
      dtype = otu.tree_dtype(tree, 'highest')
      self.assertEqual(dtype, jnp.float32)

    tree = {
        'a': {'b': jnp.array(1.0, dtype=jnp.bfloat16)},
        'c': jnp.array(2.0, dtype=jnp.float16),
    }

    with self.subTest('Check that it works when promoting mixed dtype'):
      dtype = otu.tree_dtype(tree, 'promote')
      self.assertEqual(dtype, jnp.float32)

    with self.subTest(
        'Check that it raises an error if no dtypes cannot be promoted to one'
        ' another'
    ):
      self.assertRaises(ValueError, otu.tree_dtype, tree, 'lowest')
      self.assertRaises(ValueError, otu.tree_dtype, tree, 'highest')

  @parameterized.named_parameters(
      dict(testcase_name='empty_dict', tree={}),
      dict(testcase_name='empty_list', tree=[]),
      dict(testcase_name='empty_tuple', tree=()),
      dict(testcase_name='empty_none', tree=None),
  )
  def test_tree_dtype_utilities_with_empty_trees(self, tree):
    """Test tree data type utilities on empty trees."""
    default_dtype = jnp.asarray(1.0).dtype

    with self.subTest('Check tree_dtype works with empty trees.'):
      dtype = otu.tree_dtype(tree)
      self.assertEqual(dtype, default_dtype)


if __name__ == '__main__':
  absltest.main()
