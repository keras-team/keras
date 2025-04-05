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
"""Utilities to generate random pytrees."""

from collections.abc import Callable
from typing import Optional

import chex
import jax


def tree_split_key_like(
    rng_key: chex.PRNGKey, target_tree: chex.ArrayTree
) -> chex.ArrayTree:
  """Split keys to match structure of target tree.

  Args:
    rng_key: the key to split.
    target_tree: the tree whose structure to match.

  Returns:
    a tree of rng keys.
  """
  tree_def = jax.tree.structure(target_tree)
  keys = jax.random.split(rng_key, tree_def.num_leaves)
  return jax.tree.unflatten(tree_def, keys)


def tree_random_like(
    rng_key: chex.PRNGKey,
    target_tree: chex.ArrayTree,
    sampler: Callable[
        [chex.PRNGKey, chex.Shape, chex.ArrayDType], chex.Array
    ] = jax.random.normal,
    dtype: Optional[chex.ArrayDType] = None,
) -> chex.ArrayTree:
  """Create tree with random entries of the same shape as target tree.

  Args:
    rng_key: the key for the random number generator.
    target_tree: the tree whose structure to match. Leaves must be arrays.
    sampler: the noise sampling function, by default ``jax.random.normal``.
    dtype: the desired dtype for the random numbers, passed to ``sampler``. If
      None, the dtype of the target tree is used if possible.

  Returns:
    a random tree with the same structure as ``target_tree``, whose leaves have
    distribution ``sampler``.

  .. warning::
    The possible dtypes may be limited by the sampler, for example
    ``jax.random.rademacher`` only supports integer dtypes and will raise an
    error if the dtype of the target tree is not an integer or if the dtype
    is not of integer type.

  .. versionadded:: 0.2.1
  """
  keys_tree = tree_split_key_like(rng_key, target_tree)
  return jax.tree.map(
      lambda leaf, key: sampler(key, leaf.shape, dtype or leaf.dtype),
      target_tree,
      keys_tree,
  )
