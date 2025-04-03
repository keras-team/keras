# Copyright 2021 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""A bounded functional stack implementation.

Used as a helper for expressing recursive algorithms such as QDWH-eig for
Eigendecomposition on TPU.
"""

from __future__ import annotations

from typing import Any

import jax
from jax import lax
import jax.numpy as jnp

class Stack:
  """A bounded functional stack implementation. Elements may be pytrees."""
  def __init__(self, size, data):
    """Private constructor."""
    self._size = size
    self._data = data

  def __repr__(self):
    return f"Stack({self._size}, {self._data})"

  @staticmethod
  def create(capacity: int, prototype: Any) -> Stack:
    """Creates a stack with size `capacity` with elements like `prototype`.

    `prototype` can be any JAX pytree. This function looks only at its
    structure; the specific values are ignored.
    """
    return Stack(
      jnp.array(0, jnp.int32),
      jax.tree_util.tree_map(
        lambda x: jnp.zeros((capacity,) + tuple(x.shape), x.dtype), prototype))

  def empty(self) -> Any:
    """Returns true if the stack is empty."""
    return self._size == 0

  def push(self, elem: Any) -> Stack:
    """Pushes `elem` onto the stack, returning the updated stack."""
    return Stack(
      self._size + 1,
      jax.tree_util.tree_map(
        lambda x, y: lax.dynamic_update_index_in_dim(x, y, self._size, 0),
        self._data, elem))

  def pop(self) -> tuple[Any, Stack]:
    """Pops from the stack, returning an (elem, updated stack) pair."""
    elem = jax.tree_util.tree_map(
      lambda x: lax.dynamic_index_in_dim(x, self._size - 1, 0, keepdims=False),
      self._data)
    return elem, Stack(self._size - 1, self._data)

  def flatten(self):
    leaves, treedef = jax.tree_util.tree_flatten(self._data)
    return ([self._size] + leaves), treedef

  @staticmethod
  def unflatten(treedef, leaves):
    return Stack(leaves[0], jax.tree_util.tree_unflatten(treedef, leaves[1:]))

jax.tree_util.register_pytree_node(Stack, Stack.flatten, Stack.unflatten)
