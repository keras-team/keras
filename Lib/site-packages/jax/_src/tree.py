# Copyright 2024 The JAX Authors.
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
# limitations under the License.
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar, overload

from jax._src import tree_util

T = TypeVar("T")


def all(tree: Any, *, is_leaf: Callable[[Any], bool] | None = None) -> bool:
  """Call all() over the leaves of a tree.

  Args:
    tree: the pytree to evaluate
    is_leaf : an optionally specified function that will be called at each
      flattening step. It should return a boolean, which indicates whether the
      flattening should traverse the current object, or if it should be stopped
      immediately, with the whole subtree being treated as a leaf.

  Returns:
    result: boolean True or False

  Examples:
    >>> import jax
    >>> jax.tree.all([True, {'a': True, 'b': (True, True)}])
    True
    >>> jax.tree.all([False, (True, False)])
    False

  See Also:
    - :func:`jax.tree.reduce`
    - :func:`jax.tree.leaves`
  """
  return tree_util.tree_all(tree, is_leaf=is_leaf)


def flatten(tree: Any,
            is_leaf: Callable[[Any], bool] | None = None
            ) -> tuple[list[tree_util.Leaf], tree_util.PyTreeDef]:
  """Flattens a pytree.

  The flattening order (i.e. the order of elements in the output list)
  is deterministic, corresponding to a left-to-right depth-first tree
  traversal.

  Args:
    tree: a pytree to flatten.
    is_leaf: an optionally specified function that will be called at each
      flattening step. It should return a boolean, with true stopping the
      traversal and the whole subtree being treated as a leaf, and false
      indicating the flattening should traverse the current object.

  Returns:
    A pair where the first element is a list of leaf values and the second
    element is a treedef representing the structure of the flattened tree.

  Examples:
    >>> import jax
    >>> vals, treedef = jax.tree.flatten([1, (2, 3), [4, 5]])
    >>> vals
    [1, 2, 3, 4, 5]
    >>> treedef
    PyTreeDef([*, (*, *), [*, *]])

  See Also:
    - :func:`jax.tree.leaves`
    - :func:`jax.tree.structure`
    - :func:`jax.tree.unflatten`
  """
  return tree_util.tree_flatten(tree, is_leaf)


def leaves(tree: Any,
           is_leaf: Callable[[Any], bool] | None = None
           ) -> list[tree_util.Leaf]:
  """Gets the leaves of a pytree.

  Args:
    tree: the pytree for which to get the leaves
    is_leaf : an optionally specified function that will be called at each
      flattening step. It should return a boolean, which indicates whether the
      flattening should traverse the current object, or if it should be stopped
      immediately, with the whole subtree being treated as a leaf.

  Returns:
    leaves: a list of tree leaves.

  Examples:
    >>> import jax
    >>> jax.tree.leaves([1, (2, 3), [4, 5]])
    [1, 2, 3, 4, 5]

  See Also:
    - :func:`jax.tree.flatten`
    - :func:`jax.tree.structure`
    - :func:`jax.tree.unflatten`
  """
  return tree_util.tree_leaves(tree, is_leaf)


def map(f: Callable[..., Any],
        tree: Any,
        *rest: Any,
        is_leaf: Callable[[Any], bool] | None = None) -> Any:
  """Maps a multi-input function over pytree args to produce a new pytree.

  Args:
    f: function that takes ``1 + len(rest)`` arguments, to be applied at the
      corresponding leaves of the pytrees.
    tree: a pytree to be mapped over, with each leaf providing the first
      positional argument to ``f``.
    rest: a tuple of pytrees, each of which has the same structure as ``tree``
      or has ``tree`` as a prefix.
    is_leaf: an optionally specified function that will be called at each
      flattening step. It should return a boolean, which indicates whether the
      flattening should traverse the current object, or if it should be stopped
      immediately, with the whole subtree being treated as a leaf.

  Returns:
    A new pytree with the same structure as ``tree`` but with the value at each
    leaf given by ``f(x, *xs)`` where ``x`` is the value at the corresponding
    leaf in ``tree`` and ``xs`` is the tuple of values at corresponding nodes in
    ``rest``.

  Examples:

    >>> import jax
    >>> jax.tree.map(lambda x: x + 1, {"x": 7, "y": 42})
    {'x': 8, 'y': 43}

    If multiple inputs are passed, the structure of the tree is taken from the
    first input; subsequent inputs need only have ``tree`` as a prefix:

    >>> jax.tree.map(lambda x, y: [x] + y, [5, 6], [[7, 9], [1, 2]])
    [[5, 7, 9], [6, 1, 2]]

  See Also:
    - :func:`jax.tree.leaves`
    - :func:`jax.tree.reduce`
  """
  return tree_util.tree_map(f, tree, *rest, is_leaf=is_leaf)


@overload
def reduce(function: Callable[[T, Any], T],
           tree: Any,
           *,
           is_leaf: Callable[[Any], bool] | None = None) -> T:
    ...
@overload
def reduce(function: Callable[[T, Any], T],
           tree: Any,
           initializer: T,
           is_leaf: Callable[[Any], bool] | None = None) -> T:
    ...
def reduce(function: Callable[[T, Any], T],
           tree: Any,
           initializer: Any = tree_util.no_initializer,
           is_leaf: Callable[[Any], bool] | None = None) -> T:
  """Call reduce() over the leaves of a tree.

  Args:
    function: the reduction function
    tree: the pytree to reduce over
    initializer: the optional initial value
    is_leaf : an optionally specified function that will be called at each
      flattening step. It should return a boolean, which indicates whether the
      flattening should traverse the current object, or if it should be stopped
      immediately, with the whole subtree being treated as a leaf.

  Returns:
    result: the reduced value.

  Examples:
    >>> import jax
    >>> import operator
    >>> jax.tree.reduce(operator.add, [1, (2, 3), [4, 5, 6]])
    21

  See Also:
    - :func:`jax.tree.leaves`
    - :func:`jax.tree.map`
  """
  return tree_util.tree_reduce(function, tree, initializer, is_leaf=is_leaf)


def structure(tree: Any,
              is_leaf: None | (Callable[[Any], bool]) = None) -> tree_util.PyTreeDef:
  """Gets the treedef for a pytree.

  Args:
    tree: the pytree for which to get the leaves
    is_leaf : an optionally specified function that will be called at each
      flattening step. It should return a boolean, which indicates whether the
      flattening should traverse the current object, or if it should be stopped
      immediately, with the whole subtree being treated as a leaf.

  Returns:
    pytreedef: a PyTreeDef representing the structure of the tree.

  Examples:
    >>> import jax
    >>> jax.tree.structure([1, (2, 3), [4, 5]])
    PyTreeDef([*, (*, *), [*, *]])

  See Also:
    - :func:`jax.tree.flatten`
    - :func:`jax.tree.leaves`
    - :func:`jax.tree.unflatten`
  """
  return tree_util.tree_structure(tree, is_leaf)


def transpose(outer_treedef: tree_util.PyTreeDef,
              inner_treedef: tree_util.PyTreeDef | None,
              pytree_to_transpose: Any) -> Any:
  """Transform a tree having tree structure (outer, inner) into one having structure (inner, outer).

  Args:
    outer_treedef: PyTreeDef representing the outer tree.
    inner_treedef: PyTreeDef representing the inner tree.
      If None, then it will be inferred from outer_treedef and the structure of
      pytree_to_transpose.
    pytree_to_transpose: the pytree to be transposed.

  Returns:
    transposed_pytree: the transposed pytree.

  Examples:
    >>> import jax
    >>> tree = [(1, 2, 3), (4, 5, 6)]
    >>> inner_structure = jax.tree.structure(('*', '*', '*'))
    >>> outer_structure = jax.tree.structure(['*', '*'])
    >>> jax.tree.transpose(outer_structure, inner_structure, tree)
    ([1, 4], [2, 5], [3, 6])

    Inferring the inner structure:

    >>> jax.tree.transpose(outer_structure, None, tree)
    ([1, 4], [2, 5], [3, 6])
  """
  return tree_util.tree_transpose(outer_treedef, inner_treedef, pytree_to_transpose)


def unflatten(treedef: tree_util.PyTreeDef,
              leaves: Iterable[tree_util.Leaf]) -> Any:
  """Reconstructs a pytree from the treedef and the leaves.

  The inverse of :func:`tree_flatten`.

  Args:
    treedef: the treedef to reconstruct
    leaves: the iterable of leaves to use for reconstruction. The iterable must
      match the leaves of the treedef.

  Returns:
    The reconstructed pytree, containing the ``leaves`` placed in the structure
    described by ``treedef``.

  Examples:
    >>> import jax
    >>> vals, treedef = jax.tree.flatten([1, (2, 3), [4, 5]])
    >>> newvals = [100, 200, 300, 400, 500]
    >>> jax.tree.unflatten(treedef, newvals)
    [100, (200, 300), [400, 500]]

  See Also:
    - :func:`jax.tree.flatten`
    - :func:`jax.tree.leaves`
    - :func:`jax.tree.structure`
  """
  return tree_util.tree_unflatten(treedef, leaves)


def flatten_with_path(
    tree: Any, is_leaf: Callable[[Any], bool] | None = None
) -> tuple[list[tuple[tree_util.KeyPath, Any]], tree_util.PyTreeDef]:
  """Flattens a pytree like ``tree_flatten``, but also returns each leaf's key path.

  Args:
    tree: a pytree to flatten. If it contains a custom type, it is recommended
      to be registered with ``register_pytree_with_keys``.

  Returns:
    A pair which the first element is a list of key-leaf pairs, each of
    which contains a leaf and its key path. The second element is a treedef
    representing the structure of the flattened tree.

  Examples:
    >>> import jax
    >>> path_vals, treedef = jax.tree.flatten_with_path([1, {'x': 3}])
    >>> path_vals
    [((SequenceKey(idx=0),), 1), ((SequenceKey(idx=1), DictKey(key='x')), 3)]
    >>> treedef
    PyTreeDef([*, {'x': *}])

  See Also:
    - :func:`jax.tree.flatten`
    - :func:`jax.tree.map_with_path`
    - :func:`jax.tree_util.register_pytree_with_keys`
  """
  return tree_util.tree_flatten_with_path(tree, is_leaf)


def leaves_with_path(
    tree: Any, is_leaf: Callable[[Any], bool] | None = None
) -> list[tuple[tree_util.KeyPath, Any]]:
  """Gets the leaves of a pytree like ``tree_leaves`` and returns each leaf's key path.

  Args:
    tree: a pytree. If it contains a custom type, it is recommended to be
      registered with ``register_pytree_with_keys``.

  Returns:
    A list of key-leaf pairs, each of which contains a leaf and its key path.

  Examples:
    >>> import jax
    >>> jax.tree.leaves_with_path([1, {'x': 3}])
    [((SequenceKey(idx=0),), 1), ((SequenceKey(idx=1), DictKey(key='x')), 3)]

  See Also:
    - :func:`jax.tree.leaves`
    - :func:`jax.tree.flatten_with_path`
    - :func:`jax.tree_util.register_pytree_with_keys`
  """
  return tree_util.tree_leaves_with_path(tree, is_leaf)


def map_with_path(
    f: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
  """Maps a multi-input function over pytree key path and args to produce a new pytree.

  This is a more powerful alternative of ``tree_map`` that can take the key path
  of each leaf as input argument as well.

  Args:
    f: function that takes ``2 + len(rest)`` arguments, aka. the key path and
      each corresponding leaves of the pytrees.
    tree: a pytree to be mapped over, with each leaf's key path as the first
      positional argument and the leaf itself as the second argument to ``f``.
    *rest: a tuple of pytrees, each of which has the same structure as ``tree``
      or has ``tree`` as a prefix.

  Returns:
    A new pytree with the same structure as ``tree`` but with the value at each
    leaf given by ``f(kp, x, *xs)`` where ``kp`` is the key path of the leaf at
    the corresponding leaf in ``tree``, ``x`` is the leaf value and ``xs`` is
    the tuple of values at corresponding nodes in ``rest``.

  Examples:
    >>> import jax
    >>> jax.tree.map_with_path(lambda path, x: x + path[0].idx, [1, 2, 3])
    [1, 3, 5]

  See Also:
    - :func:`jax.tree.map`
    - :func:`jax.tree.flatten_with_path`
    - :func:`jax.tree.leaves_with_path`
    - :func:`jax.tree_util.register_pytree_with_keys`
  """
  return tree_util.tree_map_with_path(f, tree, *rest, is_leaf=is_leaf)
