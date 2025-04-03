# Copyright 2024 The etils Authors.
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

"""Tree API backends."""

from __future__ import annotations

import abc
import collections
import collections.abc
import functools
import itertools
import types
from typing import Any, Callable, Optional, TypeVar

from etils import epy
from etils.etree.typing import LeafFn, Tree  # pylint: disable=g-importing-member,g-multiple-import

_T = TypeVar('_T')
_Tin = Any  # TypeVar('_Tin')
_Tout = Any  # TypeVar('_Tout')

# Structure which allow to reconstruct the tree
# * jax: TreeDef
# * tf.nest/tree: The original Tree
_TreeDef = Any


class Backend(abc.ABC):
  """Tree API backend.

  Note: The backend lazy-import the module on first call. This
  allow to use `etree` with Jax even if TF isn't installed (and
  vice-versa).
  """

  @functools.cached_property
  def module(self) -> types.ModuleType:
    """Module used by the backend."""
    try:
      module = self.import_module()
    except ImportError as e:
      epy.reraise(
          e,
          suffix=(
              'Using specific etree backend require to install extra'
              ' dependencies.'
          ),
      )
    return module

  @abc.abstractmethod
  def import_module(self) -> types.ModuleType:
    """Import and return the module."""
    raise NotImplementedError

  @abc.abstractmethod
  def map(
      self,
      map_fn: Callable[..., _Tout],  # Callable[[_Tin0, _Tin1,...], Tout]
      *trees: Tree[_Tin],  # _Tin0, _Tin1,...
      is_leaf: Optional[LeafFn] = None,
  ) -> Tree[_Tout]:
    """Like `tf.nest.map_structure`."""
    raise NotImplementedError

  @abc.abstractmethod
  def flatten(
      self,
      tree: Tree[_T],
      *,
      is_leaf: Optional[LeafFn] = None,
  ) -> tuple[list[_T], _TreeDef]:
    """Like `tf.nest.flatten`."""
    raise NotImplementedError

  @abc.abstractmethod
  def unflatten(self, structure: _TreeDef, flat_sequence: list[_T]) -> Tree[_T]:
    raise NotImplementedError

  @abc.abstractmethod
  def assert_same_structure(
      self,
      tree0: Tree[Any],
      tree1: Tree[Any],
  ) -> None:
    raise NotImplementedError


class Jax(Backend):
  """`jax.tree_util` backend."""

  def import_module(self):
    import jax  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    return jax.tree_util

  def map(self, map_fn, *trees, is_leaf=None):
    return self.module.tree_map(map_fn, *trees, is_leaf=is_leaf)

  def flatten(self, tree, *, is_leaf=None):
    flat_vals, treedef = self.module.tree_flatten(tree, is_leaf=is_leaf)
    return flat_vals, treedef

  def unflatten(self, structure, flat_sequence):
    return structure.unflatten(flat_sequence)

  def assert_same_structure(
      self,
      tree0: Tree[Any],
      tree1: Tree[Any],
  ):
    treedef0 = self.module.tree_structure(tree0)
    treedef1 = self.module.tree_structure(tree1)
    if treedef0 != treedef1:
      raise ValueError(
          "The two structures don't have the same nested structure.\n"
          f'Left: {treedef0}\nRight: {treedef1}'
      )


class Optree(Jax):
  """`optree` backend."""

  def import_module(self):
    import optree  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    return optree


class DmTree(Backend):
  """`tree` backend."""

  def import_module(self):
    import tree  # pylint: disable=g-import-not-at-top  # type: ignore

    return tree

  def map(self, map_fn, *trees, is_leaf=None):
    if is_leaf is not None:
      raise NotImplementedError('is_leaf not supported for `dm-tree` backend')
    return self.module.map_structure(map_fn, *trees)

  def flatten(self, tree, *, is_leaf=None):
    if is_leaf is not None:
      raise NotImplementedError('is_leaf not supported for `dm-tree` backend')
    return self.module.flatten(tree), tree

  def unflatten(self, structure, flat_sequence):
    return self.module.unflatten_as(structure, flat_sequence)

  def assert_same_structure(
      self,
      tree0: Tree[Any],
      tree1: Tree[Any],
  ):
    self.module.assert_same_structure(tree0, tree1)


class Nest(Backend):
  """`tf.nest` backend."""

  def import_module(self):
    import tensorflow as tf  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    return tf.nest

  def map(self, map_fn, *trees, is_leaf=None):
    if is_leaf is not None:
      raise NotImplementedError('is_leaf not supported for `nest` backend')
    return self.module.map_structure(map_fn, *trees)

  def flatten(self, tree, *, is_leaf=None):
    if is_leaf is not None:
      raise NotImplementedError('is_leaf not supported for `nest` backend')
    return self.module.flatten(tree), tree

  def unflatten(self, structure, flat_sequence):
    return self.module.pack_sequence_as(structure, flat_sequence)

  def assert_same_structure(
      self,
      tree0: Tree[Any],
      tree1: Tree[Any],
  ):
    self.module.assert_same_structure(tree0, tree1)


# TODO(epot): Also support custom `cls` like `register_pytree_node` in Jax
_SEQUENCE_TYPES = (list, tuple)
_MAPPING_TYPES = (dict, collections.abc.Mapping)
_ALL_TYPES = _SEQUENCE_TYPES + _MAPPING_TYPES


class Python(Backend):
  """Pure `Python` backend."""

  def import_module(self):
    raise RuntimeError('Python backend do not have module')

  def map(self, map_fn, *trees, is_leaf=None):
    tree0 = trees[0]
    if is_leaf is not None and is_leaf(tree0):
      return map_fn(*trees)
    elif isinstance(tree0, _SEQUENCE_TYPES):
      new_items = (self.map(map_fn, *v, is_leaf=is_leaf) for v in zip(*trees))
      if epy.is_namedtuple(tree0):
        return type(tree0)(*new_items)
      else:
        return type(tree0)(new_items)
    elif isinstance(tree0, _MAPPING_TYPES):
      new_items = (
          (k, self.map(map_fn, *v, is_leaf=is_leaf))
          for k, v in epy.zip_dict(*trees)
      )
      if isinstance(tree0, collections.defaultdict):
        new_tree = type(tree0)(tree0.default_factory)
        new_tree.update(new_items)
        return new_tree
      else:
        return type(tree0)(new_items)
    else:  # leaf
      return map_fn(*trees)

  def flatten(self, tree, *, is_leaf=None):
    return list(self._flatten(tree, is_leaf=is_leaf)), tree

  def _flatten(self, tree, is_leaf):
    """`flatten` recursive implementation."""
    if is_leaf is not None and is_leaf(tree):
      return [tree]
    elif isinstance(tree, _SEQUENCE_TYPES):
      return itertools.chain.from_iterable(
          self._flatten(v, is_leaf=is_leaf) for v in tree
      )
    elif isinstance(tree, _MAPPING_TYPES):
      return itertools.chain.from_iterable(
          self._flatten(v, is_leaf=is_leaf) for _, v in sorted(tree.items())
      )
    else:  # leaf
      return [tree]

  def unflatten(self, structure, flat_sequence):
    return self._unflatten(structure, iter(flat_sequence))

  def _unflatten(self, structure, flat_iter):
    """`unflatten` recursive implementation."""
    if isinstance(structure, _SEQUENCE_TYPES):
      new_items = (self._unflatten(v, flat_iter) for v in structure)
      if epy.is_namedtuple(structure):
        return type(structure)(*new_items)
      else:
        return type(structure)(new_items)
    elif isinstance(structure, _MAPPING_TYPES):
      # Flatten sort the keys, so reconstruct the ordered sorted
      ordered_items = {
          k: self._unflatten(v, flat_iter) for k, v in sorted(structure.items())
      }
      # Restore original dict order
      new_items = ((k, ordered_items[k]) for k in structure)

      if isinstance(structure, collections.defaultdict):
        new_tree = type(structure)(structure.default_factory)
        new_tree.update(new_items)
        return new_tree
      else:
        return type(structure)(new_items)
    else:  # leaf
      return next(flat_iter)

  def assert_same_structure(
      self,
      tree0: Tree[Any],
      tree1: Tree[Any],
  ):
    try:
      self._assert_same_structure(tree0, tree1)
    except Exception as e:  # pylint: disable=broad-except
      epy.reraise(e, prefix="The two structures don't match: ")

  def _assert_same_structure(
      self,
      tree0: Tree[Any],
      tree1: Tree[Any],
  ):
    """`assert_same_structure` recursive implementation."""
    if isinstance(tree0, _ALL_TYPES):
      if type(tree0) != type(tree1):  # pylint: disable=unidiomatic-typecheck
        raise ValueError(f'{type(tree0)} != {type(tree1)}')
    if isinstance(tree0, _SEQUENCE_TYPES):
      if len(tree0) != len(tree1):
        raise ValueError(f'{len(tree0)} != {len(tree1)}')
      for i, (v0, v1) in enumerate(zip(tree0, tree1)):
        try:
          self._assert_same_structure(v0, v1)
        except Exception as e:  # pylint: disable=broad-except
          epy.reraise(e, prefix=f'In {i}: ')
    elif isinstance(tree0, _MAPPING_TYPES):
      k0 = sorted(tree0)
      k1 = sorted(tree1)
      if k0 != k1:
        raise ValueError(f'dict keys do not match: {k0} != {k1}')
      # Flatten sort the keys, so reconstruct the ordered sorted
      for k, (v0, v1) in epy.zip_dict(tree0, tree1):
        try:
          self._assert_same_structure(v0, v1)
        except Exception as e:  # pylint: disable=broad-except
          epy.reraise(e, prefix=f'In {k}: ')
    else:  # leaf
      return
