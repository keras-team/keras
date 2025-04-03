# Copyright 2024 The Flax Authors.
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

"""Utilities for flattening and unflattening mappings.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, overload
from collections.abc import Iterable

from flax import struct


# the empty node is a struct.dataclass to be compatible with JAX.
@struct.dataclass
class _EmptyNode:
  pass


empty_node = _EmptyNode()


# TODO: In Python 3.10, use TypeAlias.
IsLeafCallable = Callable[[tuple[Any, ...], Mapping[Any, Any]], bool]


@overload
def flatten_mapping(xs: Mapping[Any, Any],
                    /,
                    *,
                    keep_empty_nodes: bool = False,
                    is_leaf: None | IsLeafCallable = None,
                    sep: None = None
                    ) -> dict[tuple[Any, ...], Any]:
  ...

@overload
def flatten_mapping(xs: Mapping[Any, Any],
                    /,
                    *,
                    keep_empty_nodes: bool = False,
                    is_leaf: None | IsLeafCallable = None,
                    sep: str,
                    ) -> dict[str, Any]:
  ...

def flatten_mapping(xs: Mapping[Any, Any],
                    /,
                    *,
                    keep_empty_nodes: bool = False,
                    is_leaf: None | IsLeafCallable = None,
                    sep: None | str = None
                    ) -> dict[Any, Any]:
  """Flatten a nested mapping.

  The nested keys are flattened to a tuple. See ``unflatten_mapping`` on how to
  restore the nested mapping.

  Example::

    >>> from flax import nnx
    >>> xs = {'foo': 1, 'bar': {'a': 2, 'b': {}}}
    >>> flat_xs = nnx.traversals.flatten_mapping(xs)
    >>> flat_xs
    {('foo',): 1, ('bar', 'a'): 2}

  Note that empty mappings are ignored and will not be restored by
  ``unflatten_mapping``.

  Args:
    xs: a nested mapping
    keep_empty_nodes: replaces empty mappings with
      ``traverse_util.empty_node``.
    is_leaf: an optional function that takes the next nested mapping and nested
      keys and returns True if the nested mapping is a leaf (i.e., should not be
      flattened further).
    sep: if specified, then the keys of the returned mapping will be
      ``sep``-joined strings (if ``None``, then keys will be tuples).
  Returns:
    The flattened mapping.
  """
  assert isinstance(
    xs, Mapping
  ), f'expected Mapping; got {type(xs).__qualname__}'

  def _key(path: tuple[Any, ...]) -> tuple[Any, ...] | str:
    if sep is None:
      return path
    return sep.join(path)

  def _flatten(xs: Any, prefix: tuple[Any, ...]) -> dict[Any, Any]:
    if not isinstance(xs, Mapping) or (
      is_leaf and is_leaf(prefix, xs)
    ):
      return {_key(prefix): xs}
    result = {}
    is_empty = True
    for key, value in xs.items():
      is_empty = False
      path = prefix + (key,)
      result.update(_flatten(value, path))
    if keep_empty_nodes and is_empty:
      if prefix == ():  # when the whole input is empty
        return {}
      return {_key(prefix): empty_node}
    return result

  return _flatten(xs, ())

def flatten_to_sequence(
  xs: Mapping[Any, Any],
  /,
  *,
  is_leaf: IsLeafCallable | None = None,
) -> list[tuple[Any, Any]]:
  """Flatten a nested mapping.

  The nested keys are flattened to a tuple. See ``unflatten_mapping`` on how to
  restore the nested mapping.

  Example::

    >>> from flax import nnx
    >>> xs = {'foo': 1, 'bar': {'a': 2, 'b': {}}}
    >>> flat_xs = nnx.traversals.flatten_to_sequence(xs)
    >>> flat_xs
    [(('foo',), 1), (('bar', 'a'), 2)]

  Note that empty mappings are ignored and will not be restored by
  ``unflatten_mapping``.

  Args:
    xs: a nested mapping
    is_leaf: an optional function that takes the next nested mapping and nested
      keys and returns True if the nested mapping is a leaf (i.e., should not be
      flattened further).

  Returns:
    The flattened mapping.
  """
  assert isinstance(
    xs, Mapping
  ), f'expected Mapping; got {type(xs).__qualname__}'
  result = []

  def _flatten(xs: Any, prefix: tuple[Any, ...]):
    if not isinstance(xs, Mapping) or (is_leaf and is_leaf(prefix, xs)):
      result.append((prefix, xs))
    else:
      for key, value in xs.items():
        _flatten(value, (*prefix, key))

  _flatten(xs, ())
  return result


@overload
def unflatten_mapping(xs: Mapping[tuple[Any, ...], Any],
                      /,
                      *,
                      sep: None = None
                      ) -> dict[Any, Any]:
  ...


@overload
def unflatten_mapping(xs: Mapping[str, Any],
                      /,
                      *,
                      sep: str
                      ) -> dict[Any, Any]:
  ...


def unflatten_mapping(xs: Any,
                      /,
                      *,
                      sep: str | None = None
                      ) -> dict[Any, Any]:
  """Unflatten a mapping.

  See ``flatten_mapping``

  Example::

    >>> from flax import nnx
    >>> flat_xs = {
    ...   ('foo',): 1,
    ...   ('bar', 'a'): 2,
    ... }
    >>> xs = nnx.traversals.unflatten_mapping(flat_xs)
    >>> xs
    {'foo': 1, 'bar': {'a': 2}}

  Args:
    xs: a flattened mapping.
    sep: separator (same as used with ``flatten_mapping()``).
  Returns:
    The nested mapping.
  """
  if isinstance(xs, Mapping):
    xs = xs.items()

  if not isinstance(xs, Iterable):
    raise TypeError(
      f'expected Mapping or Iterable; got {type(xs).__qualname__}'
    )
  result: dict[Any, Any] = {}
  for path, value in xs:
    if sep is not None:
      path = path.split(sep)
    if value is empty_node:
      value = {}
    cursor = result
    for key in path[:-1]:
      if key not in cursor:
        cursor[key] = {}
      cursor = cursor[key]
    cursor[path[-1]] = value
  return result
