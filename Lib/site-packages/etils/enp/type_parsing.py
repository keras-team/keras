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

"""Typing annotation parsing util."""

# TODO(epot): Move to `epy.typing` once more mature

from __future__ import annotations

import sys
import types
import typing
from typing import Any, Callable

import typing_extensions  # TODO(py38): Remove

TypeForm = Any
_LeafFn = Callable[[TypeForm], None]

_NoneType = type(None)


def _visit_leaf(hint: TypeForm, leaf_fn: _LeafFn):
  """Leaf node."""
  if hint == _NoneType:  # Normalize `None`
    hint = None
  return leaf_fn(hint)


def _visit_union(hint: TypeForm, leaf_fn: _LeafFn):
  """Recurse in `Union[x, y]`, `x | y`, or `Optional[x]`."""
  item_hints = typing_extensions.get_args(hint)
  for item_hint in item_hints:
    visit(item_hint, leaf_fn)


def visit(hint: TypeForm, leaf_fn: _LeafFn):
  """Recurse in the type annotation tree."""
  origin = typing_extensions.get_origin(hint)
  visit_fn = _ORIGIN_TO_VISITOR.get(origin, _visit_leaf)
  visit_fn(hint, leaf_fn)


# Currently, only support `Union` and `Optional` but could be extended
# to `dict`, `list`,...
_ORIGIN_TO_VISITOR = {
    typing.Union: _visit_union,
    None: _visit_leaf,  # Default origin
}
if sys.version_info >= (3, 10):
  _ORIGIN_TO_VISITOR[types.UnionType] = _visit_union  # In Python 3.10+: x | y


def get_leaf_types(hint: TypeForm) -> list[type[Any]]:
  """Extract the inner list of the types (`Optional[A] -> [A, None]`)."""
  all_types = []

  def _collect_leaf_types(hint):
    all_types.append(hint)

  visit(hint, leaf_fn=_collect_leaf_types)

  return all_types
