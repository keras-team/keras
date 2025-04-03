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

"""Itertools utils."""

from __future__ import annotations

import collections
import itertools

from typing import Any, Callable, Iterable, Iterator, TypeVar

# from typing_extensions import Unpack, TypeVarTuple  # pytype: disable=not-supported-yet  # pylint: disable=g-multiple-import

# TODO(pytype): Once supported, should replace
Unpack = Any
TypeVarTuple = Any

_T = TypeVar('_T')

_KeyT = TypeVar('_KeyT')
_ValuesT = Any  # TypeVarTuple('_ValuesT')

_K = TypeVar('_K')
_Tin = TypeVar('_Tin')
_Tout = TypeVar('_Tout')


def _identity(x: _Tin) -> _Tin:
  """Pass through function."""
  return x


def groupby(
    iterable: Iterable[_Tin],
    *,
    key: Callable[[_Tin], _K],
    value: Callable[[_Tin], _Tout] = _identity,
) -> dict[_K, list[_Tout]]:
  """Similar to `itertools.groupby` but return result as a `dict()`.

  Example:

  ```python
  out = epy.groupby(
      ['555', '4', '11', '11', '333'],
      key=len,
      value=int,
  )
  # Order is consistent with above
  assert out == {
      3: [555, 333],
      1: [4],
      2: [11, 11],
  }
  ```

  Other difference with `itertools.groupby`:

   * Iterable do not need to be sorted. Order of the original iterator is
     preserved in the group.
   * Transformation can be applied to the value too

  Args:
    iterable: The iterable to group
    key: Mapping applied to group the values (should return a hashable)
    value: Mapping applied to the values

  Returns:
    The dict
  """
  groups = collections.defaultdict(list)
  for v in iterable:
    groups[key(v)].append(value(v))
  return dict(groups)


def splitby(
    iterable: Iterable[_T], predicate: Callable[[_T], bool]
) -> tuple[list[_T], list[_T]]:
  """Split the iterable into 2 lists (false, true), based on the predicate.

  Example:

  ```python
  small, big = epy.splitby([100, 4, 4, 1, 200], lambda x: x > 10)
  assert small == [4, 4, 1]
  assert big == [100, 200]
  ```

  Args:
    iterable: The iterable to split
    predicate: Function applied to split

  Returns:
    False list, True list
  """
  false_list = []
  true_list = []
  for v in iterable:
    if predicate(v):
      true_list.append(v)
    else:
      false_list.append(v)
  return false_list, true_list


def zip_dict(  # pytype: disable=invalid-annotation
    *dicts: Unpack[dict[_KeyT, _ValuesT]],
) -> Iterator[_KeyT, tuple[Unpack[_ValuesT]]]:
  """Iterate over items of dictionaries grouped by their keys.

  Example:

  ```python
  d0 = {'a': 1, 'b': 2}
  d1 = {'a': 10, 'b': 20}
  d2 = {'a': 100, 'b': 200}

  list(epy.zip_dict(d0, d1, d2)) == [
      ('a', (1, 10, 100)),
      ('b', (2, 20, 200)),
  ]
  ```

  Args:
    *dicts: The dict to iterate over. Should all have the same keys

  Yields:
    The iterator of `(key, zip(*values))`

  Raises:
    KeyError: If dicts does not contain the same keys.
  """
  # Set does not keep order like dict, so only use set to compare keys
  all_keys = set(itertools.chain(*dicts))
  d0 = dicts[0]

  if len(all_keys) != len(d0):
    raise KeyError(f'Missing keys: {all_keys ^ set(d0)}')

  for key in d0:  # set merge all keys
    # Will raise KeyError if the dict don't have the same keys
    yield key, tuple(d[key] for d in dicts)
