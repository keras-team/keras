# Copyright 2024 The Orbax Authors.
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

"""Composite key-value store for checkpointing."""

from collections.abc import Iterator, KeysView, Mapping, ValuesView
from typing import Any


class Composite(Mapping):
  """A key-value store similar to a dict.

  Usage examples::

    Composite(
        state=my_train_state,
        dataset=my_dataset,
        metadata=json_metadata,
    )

    Composite(
        **{
            'state': my_train_state,
            'dataset': my_dataset,
            'metadata': json_metadata,
          }
    )

    args = Composite(...)
    args.state
    args['state']
    'state' in args

    for key, value in args.items():
      ...
  """

  _items: Mapping[str, Any]

  def __init__(self, **items: Any):
    super().__setattr__('_items', items)

    reserved_keys = set(dir(self))

    for key, value in items.items():
      # Reserve and prevent users from setting keys that start with '__'. These
      # may be used later to define options.
      if key.startswith('__'):
        raise ValueError(f'Composite keys cannot start with "__". Got: {key}')
      if key not in reserved_keys:
        # We do not raise an error if the user specifies a key that matches an
        # existing attribute (like 'keys', 'values', 'items'). These can be
        # accessed through self[key], but not self.key.
        super().__setattr__(key, value)

  def __getitem__(self, key: str) -> Any:
    if key not in self._items:
      raise KeyError(
          f'Unknown key: {key}. Available keys: {self._items.keys()}'
      )
    return self._items[key]

  def __iter__(self) -> Iterator[Any]:
    return iter(self._items)

  def __len__(self) -> int:
    return len(self._items)

  def __getattr__(self, key: str):
    return self.__getitem__(key)

  def __setattr__(self, key: str, value: Any):
    raise ValueError('Composite is immutable after initialization.')

  def get(self, key: str, default=None) -> Any | None:
    try:
      return self.__getitem__(key)
    except KeyError:
      return default

  def __and__(self, other: 'Composite') -> 'Composite':
    if isinstance(other, dict):
      other = Composite(**other)
    common_keys = self._items.keys() & other._items.keys()
    return Composite(**{key: self._items[key] for key in common_keys})

  def __or__(self, other: 'Composite') -> 'Composite':
    if isinstance(other, dict):
      other = Composite(**other)
    return Composite(**{**self._items, **other._items})

  def __repr__(self):
    return f'Composite({repr(self._items)})'

  def keys(self) -> KeysView[str]:
    return self._items.keys()

  def values(self) -> ValuesView[Any]:
    return self._items.values()

  def __contains__(self, key: str) -> bool:
    return key in self._items
