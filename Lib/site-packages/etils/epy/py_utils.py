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

"""Python utils."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import enum
import functools
import sys
import typing
from typing import Any, ClassVar, Optional, TypeVar, Union


StrOrStrList = Union[str, Sequence[str]]

_Cls = TypeVar('_Cls')


_StrEnum = (
    (enum.StrEnum,) if sys.version_info[:2] >= (3, 11) else (str, enum.Enum)
)


class StrEnum(*_StrEnum):
  """Like `Enum`, but `enum.auto()` assigns `str` rather than `int`.

  ```python
  class MyEnum(epy.StrEnum):
    SOME_ATTR = enum.auto()
    OTHER_ATTR = enum.auto()

  assert MyEnum('some_attr') is MyEnum.SOME_ATTR
  assert MyEnum.SOME_ATTR == 'some_attr'
  ```

  `StrEnum` is case insensitive.
  """

  _LOWER_TO_VAL: ClassVar[dict[str, StrEnum]]

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    cls._LOWER_TO_VAL = {e.value.lower(): e for e in cls}

  def _generate_next_value_(name, start, count, last_values) -> str:  # pylint: disable=no-self-argument
    return name.lower()

  @classmethod
  def _missing_(cls, value: str) -> StrEnum:
    if isinstance(value, str):
      enum_value = cls._LOWER_TO_VAL.get(value.lower())
      if enum_value is not None:
        return enum_value
    # Could also add `did you meant yy ?`
    all_values = [e.value for e in cls]
    raise ValueError(
        f'{value!r} is not a valid {cls.__qualname__}. '
        f'Expected one of {all_values}'
    )

  def __eq__(self, other: str) -> bool:
    if not isinstance(other, str):
      return False
    return other.lower() == self.value.lower()

  # `__ne__` is required because `str.__ne__()` exists, so it is not
  # automatically inferred.
  def __ne__(self, other: str) -> bool:
    return not self.__eq__(other)

  def __hash__(self) -> int:  # pylint: disable=useless-super-delegation
    # Somehow `hash` is not defined automatically (maybe because of
    # the `__eq__`, so define it explicitly.
    return super().__hash__()

  # Pytype is confused by EnumMeta.__iter__ vs str.__iter__
  if typing.TYPE_CHECKING:

    @classmethod
    def __iter__(cls):
      return type(enum.Enum).__iter__(cls)


def is_namedtuple(x) -> bool:
  """Returns `True` if the value is instance of `NamedTuple`.

  This is using some heuristic by checking for a `._field` attribute.

  Args:
    x: Object to check

  Returns:
    `True` if the object is a `namedtuple`
  """
  return isinstance(x, tuple) and hasattr(type(x), '_fields')


def issubclass_(
    cls: Any,
    types: Union[type[Any], tuple[type[Any], ...]],
) -> bool:
  """Like `issubclass`, but do not raise error if value is not `type`."""
  return isinstance(cls, type) and issubclass(cls, types)


def _wrap_init(init_fn):
  """`__init__` wrapper."""

  @functools.wraps(init_fn)
  def new_init(self, *args, **kwargs):
    # Do NOT use `hasattr` to support children with custom `__getattr__`
    if '_epy_is_init_done' in self.__dict__:
      # `_epy_is_init_done` already created, so it means we're
      # a `super().__init__` call.
      return init_fn(self, *args, **kwargs)
    object.__setattr__(self, '_epy_is_init_done', False)
    init_fn(self, *args, **kwargs)
    object.__setattr__(self, '_epy_is_init_done', True)

  return new_init


def _wrap_setattr(setattr_fn):
  """`__setattr__` wrapper."""

  @functools.wraps(setattr_fn)
  def new_setattr(self, name, value):
    if not hasattr(self, '_epy_is_init_done'):
      raise ValueError(
          'Child of `@epy.frozen` class should be `@epy.frozen` too. (Error'
          f' raised by {type(self)})'
      )
    if not self._epy_is_init_done:  # pylint: disable=protected-access
      return setattr_fn(self, name, value)
    else:
      raise AttributeError(
          f'Cannot assign {name!r} in `@epy.frozen` class {type(self)}'
      )

  return new_setattr


def frozen(cls: _Cls) -> _Cls:
  """Class decorator which prevent mutating attributes after `__init__`.

  Example:

  ```python
  @epy.frozen
  class A:

    def __init__(self):
      self.x = 123

  a = A()
  a.x = 456  # AttributeError
  ```

  Supports inheritance, child classes should explicitly be marked as
  `@epy.frozen` if they mutate additional attributes in `__init__`.

  Args:
    cls: The class to freeze.

  Returns:
    cls: The class object
  """
  if not isinstance(cls, type):
    raise TypeError(f'{cls.__name__} is not')

  cls.__init__ = _wrap_init(cls.__init__)
  cls.__setattr__ = _wrap_setattr(cls.__setattr__)
  return cls


def normalize_str_to_list(x: Optional[StrOrStrList]) -> list[str]:
  if x is None:
    return []
  elif isinstance(x, str):
    return [v.strip() for v in x.split(',')]
  elif not isinstance(x, (list, tuple)):
    raise TypeError(f'Expected list. Got: {x!r}')
  else:  # list/tuple
    return list(x)


def wraps_cls(wrapped: type[Any]) -> Callable[[_Cls], _Cls]:
  """Equivalent of `functools.wraps` but for classes."""

  def decorator(cls):
    cls.__name__ = wrapped.__name__
    cls.__qualname__ = wrapped.__qualname__
    cls.__doc__ = wrapped.__doc__
    cls.__module__ = wrapped.__module__
    return cls

  return decorator
