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

"""Field utils."""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Callable, Generic, Optional, Type, TypeVar

from etils import epy

_Dataclass = Any
_In = Any
_Out = Any
_InT = TypeVar('_InT')
_OutT = TypeVar('_OutT')


def field(
    *,
    validate: Optional[Callable[[_In], _OutT]] = None,
    **kwargs: Any,
) -> dataclasses.Field[_OutT]:
  """Like `dataclasses.field`, but allow `validator`.

  Args:
    validate: A callable `(x) -> x` called each time the variable is assigned.
    **kwargs: Kwargs forwarded to `dataclasses.field`

  Returns:
    The field.
  """
  if validate is None:
    return dataclasses.field(**kwargs)
  else:
    field_ = _Field(validate=validate, field_kwargs=kwargs)
    return typing.cast(dataclasses.Field, field_)  # pylint: disable=g-bare-generic


class _Field(Generic[_InT, _OutT]):
  """Field descriptor."""

  def __init__(
      self,
      validate: Callable[[_InT], _OutT],
      field_kwargs: dict[str, Any],
  ) -> None:
    """Constructor.

    Args:
      validate: A callable called each time the variable is assigned.
      field_kwargs: Kwargs forwarded to `dataclasses.field`
    """
    # Attribute name and objtype refer to the object in which the descriptor
    # is applied. E.g. if `A.x = edc.field()`:
    # * _attribute_name = 'x'
    # * _objtype = A
    self._attribute_name: Optional[str] = None
    self._objtype: Optional[Type[_Dataclass]] = None

    self._validate_fn = validate
    self._field_kwargs = field_kwargs

    # Whether `__get__` has not been called yet. See `__get__` for details.
    self._first_getattr_call: bool = True

  def __set_name__(self, objtype: Type[_Dataclass], name: str) -> None:
    """Bind the descriptor to the class (PEP 487)."""
    self._objtype = objtype
    self._attribute_name = name

  def __get__(
      self,
      obj: Optional[_Dataclass],
      objtype: Optional[Type[_Dataclass]] = None,
  ) -> _OutT:
    """Called when `MyDataclass.x` or `my_dataclass.x`."""
    # Called as `MyDataclass.my_attribute`
    if obj is None:
      if self._first_getattr_call:
        # Count the number of times `dataclasses.dataclass(cls)` calls
        # `getattr(cls, f.name)`.
        # The first time, we return a `dataclasses.Field` to let dataclass
        # do the magic.
        # The second time, `dataclasses.dataclass` delete the descriptor if
        # `isinstance(getattr(cls, f.name, None), Field)`. So it is very
        # important to return anything except a `dataclasses.Field`.
        # This rely on implementation detail, but seems to hold for python
        # 3.6-3.10.
        self._first_getattr_call = False
        return dataclasses.field(**self._field_kwargs)
      else:
        # TODO(epot): Could better handle default value: Either by returning
        # the default value, or raising an AttributeError. Currently, we just
        # return the descriptor:
        # assert isinstance(MyDataclass.my_attribute, _Field)
        return self
    else:
      # Called as `my_dataclass.my_path`
      return _getattr(obj, self._attribute_name)

  def __set__(self, obj: _Dataclass, value: _InT) -> None:
    """Called as `my_dataclass.x = value`."""
    # Validate the value during assignement
    _setattr(obj, self._attribute_name, self._validate(value))

  def _validate(self, value: _InT) -> _OutT:
    try:
      return self._validate_fn(value)
    except Exception as e:  # pylint: disable=broad-exception-caught
      epy.reraise(e, prefix=f'Error assigning {self._attribute_name!r}: ')


# Because there is one instance of the `_Field` per class, shared across all
# class instances, we need to store the per-object state somewhere.
# The simplest is to attach the state in an extra `dict[str, value]`:
# `_dataclass_field_values`.


def _getattr(
    obj: _Dataclass,
    attribute_name: str,
) -> _Out:
  """Returns the `obj.attribute_name`."""
  _init_dataclass_state(obj)
  # Accessing the attribute before it was set (e.g. before super().__init__)
  if attribute_name not in obj._dataclass_field_values:  # pylint: disable=protected-access
    raise AttributeError(
        f"type object '{type(obj).__qualname__}' has no attribute "
        f"'{attribute_name}'"
    )
  else:
    return obj._dataclass_field_values[attribute_name]  # pylint: disable=protected-access


def _setattr(
    obj: _Dataclass,
    attribute_name: str,
    value: _In,
) -> None:
  """Set the `obj.attribute_name = value`."""
  # Note: In `dataclasses.dataclass(frozen=True)`, obj.__setattr__ will
  # correctly raise a `FrozenInstanceError` before `DataclassField.__set__` is
  # called.
  _init_dataclass_state(obj)
  # fmt: off
  obj._dataclass_field_values[attribute_name] = value  # pylint: disable=protected-access
  # fmt: on


def _init_dataclass_state(obj: _Dataclass) -> None:
  """Initialize the object state containing all DataclassField values."""
  if not hasattr(obj, '_dataclass_field_values'):
    # Use object.__setattr__ for frozen dataclasses
    object.__setattr__(obj, '_dataclass_field_values', {})
