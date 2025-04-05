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

"""Contextvar util."""

from __future__ import annotations

import contextvars
import dataclasses
import functools
import typing
from typing import Any, Optional, TypeVar

from etils.edc import helpers
from typing_extensions import Annotated

_T = TypeVar('_T')

_Dataclass = Any
_DataclassT = TypeVar('_DataclassT')

_IS_CONTEXTVAR = object()


if typing.TYPE_CHECKING:
  # TODO(b/254514368): Remove hack
  class _ContextVarMeta(type):

    def __getitem__(cls, value):
      return value

  class ContextVar(metaclass=_ContextVarMeta):
    pass

else:
  ContextVar = Annotated[_T, _IS_CONTEXTVAR]  # pytype: disable=invalid-typevar


def make_contextvar_descriptor(
    field: dataclasses.Field[Any], hint: helpers.Hint
) -> _ContextvarDescriptor:
  """Replace `ContextVar[]` annotated fields with contextvar descriptor."""
  del hint
  return _ContextvarDescriptor(field)


@dataclasses.dataclass
class _ContextvarDescriptor:
  """Descriptor to read-write individual contextvar."""

  _field: dataclasses.Field[Any]
  _objtype: type[Any] = dataclasses.field(init=False)
  _attribute_name: str = dataclasses.field(init=False)

  def __set_name__(self, objtype: type[_Dataclass], name: str) -> None:
    """Bind the descriptor to the class (PEP 487)."""
    self._objtype = objtype
    self._attribute_name = name

  @functools.cached_property
  def _var(self) -> contextvars.ContextVar[Any]:
    """Contextvar."""
    default_kwargs = {}
    if self._field.default is not dataclasses.MISSING:
      default_kwargs['default'] = self._field.default
    elif self._field.default_factory is not dataclasses.MISSING:
      default_kwargs['default'] = self._field.default_factory()
    else:
      pass
    return contextvars.ContextVar(self._attribute_name, **default_kwargs)

  def __get__(
      self,
      obj: Optional[_Dataclass],
      objtype: Optional[type[_Dataclass]] = None,
  ):
    if obj is None:
      return self
    return self._var.get()

  def __set__(self, obj: _Dataclass, value: Any) -> None:
    self._var.set(value)
