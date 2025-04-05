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

"""Helper utils."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Dict, Type, TypeVar

from etils import epy
import typing_extensions
from typing_extensions import Annotated

_Cls = Type[Any]
_ClsT = TypeVar('_ClsT')

Hint = Any
_Hints = Dict[str, Hint]

Descriptor = Any


@dataclasses.dataclass
class DescriptorInfo:
  annotation: Hint
  descriptor_fn: Callable[[dataclasses.Field[Any], Hint], Descriptor]

  @functools.cached_property
  def annotated_token(self) -> object:
    """Returns the Annotated sentinel."""
    assert typing_extensions.get_origin(self.annotation) is Annotated
    (annotated_token,) = self.annotation.__metadata__
    return annotated_token


def wrap_new(cls: _ClsT, descriptor_infos: list[DescriptorInfo]) -> _ClsT:
  """`__new__` decorator to replace the fields by descriptors on first usage."""
  if not descriptor_infos:
    return cls
  cls._edc_processed = False  # pylint: disable=protected-access

  old_new_fn = cls.__new__

  @functools.wraps(old_new_fn)
  def new_new_fn(cls, *args, **kwargs):
    if old_new_fn is object.__new__:
      self = old_new_fn(cls)
    else:
      self = old_new_fn(cls, *args, **kwargs)

    # Already called, skipping initialization
    if cls.__dict__.get('_edc_processed'):
      return self

    # First time, apply to all parent classes .
    for curr_cls in cls.mro():  # Apply to all parent classes
      if cls.__dict__.get('_edc_processed', True):
        # Either:
        # This class is not a `@edc.dataclass` (but parent might)
        # This class is already processed
        continue

      _replace_field_by_descriptor(curr_cls, descriptor_infos=descriptor_infos)

    cls._edc_processed = True  # pylint: disable=protected-access
    return self

  cls.__new__ = new_new_fn
  return cls


def _replace_field_by_descriptor(
    cls: _Cls,
    *,
    descriptor_infos: list[DescriptorInfo],
):
  """Iterate over the dataclass fields and replace the fields by descriptors."""
  if not dataclasses.is_dataclass(cls):  # e.g. object
    return
  fields = {f.name: f for f in dataclasses.fields(cls)}
  hints = _get_type_hints(cls, include_extras=True)

  for name, hint in hints.items():
    if name not in cls.__annotations__:
      continue  # Only add typing from the current class
    # TODO(epot): Should create a typing parsing util.
    if typing_extensions.get_origin(hint) is not Annotated:
      continue

    hint_cls = hint.__origin__  # Unwrap the original type
    field = fields[name]

    # Make the descriptor
    for descriptor_info in descriptor_infos:
      if not any(
          a is descriptor_info.annotated_token for a in hint.__metadata__
      ):
        continue
      descriptor = descriptor_info.descriptor_fn(field, hint_cls)
      setattr(cls, name, descriptor)  # cls.__dict__[name] = cast_field
      descriptor.__set_name__(cls, name)  # Notify the descriptor


# Could merge this function with the one in `dataclass_array` in a util.
def _get_type_hints(cls, *, include_extras: bool = False) -> _Hints:
  """`get_type_hints` with better error reporting."""
  # At this point, `ForwardRef` should have been resolved.
  try:
    return _get_type_hints_fix(cls, include_extras=include_extras)
  except Exception as e:  # pylint: disable=broad-except
    msg = (
        f'Could not infer typing annotation of {cls.__qualname__} '
        f'defined in {cls.__module__}:\n'
    )
    lines = [f' * {k}: {v!r}' for k, v in cls.__annotations__.items()]
    lines = '\n'.join(lines)

    epy.reraise(e, prefix=msg + lines + '\n')  # pytype: disable=bad-return-type


def _get_type_hints_fix(cls, *, include_extras: bool = False) -> _Hints:
  """`get_type_hints` with bug fixes."""
  # TODO(py311): `get_type_hints` fail for `_: dataclasses.KW_ONLY`
  old_annotations = [_fix_annotations(subcls) for subcls in cls.mro()]
  try:
    return typing_extensions.get_type_hints(cls, include_extras=include_extras)
  finally:
    # Restore the annotations
    for subcls, annotations in zip(cls.mro(), old_annotations):
      if annotations:
        subcls.__annotations__ = annotations


def _fix_annotations(cls):
  """Remove the `_: dataclasses.KW_ONLY` annotation."""
  if cls is object or '_' not in getattr(cls, '__annotations__', {}):
    return
  old_annotations = dict(cls.__annotations__)
  cls.__annotations__.pop('_')
  return old_annotations
