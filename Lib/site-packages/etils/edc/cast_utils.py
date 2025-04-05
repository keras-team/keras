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

"""Auto-apply normalization to a dataclass fields."""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, TypeVar

from etils.edc import field_utils
from etils.edc import helpers
from typing_extensions import Annotated


_T = TypeVar('_T')

_IS_NORMALIZED = object()


if typing.TYPE_CHECKING:
  # TODO(b/254514368): Remove hack
  class _AutoCastMeta(type):

    def __getitem__(cls, value):
      return value

  class AutoCast(metaclass=_AutoCastMeta):
    pass

else:
  AutoCast = Annotated[_T, _IS_NORMALIZED]  # pytype: disable=invalid-typevar


def make_auto_cast_descriptor(
    field: dataclasses.Field[Any], hint: helpers.Hint
) -> helpers.Descriptor:
  """Apply the auto-casting magic to a single class."""
  # TODO(epot): Support `Optional`
  if field.default_factory is not dataclasses.MISSING:
    raise ValueError(
        f'dataclass field {field.name} cannot be both `AutoCast` and'
        ' `default_factory=`'
    )
  # TODO(epot): Propagate other field_kwargs (through likely not necessary)
  return field_utils.field(validate=hint)
