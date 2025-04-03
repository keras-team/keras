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

"""Epath FLAGS utils."""

from __future__ import annotations

import os
import sys
import typing
from typing import Optional

from etils.epath import abstract_path
from etils.epath import typing as epath_typing
from typing_extensions import Literal

if typing.TYPE_CHECKING:
  from absl import flags

if 'absl.flags' in sys.modules:
  from absl import flags  # pylint: disable=g-import-not-at-top]
  # Skip this module when detecting in which module the flag is defined.
  # This is required to avoid duplicate flag issues when reloading adhoc
  # imports.
  flags.disclaim_key_flags()  # pylint: disable=used-before-assignment


# required=True -> Path
@typing.overload
def DEFINE_path(  # pylint: disable=invalid-name
    name: str,
    default: None,
    help: str,  # pylint: disable=redefined-builtin
    flag_values: flags.FlagValues = ...,
    *,
    required: Literal[True],
    **kwargs,
) -> flags.FlagHolder[abstract_path.Path]:
  ...


# required=False, default=None -> Path | None
@typing.overload
def DEFINE_path(  # For consistency with other flags, pylint: disable=invalid-name
    name: str,
    default: None,
    help: str,  # pylint: disable=redefined-builtin
    flag_values: flags.FlagValues = ...,
    *,
    required: Literal[False] = False,
    **kwargs,
) -> flags.FlagHolder[Optional[abstract_path.Path]]:
  ...


# required=False, default='/path' -> Path
@typing.overload
def DEFINE_path(  # For consistency with other flags, pylint: disable=invalid-name
    name: str,
    default: epath_typing.PathLike,
    help: str,  # pylint: disable=redefined-builtin
    flag_values: flags.FlagValues = ...,
    *,
    required: Literal[False] = False,
    **kwargs,
) -> flags.FlagHolder[abstract_path.Path]:
  ...


def DEFINE_path(  # pylint: disable=invalid-name
    name,
    default,
    help,  # pylint: disable=redefined-builtin
    flag_values=None,
    *,
    required=False,
    **kwargs,
):
  """Defines a flag containing a epath.Path value."""

  # Lazy-import as absl is an optional dep
  from absl import flags  # pylint: disable=g-import-not-at-top

  if flag_values is None:
    flag_values = flags.FLAGS

  class _PathParser(flags.ArgumentParser):

    def parse(self, value):
      return abstract_path.Path(value)

  class _PathSerializer(flags.ArgumentSerializer):

    def serialize(self, value):
      return os.fspath(value)

  return flags.DEFINE(
      _PathParser(),
      name,
      default,
      help,
      flag_values,
      _PathSerializer(),
      required=required,
      **kwargs,
  )  # pytype: disable=bad-return-type
