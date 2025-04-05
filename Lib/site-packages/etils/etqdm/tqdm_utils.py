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

"""Wrapper for tdqm."""

import typing
from typing import Optional, TypeVar

from etils import epy
from etils.epy import _internal

with _internal.check_missing_deps():
  # pylint: disable=g-import-not-at-top
  from absl import logging
  import tqdm as tqdm_base
  # pylint: enable=g-import-not-at-top

_IterableT = TypeVar('_IterableT')


class _LogFile:
  """A File-like object that log to INFO."""

  def write(self, message):
    logging.info(message)

  def flush(self):
    pass

  def close(self):
    pass


# TODO(epot): Mock the original `tqdm`, (in `__main__`), rather than
# having to change the import.
def tqdm(iterable: Optional[_IterableT] = None, **kwargs) -> _IterableT:
  """Add a progressbar to the iterable."""
  return tqdm_base.tqdm(iterable=iterable, **kwargs)


if typing.TYPE_CHECKING:
  # API is the same as open-source
  tqdm = tqdm_base.tqdm
