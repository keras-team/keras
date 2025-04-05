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

"""JsonCheckpointHandler class.

Implementation of CheckpointHandler interface.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, List, Mapping, Optional

from etils import epath
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import options as options_lib
from orbax.checkpoint import utils
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.handlers import async_checkpoint_handler

CheckpointArgs = checkpoint_args.CheckpointArgs
register_with_handler = checkpoint_args.register_with_handler


class JsonCheckpointHandler(async_checkpoint_handler.AsyncCheckpointHandler):
  """Saves nested dictionary using json."""

  def __init__(
      self,
      filename: Optional[str] = None,
      *,
      multiprocessing_options: options_lib.MultiprocessingOptions = options_lib.MultiprocessingOptions(),
  ):
    """Initializes JsonCheckpointHandler.

    Args:
      filename: optional file name given to the written file; defaults to
        'metadata'
      multiprocessing_options: See orbax.checkpoint.options.
    """
    self._filename = filename or 'metadata'
    self._primary_host = multiprocessing_options.primary_host

  async def _save_fn(self, x, directory):
    if utils.is_primary_host(self._primary_host):
      path = directory / self._filename
      path.write_text(json.dumps(x))

  async def async_save(
      self,
      directory: epath.Path,
      item: Optional[Mapping[str, Any]] = None,
      args: Optional[JsonSaveArgs] = None,
  ) -> Optional[List[future.Future]]:
    """Saves the given item.

    Args:
      directory: save location directory.
      item: Deprecated, use `args` instead.
      args: JsonSaveArgs (see below).

    Returns:
      A list of commit futures.
    """
    if isinstance(item, CheckpointArgs):
      raise ValueError(
          'Make sure to specify kwarg name `args=` when providing'
          ' `JsonSaveArgs`.'
      )
    if args is not None:
      item = args.item
    return [
        future.CommitFutureAwaitingContractedSignals(
            self._save_fn(item, directory), name='json_ch_save'
        )
    ]

  def save(
      self,
      directory: epath.Path,
      item: Optional[Mapping[str, Any]] = None,
      args: Optional[JsonSaveArgs] = None,
  ):
    async def async_save(directory, item, args):
      commit_futures = await self.async_save(directory, item, args)
      if commit_futures:
        for f in commit_futures:
          f.result()

    asyncio_utils.run_sync(async_save(directory, item, args))

  def restore(
      self,
      directory: epath.Path,
      item: Optional[Mapping[str, Any]] = None,
      args: Optional[JsonRestoreArgs] = None,
  ) -> Mapping[str, Any]:
    """Restores json mapping from directory.

    `item` is unused.

    Args:
      directory: restore location directory.
      item: unused
      args: unused

    Returns:
      JSON dict.

    Raises:
      FileNotFoundError: if the file does not exist.
    """
    del item
    del args
    path = directory / self._filename
    if not path.exists():
      raise FileNotFoundError(f'File {path} not found.')
    return json.loads(path.read_text())


@register_with_handler(JsonCheckpointHandler, for_save=True)
@dataclasses.dataclass
class JsonSaveArgs(CheckpointArgs):
  """Parameters for saving to json.

  Attributes:
    item (required): a nested dictionary.
  """

  item: Mapping[str, Any]


@register_with_handler(JsonCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class JsonRestoreArgs(CheckpointArgs):
  """Json restore args.

  Attributes:
    item: unused, but included for legacy-compatibility reasons.
  """

  item: Optional[bytes] = None
