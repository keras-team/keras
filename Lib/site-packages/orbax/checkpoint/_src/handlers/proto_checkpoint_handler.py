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

"""ProtoCheckpointHandler class.

Implementation of CheckpointHandler interface.
"""

import dataclasses
from typing import List, Optional, Type

from etils import epath
from google.protobuf import message
from google.protobuf import text_format
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import options as options_lib
from orbax.checkpoint import utils
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.handlers import async_checkpoint_handler


CheckpointArgs = checkpoint_args.CheckpointArgs
register_with_handler = checkpoint_args.register_with_handler

_DEFAULT_FILENAME = "proto.pbtxt"


class ProtoCheckpointHandler(async_checkpoint_handler.AsyncCheckpointHandler):
  """Serializes/deserializes protocol buffers."""

  def __init__(
      self,
      filename: str = _DEFAULT_FILENAME,
      *,
      multiprocessing_options: options_lib.MultiprocessingOptions = options_lib.MultiprocessingOptions()
  ):
    """Initializes ProtoCheckpointHandler.

    Args:
      filename: file name given to the written file.
      multiprocessing_options: See orbax.checkpoint.options.
    """
    self._filename = filename
    self._primary_host = multiprocessing_options.primary_host

  async def async_save(
      self,
      directory: epath.Path,
      item: Optional[message.Message] = None,
      args: Optional["ProtoSaveArgs"] = None,
  ) -> Optional[List[future.Future]]:
    """Saves the given proto.

    Args:
      directory: save location directory.
      item: Deprecated, use `args`.
      args: ProtoSaveArgs (see below).

    Returns:
      A commit future.
    """
    if args is not None:
      item = args.item

    async def _save_fn(x):
      if utils.is_primary_host(self._primary_host):
        path = directory / self._filename
        path.write_text(text_format.MessageToString(x))

    return [
        future.CommitFutureAwaitingContractedSignals(
            _save_fn(item), name="proto_ch_save"
        )
    ]

  def save(self, *args, **kwargs):
    """Saves the provided item."""

    async def async_save():
      commit_futures = await self.async_save(*args, **kwargs)
      if commit_futures:
        for f in commit_futures:
          f.result()

    asyncio_utils.run_sync(async_save())

  def restore(
      self,
      directory: epath.Path,
      item: Optional[Type[message.Message]] = None,
      args: Optional["ProtoRestoreArgs"] = None,
  ):
    """Restores the proto from directory.

    Args:
      directory: restore location directory.
      item: Deprecated, use `args`.
      args: ProtoRestoreArgs (see below).

    Returns:
      The deserialized proto read from `directory` if item is not None
    """
    if not args and not item:
      raise ValueError(
          "Must provide `item` in order to deserialize proto to the correct"
          " type."
      )
    if args:
      item = args.item
    path = directory / self._filename
    return text_format.Parse(path.read_text(), item())


@register_with_handler(ProtoCheckpointHandler, for_save=True)
@dataclasses.dataclass
class ProtoSaveArgs(CheckpointArgs):
  """Parameters for saving a proto.

  Attributes:
    item (required): the proto to serialize.
  """

  item: message.Message


@register_with_handler(ProtoCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class ProtoRestoreArgs(CheckpointArgs):
  """Proto restore args.

  Attributes:
    item (required): the proto class
  """

  item: Type[message.Message]
