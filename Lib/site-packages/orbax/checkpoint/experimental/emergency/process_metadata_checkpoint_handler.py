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

"""ProcessMetadataCheckpointHandler class.

Implementation of CheckpointHandler interface.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

from etils import epath
import jax
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.handlers import async_checkpoint_handler
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint.experimental.emergency import mesh_consistency


CheckpointArgs = checkpoint_args.CheckpointArgs
PreviousDistributedToDeviceIds = List[List[int]]
PreviousDeviceIds = List[int]
register_with_handler = checkpoint_args.register_with_handler


class ProcessMetadataCheckpointHandler(
    async_checkpoint_handler.AsyncCheckpointHandler
):
  """Saves processmetadata."""

  def __init__(
      self,
      multiprocessing_options: options_lib.MultiprocessingOptions = options_lib.MultiprocessingOptions(),
  ):
    """Initializes ProcessMetadataCheckpointHandler."""

  async def async_save(
      self,
      directory: epath.Path,
      args: ProcessMetadataSaveArgs,
  ) -> Optional[List[future.Future]]:
    """Saves the given process metadata.

    Args:
      directory: save location directory.
      args: ProcessMetadataSaveArgs (see below).

    Returns:
      A list of commit futures.
    """
    return [
        future.CommitFutureAwaitingContractedSignals(
            mesh_consistency.save_process_metadata(
                directory,
                args.global_mesh,
                multihost.distributed_to_device_ids(),
                ),
            name='process_metadata_ch_save',
        )
    ]

  def save(
      self,
      directory: epath.Path,
      args: ProcessMetadataSaveArgs,
  ):
    async def async_save(directory, args):
      commit_futures = await self.async_save(directory, args)
      if commit_futures:
        for f in commit_futures:
          f.result()

    asyncio_utils.run_sync(async_save(directory, args))

  def restore(
      self,
      directory: epath.Path,
      args: ProcessMetadataRestoreArgs,
  ) -> Tuple[PreviousDistributedToDeviceIds, PreviousDeviceIds]:
    """Restores metadata from directory.

    Args:
      directory: restore location directory.
      args: ProcessMetadataRestoreArgs (see below).

    Returns:
      A tuple of previous distributed to device ids and previous device ids.
    """
    return mesh_consistency.read_process_metadata(directory)


@register_with_handler(ProcessMetadataCheckpointHandler, for_save=True)
@dataclasses.dataclass
class ProcessMetadataSaveArgs(CheckpointArgs):
  """Parameters for saving process metadata.

  Attributes:
    global_mesh: the global mesh.
  """

  global_mesh: jax.sharding.Mesh


@register_with_handler(ProcessMetadataCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class ProcessMetadataRestoreArgs(CheckpointArgs):
  """Parameters for restoring process metadata."""
