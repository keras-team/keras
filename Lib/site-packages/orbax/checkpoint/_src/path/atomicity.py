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

"""Utils for creating and finalizing temporary paths.

Note that the configurability provided by this feature does
not leave users free to define their own temporary path structure. The current
implementation is mainly a refactoring of old logic that separately created
temp directories and finalized them. It does not touch other logic that detects
temp checkpoints and cleans them up (primarily located in
orbax.checkpoint.path.step and CheckpointManager).

Ordinarily, atomic logic defaults to `AtomicRenameTemporaryPath`, which uses an
atomic rename to indicate checkpoint completion. However, not all filesystems
support atomic rename, so `CommitFileTemporaryPath` is provided as an
alternative, which uses a "commit_success" file to indicate completion.

Ideally, we would standardize on a single behavior, but it is difficult, largely
for legacy reasons, to achieve this. Furthermore, there are many other
alternative ways of ensuring save atomicity. As such, we have opted to provide
a more flexible approach that allows users to configure the behavior they want.

Configuration can be done in the following way::

  AsyncCheckpointer(
      StandardCheckpointHandler(),
      temporary_path_class=CommitFileTemporaryPath,
  )

  # OR

  CheckpointManager(
      directory,
      item_names=('state', 'dataset',),
      options=CheckpointManagerOptions(
          temporary_path_class=atomicity.CommitFileTemporaryPath
      ),
  )
"""

from __future__ import annotations

import asyncio
import itertools
import re
import threading
import time
from typing import Awaitable, Optional, Protocol, Sequence

from absl import logging
from etils import epath
import jax
from orbax.checkpoint import options as options_lib
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint._src.metadata import checkpoint as checkpoint_metadata
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import async_utils
from orbax.checkpoint._src.path import atomicity_types
from orbax.checkpoint._src.path import step as step_lib
from orbax.checkpoint._src.path import utils


TMP_DIR_SUFFIX = step_lib.TMP_DIR_SUFFIX
COMMIT_SUCCESS_FILE = step_lib._COMMIT_SUCCESS_FILE  # pylint: disable=protected-access


_module_unique_count = itertools.count()


class AsyncMakeDirFunc(Protocol):

  def __call__(
      self,
      path: epath.Path,
      parents: bool = False,
      exist_ok: bool = False,
      mode: int = step_lib.WORLD_READABLE_MODE,
      **kwargs,
  ) -> Awaitable[None]:
    """Creates the directory at path."""
    pass


async def _create_tmp_directory(
    async_makedir_func: AsyncMakeDirFunc,
    tmp_dir: epath.Path,
    *,
    primary_host: Optional[int] = 0,
    path_permission_mode: int = step_lib.WORLD_READABLE_MODE,
    checkpoint_metadata_store: Optional[
        checkpoint_metadata.MetadataStore
    ] = None,
    **kwargs,
) -> epath.Path:
  """Creates a non-deterministic tmp directory for saving for given `final_dir`.

  Also writes checkpoint metadata in the tmp directory.

  Args:
    async_makedir_func: An implementation of AsyncMakeDirFunc to call.
    tmp_dir: The temporary directory path.
    primary_host: primary host id, default=0.
    path_permission_mode: Path permission mode for the temp directory. e.g.
      0o750. Please check
      https://github.com/google/etils/blob/main/etils/epath/backend.py if your
        path is supported.
    checkpoint_metadata_store: optional `CheckpointMetadataStore` instance. If
      present then it is used to create `StepMetadata` with current timestamp.
    **kwargs: Optional. Additional kwargs to pass to `async_makedir_func`

  Returns:
    The tmp directory.

  Raises:
    FileExistsError: if tmp directory already exists.
  """
  if multihost.is_primary_host(primary_host):
    if await async_utils.async_exists(tmp_dir):
      if await async_utils.async_is_tmp_checkpoint(tmp_dir):
        logging.warning(
            'Attempted to create temporary directory %s which already exists.'
            ' Removing existing directory since it is not finalized.',
            tmp_dir,
        )
        await async_utils.async_rmtree(tmp_dir)
      else:
        raise FileExistsError(
            f'Attempted to create temporary directory {tmp_dir} which already'
            ' exists but appears a non-temporary checkpoint.'
        )
    logging.info('Creating tmp directory %s', tmp_dir)
    await async_makedir_func(
        tmp_dir,
        parents=True,
        exist_ok=False,
        mode=path_permission_mode,
        **kwargs,
    )
    if checkpoint_metadata_store is not None:
      checkpoint_metadata_store.write(
          file_path=checkpoint_metadata.step_metadata_file_path(tmp_dir),
          metadata=step_metadata_serialization.serialize(
              checkpoint_metadata.StepMetadata(
                  init_timestamp_nsecs=time.time_ns()
              )
          ),
      )

  return tmp_dir


def _get_tmp_directory(final_path: epath.Path) -> epath.Path:
  # Path may not be completely unique if a preemption occurs. We rely on the
  # existing tmp directory being deleted elsewhere.
  return epath.Path(final_path.parent) / (
      final_path.name + TMP_DIR_SUFFIX + str(next(_module_unique_count))
  )


def _get_tmp_directory_pattern(final_path_name: Optional[str] = None) -> str:
  suffix = r'\.orbax-checkpoint-tmp-.+'
  if final_path_name is None:
    return '(.+)' + suffix
  else:
    return final_path_name + suffix


class AtomicRenameTemporaryPath(atomicity_types.TemporaryPath):
  """TemporaryPath implementation that uses atomic rename."""

  def __init__(
      self,
      temporary_path: epath.Path,
      final_path: epath.Path,
      *,
      checkpoint_metadata_store: Optional[
          checkpoint_metadata.MetadataStore
      ] = None,
      file_options: Optional[options_lib.FileOptions] = None,
      multiprocessing_options: Optional[
          options_lib.MultiprocessingOptions
      ] = None,
  ):
    self._tmp_path = temporary_path
    self._final_path = final_path

    multiprocessing_options = (
        multiprocessing_options or options_lib.MultiprocessingOptions()
    )
    file_options = file_options or options_lib.FileOptions()
    self._checkpoint_metadata_store = checkpoint_metadata_store
    self._primary_host = multiprocessing_options.primary_host
    self._active_processes = multiprocessing_options.active_processes
    self._barrier_sync_key_prefix = (
        multiprocessing_options.barrier_sync_key_prefix
    )
    self._path_permission_mode = file_options.path_permission_mode

  @classmethod
  def from_final(
      cls,
      final_path: epath.Path,
      *,
      checkpoint_metadata_store: Optional[
          checkpoint_metadata.MetadataStore
      ] = None,
      file_options: Optional[options_lib.FileOptions] = None,
      multiprocessing_options: Optional[
          options_lib.MultiprocessingOptions
      ] = None,
  ) -> AtomicRenameTemporaryPath:
    return cls(
        _get_tmp_directory(final_path),
        final_path,
        checkpoint_metadata_store=checkpoint_metadata_store,
        file_options=file_options,
        multiprocessing_options=multiprocessing_options,
    )

  @classmethod
  def match(cls, temporary_path: epath.Path, final_path: epath.Path) -> bool:
    if re.match(
        _get_tmp_directory_pattern(final_path.name),
        temporary_path.name,
    ):
      return temporary_path.parent == final_path.parent
    return False

  def get(self) -> epath.Path:
    return self._tmp_path

  def get_final(self) -> epath.Path:
    return self._final_path

  async def create(
      self,
      *,
      file_options: options_lib.FileOptions = options_lib.FileOptions(),
  ) -> epath.Path:
    """Creates a non-deterministic tmp directory for saving for given `final_dir`.

    Also writes checkpoint metadata in the tmp directory.

    NOTE: This function does not include any barrier syncs, and calling it
    directly from multiprocess code can lead to race conditions. Prefer to
    use `atomicity.create_all` in such cases.

    Args:
      file_options: FileOptions object.

    Returns:
      The tmp directory.

    Raises:
      FileExistsError: if tmp directory already exists.
    """
    mode = step_lib.WORLD_READABLE_MODE  # pylint: disable=unused-variable
    mode = (
        file_options.path_permission_mode or self._path_permission_mode or mode
    )
    return await _create_tmp_directory(
        async_utils.async_makedirs,
        self._tmp_path,
        primary_host=self._primary_host,
        path_permission_mode=mode,
        checkpoint_metadata_store=self._checkpoint_metadata_store,
    )

  def finalize(
      self,
  ):
    """Finalizes atomic save by renaming tmp_dir or writing a success file.

    Updates checkpoint metadata with commit_timestamp_nsecs.

    """
    logging.info('Renaming %s to %s', self._tmp_path, self._final_path)
    if self._checkpoint_metadata_store:
      self._checkpoint_metadata_store.wait_until_finished()
      self._checkpoint_metadata_store.update(
          file_path=checkpoint_metadata.step_metadata_file_path(self._tmp_path),
          commit_timestamp_nsecs=time.time_ns(),
      )
      self._checkpoint_metadata_store.wait_until_finished()
    self._tmp_path.rename(self._final_path)

  def __repr__(self) -> str:
    return (
        f'AtomicRenameTemporaryPath(tmp="{self._tmp_path.name}",'
        f' final="{self._final_path.name}",'
        f' directory="{self._final_path.parent}")'
    )


class CommitFileTemporaryPath(atomicity_types.TemporaryPath):
  """TemporaryPath implementation that uses a commit file."""

  def __init__(
      self,
      temporary_path: epath.Path,
      final_path: epath.Path,
      *,
      checkpoint_metadata_store: Optional[
          checkpoint_metadata.MetadataStore
      ] = None,
      file_options: Optional[options_lib.FileOptions] = None,
      multiprocessing_options: Optional[
          options_lib.MultiprocessingOptions
      ] = None,
  ):
    self._tmp_path = temporary_path
    self._final_path = final_path

    multiprocessing_options = (
        multiprocessing_options or options_lib.MultiprocessingOptions()
    )
    file_options = file_options or options_lib.FileOptions()
    self._checkpoint_metadata_store = checkpoint_metadata_store
    self._primary_host = multiprocessing_options.primary_host
    self._active_processes = multiprocessing_options.active_processes
    self._barrier_sync_key_prefix = (
        multiprocessing_options.barrier_sync_key_prefix
    )
    self._path_permission_mode = file_options.path_permission_mode

  @classmethod
  def from_final(
      cls,
      final_path: epath.Path,
      *,
      checkpoint_metadata_store: Optional[
          checkpoint_metadata.MetadataStore
      ] = None,
      file_options: Optional[options_lib.FileOptions] = None,
      multiprocessing_options: Optional[
          options_lib.MultiprocessingOptions
      ] = None,
  ) -> CommitFileTemporaryPath:
    return cls(
        final_path,
        final_path,
        checkpoint_metadata_store=checkpoint_metadata_store,
        file_options=file_options,
        multiprocessing_options=multiprocessing_options,
    )

  @classmethod
  def match(cls, temporary_path: epath.Path, final_path: epath.Path) -> bool:
    return (
        temporary_path.name == final_path.name
        and temporary_path.parent == final_path.parent
    )

  def get(self) -> epath.Path:
    return self._tmp_path

  def get_final(self) -> epath.Path:
    return self._final_path

  async def create(
      self,
      *,
      file_options: options_lib.FileOptions = options_lib.FileOptions(),
  ) -> epath.Path:
    """Creates a non-deterministic tmp directory for saving for given `final_dir`.

    Also writes checkpoint metadata in the tmp directory.

    NOTE: This function does not include any barrier syncs, and calling it
    directly from multiprocess code can lead to race conditions. Prefer to
    use `atomicity.create_all` in such cases.

    Args:
      file_options: FileOptions object.

    Returns:
      The tmp directory.

    Raises:
      FileExistsError: if tmp directory already exists.
    """
    mode = step_lib.WORLD_READABLE_MODE
    mode = (
        file_options.path_permission_mode or self._path_permission_mode or mode
    )
    return await _create_tmp_directory(
        async_utils.async_makedirs,
        self._tmp_path,
        primary_host=self._primary_host,
        path_permission_mode=mode,
        checkpoint_metadata_store=self._checkpoint_metadata_store,
    )

  def finalize(
      self,
  ):
    """Finalizes atomic save by renaming tmp_dir or writing a success file.

    Updates checkpoint metadata with commit_timestamp_nsecs.

    """
    logging.info('Finalizing %s', self._tmp_path)
    if self._checkpoint_metadata_store:
      self._checkpoint_metadata_store.wait_until_finished()
      self._checkpoint_metadata_store.update(
          file_path=checkpoint_metadata.step_metadata_file_path(self._tmp_path),
          commit_timestamp_nsecs=time.time_ns(),
      )
      self._checkpoint_metadata_store.wait_until_finished()
    commit_success_file = self._final_path / COMMIT_SUCCESS_FILE
    commit_success_file.write_text(
        f'Checkpoint commit was successful to {self._final_path}'
    )


async def create_all(
    paths: Sequence[atomicity_types.TemporaryPath],
    *,
    multiprocessing_options: Optional[
        options_lib.MultiprocessingOptions
    ] = None,
):
  """Creates all temporary paths in parallel."""
  start = time.time()
  multiprocessing_options = (
      multiprocessing_options or options_lib.MultiprocessingOptions()
  )
  barrier_sync_key_prefix = multiprocessing_options.barrier_sync_key_prefix
  active_processes = multiprocessing_options.active_processes
  # Sync before existence is checked and directory is created because there are
  # additional existence checks happening in the callers of this function.
  multihost.sync_global_processes(
      multihost.unique_barrier_key(
          'create_tmp_directory:pre',
          prefix=barrier_sync_key_prefix,
      ),
      timeout=multihost.DIRECTORY_CREATION_TIMEOUT,
      processes=active_processes,
  )
  await asyncio.gather(*[path.create() for path in paths])
  multihost.sync_global_processes(
      multihost.unique_barrier_key(
          'create_tmp_directory:post',
          prefix=barrier_sync_key_prefix,
      ),
      timeout=multihost.DIRECTORY_CREATION_TIMEOUT,
      processes=active_processes,
  )
  directory_creation_secs = time.time() - start
  jax.monitoring.record_event_duration_secs(
      '/jax/orbax/write/directory_creation_secs', directory_creation_secs
  )
  logging.vlog(
      1,
      'Synchronous directory creation took %s seconds',
      directory_creation_secs,
  )


def create_all_async(
    paths: Sequence[atomicity_types.TemporaryPath],
    completion_signals: Sequence[synchronization.HandlerAwaitableSignal],
    *,
    multiprocessing_options: Optional[
        options_lib.MultiprocessingOptions
    ] = None,
) -> future.Future:
  """Creates all temporary paths in parallel asynchronously.

  Args:
    paths: Sequence of temporary paths to create.
    completion_signals: Sequence of signals to send when all paths are created.
      Also adds them to the awaitable signals contract.
    multiprocessing_options: MultiprocessingOptions to use for barrier syncs and
      primary host.

  Returns:
    A future that which sends the completion signals when all paths are created.
  """
  multiprocessing_options = (
      multiprocessing_options or options_lib.MultiprocessingOptions()
  )
  barrier_sync_key_prefix = multiprocessing_options.barrier_sync_key_prefix
  active_processes = multiprocessing_options.active_processes
  primary_host = multiprocessing_options.primary_host
  # Sync for existence check to complete on all hosts before directory
  # creation starts.
  multihost.sync_global_processes(
      multihost.unique_barrier_key(
          'create_tmp_directory:post_existence_check',
          prefix=barrier_sync_key_prefix,
      ),
      timeout=multihost.DIRECTORY_CREATION_TIMEOUT,
      processes=active_processes,
  )

  commit_future = future.NoopFuture()
  if multihost.is_primary_host(primary_host):
    commit_future = future.CommitFutureAwaitingContractedSignals(
        _create_paths(paths),
        send_signals=completion_signals,
        timeout_secs=multihost.DIRECTORY_CREATION_TIMEOUT,
    )
    future.add_to_awaitable_signals_contract(completion_signals)

  # Sync to enusre that all hosts have the same awaitable signals contract.
  multihost.sync_global_processes(
      multihost.unique_barrier_key(
          'add_to_awaitable_signals_contract',
          prefix=barrier_sync_key_prefix,
      ),
      timeout=multihost.DIRECTORY_CREATION_TIMEOUT,
      processes=active_processes,
  )
  return commit_future


async def _create_paths(
    paths: Sequence[atomicity_types.TemporaryPath],
):
  """Creates all temporary paths in parallel."""
  start = time.time()
  await asyncio.gather(*[path.create() for path in paths])
  directory_creation_secs = time.time() - start
  jax.monitoring.record_event_duration_secs(
      '/jax/orbax/write/directory_creation_secs',
      directory_creation_secs,
  )
  logging.vlog(
      1,
      'Asynchronous directory creation took %s seconds',
      directory_creation_secs,
  )


def on_commit_callback(
    tmp_dir: atomicity_types.TemporaryPath,
    *,
    checkpoint_start_time: float,
):
  """To commit save operation, atomically finalizes step dir.

  Records save duration and lineage-logs step dir.

  Args:
    tmp_dir: A temporary checkpoint directory, where the checkpoint data is
      currently saved.
    checkpoint_start_time: The time at which checkpoint saving began.
  """
  tmp_dir.finalize(
  )
  step_lib.record_saved_duration(checkpoint_start_time)
  jax.monitoring.record_event('/jax/orbax/write/success')
  logging.info(
      '[process=%s][thread=%s] Finished saving checkpoint (finalized tmp dir)'
      ' to `%s`.',
      multihost.process_index(),
      threading.current_thread().name,
      tmp_dir.get_final(),
  )
