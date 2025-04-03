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

"""Synchronous Checkpointer implementation."""

import time
from typing import Any, Iterable, Optional, Type

from absl import logging
from etils import epath
from etils import epy
import jax
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint import options as options_lib
from orbax.checkpoint import utils
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.checkpointers import abstract_checkpointer
from orbax.checkpoint._src.futures import synchronization
from orbax.checkpoint._src.handlers import checkpoint_handler
from orbax.checkpoint._src.handlers import composite_checkpoint_handler
from orbax.checkpoint._src.metadata import checkpoint
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import atomicity
from orbax.checkpoint._src.path import atomicity_defaults
from orbax.checkpoint._src.path import atomicity_types
from typing_extensions import Self  # for Python version < 3.11



CheckpointArgs = checkpoint_args.CheckpointArgs
register_with_handler = checkpoint_args.register_with_handler
get_legacy_handler_wrapper = (
    composite_checkpoint_handler.get_legacy_handler_wrapper
)
StepMetadata = checkpoint.StepMetadata


def construct_checkpoint_args(
    handler: checkpoint_handler.CheckpointHandler,
    for_save: bool,
    *args,
    **kwargs,
) -> checkpoint_args.CheckpointArgs:
  """Constructs `CheckpointArgs` for save or restore for the handler."""
  for arg in args:
    if isinstance(arg, checkpoint_args.CheckpointArgs):
      return arg
  for arg in kwargs.values():
    if isinstance(arg, checkpoint_args.CheckpointArgs):
      return arg
  jax.monitoring.record_event('/jax/orbax/deprecation/checkpointer_legacy_args')
  save_arg_cls, restore_arg_cls = checkpoint_args.get_registered_args_cls(
      handler
  )
  if for_save:
    return save_arg_cls(*args, **kwargs)
  else:
    return restore_arg_cls(*args, **kwargs)


class Checkpointer(
    abstract_checkpointer.AbstractCheckpointer, epy.ContextManager
):
  """A synchronous implementation of AbstractCheckpointer.

  This class saves synchronously to a given directory using an underlying
  `CheckpointHandler`. Atomicity of the operation is guaranteed.

  IMPORTANT: Async checkpointing can often be faster for saving. Strongly
  consider using `AsyncCheckpointer` instead.

  IMPORTANT: Remember that to save and restore a checkpoint, one should always
  use an `AbstractCheckpointer` coupled with a `CheckpointHandler`. The specific
  `CheckpointHandler` to use depends on the object being saved or restored.

  Basic example::

    ckptr = Checkpointer(StandardCheckpointHandler())
    args = ocp.args.StandardSave(item=pytree_of_arrays)
    ckptr.save(path, args=args)
    args = ocp.args.StandardRestore(item=abstract_pytree_target)
    ckptr.restore(path, args=args)

  Each handler includes `...SaveArgs` and `...RestoreArgs` classes that document
  what arguments are expected. When using `Checkpointer`, you can either use
  this dataclass directly, or you can provide the arguments in keyword form.

  For example::

    ckptr = Checkpointer(StandardCheckpointHandler())
    ckptr.save(path, state=pytree_of_arays)
    ckptr.restore(path, state=abstract_pytree_target)
  """

  def __init__(
      self,
      handler: checkpoint_handler.CheckpointHandler,
      *,
      multiprocessing_options: options_lib.MultiprocessingOptions = options_lib.MultiprocessingOptions(),
      file_options: options_lib.FileOptions = options_lib.FileOptions(),
      checkpoint_metadata_store: Optional[checkpoint.MetadataStore] = None,
      temporary_path_class: Optional[
          Type[atomicity_types.TemporaryPath]
      ] = None,
  ):
    if not checkpoint_args.has_registered_args(handler):
      logging.warning(
          'No registered CheckpointArgs found for handler type: %s',
          type(handler),
      )
      handler = get_legacy_handler_wrapper(handler)
    self._handler = handler
    self._primary_host = multiprocessing_options.primary_host
    self._active_processes = multiprocessing_options.active_processes
    self._barrier_sync_key_prefix = (
        multiprocessing_options.barrier_sync_key_prefix
    )
    self._multiprocessing_options = options_lib.MultiprocessingOptions(
        primary_host=self._primary_host,
        active_processes=self._active_processes,
        barrier_sync_key_prefix=self._barrier_sync_key_prefix,
    )
    self._file_options = file_options
    self._temporary_path_class = temporary_path_class

    # If not provided then use checkpoint_metadata_store with blocking writes.
    self._metadata_store = (
        checkpoint_metadata_store
        or checkpoint.metadata_store(enable_write=True, blocking_write=True)
    )
    if not self._metadata_store.is_blocking_writer():
      raise ValueError('Checkpoint metadata store must be blocking writer.')

    jax.monitoring.record_event('/jax/orbax/checkpointer/init')

  def get_temporary_path(
      self, directory: epath.Path
  ) -> atomicity_types.TemporaryPath:
    temporary_path_class = (
        self._temporary_path_class
        or atomicity_defaults.get_default_temporary_path_class(directory)
    )
    tmpdir = temporary_path_class.from_final(
        directory,
        checkpoint_metadata_store=self._metadata_store,
        multiprocessing_options=self._multiprocessing_options,
        file_options=self._file_options,
    )
    return tmpdir

  async def create_temporary_path(
      self, temporary_path: atomicity_types.TemporaryPath
  ):
    await atomicity.create_all(
        [temporary_path],
        multiprocessing_options=self._multiprocessing_options,
    )

  def synchronize_next_awaitable_signal_operation_id(self):
    synchronization.HandlerAwaitableSignalOperationIdGenerator.next_operation_id()

    multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'next_awaitable_signal_operation_id:sync',
            prefix=self._barrier_sync_key_prefix,
        ),
        timeout=multihost.DIRECTORY_CREATION_TIMEOUT,
        processes=self._active_processes,
    )

  def save(
      self,
      directory: epath.PathLike,
      *args,
      force: bool = False,
      custom_metadata: dict[str, Any] | None = None,
      **kwargs,
  ):
    """Saves the given item to the provided directory.

    Delegates to the underlying CheckpointHandler. Ensures save operation
    atomicity.

    This method should be called by all hosts - process synchronization and
    actions that need to be performed on only one host are managed internally.

    Args:
      directory: a path to which to save.
      *args: additional args to provide to the CheckpointHandler's save method.
      force: if True, allows overwriting an existing directory. May add overhead
        due to the need to delete any existing files.
      custom_metadata: a dictionary of custom metadata to be written to the
        checkpoint directory via StepMetadata.
      **kwargs: additional keyword args to provide to the CheckpointHandler's
        save method.

    Raises:
      ValueError if the provided directory already exists.
    """
    checkpoint_start_time = time.time()
    multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'Checkpointer:save_start',
            prefix=self._barrier_sync_key_prefix,
        ),
        processes=self._active_processes,
        record_event_name=(
            '/jax/orbax/write/checkpoint_start_sync_duration_secs'
        ),
    )
    directory = epath.Path(directory)

    jax.monitoring.record_event('/jax/orbax/write/start')
    logging.info(
        '[process=%s] Started saving checkpoint to %s.',
        multihost.process_index(),
        directory,
    )
    self.synchronize_next_awaitable_signal_operation_id()

    if directory.exists():
      if force:
        if utils.is_primary_host(self._primary_host):
          logging.info('Specified `force`: removing existing directory.')
          directory.rmtree()  # Post-sync handled by create_tmp_directory.
      else:
        raise ValueError(f'Destination {directory} already exists.')
    ckpt_args = construct_checkpoint_args(self._handler, True, *args, **kwargs)
    tmpdir = self.get_temporary_path(directory)
    # tmpdir creation also does an initial StepMetadata save.
    asyncio_utils.run_sync(self.create_temporary_path(tmpdir))
    self._handler.save(tmpdir.get(), args=ckpt_args)
    if utils.is_primary_host(self._primary_host):
      # Update StepMetadata after the handler save is complete. (blocking write)
      self._save_step_metadata(tmpdir.get(), custom_metadata=custom_metadata)
    multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'Checkpointer:save',
            prefix=self._barrier_sync_key_prefix,
        ),
        processes=self._active_processes,
    )

    # Ensure save operation atomicity and record time saved by checkpoint.
    if utils.is_primary_host(self._primary_host):
      # finalize does a final StepMetadata update.
      self._handler.finalize(tmpdir.get())
      atomicity.on_commit_callback(
          tmpdir,
          checkpoint_start_time=checkpoint_start_time,
      )
    multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'Checkpointer:finalize',
            prefix=self._barrier_sync_key_prefix,
            # suffix=directory.name,
        ),
        processes=self._active_processes,
    )
    save_duration_secs = time.time() - checkpoint_start_time
    logging.info(
        'Finished synchronous save in %.2f seconds to %s',
        save_duration_secs,
        directory,
    )

  def restore(self, directory: epath.PathLike, *args, **kwargs) -> Any:
    """See superclass documentation."""
    restore_start_time = time.time()
    directory = epath.Path(directory)
    if not directory.exists():
      raise FileNotFoundError(f'Checkpoint at {directory} not found.')
    if not utils.is_checkpoint_finalized(directory):
      raise ValueError(f'Found incomplete checkpoint at {directory}.')
    logging.info('Restoring checkpoint from %s.', directory)
    ckpt_args = construct_checkpoint_args(self._handler, False, *args, **kwargs)
    restored = self._restore(directory, args=ckpt_args)
    multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'Checkpointer:restore',
            prefix=self._barrier_sync_key_prefix,
        ),
        processes=self._active_processes,
    )
    restore_duration_secs = time.time() - restore_start_time
    logging.info(
        'Finished restoring checkpoint in %.2f seconds from %s.',
        restore_duration_secs,
        directory,
    )
    return restored

  def _restore(
      self, directory: epath.PathLike, args: checkpoint_args.CheckpointArgs
  ) -> Any:
    return self._handler.restore(directory, args=args)

  def metadata(self, directory: epath.PathLike) -> StepMetadata | Any | None:
    """See superclass documentation."""
    directory = epath.Path(directory)
    return self._handler.metadata(directory)

  def _save_step_metadata(
      self, directory: epath.Path, custom_metadata: dict[str, Any] | None
  ):
    """Saves StepMetadata to the checkpoint directory."""
    update_dict = {
        'custom_metadata': custom_metadata,
    }
    if isinstance(
        self._handler, composite_checkpoint_handler.CompositeCheckpointHandler
    ):
      try:
        # get item_handlers from handler
        partial_metadata: StepMetadata = (
            self._handler.metadata_from_temporary_paths(directory)
        )
      except (FileNotFoundError, NotImplementedError, ValueError, TypeError):
        logging.warning(
            'Failed to get per-item metadata from directory %s. Handler types '
            'will not be saved.',
            directory,
        )
      else:
        update_dict['item_handlers'] = partial_metadata.item_handlers
    else:
      try:
        item_handler = self._handler.typestr()
      except (NotImplementedError, AttributeError):
        logging.warning(
            'Failed to get item handler typestr from directory %s. Backup '
            'handler type will be saved.',
            directory,
        )
        item_handler = (
            f'{self._handler.__module__}.{self._handler.__class__.__qualname__}'
        )
      update_dict['item_handlers'] = item_handler
    self._metadata_store.update(
        file_path=checkpoint.step_metadata_file_path(directory),
        **step_metadata_serialization.serialize_for_update(**update_dict),
    )

  def close(self):
    """Closes the underlying CheckpointHandler."""
    self._handler.close()
    self._metadata_store.close()

  @property
  def handler(self) -> checkpoint_handler.CheckpointHandler:
    return self._handler

  def __contextmanager__(
      self,
  ) -> Iterable[Self]:
    try:
      yield self
    finally:
      self.close()
