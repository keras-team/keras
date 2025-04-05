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

"""ReplicatorCheckpointManager for emergency checkpoint. See details below.

WARNING: Do not use without specific approval. The API and implementation are
subject to change without notice.
"""

import dataclasses
from typing import Any, Callable, Iterable, List, Sequence, Tuple

from absl import logging
from etils import epath
from etils import epy
import jax
from orbax.checkpoint import abstract_checkpoint_manager
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint._src import composite
from orbax.checkpoint._src.handlers import handler_registration
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint.experimental.emergency import mesh_consistency
from orbax.checkpoint.experimental.emergency import process_metadata_checkpoint_handler
from orbax.checkpoint.path import step as step_lib
from typing_extensions import Self  # for Python version < 3.11


PyTree = Any
DefaultCheckpointHandlerRegistry = (
    handler_registration.DefaultCheckpointHandlerRegistry
)
PyTreeCheckpointHandler = pytree_checkpoint_handler.PyTreeCheckpointHandler
RootMetadata = checkpoint_manager.RootMetadata
StepMetadata = checkpoint_manager.StepMetadata
ProcessMetadataCheckpointHandler = (
    process_metadata_checkpoint_handler.ProcessMetadataCheckpointHandler
)


_UNNAMED_ITEM_NAME = 'state'
_PROCESS_METADATA_NAME = 'process_metadata'


def _local_checkpoint_handler(
    multiprocessing_options: checkpoint_manager.MultiprocessingOptions,
) -> Tuple[PyTreeCheckpointHandler, ProcessMetadataCheckpointHandler]:
  """Create a PyTreeCheckpointHandler for local checkpoints."""
  if multiprocessing_options.primary_host is not None:
    raise ValueError(
        'multiprocessing_options.primary_host must be set to None for local'
        ' checkpoints.'
    )
  local_registry = type_handlers.create_type_handler_registry(
      (
          jax.Array,
          type_handlers.ArrayHandler(
              primary_host=None, replica_id=None, use_replica_parallel=False
          ),
      ),
  )
  pytree_handler = PyTreeCheckpointHandler(
      use_ocdbt=True,
      use_zarr3=True,
      multiprocessing_options=multiprocessing_options,
      type_handler_registry=local_registry,
  )
  metadata_handler = ProcessMetadataCheckpointHandler()
  return pytree_handler, metadata_handler


@dataclasses.dataclass
class ReplicatorCheckpointManagerOptions:
  save_interval_steps: int = 1
  step_name_format: step_lib.NameFormat[step_lib.Metadata] | None = None
  should_save_fn: Callable[[int, int | None], bool] | None = None


class ReplicatorCheckpointManager(
    abstract_checkpoint_manager.AbstractCheckpointManager, epy.ContextManager
):
  """ReplicatorCheckpointManager.

  This class is intended for use in emergency checkpointing, but the system must
  conform to restrictive assumptions in order to work. Consider using
  `emergency.checkpoint_manager.CheckpointManager` if these assumptions are not
  met.

  This class assumes that an independent service is responsible for reflecting
  any checkpoints saved in local storage (e.g. RAMFS) to a persistent global
  storage (e.g. GCS). Thus, this class only saves checkpoints to process-local
  storage. Each process saves addressable data to its own storage location, on
  every process.

  When restoring a checkpoint, this class assumes that the independent
  replicator service will ensure consistency across all process-local storages.
  This means that if a restart causes data loss on one or more processes, the
  replicator must ensure that the corresponding data from a process with intact
  data is copied to the process with data loss. Specifically, the processes must
  be "peers" in the sense that they share the same index into the global array
  data.

  Users can control a few properties of the checkpointing behavior like the save
  interval. The period at which checkpoints are persisted to global storage, or
  the period at which they are
  garbage-collected, must be controlled by the replicator service.
  """

  def __init__(
      self,
      local_directory: epath.Path,
      options: ReplicatorCheckpointManagerOptions | None = None,
      *,
      global_mesh: jax.sharding.Mesh,
  ):
    self._global_mesh = global_mesh
    options = options or ReplicatorCheckpointManagerOptions()
    multiprocessing_options = checkpoint_manager.MultiprocessingOptions(
        primary_host=None
    )
    self._options = options
    self._step_name_format = (
        options.step_name_format or step_lib.standard_name_format()
    )
    self._options = dataclasses.replace(
        self._options,
        step_name_format=self._step_name_format,
    )
    [state_handler, process_metadata_handler] = _local_checkpoint_handler(
        multiprocessing_options
    )
    options = checkpoint_manager.CheckpointManagerOptions(
        save_interval_steps=options.save_interval_steps,
        max_to_keep=None,  # No garbage collection.
        step_name_format=options.step_name_format,
        should_save_fn=options.should_save_fn,
        multiprocessing_options=multiprocessing_options,
        create=True,
        cleanup_tmp_directories=False,  # Handled separately below.
        enable_background_delete=False,
        enable_async_checkpointing=True,
    )

    self._handler_registry = DefaultCheckpointHandlerRegistry()
    self._handler_registry.add(None, args_lib.PyTreeSave, state_handler)
    self._handler_registry.add(None, args_lib.PyTreeRestore, state_handler)
    self._handler_registry.add(
        None,
        process_metadata_checkpoint_handler.ProcessMetadataSaveArgs,
        process_metadata_handler,
    )
    self._handler_registry.add(
        None,
        process_metadata_checkpoint_handler.ProcessMetadataRestoreArgs,
        process_metadata_handler,
    )
    self._impl = checkpoint_manager.CheckpointManager(
        local_directory,
        options=options,
        handler_registry=self._handler_registry,
    )
    self._run_initial_garbage_collection()

  def _run_initial_garbage_collection(self):
    """Remove steps that might be left over from previous runs."""
    logging.info('Running initial garbage collection at %s.', self.directory)
    logging.info('Cleaning up existing temporary directories.')
    tmp_files = step_lib.tmp_checkpoints(self.directory)
    logging.info('Found tmp files: %s', tmp_files)
    for tmp_file in tmp_files:
      if (
          tmp_file
          == mesh_consistency.process_metadata_folder(self.directory).name
      ):
        continue
      (self.directory / tmp_file).rmtree()

  @property
  def directory(self) -> epath.Path:
    return self._impl.directory

  @property
  def global_mesh(self) -> jax.sharding.Mesh:
    return self._global_mesh

  def all_steps(self, read: bool = False) -> Sequence[int]:
    return self._impl.all_steps(read=read)

  def latest_step(self) -> int | None:
    return self._impl.latest_step()

  def best_step(self) -> int | None:
    raise NotImplementedError()

  def reload(self):
    return self._impl.reload()

  def reached_preemption(self, step: int) -> bool:
    return self._impl.reached_preemption(step)

  def should_save(self, step: int) -> bool:
    return self._impl.should_save(step)

  def delete(self, step: int):
    return self._impl.delete(step)

  def _validate_and_standardize_args(
      self,
      args: args_lib.CheckpointArgs | None,
  ) -> args_lib.Composite:
    if not isinstance(args, args_lib.CheckpointArgs):
      raise ValueError(
          f'Expected args to be a `CheckpointArgs`, but got {type(args)}.'
      )
    if not isinstance(args, composite.Composite):
      args = args_lib.Composite(**{_UNNAMED_ITEM_NAME: args})
    for a in args.values():
      assert isinstance(a, args_lib.CheckpointArgs)
      if not self._handler_registry.has(None, a):
        raise ValueError(
            f'{type(a)} is not supported by this CheckpointManager. This is'
            ' likely because it does not yet implement support for local'
            ' checkpointing.'
        )
    return args

  def save(
      self,
      step: int,
      args: args_lib.CheckpointArgs,
      *,
      force: bool = False,
  ) -> bool:
    args = self._validate_and_standardize_args(args)
    multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'CheckpointManager:save_start',
            prefix='replicator_checkpoint_manager',
        ),
        record_event_name=(
            '/jax/orbax/write/checkpoint_start_sync_duration_secs'
        ),
    )
    process_metadata_args = (
        process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
            global_mesh=self._global_mesh,
        )
    )
    args_dict = dict(args.items())
    args_dict[_PROCESS_METADATA_NAME] = process_metadata_args
    args = args_lib.Composite(**args_dict)
    saved = self._impl.save(step, args=args, force=force)
    return saved

  def _get_mesh_consistent_args(
      self,
      previous_distributed_to_device_ids: List[List[int]],
      previous_device_ids: List[int],
      args: args_lib.Composite,
  ) -> tuple[args_lib.Composite, args_lib.Composite]:
    restore_mesh = mesh_consistency.consistent_restore_mesh_from_metadata(
        self._global_mesh,
        multihost.distributed_to_device_ids(),
        previous_distributed_to_device_ids=previous_distributed_to_device_ids,
        previous_device_ids=previous_device_ids,
    )

    def _replace_sharding(arg: type_handlers.ArrayRestoreArgs):
      if arg.sharding is None:
        raise ValueError('ArrayRestoreArgs sharding cannot be None.')
      if not isinstance(arg.sharding, jax.sharding.NamedSharding):
        raise ValueError(
            'ArrayRestoreArgs sharding must be a NamedSharding, but got'
            f' {type(arg.sharding)}.'
        )
      return dataclasses.replace(
          arg,
          sharding=jax.sharding.NamedSharding(
              mesh=restore_mesh, spec=arg.sharding.spec
          ),
      )

    original_args = args
    consistent_args = {}
    for k, a in args.items():
      if isinstance(a, args_lib.PyTreeRestore):
        if a.restore_args is None:
          raise ValueError('`restore_args` cannot be None.')
        consistent_args[k] = dataclasses.replace(
            a, restore_args=jax.tree.map(_replace_sharding, a.restore_args)
        )
    return original_args, args_lib.Composite(**consistent_args)

  def _get_mesh_consistent_result(
      self,
      original_args: args_lib.Composite,
      consistent_result: args_lib.Composite,
      *,
      default_item_mode: bool,
  ):
    result = {}
    for k, a in original_args.items():
      item_result = consistent_result[k]
      if isinstance(a, args_lib.PyTreeRestore):
        original_shardings = jax.tree.map(
            lambda arg: arg.sharding, a.restore_args
        )
        item_result = mesh_consistency.consistent_restore_mesh_to_global_mesh(
            item_result, original_shardings
        )
      result[k] = item_result

    result = args_lib.Composite(**result)
    if default_item_mode:
      assert len(result) == 1
      return result[_UNNAMED_ITEM_NAME]
    else:
      return result

  def restore(
      self,
      step: int | None,
      args: args_lib.CheckpointArgs | None = None,
  ) -> Any:
    if step is None:
      step = self.latest_step()
      if step is None:
        raise FileNotFoundError(f'No steps found in {self.directory}.')
    default_item_mode = (
        checkpoint_manager.determine_default_item_mode_from_args(args)
    )
    args = self._validate_and_standardize_args(args)
    process_meatadata_args = args_lib.Composite(**{
        _PROCESS_METADATA_NAME: (
            process_metadata_checkpoint_handler.ProcessMetadataRestoreArgs()
        )
    })
    process_metadata_restored = self._impl.restore(
        step, args=process_meatadata_args
    )
    previous_distributed_to_device_ids, previous_device_ids = (
        process_metadata_restored[_PROCESS_METADATA_NAME]
    )
    original_args, consistent_args = self._get_mesh_consistent_args(
        previous_distributed_to_device_ids, previous_device_ids, args
    )
    restored = self._impl.restore(step, args=consistent_args)
    return self._get_mesh_consistent_result(
        original_args, restored, default_item_mode=default_item_mode
    )

  def item_metadata(self, step: int) -> Any:
    return self._impl.item_metadata(step)

  def metadata(self, step: int | None = None) -> RootMetadata | StepMetadata:
    return self._impl.metadata(step)

  def metrics(self, step: int) -> PyTree | None:
    raise NotImplementedError()

  def wait_until_finished(self):
    return self._impl.wait_until_finished()

  def check_for_errors(self):
    return self._impl.check_for_errors()

  def close(self):
    return self._impl.close()

  def __contextmanager__(
      self,
  ) -> Iterable[Self]:
    try:
      yield self
    finally:
      self.close()
