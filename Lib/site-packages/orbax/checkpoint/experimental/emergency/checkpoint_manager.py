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

"""A class providing emergency checkpoint management.


WARNING: This class is experimental; do not use without specific approval.

NOTE: All classes within this module should be called across all *relevant*
processes. CheckpointManager is designed to be created and called across
*every* process. LocalCheckpointManager is designed to be created and called
across every process within *non-primary* slices. Similarly, a CheckpointManager
intended to work only with the persistent checkpoint on the primary slice should
always be called across all processes within the primary slice.
"""

import asyncio
import collections
import dataclasses
import enum
import functools
import itertools
import operator
import time
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Set

from absl import logging
from etils import epath
from etils import epy
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import abstract_checkpoint_manager
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import checkpoint_manager
from orbax.checkpoint import checkpoint_utils
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.logging import abstract_logger
from orbax.checkpoint._src.logging import standard_logger
from orbax.checkpoint._src.logging import step_statistics
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.path import step as step_lib
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint.experimental.emergency import local_checkpoint_data_debugging
from orbax.checkpoint.experimental.emergency import mesh_consistency
from orbax.checkpoint.experimental.emergency import process_metadata_checkpoint_handler
from typing_extensions import Self  # for Python version < 3.11


PyTree = checkpoint_manager.PyTree
CheckpointHandler = checkpoint_manager.CheckpointHandler
P = jax.sharding.PartitionSpec
PyTreeCheckpointHandler = pytree_checkpoint_handler.PyTreeCheckpointHandler
ProcessMetadataCheckpointHandler = (
    process_metadata_checkpoint_handler.ProcessMetadataCheckpointHandler
)
unique_barrier_key = multihost._unique_barrier_key  # pylint: disable=protected-access
ChunkId = local_checkpoint_data_debugging.ChunkId
get_present_and_missing_chunks = (
    local_checkpoint_data_debugging.get_present_and_missing_chunks
)
RootMetadata = checkpoint_manager.RootMetadata
StepMetadata = checkpoint_manager.StepMetadata

_PRIMARY_REPLICA_ID = 0
_SECONDARY_REPLICA_ID = 1
_UNNAMED_ITEM_NAME = 'state'
_PROCESS_METADATA_NAME = 'process_metadata'


local_all_steps_broadcast_counter = itertools.count()
find_complete_slice_broadcast_counter = itertools.count()


def _local_checkpoint_handler(
    multiprocessing_options: checkpoint_manager.MultiprocessingOptions,
) -> PyTreeCheckpointHandler:
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
              primary_host=None,
              replica_id=None,
              use_replica_parallel=False,
          ),
      ),
  )
  return PyTreeCheckpointHandler(
      use_ocdbt=True,
      use_zarr3=True,
      multiprocessing_options=multiprocessing_options,
      type_handler_registry=local_registry,
  )


def _persistent_checkpoint_handler(
    multiprocessing_options: checkpoint_manager.MultiprocessingOptions,
) -> PyTreeCheckpointHandler:
  """Create a PyTreeCheckpointHandler for local checkpoints."""
  # TODO(b/372291557) Selection of replica_id=0 could be problematic if we can't
  # guarantee that the primary slice (in which the primary process is present)
  # always has the shard with shard.replica_id=0 for all available arrays.
  registry = type_handlers.create_type_handler_registry(
      (
          jax.Array,
          type_handlers.ArrayHandler(
              primary_host=multiprocessing_options.primary_host,
              replica_id=0,
              use_replica_parallel=False,
          ),
      ),
  )
  return PyTreeCheckpointHandler(
      use_ocdbt=True,
      use_zarr3=True,
      multiprocessing_options=multiprocessing_options,
      type_handler_registry=registry,
  )


@dataclasses.dataclass
class LocalCheckpointOptions:
  """Optional CheckpointManager arguments for saving local checkpoints.

  save_interval_steps:
    The interval at which checkpoints should be saved to local storage.
    Ensures checkpoints will only be saved every m steps. Defaults to 10.
  max_to_keep:
    Specifies the maximum number of local checkpoints to
    keep aside from the one currently being written. Older checkpoints are
    removed. When set, no more than (`max_to_keep` + 1) checkpoints will be
    present at any one time.
  read_only:
    If True, the local checkpoint manager will not save any checkpoints.
  should_save_fn:
    Predicate callable to check if given step can be saved. This callable
    accepts step number and optional latest step number as param and returns
    bool. If present then `save_interval_steps` and `save_on_steps` options are
    ignored.
  """

  save_interval_steps: int = 10
  max_to_keep: int = 1
  read_only: bool = False
  should_save_fn: Optional[Callable[[int, Optional[int]], bool]] = None

  debug_use_full_global_mesh: bool = False


@dataclasses.dataclass
class PersistentCheckpointOptions:
  """Optional CheckpointManager arguments for saving persistent checkpoints.

  save_interval_steps:
    The interval at which checkpoints should be saved to persistent storage.
    Ensures checkpoints will only be saved every n steps. Defaults to 1000.
  max_to_keep:
    If provided, specifies the maximum number of persistent checkpoints to
    keep. Older checkpoints are removed. By default, does not remove any old
    checkpoints. Must be None or non-negative. When set, checkpoints
    may be considered for deletion when there are more than `max_to_keep`
    checkpoints present.
  should_save_fn:
    Predicate callable to check if given step can be saved. This callable
    accepts step number and optional latest step number as param and returns
    bool. If present then `save_interval_steps` and `save_on_steps` options are
    ignored.
  """

  save_interval_steps: int = 1000
  max_to_keep: Optional[int] = None
  should_save_fn: Optional[Callable[[int, Optional[int]], bool]] = None


@dataclasses.dataclass
class MultiprocessingOptions:
  """Options used to configure multiprocessing behavior.

  coordination_timeout_secs: The timeout in seconds for inter-process
    coordination. Essentially, this should represent the maximum amount of time
    that different processes can be "out of sync" by.
  """

  coordination_timeout_secs: int = 120


@dataclasses.dataclass
class CheckpointManagerOptions:
  """Optional arguments for CheckpointManager.

  local:
    Options relevant to the local checkpoints.
    See `LocalCheckpointOptions`.
  persistent:
    Options relevant to the persistent checkpoints. See
    `PersistentCheckpointOptions`.
  replica_axis_index:
    The index of the replica axis in the global mesh.
  step_name_format:
    NameFormat to build or find steps under input root directory. If provided,
    `step_prefix`, `step_format_fixed_length` are ignored.
  cleanup_tmp_directories:
    If True, cleans up any existing temporary directories
    on CheckpointManager creation.
  enable_async_checkpointing:
    Enable async saving.
  async_options:
    Used to configure properties of async behavior. See above.
  """

  local: LocalCheckpointOptions = dataclasses.field(
      default_factory=LocalCheckpointOptions
  )
  persistent: PersistentCheckpointOptions = dataclasses.field(
      default_factory=PersistentCheckpointOptions
  )

  replica_axis_index: int = 0

  step_name_format: step_lib.NameFormat[step_lib.Metadata] = (
      step_lib.standard_name_format()
  )
  cleanup_tmp_directories: bool = False
  enable_async_checkpointing: bool = True
  async_options: Optional[checkpoint_manager.AsyncOptions] = None
  multiprocessing_options: Optional[MultiprocessingOptions] = None


class _BarrierIdentifier(enum.Enum):
  """Identifies the barrier being run."""

  LOCAL_ALL_STEPS = 'local_all_steps'
  FIND_COMPLETE_SLICE = 'find_complete_slice'

  def get_counter(self) -> str:
    if self.name == self.LOCAL_ALL_STEPS.name:
      return str(next(local_all_steps_broadcast_counter))
    elif self.name == self.FIND_COMPLETE_SLICE.name:
      return str(next(find_complete_slice_broadcast_counter))
    else:
      raise ValueError(f'Unknown barrier identifier: {self.name}')


def _common_values_per_slice(
    per_process_values: Dict[int, Set[int]],
    global_mesh: jax.sharding.Mesh,
    *,
    replica_axis_index: int,
) -> Dict[int, Set[int]]:
  """Obtains values shared in common across all processes in each slice.

  Args:
    per_process_values: A mapping of process index to a list of values local to
      that process.
    global_mesh: The global mesh.
    replica_axis_index: The index of the replica axis in the global mesh.

  Returns:
    A mapping of slice index to a set of values shared in common across all
    processes in that slice. A value appearing in one process but not another
    in the same slice will not appear in the output.
  """
  total_num_slices = multislice.slice_count(
      global_mesh, replica_axis_index=replica_axis_index
  )
  num_processes_per_slice = (
      global_mesh.devices.size // total_num_slices // jax.local_device_count()
  )
  per_slice_values = collections.defaultdict(list)
  for pid, values in per_process_values.items():
    slice_id = multislice.process_slice_id(
        pid, global_mesh, replica_axis_index=replica_axis_index
    )
    per_slice_values[slice_id].extend(values)

  for slice_id, values in per_slice_values.items():
    counter = collections.Counter(values)
    common_values = [
        k for k in counter if counter[k] == num_processes_per_slice
    ]
    # Here `len(common_values)`` will be less than or equal to `len(values)`
    # because a value can only appear in `common_values` if it occurs
    # `num_processes_per_slice` times in `values`.
    if len(common_values) > len(values):
      raise AssertionError(
          f' len(common_values) ({common_values}) exceeded length of input'
          f' values ({values}).'
      )
    per_slice_values[slice_id] = common_values

  return {k: set(v) for k, v in per_slice_values.items()}


def _pad_steps(steps, target):
  return steps + [-1] * (target - len(steps))


def _process_local_to_global(
    values: Set[int],
    barrier_processes: Set[int],
    *,
    timeout: int,
    barrier_id: _BarrierIdentifier,
    slice_id: Optional[int] = None,
) -> Dict[int, Set[int]]:
  """Shares a sequence of host-local integers across given processes.

  Args:
    values: A set of local values. Each process has its own set of values.
    barrier_processes: A set of processes to share the set of values with.
    timeout: The timeout in seconds for inter-process coordination.
    barrier_id: Barrier identifier.
    slice_id: The slice id. Only needed if multiple slices need to run the same
      barrier in parallel, but only sync intra-slice, not inter-slice.

  Returns:
    A mapping of process index to the sequence of local values on that process.
    The result will have an entry for every process in `barrier_processes`.
  """
  barrier_name = (
      f'{barrier_id.name}_{slice_id}' if slice_id else barrier_id.name
  )
  # Need to include barrier processes because there can be identical calls, just
  # with different participating processes.
  barrier_processes_as_string = ''.join(str(p) for p in barrier_processes)
  barrier_name_and_id = (
      f'{barrier_name}_{barrier_id.get_counter()}_{barrier_processes_as_string}'
  )

  client = multihost.get_jax_distributed_client()
  broadcast_dir_key = f'broadcast_{barrier_name_and_id}/'
  broadcast_dir_key = unique_barrier_key(broadcast_dir_key) + '/'
  broadcast_key = broadcast_dir_key + str(multihost.process_index())
  client.key_value_set(broadcast_key, ','.join([str(s) for s in values]))

  barrier_key = f'barrier_{barrier_name_and_id}'
  barrier_key = unique_barrier_key(barrier_key)
  logging.vlog(
      1,
      '[process=%s] Waiting at barrier %s',
      multihost.process_index(),
      barrier_key,
  )
  logging.vlog(
      1,
      '[process=%s] Barrier processes: %s',
      multihost.process_index(),
      barrier_processes,
  )
  client.wait_at_barrier(
      barrier_key,
      process_ids=list(barrier_processes),
      timeout_in_ms=timeout * 1000,
  )

  per_process_values = {
      int(k.split('/')[-1]): {int(s) for s in v.split(',')} if v else set()
      for k, v in client.key_value_dir_get(broadcast_dir_key)
  }
  assert set(per_process_values.keys()) == barrier_processes
  return per_process_values


def _all_devices_excepting_slice(
    devices: np.ndarray,
    *,
    replica_id: int = 0,
    replica_axis_index: int = 0,
) -> np.ndarray:
  if hasattr(jax.devices()[0], 'slice_index'):
    get_slice_id = np.vectorize(lambda x: x.slice_index)
    return devices[get_slice_id(devices) != replica_id]
  else:
    return np.delete(devices, replica_id, axis=replica_axis_index)


def _get_global_broadcast_fn() -> Callable[[jax.Array], jax.Array]:
  """Returns a global broadcast function. More efficient to compile once."""
  # Compile function for global broadcast.
  slice_mesh = jax.sharding.Mesh(
      np.asarray(jax.devices()).reshape(
          multihost.process_count(), jax.local_device_count()
      ),
      ['host', 'dev'],
  )
  return jax.jit(
      lambda x: x,
      out_shardings=jax.sharding.NamedSharding(slice_mesh, P()),
  )


def _global_max(
    values: list[int], global_broadcast_fn: Callable[[jax.Array], jax.Array]
) -> list[int]:
  """Computes the global max of a list of values."""
  num_hosts = multihost.process_count()
  num_devices_per_host = jax.local_device_count()
  slice_mesh = jax.sharding.Mesh(
      np.asarray(jax.devices()).reshape(num_hosts, num_devices_per_host),
      ['host', 'dev'],
  )

  sdas = []
  for d in jax.local_devices():
    sdas.append(
        jax.device_put(np.asarray(values).reshape((1, 1, len(values))), d)
    )
  sharding = jax.sharding.NamedSharding(slice_mesh, P('host', 'dev'))
  # TODO(cpgaffney): Use jax.make_array_from_process_local_data.
  g_arr = jax.make_array_from_single_device_arrays(
      (num_hosts, num_devices_per_host, len(values)), sharding, sdas
  )
  result_arr = global_broadcast_fn(g_arr)
  result_arr = np.asarray(result_arr.addressable_data(0))

  # Checks that for every host, values are equal across local devices.
  assert (
      np.sum(result_arr, axis=1) / num_devices_per_host == result_arr[:, 0, :]
  ).all()
  # Select values from first device and compute max for each value across
  # hosts.
  return list(np.max(result_arr[:, 0, :], axis=0).astype(int))


class _LocalCheckpointManager(checkpoint_manager.CheckpointManager):
  """A checkpoint manager that checkpoints to local storage.

  Attributes:
    device_array: an ndarray representing all the devices running
      LocalCheckpointManager in the same global jax Mesh, importantly the first
      axis of the device_array is assumed to be the direction of device slices
      across which the Data Parallelism is happening.
  """

  # TODO: b/330585086 - Allow configuration of global mesh describing slices.
  # Validate against global meshes used for arrays in state.
  # TODO: b/330585086 - Support arbitrary items beyond state. We will have
  # to evaluate whether arbitrary items can be a good fit for local
  # checkpointing, given restore+broadcast requirements.
  def __init__(
      self,
      directory: epath.PathLike,
      global_mesh: jax.sharding.Mesh,
      *,
      primary_replica_id: int = _PRIMARY_REPLICA_ID,
      options: Optional[CheckpointManagerOptions] = None,
      metadata: Optional[dict[str, Any]] = None,
      logger: Optional[abstract_logger.AbstractLogger] = None,
  ):
    # TODO: b/330585086 - Fully support options.
    options = options or CheckpointManagerOptions()
    self._global_mesh = global_mesh
    self._replica_axis_index = options.replica_axis_index

    devices = np.asarray(self._global_mesh.devices)
    # Select all devices except those belonging to the primary replica.
    if not options.local.debug_use_full_global_mesh:
      devices = _all_devices_excepting_slice(
          devices,
          replica_id=primary_replica_id,
          replica_axis_index=self._replica_axis_index,
      )

    self._active_processes = multihost.unique_processes_from_devices(devices)
    multiprocessing_options = checkpoint_manager.MultiprocessingOptions(
        primary_host=None,
        active_processes=self._active_processes,
        barrier_sync_key_prefix='local',
    )
    local_options = checkpoint_manager.CheckpointManagerOptions(
        save_interval_steps=options.local.save_interval_steps,
        max_to_keep=options.local.max_to_keep,
        step_name_format=options.step_name_format,
        should_save_fn=options.local.should_save_fn,
        create=False,
        # we always clean up local tmp directories explicitly
        cleanup_tmp_directories=False,
        async_options=options.async_options,
        multiprocessing_options=multiprocessing_options,
        enable_async_checkpointing=options.enable_async_checkpointing,
        read_only=options.local.read_only,
        single_host_load_and_broadcast=False,
        # enable_background_delete set to False to ensure gc is done before save
        enable_background_delete=False,
        save_root_metadata=False,
    )
    self._logger = logger or standard_logger.StandardLogger()
    self._coordination_timeout_secs = (
        options.multiprocessing_options or MultiprocessingOptions()
    ).coordination_timeout_secs
    super().__init__(
        directory,
        options=local_options,
        metadata=metadata,
        item_handlers=dict(
            state=_local_checkpoint_handler(multiprocessing_options),
            process_metadata=ProcessMetadataCheckpointHandler,
        ),
        logger=self._logger,
    )
    self._run_initial_garbage_collection()
    # Set additional properties.
    self._max_to_keep = options.local.max_to_keep
    self._local_options = options.local
    self._steps = list(self.all_steps(read=True))

  def _run_initial_garbage_collection(self):
    """Remove steps that might be left over from previous runs."""
    steps_to_remove = self._get_old_steps_to_remove()
    self._checkpoints.delete_if(lambda info: info.step in steps_to_remove)
    self._checkpoint_deleter.delete_steps(steps_to_remove)

  def local_host_steps(self, read: bool) -> Sequence[int]:
    """Returns steps known to local host."""
    # List of steps present in individual host storage.
    local_steps = list(super().all_steps(read))
    logging.info(
        'Found steps: %s in local host storage: %s.',
        local_steps,
        self.directory,
    )

    if len(local_steps) > self._max_to_keep:
      logging.error(
          'local_step on host %d exceeded `max_to_keep` %d',
          multihost.process_index(),
          self._max_to_keep,
      )

    return _pad_steps(local_steps, self._max_to_keep)

  def all_steps(self, read: bool = False) -> Sequence[int]:
    """Returns all steps tracked by the manager.

    Includes steps located in local storage.

    Args:
      read: If True, forces a read directly from the storage location.
        Otherwise, a cached result can be returned.

    Returns:
      A sequence of steps (integers)
    """
    if read:
      local_steps = set(self.local_host_steps(read))
      # Per-process mapping of the local steps that each process knows about.
      per_process_steps = _process_local_to_global(
          local_steps,
          self._active_processes,
          timeout=self._coordination_timeout_secs,
          barrier_id=_BarrierIdentifier.LOCAL_ALL_STEPS,
      )
      slice_id = multislice.process_slice_id(
          multihost.process_index(),
          self._global_mesh,
          replica_axis_index=self._replica_axis_index,
      )
      per_slice_steps = _common_values_per_slice(
          per_process_steps,
          self._global_mesh,
          replica_axis_index=self._replica_axis_index,
      )
      logging.info(
          'After broadcast, found steps %s shared between local slice'
          ' processes.',
          per_slice_steps[slice_id],
      )
      steps = functools.reduce(operator.ior, per_slice_steps.values(), set())
      self._steps = [x for x in steps if x != -1]
    return self._steps

  def latest_step(self) -> Optional[int]:
    """Returns the latest step saved in the local storage.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    return max(self._steps) if self._steps else None

  def save(
      self,
      step: int,
      args: Optional[args_lib.CheckpointArgs] = None,
      metrics: Optional[PyTree] = None,
      force: Optional[bool] = False,
  ) -> bool:
    """Saves the checkpoint at the given step."""
    saved = super().save(step, args=args, metrics=metrics, force=force)
    if saved:
      self._steps.append(step)
      self._steps = self._steps[-self._max_to_keep :]
    return saved

  def reload(self):
    """Reloads internal properties.

    refreshes the cached list of globally available local checkpointed steps.
    """
    super().reload()
    self._steps = list(self.all_steps(read=True))


def _get_single_slice_sharding(
    mesh: jax.sharding.Mesh,
    pspec: jax.sharding.PartitionSpec,
    replica_id: int,
    replica_axis_index: int,
):
  """Get sharding for a single slice."""
  slice_devices = multislice.slice_devices(
      mesh,
      replica_id=replica_id,
      replica_axis_index=replica_axis_index,
  )
  single_slice_mesh_shape = [
      1 if i == replica_axis_index else d
      for i, d in enumerate(mesh.devices.shape)
  ]
  slice_mesh = jax.sharding.Mesh(
      slice_devices.reshape(single_slice_mesh_shape), mesh.axis_names
  )
  return jax.sharding.NamedSharding(slice_mesh, pspec)


def _get_persistent_options(
    options: CheckpointManagerOptions,
    multiprocessing_options: checkpoint_manager.MultiprocessingOptions,
) -> checkpoint_manager.CheckpointManagerOptions:
  """Get options for persistent checkpoint manager."""
  return checkpoint_manager.CheckpointManagerOptions(
      save_interval_steps=options.persistent.save_interval_steps,
      max_to_keep=options.persistent.max_to_keep,
      step_name_format=options.step_name_format,
      create=False,
      cleanup_tmp_directories=options.cleanup_tmp_directories,
      async_options=options.async_options,
      multiprocessing_options=multiprocessing_options,
      enable_async_checkpointing=options.enable_async_checkpointing,
      should_save_fn=options.persistent.should_save_fn,
      save_root_metadata=False,
  )


class _MultisliceCheckpointManager(
    abstract_checkpoint_manager.AbstractCheckpointManager, epy.ContextManager
):
  """Provides both checkpoint management and emergency checkpointings.

  This class is an implementation layer for handling multislice checkpointing.

  This class composes a local and a persistent checkpoint managers. The local
  manager saves checkpoints frequently to a fast local storage (like RAMFS).
  When a complete checkpoint exists at least one slice, restoration is possible,
  and the slice broadcasts the checkpoint to others. Additionally, the
  persistent manager checkpoints less frequently to a remote file system (e.g.,
  GCS),
  providing a fail-safe if local checkpoints become unavailable due to issues
  like hardware failure or preemption.

  Usage::
    options = CheckpointManagerOptions(
        local=LocalCheckpointOptions(save_interval_steps=2, max_to_keep=2),
        persistent=PersistentCheckpointOptions(
            save_interval_steps=5, max_to_keep=3
        ),
        enable_async_checkpointing=use_async,
    )
    return _MultisliceCheckpointManager(
        local_directory=local_directory,
        persistent_directory=persistent_directory,
        global_mesh=global_mesh,
        abstract_state=abstract_state,
        options=options,
    )
  """

  # TODO: b/330585086 - Support arbitrary items beyond state. We will have
  # to evaluate whether arbitrary items can be a good fit for local
  # checkpointing, given restore+broadcast requirements.
  def __init__(
      self,
      local_directory: epath.PathLike,
      persistent_directory: epath.PathLike,
      global_mesh: jax.sharding.Mesh,
      abstract_state: PyTree,  # a single PyTree describing the state structure
      *,
      options: Optional[CheckpointManagerOptions] = None,
      metadata: Optional[dict[str, Any]] = None,
      logger: Optional[abstract_logger.AbstractLogger] = None,
  ):
    self._local_directory = epath.Path(local_directory)
    self._persistent_directory = epath.Path(persistent_directory)
    if not self._local_directory.exists():
      raise FileNotFoundError(
          f'Local directory {self._local_directory} must be created by the'
          ' caller.'
      )
    if not self._persistent_directory.exists():
      raise FileNotFoundError(
          f'Persistent directory {self._persistent_directory} must be created'
          ' by the caller.'
      )

    self._logger = logger or standard_logger.StandardLogger()
    # TODO: b/330585086 - Fully support options.
    options = options or CheckpointManagerOptions()
    self._replica_axis_index = options.replica_axis_index
    self._global_mesh = global_mesh
    logging.info(
        'Configured emergency.CheckpointManager with replica_axis_index=%d,'
        ' corresponding to "%s" in the global mesh.',
        self._replica_axis_index,
        self._global_mesh.axis_names[self._replica_axis_index],
    )

    self._abstract_state = abstract_state
    self._slice_id = multislice.process_slice_id(
        multihost.process_index(),
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
    )
    self._options = options
    self._metadata = metadata

    self._persistent_max_to_keep = self._options.persistent.max_to_keep
    self._local_max_to_keep = self._options.local.max_to_keep
    self._coordination_timeout_secs = (
        options.multiprocessing_options or MultiprocessingOptions()
    ).coordination_timeout_secs

    if len(global_mesh.devices.shape) <= self._replica_axis_index:
      raise AssertionError(
          f'replica_axis_index {self._replica_axis_index} is out of bound for'
          f' global_mesh.devices.shape {global_mesh.devices.shape}'
      )
    if (
        multislice.slice_count(
            global_mesh, replica_axis_index=self._replica_axis_index
        )
        <= 1
    ):
      raise AssertionError(
          'To use this CheckpointManager, at least 2 data-parallel replicas are'
          ' needed.'
      )

    primary_replica_id = _PRIMARY_REPLICA_ID
    secondary_replica_id = _SECONDARY_REPLICA_ID

    self._persistent_primary_host = multislice.primary_process_in_slice(
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
        replica_id=primary_replica_id,
    )
    self._local_primary_host = multislice.primary_process_in_slice(
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
        replica_id=secondary_replica_id,
    )
    self._in_primary_slice = multislice.in_slice(
        multihost.process_index(),
        global_mesh,
        replica_axis_index=self._replica_axis_index,
        replica_id=primary_replica_id,
    )

    if self._in_primary_slice:
      persistent_multiprocessing_options = (
          checkpoint_manager.MultiprocessingOptions(
              primary_host=self._persistent_primary_host,
              active_processes=multihost.unique_processes_from_devices(
                  multislice.slice_devices(
                      self._global_mesh,
                      replica_axis_index=self._replica_axis_index,
                      replica_id=primary_replica_id,
                  )
              ),
              barrier_sync_key_prefix='persistent',
          )
      )
      self._persistent_checkpoint_manager = (
          self._make_persistent_checkpoint_manager(
              persistent_multiprocessing_options
          )
      )
    else:
      self._local_checkpoint_manager = self._make_local_checkpoint_manager(
          primary_replica_id
      )

    self._local_steps = []
    self._persistent_steps = []
    # clean up tmp directories in ram
    self._cleanup_local_tmp_directories()

    # Initialize step cache.
    self.all_steps(read=True)

    self._global_broadcast_fn = _get_global_broadcast_fn()

    logging.info(
        'Created emergency.CheckpointManager with slice_id=%d,'
        ' process_index=%d, jax.process_index=%d',
        self._slice_id,
        multihost.process_index(),
        jax.process_index(),
    )
    logging.vlog(1, 'Local devices: %s', jax.local_devices())

  def _cleanup_local_tmp_directories(self):
    logging.info(
        'Cleaning up existing temporary directories at %s.',
        self._local_directory,
    )
    tmp_files = step_lib.tmp_checkpoints(self._local_directory)
    for tmp_file in tmp_files:
      logging.info('Deleting temporary checkpoint: %s.', tmp_file)
      (self._local_directory / tmp_file).rmtree()

  def _make_persistent_checkpoint_manager(
      self,
      persistent_multiprocessing_options: checkpoint_manager.MultiprocessingOptions,
  ) -> checkpoint_manager.CheckpointManager:
    return checkpoint_manager.CheckpointManager(
        self._persistent_directory,
        options=_get_persistent_options(
            self._options, persistent_multiprocessing_options
        ),
        metadata=self._metadata,
        item_handlers=_persistent_checkpoint_handler(
            persistent_multiprocessing_options
        ),
        logger=self._logger,
    )

  def _make_local_checkpoint_manager(
      self, primary_replica_id: int = _PRIMARY_REPLICA_ID
  ) -> _LocalCheckpointManager:
    return _LocalCheckpointManager(
        self._local_directory,
        global_mesh=self._global_mesh,
        primary_replica_id=primary_replica_id,
        options=self._options,
        metadata=self._metadata,
        logger=self._logger,
    )

  @property
  def directory(self) -> epath.Path:
    return self._persistent_directory

  @property
  def in_primary_slice(self) -> bool:
    return self._in_primary_slice

  @property
  def global_mesh(self) -> jax.sharding.Mesh:
    return self._global_mesh

  def all_steps(self, read: bool = False) -> Sequence[int]:
    """Returns all steps tracked by the manager.

    Includes steps located in local as well as persistent storage.

    Args:
      read: If True, forces a read directly from the storage location.
        Otherwise, a cached result can be returned.

    Returns:
      A sequence of steps (integers)
    """
    logging.info('Retrieving all steps.')
    if read:
      per_slice_local_steps = self._get_per_slice_local_steps()
      self._local_steps = list(set.union(*per_slice_local_steps.values()))
      self._persistent_steps = step_lib.checkpoint_steps(
          self._persistent_directory
      )
    return list(set(self._local_steps) | set(self._persistent_steps))

  def latest_step(self) -> Optional[int]:
    """Returns the latest step saved.

    Includes steps located in local as well as persistent storage.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    logging.info('Retrieving latest step.')
    all_steps = self.all_steps()
    return max(all_steps) if all_steps else None

  def best_step(self) -> Optional[int]:
    """Returns the best step saved, as defined by `options.best_fn`.

    Includes steps located in local as well as persistent storage.

    Returns None if no steps have been saved.

    Returns:
      A step (int) or None if no steps are present.
    """
    raise NotImplementedError(
        'Metrics tracking not yet implemented for emergency.CheckpointManager.'
    )

  def reload(self):
    """Performs disk reads to ensure internal properties are up to date."""
    if self.in_primary_slice:
      self._persistent_checkpoint_manager.reload()
    else:
      self._local_checkpoint_manager.reload()

  def reached_preemption(self, step: int) -> bool:
    """Returns True if a preemption sync point has been reached."""
    return multihost.reached_preemption(step)

  def should_save(self, step: int) -> bool:
    """Returns True if a checkpoint should be saved for the current step.

    This depends the previous step and save interval.

    Args:
      step: int

    Returns:
      True if the checkpoint should be saved.
    """
    logging.info('Checking should_save at step: %d.', step)
    if self.in_primary_slice:
      should_save = self._persistent_checkpoint_manager.should_save(step)
    else:
      should_save = self._local_checkpoint_manager.should_save(step)
    return bool(_global_max([int(should_save)], self._global_broadcast_fn)[0])

  def delete(self, step: int):
    """Deletes a step checkpoint."""
    raise NotImplementedError(
        'Delete not yet implemented for emergency.CheckpointManager.'
    )

  def save(
      self,
      step: int,
      args: args_lib.CheckpointArgs,
      *,
      force: bool = False,
  ) -> bool:
    """Returns True if a checkpoint was saved either locally or persistently."""
    multihost.sync_global_processes(
        multihost.unique_barrier_key(
            'CheckpointManager:save_start',
            prefix='emergency_checkpoint_manager',
        ),
        record_event_name=(
            '/jax/orbax/write/checkpoint_start_sync_duration_secs'
        ),
    )
    # TODO: b/330608746 - implement save op on different slices
    persistent_saved = False
    local_saved = False
    if self.in_primary_slice:
      logging.info('Maybe saving at step %d (persistent).', step)
      persistent_saved = self._persistent_checkpoint_manager.save(
          step, args=args, force=force
      )
    else:
      logging.info('Maybe saving at step %d (local).', step)

      args = args_lib.Composite(**{
          _UNNAMED_ITEM_NAME: args,
          _PROCESS_METADATA_NAME: (
              process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self._global_mesh
              )
          ),
      })

      local_saved = self._local_checkpoint_manager.save(
          step, args=args, force=force
      )

    start = time.time()
    saved = tuple(
        bool(e)
        for e in _global_max(
            [int(persistent_saved), int(local_saved)], self._global_broadcast_fn
        )
    )
    persistent_saved, local_saved = saved
    logging.info('Broadcast `saved` bool in %f seconds.', time.time() - start)

    if persistent_saved:
      self._persistent_steps.append(step)
      if self._persistent_max_to_keep is not None:
        self._persistent_steps = self._persistent_steps[
            -self._persistent_max_to_keep :
        ]
    if local_saved:
      self._local_steps.append(step)
      self._local_steps = self._local_steps[-self._local_max_to_keep :]

    return persistent_saved or local_saved

  def _get_per_slice_local_steps(self) -> Dict[int, Set[int]]:
    local_steps = set(step_lib.checkpoint_steps(self._local_directory))
    logging.info(
        'Found steps: %s in local host storage: %s.',
        local_steps,
        self._local_directory,
    )

    # The steps that each process actually has in local storage.
    per_process_steps = _process_local_to_global(
        local_steps,
        set(range(jax.process_count())),
        timeout=self._coordination_timeout_secs,
        barrier_id=_BarrierIdentifier.FIND_COMPLETE_SLICE,
    )
    logging.vlog(1, 'per_process_steps=%s', per_process_steps)
    per_slice_steps = _common_values_per_slice(
        per_process_steps,
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
    )
    logging.vlog(1, 'per_slice_steps=%s', per_slice_steps)
    return per_slice_steps

  def _find_slice_with_complete_local_checkpoint(self, step: int) -> int:
    """Return the slice id which has the step."""
    per_slice_steps = self._get_per_slice_local_steps()

    for slice_id, steps in per_slice_steps.items():
      if step in steps:
        return slice_id
    return -1

  def _restore_from_local(
      self,
      step: int,
      restoring_slice_id: int,
      directory: Optional[epath.PathLike] = None,
  ) -> Any:
    logging.info(
        'emergency.CheckpointManager: restoring step=%s from local checkpoint'
        ' using slice_id: %s',
        step,
        restoring_slice_id,
    )
    step_stats = step_statistics.EmergencyRestoreStepStatistics()
    step_stats.checkpoint_manager_start_time = time.time()
    step_stats.step = step
    is_restoring_slice = restoring_slice_id == self._slice_id
    step_stats.is_restoring_slice = is_restoring_slice
    step_stats.in_primary_slice = self.in_primary_slice

    shape_dtypes, tree_defs = jax.tree.flatten(self._abstract_state)
    original_single_slice_shardings = jax.tree.map(
        lambda arr: _get_single_slice_sharding(
            mesh=arr.sharding.mesh,
            pspec=arr.sharding.spec,
            replica_id=self._slice_id,
            replica_axis_index=self._replica_axis_index,
        ),
        self._abstract_state,
    )
    original_single_slice_shardings_tuple = tuple(
        jax.tree.flatten(original_single_slice_shardings)[0]
    )

    # Debug logging for sharding information.
    if logging.vlog_is_on(1):
      logging.vlog(
          1,
          'Debugging global restore_args based on abstract state. This uses the'
          ' user-provided mesh, and is not actually used for restoration.',
      )
      local_checkpoint_data_debugging.print_devices_indices_debug_info(
          checkpoint_utils.construct_restore_args(self._abstract_state)
      )
      logging.vlog(
          1,
          'Debugging single-slice restore_args based on abstract state. This'
          ' uses the user-provided mesh, and is not actually used for'
          ' restoration.',
      )
      local_checkpoint_data_debugging.print_devices_indices_debug_info(
          checkpoint_utils.construct_restore_args(
              self._abstract_state, original_single_slice_shardings
          )
      )

    if is_restoring_slice:
      logging.vlog(
          1, 'emergency.CheckpointManager: restoring from local checkpoint.'
      )
      restore_directory = self._options.step_name_format.find_step(
          epath.Path(directory or self._local_directory), step
      ).path
      step_stats.directory = str(restore_directory)
      (
          previous_distributed_to_device_ids,
          previous_device_ids,
      ) = ProcessMetadataCheckpointHandler().restore(
          restore_directory / _PROCESS_METADATA_NAME,
          process_metadata_checkpoint_handler.ProcessMetadataRestoreArgs(),
      )
      restore_mesh = mesh_consistency.consistent_restore_mesh_from_metadata(
          self._global_mesh,
          multihost.distributed_to_device_ids(),
          previous_distributed_to_device_ids=previous_distributed_to_device_ids,
          previous_device_ids=previous_device_ids,
      )
      restoring_processes = multihost.unique_processes_from_devices(
          multislice.slice_devices(
              restore_mesh,
              replica_id=self._slice_id,
              replica_axis_index=self._replica_axis_index,
          )
      )
      multiprocessing_options = checkpoint_manager.MultiprocessingOptions(
          primary_host=None,
          active_processes=restoring_processes,
          barrier_sync_key_prefix='local_restoring_slice',
      )
      local_state_handler = _local_checkpoint_handler(multiprocessing_options)

      restore_single_slice_shardings = jax.tree.map(
          lambda arr: _get_single_slice_sharding(
              mesh=restore_mesh,
              pspec=arr.sharding.spec,
              replica_id=self._slice_id,
              replica_axis_index=self._replica_axis_index,
          ),
          self._abstract_state,
      )
      single_slice_restore_args = checkpoint_utils.construct_restore_args(
          self._abstract_state, restore_single_slice_shardings
      )

      # Directly use CheckpointHandler to restore. This is undesirable, but
      # allows us to avoid barrier issues that occur when calling
      # LocalCheckpointManager a different number of times on the non-primary
      # slices, which leads to
      # _module_unique_count getting out of sync.
      logging.vlog(
          1,
          'Restoring from %s',
          restore_directory / _UNNAMED_ITEM_NAME,
      )
      if logging.vlog_is_on(1):
        asyncio.run(
            local_checkpoint_data_debugging.print_chunk_debug_info(
                restore_directory / _UNNAMED_ITEM_NAME,
                single_slice_restore_args,
            )
        )
        local_checkpoint_data_debugging.print_devices_indices_debug_info(
            single_slice_restore_args
        )

      step_stats.checkpointer_start_time = time.time()
      args = args_lib.PyTreeRestore(
          item=self._abstract_state,
          restore_args=checkpoint_utils.construct_restore_args(
              self._abstract_state
          ),
      )
      single_slice_pytree = local_state_handler.restore(
          restore_directory / _UNNAMED_ITEM_NAME,
          args=dataclasses.replace(
              args, restore_args=single_slice_restore_args
          ),
      )
      step_stats.checkpointer_duration_secs = (
          time.time() - step_stats.checkpointer_start_time
      )
      in_tree = tuple(jax.tree.flatten(single_slice_pytree)[0])
      if not np.array_equal(
          restore_mesh.device_ids, self._global_mesh.device_ids
      ):
        # User-provided mesh is usually optimized for performance.
        # But we permuted the original mesh so that we can read each locally
        # available shard correctly. This may cause performance issues.

        # Thus, we re-shard the array to follow the original mesh and layout.
        in_tree = mesh_consistency.consistent_restore_mesh_to_global_mesh(
            in_tree, original_single_slice_shardings_tuple
        )
    else:
      logging.vlog(
          1,
          'emergency.CheckpointManager: secondary slice, create zeros and'
          ' wait for broacast.',
      )

      @functools.partial(
          jax.jit,
          static_argnums=0,
          out_shardings=original_single_slice_shardings_tuple,
      )
      def create_zeros(shape_dtype_tup):
        return jax.tree.map(
            lambda sd: jnp.zeros(sd.shape, dtype=sd.dtype), shape_dtype_tup
        )

      zeros_pytree = create_zeros(tuple(shape_dtypes))
      in_tree = tuple(zeros_pytree)

    multihost.sync_global_processes('local_restore_pre_broadcast')

    start_broadcast = time.time()
    shared_states, _ = multislice.broadcast_one_replica_to_all(
        in_tree,
        self._global_mesh,
        replica_axis_index=self._replica_axis_index,
        is_source=is_restoring_slice,
    )
    broadcast_elapsed_s = time.time() - start_broadcast
    jax.monitoring.record_event_duration_secs(
        '/orbax/emergency/checkpoint/read/broadcast_duration_secs',
        broadcast_elapsed_s,
    )
    step_stats.broadcast_start_time = start_broadcast
    step_stats.broadcast_duration_secs = broadcast_elapsed_s
    step_stats.checkpoint_manager_duration_secs = (
        time.time() - step_stats.checkpoint_manager_start_time
    )
    self._logger.log_entry(dataclasses.asdict(step_stats))

    logging.info('Finished broadcasting in %.2f', broadcast_elapsed_s)

    return jax.tree.unflatten(tree_defs, shared_states)

  def _restore_from_persistent(
      self,
      step: int,
      directory: Optional[epath.PathLike] = None,
  ) -> Any:
    logging.info(
        'emergency.CheckpointManager: restoring step=%s from persistent'
        ' checkpoint in directory=%s',
        step,
        directory or self._persistent_directory,
    )
    args = args_lib.PyTreeRestore(
        item=self._abstract_state,
        restore_args=checkpoint_utils.construct_restore_args(
            self._abstract_state
        ),
    )

    # Create a temporarily read-only PersistentCheckpointManager that will
    # synchronize the restoration with global processes.
    persistent_options = checkpoint_manager.CheckpointManagerOptions(
        step_name_format=self._options.step_name_format,
        create=False,
        cleanup_tmp_directories=False,
        read_only=True,
        enable_async_checkpointing=False,
        multiprocessing_options=checkpoint_manager.MultiprocessingOptions(
            barrier_sync_key_prefix='persistent_global',
        ),
    )
    with checkpoint_manager.CheckpointManager(
        self._persistent_directory,
        options=persistent_options,
        metadata=self._metadata,
        item_handlers=PyTreeCheckpointHandler(
            use_ocdbt=True,
            use_zarr3=True,
        ),
    ) as pcm:
      try:
        return pcm.restore(step, args=args, directory=directory)
      except FileNotFoundError as e:
        raise FileNotFoundError(
            'No steps found in either local or persistent storage when'
            f' requesting restoration of step {step}.'
        ) from e

  def restore(
      self,
      step: Optional[int],
      args: args_lib.CheckpointArgs | None = None,
  ) -> Any:
    del args
    if step is None:
      step = self.latest_step()
      if step is None:
        raise FileNotFoundError(
            'No steps found in persistent or local storage.'
        )
    logging.info('Restoring at step %d.', step)
    restoring_slice_id = self._find_slice_with_complete_local_checkpoint(step)
    if restoring_slice_id > -1:
      # restore from LCM
      return self._restore_from_local(
          step=step,
          restoring_slice_id=restoring_slice_id,
      )

    return self._restore_from_persistent(step=step)

  def item_metadata(self, step: int) -> Any:
    raise NotImplementedError(
        'Item metadata not yet implemented for emergency.CheckpointManager.'
    )

  def metadata(self, step: int | None = None) -> RootMetadata | StepMetadata:
    """Returns CheckpointManager level metadata if present, empty otherwise."""
    raise NotImplementedError(
        'Metadata not yet implemented for emergency.CheckpointManager.'
    )

  def metrics(self, step: int) -> Optional[PyTree]:
    """Returns metrics for step, if present."""
    raise NotImplementedError(
        'Metrics not yet implemented for emergency.CheckpointManager.'
    )

  def wait_until_finished(self):
    """Blocks until any incomplete save operations are completed.

    Note that this method will typically be a no-op if all checkpointers are
    synchronous, since old checkpoints are already cleaned up immediately after
    completing `save`, and there is no background thread to wait for.

    If some checkpointers are of type AsyncCheckpointer, however, this method
    will wait until each of these checkpointers is finished.
    """
    logging.info('Waiting for checkpoint to complete.')
    if self.in_primary_slice:
      self._persistent_checkpoint_manager.wait_until_finished()
    else:
      self._local_checkpoint_manager.wait_until_finished()

  def check_for_errors(self):
    """Checks for any outstanding errors in completed asynchronous save operations.

    Delegates to underlying Checkpointer.
    """
    if self.in_primary_slice:
      self._persistent_checkpoint_manager.check_for_errors()
    else:
      self._local_checkpoint_manager.check_for_errors()

  def close(self):
    """Waits for outstanding operations to finish and closes Checkpointers."""
    logging.info('Closing CheckpointManager.')
    self.wait_until_finished()
    if self.in_primary_slice:
      self._persistent_checkpoint_manager.close()
    else:
      self._local_checkpoint_manager.close()

  def __contextmanager__(
      self,
  ) -> Iterable[Self]:
    try:
      yield self
    finally:
      self.close()


class CheckpointManager(
    abstract_checkpoint_manager.AbstractCheckpointManager, epy.ContextManager
):
  """Provides both checkpoint management and emergency checkpointings.

  This class composes a local and a persistent checkpoint managers. The local
  manager saves checkpoints frequently to a fast local storage (like RAMFS).
  When a complete checkpoint exists at least one slice, restoration is possible,
  and the slice broadcasts the checkpoint to others. Additionally, the
  persistent manager checkpoints less frequently to a remote file system (e.g.,
  GCS),
  providing a fail-safe if local checkpoints become unavailable due to issues
  like hardware failure or preemption.

  Usage::

    options = CheckpointManagerOptions(
        local=LocalCheckpointOptions(save_interval_steps=2, max_to_keep=2),
        persistent=PersistentCheckpointOptions(
            save_interval_steps=5, max_to_keep=3
        ),
        enable_async_checkpointing=use_async,
    )
    return CheckpointManager(
        local_directory=local_directory,
        persistent_directory=persistent_directory,
        global_mesh=global_mesh,
        abstract_state=abstract_state,
        options=options,
    )
  """

  # TODO: b/330585086 - Support arbitrary items beyond state. We will have
  # to evaluate whether arbitrary items can be a good fit for local
  # checkpointing, given restore+broadcast requirements.
  def __init__(
      self,
      local_directory: epath.PathLike,
      persistent_directory: epath.PathLike,
      global_mesh: jax.sharding.Mesh,
      abstract_state: PyTree,  # a single PyTree describing the state structure
      *,
      options: Optional[CheckpointManagerOptions] = None,
      metadata: Optional[dict[str, Any]] = None,
      logger: Optional[abstract_logger.AbstractLogger] = None,
  ):
    options = options or CheckpointManagerOptions()
    self._global_mesh = global_mesh
    self._abstract_state = abstract_state
    self._slice_count = multislice.slice_count(
        global_mesh, replica_axis_index=options.replica_axis_index
    )
    checkpoint_manager._create_root_directory(
        persistent_directory,
        multiprocessing_options=checkpoint_manager.MultiprocessingOptions(),
    )
    if self._slice_count <= 0:
      raise ValueError(
          'Slice count must be positive, but got'
          f' {self._slice_count} for mesh {global_mesh}.'
      )
    elif self._slice_count == 1:
      del local_directory
      self._checkpoint_manager = checkpoint_manager.CheckpointManager(
          persistent_directory,
          options=_get_persistent_options(
              options, checkpoint_manager.MultiprocessingOptions()
          ),
          metadata=metadata,
          logger=logger,
      )
    else:
      self._checkpoint_manager = _MultisliceCheckpointManager(
          local_directory=local_directory,
          persistent_directory=persistent_directory,
          global_mesh=global_mesh,
          abstract_state=abstract_state,
          options=options,
          metadata=metadata,
          logger=logger,
      )

  @property
  def directory(self) -> epath.Path:
    return self._checkpoint_manager.directory

  @property
  def in_primary_slice(self) -> bool:
    if self._slice_count == 1:
      return True
    else:
      assert isinstance(self._checkpoint_manager, _MultisliceCheckpointManager)
      return self._checkpoint_manager.in_primary_slice

  @property
  def global_mesh(self) -> jax.sharding.Mesh:
    return self._global_mesh

  def all_steps(self, read: bool = False) -> Sequence[int]:
    return self._checkpoint_manager.all_steps(read=read)

  def latest_step(self) -> Optional[int]:
    return self._checkpoint_manager.latest_step()

  def best_step(self) -> Optional[int]:
    raise NotImplementedError(
        'Metrics tracking not yet implemented for emergency.CheckpointManager.'
    )

  def reload(self):
    return self._checkpoint_manager.reload()

  def reached_preemption(self, step: int) -> bool:
    return self._checkpoint_manager.reached_preemption(step)

  def should_save(self, step: int) -> bool:
    return self._checkpoint_manager.should_save(step)

  def delete(self, step: int):
    """Deletes a step checkpoint."""
    raise NotImplementedError(
        'Delete not yet implemented for emergency.CheckpointManager.'
    )

  def save(
      self,
      step: int,
      args: args_lib.CheckpointArgs,
      *,
      force: bool = False,
  ) -> bool:
    return self._checkpoint_manager.save(step, args=args, force=force)

  def restore(
      self,
      step: int | None,
      args: args_lib.CheckpointArgs | None = None,
  ) -> Any:
    del args
    args = args_lib.PyTreeRestore(
        item=self._abstract_state,
        restore_args=checkpoint_utils.construct_restore_args(
            self._abstract_state
        ),
    )
    return self._checkpoint_manager.restore(step, args=args)

  def item_metadata(self, step: int) -> Any:
    raise NotImplementedError(
        'Item metadata not implemented for emergency.CheckpointManager.'
    )

  def metadata(self, step: int | None = None) -> RootMetadata | StepMetadata:
    """Returns CheckpointManager level metadata if present, empty otherwise."""
    raise NotImplementedError(
        'Metadata not yet implemented for emergency.CheckpointManager.'
    )

  def metrics(self, step: int) -> Optional[PyTree]:
    """Returns metrics for step, if present."""
    raise NotImplementedError(
        'Metrics not yet implemented for emergency.CheckpointManager.'
    )

  def wait_until_finished(self):
    return self._checkpoint_manager.wait_until_finished()

  def check_for_errors(self):
    return self._checkpoint_manager.check_for_errors()

  def close(self):
    return self._checkpoint_manager.close()

  def __contextmanager__(
      self,
  ) -> Iterable[Self]:
    try:
      yield self
    finally:
      self.close()
