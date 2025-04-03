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

"""Simple script to test multi-slice emergency restore."""
# pylint: disable=protected-access

import os
import re
import sys
import time
from typing import Any

from absl import app
from absl import flags
from absl import logging
from etils import epath
import jax
import orbax.checkpoint as ocp
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint.experimental.emergency import checkpoint_manager
from orbax.checkpoint.experimental.emergency import integration_test_utils


PyTree = Any


_TASK_HANDLE_RE = re.compile(r'(?:logs\.)?(\d+)\.(.*)\.([^.]+)\.\d+')


_PERSISTENT_DIRECTORY = flags.DEFINE_string(
    'persistent_directory',
    None,
    'Directory on CNS.',
)

_LOCAL_DIRECTORY = flags.DEFINE_string(
    'local_directory',
    None,
    'Directory for local checkpoints. If not specified, defaults to a'
    ' subdirectory of the persistent directory. Each process will create its'
    ' own subdirectory of the given directory.',
)

_USE_ASYNC = flags.DEFINE_bool(
    'use_async',
    False,
    'Enable async checkpointing.',
)

_REPLICA_AXIS_INDEX = flags.DEFINE_integer(
    'replica_axis_index',
    0,
    'Replica axis index.',
)


_TENSOR_PARALLELISM = flags.DEFINE_integer(
    'tensor_parallelism',
    None,
    'Tensor parallelism.',
)

_FSDP_PARALLELISM = flags.DEFINE_integer(
    'fsdp_parallelism',
    None,
    'FSDP parallelism.',
)

_NUM_SLICES = flags.DEFINE_integer(
    'num_slices',
    None,
    'num_slices',
)


CheckpointManager = checkpoint_manager.CheckpointManager
CheckpointManagerOptions = checkpoint_manager.CheckpointManagerOptions
LocalCheckpointOptions = checkpoint_manager.LocalCheckpointOptions
PersistentCheckpointOptions = checkpoint_manager.PersistentCheckpointOptions


def train_step(state, inc=1):
  time.sleep(2)

  def increment(tree):
    return jax.tree.map(lambda x: x + inc, tree)

  # Avoid out shardings from changing
  shardings = jax.tree.map(lambda x: x.sharding, state)
  return jax.jit(increment, in_shardings=(shardings,), out_shardings=shardings)(
      state
  )


class TestClass:
  """Class to run equality assertions."""

  def assertSameElements(self, a, b):  # pylint: disable=invalid-name
    assert len(a) == len(b)
    for x, y in zip(a, b):
      assert x == y

  def assertEqual(self, a, b):  # pylint: disable=invalid-name
    assert a == b

  def assertIsInstance(self, a, b):  # pylint: disable=invalid-name
    assert isinstance(a, b)


def _log_and_check_restored(
    test_class: TestClass, restored: PyTree, expected: PyTree
):
  logging.info('Restored state:')
  _log_array_tree(restored)
  logging.info('Expected state:')
  _log_array_tree(expected)
  test_utils.assert_tree_equal(test_class, restored, expected)


def save_and_restore(
    persistent_directory,
    local_directory,
    global_mesh,
    state,
    replica_axis_index,
):
  """Tests save and restore. Saves a final step 10."""
  abstract_state = jax.tree.map(
      ocp.utils.to_shape_dtype_struct, state
  )
  restore_args = ocp.checkpoint_utils.construct_restore_args(abstract_state)
  test_class = TestClass()
  for step in range(6):
    logging.info('Training step %d', step)
    # Create at every step, since we are messing with directories under the
    # hood.
    manager = integration_test_utils.create_checkpoint_manager(
        global_mesh,
        abstract_state,
        local_directory,
        persistent_directory,
        use_async=True,
        replica_axis_index=replica_axis_index,
    )

    manager.save(step, args=ocp.args.PyTreeSave(state))
    manager.wait_until_finished()
    # Without this, we don't wait for persistent + local save to complete
    # everywhere.
    jax.experimental.multihost_utils.sync_global_devices('save')

    if step % 5 == 0:
      assert (persistent_directory / str(step)).exists()

      logging.info('[test] Restoring from local checkpoint.')
      # First restore from local checkpoint.
      restored = manager.restore(
          step, args=ocp.args.PyTreeRestore(restore_args=restore_args)
      )
      _log_and_check_restored(test_class, restored, state)
      restored = jax.tree.map(lambda x: None, restored)  # free memory
      del restored

      logging.info('[test] Restoring from persistent checkpoint.')
      # Fall back on persistent.
      (local_directory / str(step)).rmtree(missing_ok=True)
      manager.reload()
      restored = manager.restore(
          step, args=ocp.args.PyTreeRestore(restore_args=restore_args)
      )
      _log_and_check_restored(test_class, restored, state)
      restored = jax.tree.map(lambda x: None, restored)  # free memory
      del restored
    elif step % 2 == 0:
      if not multislice.in_slice(
          multihost.process_index(),
          global_mesh,
          replica_id=checkpoint_manager._PRIMARY_REPLICA_ID,
          replica_axis_index=replica_axis_index,
      ):
        assert (local_directory / str(step)).exists()

      # Delete one of the local checkpoints on a particular slice.
      if len(global_mesh.devices) > 2:
        logging.info('[test] Deleting checkpoint in slice.')
        if multislice.in_slice(
            multihost.process_index(),
            global_mesh,
            replica_id=checkpoint_manager._SECONDARY_REPLICA_ID,
            replica_axis_index=replica_axis_index,
        ):
          (local_directory / str(step)).rmtree(missing_ok=False)
        logging.info(
            '[test] Restoring from local checkpoint with one slice missing.'
        )
        manager.reload()
        restored = manager.restore(
            step, args=ocp.args.PyTreeRestore(restore_args=restore_args)
        )
        _log_and_check_restored(test_class, restored, state)
        restored = jax.tree.map(lambda x: None, restored)  # free memory
        del restored

    logging.info('Beginning train step %d', step)
    state = train_step(state)
    logging.info('Finished train step %d', step)

  logging.info('Finished training')

  # Do a final save.
  final_state = integration_test_utils.create_train_state(global_mesh)
  step = 10
  final_state = train_step(final_state, inc=step)
  manager = integration_test_utils.create_checkpoint_manager(
      global_mesh,
      abstract_state,
      local_directory,
      persistent_directory,
      use_async=True,
      replica_axis_index=replica_axis_index,
  )
  manager.save(step, args=ocp.args.PyTreeSave(final_state), force=True)
  manager.wait_until_finished()
  restored = manager.restore(
      step, args=ocp.args.PyTreeRestore(restore_args=restore_args)
  )
  _log_and_check_restored(test_class, restored, final_state)
  manager.close()


def _log_array_tree(tree):
  def _log(key, arr):
    logging.info('%s: %s', key, [s.data for s in arr.addressable_shards])

  jax.tree_util.tree_map_with_path(_log, tree)


def _replicate_tree(mesh, tree):
  replicate_fn = jax.jit(
      lambda x: x,
      out_shardings=jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec()
      ),
  )
  return jax.tree.map(replicate_fn, tree)


def simple_restore(
    persistent_directory,
    local_directory,
    global_mesh,
    state,
    replica_axis_index,
):
  """Tests a single restore of latest step."""
  logging.info(
      'Using global_mesh: %s',
      global_mesh.devices,
  )

  abstract_state = jax.tree.map(ocp.utils.to_shape_dtype_struct, state)
  restore_args = ocp.checkpoint_utils.construct_restore_args(abstract_state)
  test_class = TestClass()
  expected_state = integration_test_utils.create_train_state(global_mesh)
  manager = integration_test_utils.create_checkpoint_manager(
      global_mesh,
      abstract_state,
      local_directory,
      persistent_directory,
      use_async=True,
      replica_axis_index=replica_axis_index,
  )
  step = manager.latest_step()
  if step is None:
    raise ValueError('No steps found.')
  expected_state = train_step(expected_state, inc=step)

  restored = manager.restore(
      step, args=ocp.args.PyTreeRestore(restore_args=restore_args)
  )

  _log_and_check_restored(test_class, restored, expected_state)

  replicated_expected_state = _replicate_tree(global_mesh, expected_state)
  replicated_restored = _replicate_tree(global_mesh, restored)
  _log_and_check_restored(
      test_class, replicated_restored, replicated_expected_state
  )

  # Save again.
  state_to_save = train_step(expected_state, inc=1)
  manager.save(step + 1, args=ocp.args.PyTreeSave(state_to_save), force=True)

  manager.close()


def create_directories(persistent_directory, local_directory):
  persistent_directory.mkdir(parents=True, exist_ok=True)
  local_directory.mkdir(parents=True, exist_ok=False)
  jax.experimental.multihost_utils.sync_global_devices('create directories')


def main(_):
  host_count = jax.process_count()
  local_device_count = jax.local_device_count()
  logging.info(
      'Device count: %d, host count: %d, local device count: %d',
      jax.device_count(),
      host_count,
      local_device_count,
  )
  flags.FLAGS.experimental_orbax_use_distributed_process_id = True
  if not multihost.is_runtime_to_distributed_ids_initialized():
    multihost.initialize_runtime_to_distributed_ids()

  if not multihost.is_distributed_to_device_ids_initialized():
    multihost.initialize_distributed_to_device_ids()

  replica_axis_index = _REPLICA_AXIS_INDEX.value
  num_slices = _NUM_SLICES.value
  if not num_slices:
    raise ValueError(f'Invalid `num_slices` value: {num_slices}')
  global_mesh = integration_test_utils.create_global_mesh(
      num_slices,
      replica_axis_index=replica_axis_index,
      tensor_parallelism=_TENSOR_PARALLELISM.value,
      fsdp_parallelism=_FSDP_PARALLELISM.value,
  )
  logging.info('global_mesh shape: %s', global_mesh.devices.shape)
  state = integration_test_utils.create_train_state(global_mesh)

  persistent_directory = epath.Path(_PERSISTENT_DIRECTORY.value)
  if _LOCAL_DIRECTORY.value is None:
    local_directory = (
        persistent_directory / 'local' / f'process_{multihost.process_index()}/'
    )
  else:
    local_directory = epath.Path(_LOCAL_DIRECTORY.value)

  persistent_directory_exists = persistent_directory.exists()
  # Ensure all processes check for existence before proceeding.
  jax.experimental.multihost_utils.sync_global_devices('directory exists')

  if persistent_directory_exists:
    logging.info('Simple restoring from %s.', persistent_directory)
    simple_restore(
        persistent_directory,
        local_directory,
        global_mesh,
        state,
        replica_axis_index,
    )
  else:
    logging.info('Running save and restore.')
    create_directories(persistent_directory, local_directory)
    save_and_restore(
        persistent_directory,
        local_directory,
        global_mesh,
        state,
        replica_axis_index,
    )
  logging.info('Completed test.')


if __name__ == '__main__':
  app.run(main)
