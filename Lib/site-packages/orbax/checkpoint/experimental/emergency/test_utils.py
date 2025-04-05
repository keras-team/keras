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

"""Test base classes for emergency.CheckpointManager."""

# pylint: disable=protected-access

import functools
import json
from typing import Any, Optional
from unittest import mock

from absl import logging
from absl.testing import flagsaver
from absl.testing import parameterized
from etils import epath
import jax
import numpy as np
import optax
from orbax.checkpoint import args as args_lib
from orbax.checkpoint import test_utils
from orbax.checkpoint import utils
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint.experimental.emergency import checkpoint_manager
from orbax.checkpoint.experimental.emergency import mesh_consistency
from orbax.checkpoint.experimental.emergency import process_metadata_checkpoint_handler

PyTree = Any
PyTreeCheckpointHandler = pytree_checkpoint_handler.PyTreeCheckpointHandler
PyTreeSaveArgs = pytree_checkpoint_handler.PyTreeSaveArgs
PyTreeRestoreArgs = pytree_checkpoint_handler.PyTreeRestoreArgs

CheckpointManager = checkpoint_manager.CheckpointManager
LocalCheckpointManager = checkpoint_manager._LocalCheckpointManager  # pylint: disable=protected-access
CheckpointManagerOptions = checkpoint_manager.CheckpointManagerOptions
LocalCheckpointOptions = checkpoint_manager.LocalCheckpointOptions
PersistentCheckpointOptions = checkpoint_manager.PersistentCheckpointOptions
barrier_compatible_test = test_utils.barrier_compatible_test
assert_tree_equal = test_utils.assert_tree_equal
get_fake_global_mesh_for_slices = test_utils.get_fake_global_mesh_for_slices


def swap_slices_in_mesh(
    mesh: jax.sharding.Mesh, *, replica_axis_index: int = 0
) -> jax.sharding.Mesh:
  """Reverses the ordering of devices such that slices swap IDs."""
  devices = []
  for slice_id in range(
      multislice.slice_count(mesh, replica_axis_index=replica_axis_index)
  ):
    devices.append(
        multislice.slice_devices(
            mesh, replica_id=slice_id, replica_axis_index=replica_axis_index
        )
    )
  devices.reverse()
  devices = np.stack(devices, axis=replica_axis_index)
  return jax.sharding.Mesh(devices, mesh.axis_names)


def _replace_abstract_array_sharding_with_mesh(
    sds: jax.ShapeDtypeStruct, mesh: jax.sharding.Mesh
) -> jax.ShapeDtypeStruct:
  assert isinstance(sds.sharding, jax.sharding.NamedSharding)
  sds.sharding = jax.sharding.NamedSharding(mesh, sds.sharding.spec)
  return sds


class LocalCheckpointManagerTestBase:
  """Test case."""

  @barrier_compatible_test
  class Test(parameterized.TestCase):
    """Test case."""

    def make_global_mesh(
        self, replica_axis_index: int = 0
    ) -> jax.sharding.Mesh:
      """Creates a global mesh for testing purposes.

      This method sets up a global mesh with 8 devices across 4 processes,
      where each process has 2 local devices. The mesh is constructed based
      on provided slice processes and a specified replica axis.

      Args:
        replica_axis_index: The index of the replica axis in the resulting mesh.
          Must be either 0 or 1. Defaults to 0.

      Returns:
        A `jax.sharding.Mesh` object representing the fake global mesh.

      Raises:
        ValueError: If `replica_axis_index` is not 0 or 1.
      """
      if replica_axis_index not in [0, 1]:
        raise ValueError(
            'replica_axis_index must be 0 or 1 for this test. Got:'
            f' {replica_axis_index}.'
        )
      self.assertEqual(jax.device_count(), 8)
      self.assertEqual(jax.process_count(), 4)
      self.assertEqual(jax.local_device_count(), 2)

      # setup global mesh info for 2-slice tests
      slice_processes = [{0, 1}, {2, 3}]
      mesh = test_utils.get_fake_global_mesh_for_slices(
          slice_processes, replica_axis_index
      )
      if replica_axis_index == 0:
        assert mesh.devices.shape == (2, 4), mesh.devices.shape
      if replica_axis_index == 1:
        assert mesh.devices.shape == (4, 2), mesh.devices.shape
      return mesh

    def setUp(self):
      super().setUp()
      self.enter_context(
          flagsaver.flagsaver(
              experimental_orbax_use_distributed_process_id=True
          )
      )
      if not multihost.is_runtime_to_distributed_ids_initialized():
        multihost.initialize_runtime_to_distributed_ids()
        multihost.initialize_distributed_to_device_ids()

      self.global_mesh = self.make_global_mesh()

      self._fn = lambda ty: issubclass(ty, jax.Array)

      pytree, mesh_tree, axes_tree = test_utils.setup_sharded_pytree()
      doubled_pytree = test_utils.apply_function(pytree, lambda x: x * 2)

      self.pytree = pytree
      self.doubled_pytree = doubled_pytree
      self.mesh_tree = mesh_tree
      self.axes_tree = axes_tree

      # make sure each process is working on different directories
      self.local_directory = epath.Path(
          self.create_tempdir(
              name=f'checkpointing_test_pid{multihost.process_index()}'
          ).full_path
      )
      logging.info(
          'self.directory=%s',
          self.local_directory,
      )
      test_utils.set_tensorstore_driver_for_test()

      test_utils.sync_global_processes(
          'LocalCheckpointManagerTest:setup_complete'
      )

    def tearDown(self):
      super().tearDown()
      test_utils.sync_global_processes(
          'LocalCheckpointManagerTest:teardown_complete'
      )

    def test_local_save_restore(self):
      """Test case."""
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(
              save_interval_steps=1,
              max_to_keep=1,
              debug_use_full_global_mesh=True,
          ),
          persistent=PersistentCheckpointOptions(
              save_interval_steps=1, max_to_keep=1
          ),
      )
      manager = LocalCheckpointManager(
          directory=self.local_directory,
          global_mesh=self.global_mesh,
          options=options,
      )

      manager.save(
          0,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )

      manager.wait_until_finished()

      restored = manager.restore(
          0,
          args=args_lib.Composite(
              state=PyTreeRestoreArgs(),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataRestoreArgs(),
          ),
      )
      test_utils.assert_tree_equal(self, self.pytree, restored.state)

    def test_local_auto_cleanup(self):
      """Test case."""
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(
              save_interval_steps=1,
              max_to_keep=1,
              debug_use_full_global_mesh=True,
          ),
          persistent=PersistentCheckpointOptions(
              save_interval_steps=1, max_to_keep=1
          ),
      )
      manager = LocalCheckpointManager(
          directory=self.local_directory,
          global_mesh=self.global_mesh,
          options=options,
      )

      manager.save(
          0,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      manager.save(
          1,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.doubled_pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      # step 0 should have been removed
      with self.assertRaisesRegex(ValueError, 'No step path found'):
        manager.restore(
            0,
            args=args_lib.Composite(
                state=PyTreeRestoreArgs(),
                process_metadata=process_metadata_checkpoint_handler.ProcessMetadataRestoreArgs(),
            ),
        )

      restored = manager.restore(
          1,
          args=args_lib.Composite(
              state=PyTreeRestoreArgs(),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataRestoreArgs,
          ),
      )

      test_utils.assert_tree_equal(self, self.doubled_pytree, restored.state)

    def test_default_max_to_keep_is_1(self):
      """Test case."""
      # the default max_to_keep should be 1 because we want to ensure that by
      # default, up to 2 steps are in local storage at the same time.
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(
              save_interval_steps=1,
              # max_to_keep unset to use default value
              debug_use_full_global_mesh=True,
          ),
      )
      manager = LocalCheckpointManager(
          directory=self.local_directory,
          global_mesh=self.global_mesh,
          options=options,
      )
      manager.save(
          0,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.save(
          1,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.doubled_pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      self.assertEqual(manager.all_steps(), [1])
      self.assertFalse((self.local_directory / '0').exists())
      self.assertTrue((self.local_directory / '1' / 'state').exists())
      self.assertTrue(
          (self.local_directory / '1' / 'process_metadata').exists()
      )

    @parameterized.parameters(
        (False),
        (True,),
    )
    def test_local_startup_cleanup(self, enable_async_checkpointing: bool):
      """Test case."""
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(
              save_interval_steps=1,
              max_to_keep=3,
              debug_use_full_global_mesh=True,
          ),
          enable_async_checkpointing=enable_async_checkpointing,
      )
      manager = LocalCheckpointManager(
          directory=self.local_directory,
          global_mesh=self.global_mesh,
          options=options,
      )

      manager.save(
          0,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.save(
          1,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.save(
          2,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.doubled_pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      # 3 steps should exist
      self.assertEqual(manager.all_steps(), [0, 1, 2])
      for i in range(3):
        self.assertTrue((self.local_directory / str(i)).exists())
        self.assertTrue(
            mesh_consistency.process_metadata_folder(
                self.local_directory / str(i) / 'process_metadata'
            ).exists()
        )

      # create a new manager with max_to_keep=1
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(
              save_interval_steps=1,
              max_to_keep=1,
              debug_use_full_global_mesh=True,
          ),
          enable_async_checkpointing=enable_async_checkpointing,
      )
      manager2 = LocalCheckpointManager(
          directory=self.local_directory,
          global_mesh=self.global_mesh,
          options=options,
      )

      # manager2 should have gc'd step 0 and 1
      self.assertEqual(manager2.all_steps(), [2])
      self.assertFalse((self.local_directory / '0').exists())
      self.assertFalse((self.local_directory / '1').exists())
      self.assertTrue((self.local_directory / '2' / 'state').exists())
      self.assertTrue(
          (self.local_directory / '2' / 'process_metadata').exists()
      )

      # new step save should work
      manager2.save(
          3,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.doubled_pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager2.wait_until_finished()
      self.assertEqual(manager2.all_steps(), [3])
      self.assertFalse((self.local_directory / '2').exists())
      self.assertTrue((self.local_directory / '3' / 'state').exists())
      self.assertTrue(
          (self.local_directory / '3' / 'process_metadata').exists()
      )

    @parameterized.parameters(
        (False),
        (True,),
    )
    def test_local_garbage_collection(self, enable_async_checkpointing: bool):
      """Test case."""
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(
              save_interval_steps=1,
              max_to_keep=2,
              debug_use_full_global_mesh=True,
          ),
          persistent=PersistentCheckpointOptions(
              save_interval_steps=1, max_to_keep=1
          ),
          enable_async_checkpointing=enable_async_checkpointing,
      )
      manager = LocalCheckpointManager(
          directory=self.local_directory,
          global_mesh=self.global_mesh,
          options=options,
      )

      manager.save(
          0,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.save(
          1,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.save(
          2,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.doubled_pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()
      self.assertFalse((self.local_directory / '0').exists())
      self.assertTrue((self.local_directory / '1' / 'state').exists())
      self.assertTrue(
          (self.local_directory / '1' / 'process_metadata').exists()
      )

      self.assertTrue((self.local_directory / '2' / 'state').exists())
      self.assertTrue(
          (self.local_directory / '2' / 'process_metadata').exists()
      )

      if multihost.process_index() == 0:
        test_utils.empty_directory(self.local_directory)

      manager.reload()

      if multihost.process_index() == 0:
        self.assertFalse((self.local_directory / '1').exists())
        self.assertFalse((self.local_directory / '2').exists())
      else:
        self.assertTrue((self.local_directory / '1' / 'state').exists())
        self.assertTrue(
            (self.local_directory / '1' / 'process_metadata').exists()
        )
        self.assertTrue((self.local_directory / '2' / 'state').exists())
        self.assertTrue(
            (self.local_directory / '2' / 'process_metadata').exists()
        )

      # subsequent save should be successful and gc should work
      manager.save(
          3,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      if multihost.process_index() == 0:
        self.assertFalse((self.local_directory / '2').exists())
        self.assertTrue((self.local_directory / '3' / 'state').exists())
        self.assertTrue(
            (self.local_directory / '3' / 'process_metadata').exists()
        )
      else:
        self.assertFalse((self.local_directory / '1').exists())
        self.assertTrue((self.local_directory / '2' / 'state').exists())
        self.assertTrue(
            (self.local_directory / '2' / 'process_metadata').exists()
        )
        self.assertTrue((self.local_directory / '3' / 'state').exists())
        self.assertTrue(
            (self.local_directory / '3' / 'process_metadata').exists()
        )

      manager.save(
          4,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.doubled_pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.save(
          5,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      self.assertFalse((self.local_directory / '2').exists())
      self.assertFalse((self.local_directory / '3').exists())
      self.assertTrue((self.local_directory / '4' / 'state').exists())
      self.assertTrue(
          (self.local_directory / '4' / 'process_metadata').exists()
      )
      self.assertTrue((self.local_directory / '5' / 'state').exists())
      self.assertTrue(
          (self.local_directory / '5' / 'process_metadata').exists()
      )

    @parameterized.parameters(
        ([[], [], [], []], {0: {}, 1: {}}),
        ([[1], [1], [0], [0]], {0: {1}, 1: {0}}),
        ([[], [], [0], [0]], {0: {}, 1: {0}}),
        ([[], [], [0], []], {0: {}, 1: {}}),
        ([[], [], [0], [1]], {0: {}, 1: {}}),
        ([[], [0], [], [0]], {0: {}, 1: {}}),
        ([[0], [1], [0], [1]], {0: {}, 1: {0}}),
        ([[1, 2], [1, 2], [4], [4, 5]], {0: {1, 2}, 1: {4}}),
        ([[-1, 0], [-1, 0], [1, 2], [1, 2]], {0: {-1, 0}, 1: {1, 2}}),
        (
            [[-1, 0, 1], [-1, 0, 1], [0, 1, -1], [1, 0, -1]],
            {0: {-1, 0, 1}, 1: {-1, 0, 1}},
        ),
    )
    def test_common_steps_per_slice(self, process_steps, expectation):
      """Test case."""
      per_process_steps = {
          pid: steps for pid, steps in enumerate(process_steps)
      }
      result = checkpoint_manager._common_values_per_slice(  # pylint: disable=protected-access
          per_process_steps, self.global_mesh, replica_axis_index=0
      )
      self.assertSameElements(result, expectation)

    @parameterized.parameters((True, []), (True, [0]), (False, [0, 3]))
    def test_local_all_steps(self, steps_found, processes_to_clear):
      """Test case."""
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(
              save_interval_steps=1,
              max_to_keep=4,
              debug_use_full_global_mesh=True,
          ),
      )
      manager = LocalCheckpointManager(
          directory=self.local_directory,
          global_mesh=self.global_mesh,
          options=options,
      )

      manager.save(
          0,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      manager.save(
          1,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.doubled_pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      manager.save(
          2,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      if multihost.process_index() in processes_to_clear:
        test_utils.empty_directory(self.local_directory)
      manager.reload()

      self.assertEqual(manager.all_steps(), [0, 1, 2] if steps_found else [])

      # create a new manager
      manager2 = LocalCheckpointManager(
          directory=self.local_directory,
          global_mesh=self.global_mesh,
          options=options,
      )
      self.assertEqual(manager2.all_steps(), [0, 1, 2] if steps_found else [])

    def test_local_look_for_all_partial_slice(self):
      p1 = self.global_mesh.devices[0][0].process_index
      p2 = self.global_mesh.devices[1][-1].process_index
      empty_process = [p1, p2]

      # steps for processes:
      # [2, -1], [1, 2]
      # [1, 2], [2, -1]
      # end result should be 2 since it's the only common step
      self.save_local_all_and_remove_partial(False, empty_process)

    def test_local_look_for_all_one_full_slice(self):
      empty_process = [0]

      # steps for processes:
      # [2, -1], [1, 2]
      # [1, 2], [1, 2]
      # end result should be [1, 2] since it's the common step in the bottom
      # slice.
      self.save_local_all_and_remove_partial(True, empty_process)

    def save_local_all_and_remove_partial(self, keep, empty_process):
      """Test case."""
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(
              save_interval_steps=1,
              max_to_keep=2,
              debug_use_full_global_mesh=True,
          ),
      )
      manager = LocalCheckpointManager(
          directory=self.local_directory,
          global_mesh=self.global_mesh,
          options=options,
      )

      manager.save(
          0,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      manager.save(
          1,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.doubled_pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      if multihost.process_index() in empty_process:
        test_utils.empty_directory(self.local_directory)

      manager.reload()

      manager.save(
          2,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      self.assertEqual(manager.all_steps(), [1, 2] if keep else [2])

      # create a new manager
      manager2 = LocalCheckpointManager(
          directory=self.local_directory,
          global_mesh=self.global_mesh,
          options=options,
      )

      all_step = manager2.all_steps()

      self.assertEqual(all_step, [1, 2] if keep else [2])

    @parameterized.parameters(
        (1, []), (1, [0]), (None, [0, 3]), (None, [0, 1, 2, 3])
    )
    def test_local_latest_step(self, expected, empty_process):
      """Test case."""
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(
              save_interval_steps=1,
              max_to_keep=1,
              debug_use_full_global_mesh=True,
          ),
          persistent=PersistentCheckpointOptions(
              save_interval_steps=1, max_to_keep=1
          ),
      )
      manager = LocalCheckpointManager(
          directory=self.local_directory,
          global_mesh=self.global_mesh,
          options=options,
      )

      manager.save(
          0,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.save(
          1,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.doubled_pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      # confirm all hosts have latest step as 1
      latest_step = manager.latest_step()
      self.assertEqual(latest_step, 1)

      if multihost.process_index() in empty_process:
        test_utils.empty_directory(self.local_directory)
      manager.reload()

      self.assertEqual(manager.latest_step(), expected)

      # create a new manager
      manager2 = LocalCheckpointManager(
          directory=self.local_directory,
          global_mesh=self.global_mesh,
          options=options,
      )

      latest_step = manager2.latest_step()

      self.assertEqual(latest_step, expected)

    @parameterized.parameters((1, [3]), (2, [2, 3]))
    def test_local_latest_step_cached(self, max_to_keep, expected_steps):
      """Test case."""
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(
              save_interval_steps=1,
              max_to_keep=max_to_keep,
              debug_use_full_global_mesh=True,
          )
      )
      manager = LocalCheckpointManager(
          directory=self.local_directory,
          global_mesh=self.global_mesh,
          options=options,
      )
      self.assertEmpty(manager._steps)

      manager.save(
          0,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      # confirm all hosts have latest step as 0
      self.assertEqual(manager._steps, [0])
      self.assertEqual(manager.latest_step(), 0)

      test_utils.empty_directory(self.local_directory)

      self.assertEqual(manager.latest_step(), 0)
      manager.reload()
      # after reload, latest_step cached should be false
      self.assertEmpty(manager._steps)

      # latest_step should be None and cached after calling latest_step
      self.assertIsNone(manager.latest_step())
      self.assertEqual(manager._steps, [])

      manager.save(
          1,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.doubled_pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.save(
          2,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.save(
          3,
          args=args_lib.Composite(
              state=PyTreeSaveArgs(self.doubled_pytree),
              process_metadata=process_metadata_checkpoint_handler.ProcessMetadataSaveArgs(
                  global_mesh=self.global_mesh
              ),
          ),
      )
      manager.wait_until_finished()

      if multihost.process_index() == 0:
        test_utils.empty_directory(self.local_directory)
      # still should be cache hit
      self.assertEqual(manager._steps, expected_steps)
      self.assertEqual(manager.latest_step(), 3)

      manager = LocalCheckpointManager(
          directory=self.local_directory,
          global_mesh=self.global_mesh,
          options=options,
      )
      self.assertEqual(manager._steps, expected_steps)
      self.assertEqual(manager.latest_step(), 3)


class CheckpointManagerTestBase:
  """Test case."""

  @barrier_compatible_test
  class Test(parameterized.TestCase):
    """Test case."""

    def make_global_mesh(
        self, replica_axis_index: int = 0
    ) -> jax.sharding.Mesh:
      raise NotImplementedError()

    def setUp(self):
      super().setUp()
      self.enter_context(
          flagsaver.flagsaver(
              experimental_orbax_use_distributed_process_id=True
          )
      )
      if not multihost.is_runtime_to_distributed_ids_initialized():
        multihost.initialize_runtime_to_distributed_ids()
        multihost.initialize_distributed_to_device_ids()

      # make sure each process is working on different directories
      self.local_directory = epath.Path(
          self.create_tempdir(
              name=f'local_checkpointing_test_pid{multihost.process_index()}'
          ).full_path
      )
      self.persistent_directory = epath.Path(
          self.create_tempdir(name='persistent_checkpointing_test').full_path
      )
      logging.info(
          'self.local_directory=%s, self.persistent_directory=%s',
          self.local_directory,
          self.persistent_directory,
      )

      test_utils.set_tensorstore_driver_for_test()
      test_utils.sync_global_processes('CheckpointManagerTest:setup_complete')

    def tearDown(self):
      super().tearDown()
      test_utils.sync_global_processes(
          'CheckpointManagerTest:teardown_complete'
      )

    def setup_pytree(self, global_mesh: jax.sharding.Mesh):
      """Create mesh and pytree."""
      pytree = {
          'a': test_utils.create_sharded_array(
              np.arange(8), global_mesh, jax.sharding.PartitionSpec(None)
          ),
          'b': test_utils.create_sharded_array(
              np.arange(16),
              global_mesh,
              jax.sharding.PartitionSpec('data'),
          ),
          'scalar': test_utils.create_sharded_array(
              123, global_mesh, jax.sharding.PartitionSpec()
          ),
          'empty_dict': {},
          'empty_node': optax.EmptyState(),
      }
      return global_mesh, pytree

    @parameterized.parameters(
        (0, 1, 1, True),
        (0, 2, 2, True),
        (1, 1, 1, True),
        (1, 1, 2, True),
        (1, 2, 2, False),
        (3, 5, 7, False),
        (2, 1, 2, True),
        (3, 2, 3, True),
    )
    def test_should_save(
        self,
        step: int,
        local_interval: int,
        persistent_interval: int,
        expectation: bool,
    ):
      """Test case."""
      global_mesh, pytree = self.setup_pytree(self.make_global_mesh())
      abstract_state = jax.tree.map(utils.to_shape_dtype_struct, pytree)
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(
              save_interval_steps=local_interval, max_to_keep=1
          ),
          persistent=PersistentCheckpointOptions(
              save_interval_steps=persistent_interval, max_to_keep=4
          ),
      )
      manager = CheckpointManager(
          local_directory=self.local_directory,
          persistent_directory=self.persistent_directory,
          global_mesh=global_mesh,
          abstract_state=abstract_state,
          options=options,
      )

      for i in range(step):
        manager.save(i, args=PyTreeSaveArgs(pytree))
        manager.wait_until_finished()

      self.assertEqual(manager.should_save(step), expectation)

    @parameterized.parameters(
        ([-1, -1, -1, -1], -1),
        ([0, 0, 0, -1], 0),
        ([2, 1, 0, -90000], 2),
        ([0, 1, 2, 3], 3),
        ([0, 0, 0, 0], 0),
        ([[0, 0], [1, 1], [2, 2], [3, 3]], [3, 3]),
        ([[0, -1], [1, 2], [0, 2], [-1, 2]], [1, 2]),
        ([False, False, False, False], 0),
        ([False, False, False, True], 1),
        ([False, True, True, True], 1),
        ([True, True, True, True], 1),
    )
    def test_global_max(self, inputs, expectation):
      """Test case."""
      local_host_inputs = inputs[multihost.process_index()]
      if not isinstance(local_host_inputs, list):
        local_host_inputs = [local_host_inputs]
        expectation = [expectation]
      self.assertEqual(
          checkpoint_manager._global_max(
              local_host_inputs, checkpoint_manager._get_global_broadcast_fn()
          ),
          expectation,
      )

    @parameterized.parameters(
        (1, 2, 2, 4, 10, [2, 4, 6, 8, 9]),
        (1, 3, 2, 4, 10, [2, 4, 6, 7, 8, 9]),
        (1, 2, 2, 5, 10, [0, 2, 4, 6, 8, 9]),
        (1, 2, 10, 4, 10, [0, 8, 9]),
    )
    def test_all_steps(
        self,
        local_interval,
        local_max_to_keep,
        persistent_interval,
        persistent_max_to_keep,
        total_steps,
        expectation,
    ):
      """Test case."""
      global_mesh, pytree = self.setup_pytree(self.make_global_mesh())
      abstract_state = jax.tree.map(utils.to_shape_dtype_struct, pytree)
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(
              save_interval_steps=local_interval, max_to_keep=local_max_to_keep
          ),
          persistent=PersistentCheckpointOptions(
              save_interval_steps=persistent_interval,
              max_to_keep=persistent_max_to_keep,
          ),
      )
      manager = CheckpointManager(
          local_directory=self.local_directory,
          persistent_directory=self.persistent_directory,
          global_mesh=global_mesh,
          abstract_state=abstract_state,
          options=options,
      )

      for i in range(total_steps):
        manager.save(i, args=PyTreeSaveArgs(pytree))
        manager.wait_until_finished()

      self.assertEqual(sorted(manager.all_steps()), expectation)

      # create a new manager
      manager2 = CheckpointManager(
          local_directory=self.local_directory,
          persistent_directory=self.persistent_directory,
          global_mesh=global_mesh,
          abstract_state=abstract_state,
          options=options,
      )

      self.assertEqual(sorted(manager2.all_steps()), expectation)

    @parameterized.parameters(
        (1, 4, 3),
        (2, 3, 3),
        (2, 4, 2),
        (4, 4, 0),
    )
    def test_latest_step(
        self,
        local_interval,
        persistent_interval,
        expectation,
        total_steps: Optional[int] = 4,
    ):
      """Test case."""
      global_mesh, pytree = self.setup_pytree(self.make_global_mesh())
      abstract_state = jax.tree.map(utils.to_shape_dtype_struct, pytree)
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(
              save_interval_steps=local_interval, max_to_keep=2
          ),
          persistent=PersistentCheckpointOptions(
              save_interval_steps=persistent_interval, max_to_keep=4
          ),
      )
      manager = CheckpointManager(
          local_directory=self.local_directory,
          persistent_directory=self.persistent_directory,
          global_mesh=global_mesh,
          abstract_state=abstract_state,
          options=options,
      )

      for i in range(total_steps):
        manager.save(i, args=PyTreeSaveArgs(pytree))
        manager.wait_until_finished()

      self.assertEqual(manager.latest_step(), expectation)

      # create a new manager
      manager2 = CheckpointManager(
          local_directory=self.local_directory,
          persistent_directory=self.persistent_directory,
          global_mesh=global_mesh,
          abstract_state=abstract_state,
          options=options,
      )

      self.assertEqual(manager2.latest_step(), expectation)

    @parameterized.parameters(
        (False,),
        (True,),
    )
    def test_save(self, enable_async_checkpointing: bool):
      """Test case."""
      global_mesh, pytree = self.setup_pytree(self.make_global_mesh())
      abstract_state = jax.tree.map(utils.to_shape_dtype_struct, pytree)
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(save_interval_steps=3, max_to_keep=2),
          persistent=PersistentCheckpointOptions(
              save_interval_steps=5, max_to_keep=3
          ),
          enable_async_checkpointing=enable_async_checkpointing,
      )
      manager = CheckpointManager(
          local_directory=self.local_directory,
          persistent_directory=self.persistent_directory,
          global_mesh=global_mesh,
          abstract_state=abstract_state,
          options=options,
      )

      for i in range(17):
        manager.save(i, args=PyTreeSaveArgs(pytree))
      manager.wait_until_finished()

      self.assertEqual(manager.all_steps(), [5, 10, 12, 15])

    @parameterized.parameters(
        (False, 0),
        (True, 0),
        (True, 1),
    )
    def test_local_restore_by_broadcast(
        self, enable_async_checkpointing: bool, replica_axis_index: int
    ):
      """Test restore successfully."""
      global_mesh, pytree = self.setup_pytree(
          self.make_global_mesh(replica_axis_index)
      )
      abstract_state = jax.tree.map(utils.to_shape_dtype_struct, pytree)
      pytree_double = test_utils.apply_function(pytree, lambda x: x * 2)
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(save_interval_steps=1, max_to_keep=2),
          persistent=PersistentCheckpointOptions(
              save_interval_steps=1, max_to_keep=2
          ),
          enable_async_checkpointing=enable_async_checkpointing,
          replica_axis_index=replica_axis_index,
      )

      with mock.patch.object(
          checkpoint_manager._MultisliceCheckpointManager,
          '_restore_from_persistent',
          autospec=True,
          return_value=None,
      ) as pcm_restore, CheckpointManager(
          local_directory=self.local_directory,
          persistent_directory=self.persistent_directory,
          global_mesh=global_mesh,
          options=options,
          abstract_state=abstract_state,
      ) as manager:

        manager.save(0, args=PyTreeSaveArgs(pytree))
        manager.save(1, args=PyTreeSaveArgs(pytree))
        manager.save(2, args=PyTreeSaveArgs(pytree_double))
        manager.wait_until_finished()

        if multihost.process_index() == 0:
          test_utils.empty_directory(self.persistent_directory)
        restored = manager.restore(2)
        test_utils.assert_tree_equal(self, pytree_double, restored)
        pcm_restore.assert_not_called()

    @parameterized.parameters(
        (0,),
        (1,),
    )
    def test_slice_setup(self, replica_axis_index: int):
      """Test restoration after slice swap (via global mesh)."""
      global_mesh, pytree = self.setup_pytree(
          self.make_global_mesh(replica_axis_index)
      )
      abstract_state = jax.tree.map(utils.to_shape_dtype_struct, pytree)
      pytree_double = test_utils.apply_function(pytree, lambda x: x * 2)
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(save_interval_steps=1, max_to_keep=2),
          persistent=PersistentCheckpointOptions(
              save_interval_steps=1, max_to_keep=2
          ),
          replica_axis_index=replica_axis_index,
      )

      with CheckpointManager(
          local_directory=self.local_directory,
          persistent_directory=self.persistent_directory,
          global_mesh=global_mesh,
          options=options,
          abstract_state=abstract_state,
      ) as manager:
        self.assertEqual(
            manager._checkpoint_manager._persistent_primary_host, 0
        )
        self.assertEqual(manager._checkpoint_manager._local_primary_host, 2)
        slice_id = multislice.process_slice_id(
            multihost.process_index(),
            global_mesh,
            replica_axis_index=replica_axis_index,
        )
        self.assertEqual(slice_id == 0, manager.in_primary_slice)
        manager.save(0, args=PyTreeSaveArgs(pytree))
        manager.save(1, args=PyTreeSaveArgs(pytree))
        manager.save(2, args=PyTreeSaveArgs(pytree_double))

      self.assertEqual(jax.process_count(), 4)
      self.assertEqual(
          multislice.process_slice_id(
              0, global_mesh, replica_axis_index=replica_axis_index
          ),
          0,
      )
      self.assertEqual(
          multislice.process_slice_id(
              1, global_mesh, replica_axis_index=replica_axis_index
          ),
          0,
      )
      self.assertEqual(
          multislice.process_slice_id(
              2, global_mesh, replica_axis_index=replica_axis_index
          ),
          1,
      )
      self.assertEqual(
          multislice.process_slice_id(
              3, global_mesh, replica_axis_index=replica_axis_index
          ),
          1,
      )
      new_global_mesh = swap_slices_in_mesh(
          global_mesh, replica_axis_index=replica_axis_index
      )
      self.assertEqual(
          multislice.process_slice_id(
              0, new_global_mesh, replica_axis_index=replica_axis_index
          ),
          1,
      )
      self.assertEqual(
          multislice.process_slice_id(
              1, new_global_mesh, replica_axis_index=replica_axis_index
          ),
          1,
      )
      self.assertEqual(
          multislice.process_slice_id(
              2, new_global_mesh, replica_axis_index=replica_axis_index
          ),
          0,
      )
      self.assertEqual(
          multislice.process_slice_id(
              3, new_global_mesh, replica_axis_index=replica_axis_index
          ),
          0,
      )

    @parameterized.parameters(
        (0,),
        (1,),
    )
    def test_slice_swap_check_steps(self, replica_axis_index: int):
      """Test step detection across slice swap (via mesh)."""
      global_mesh, pytree = self.setup_pytree(
          self.make_global_mesh(replica_axis_index)
      )
      abstract_state = jax.tree.map(utils.to_shape_dtype_struct, pytree)
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(save_interval_steps=1, max_to_keep=10),
          persistent=PersistentCheckpointOptions(
              save_interval_steps=10, max_to_keep=10
          ),
          replica_axis_index=replica_axis_index,
      )

      with CheckpointManager(
          local_directory=self.local_directory,
          persistent_directory=self.persistent_directory,
          global_mesh=global_mesh,
          options=options,
          abstract_state=abstract_state,
      ) as manager:
        self.assertEqual(
            manager._checkpoint_manager._persistent_primary_host, 0
        )
        self.assertEqual(manager._checkpoint_manager._local_primary_host, 2)
        slice_id = multislice.process_slice_id(
            multihost.process_index(),
            global_mesh,
            replica_axis_index=replica_axis_index,
        )
        self.assertEqual(slice_id == 0, manager.in_primary_slice)
        manager.save(0, args=PyTreeSaveArgs(pytree))
        manager.save(1, args=PyTreeSaveArgs(pytree))
        manager.save(2, args=PyTreeSaveArgs(pytree))
        manager.wait_until_finished()
        self.assertSameElements([0, 1, 2], manager.all_steps())
        self.assertEqual(2, manager.latest_step())

        # Delete the checkpoint on the *current* primary slice (it will
        # become the secondary below, and the only checkpoint available will be
        # on the primary slice).
        if manager.in_primary_slice:
          test_utils.empty_directory(self.local_directory)
        test_utils.sync_global_processes('sync_after_local_dir_empty')

      new_global_mesh = swap_slices_in_mesh(
          global_mesh, replica_axis_index=replica_axis_index
      )
      abstract_state = jax.tree.map(
          functools.partial(
              _replace_abstract_array_sharding_with_mesh,
              mesh=new_global_mesh,
          ),
          abstract_state,
      )

      with CheckpointManager(
          local_directory=self.local_directory,
          persistent_directory=self.persistent_directory,
          global_mesh=new_global_mesh,
          options=options,
          abstract_state=abstract_state,
      ) as manager:
        self.assertEqual(
            manager._checkpoint_manager._persistent_primary_host, 2
        )
        self.assertEqual(manager._checkpoint_manager._local_primary_host, 0)
        slice_id = multislice.process_slice_id(
            multihost.process_index(),
            new_global_mesh,
            replica_axis_index=replica_axis_index,
        )
        # It is able to recover the steps even though only the primary slice
        # has local checkpoints. This has caused problems in the past because
        # the primary ordinarily is not responsible for dealing with local
        # checkpoints.
        if manager.in_primary_slice:
          self.assertNotEmpty(list(self.local_directory.iterdir()))
        else:
          self.assertEmpty(list(self.local_directory.iterdir()))
        self.assertEqual(slice_id == 0, manager.in_primary_slice)
        self.assertSameElements([0, 1, 2], manager.all_steps())
        self.assertEqual(2, manager.latest_step())

    @parameterized.parameters(
        (False, 0),
        (True, 0),
        (True, 1),
    )
    def test_local_missing_checkpoint(
        self, enable_async_checkpointing: bool, replica_axis_index: int
    ):
      """Test restore successfully."""
      global_mesh, pytree = self.setup_pytree(
          self.make_global_mesh(replica_axis_index)
      )
      abstract_state = jax.tree.map(utils.to_shape_dtype_struct, pytree)
      pytree_double = test_utils.apply_function(pytree, lambda x: x * 2)
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(save_interval_steps=1, max_to_keep=2),
          persistent=PersistentCheckpointOptions(
              save_interval_steps=2, max_to_keep=2
          ),
          enable_async_checkpointing=enable_async_checkpointing,
          replica_axis_index=replica_axis_index,
      )

      with CheckpointManager(
          local_directory=self.local_directory,
          persistent_directory=self.persistent_directory,
          global_mesh=global_mesh,
          abstract_state=abstract_state,
          options=options,
      ) as manager:
        manager.save(0, args=PyTreeSaveArgs(pytree))  # both
        manager.save(1, args=PyTreeSaveArgs(pytree))  # local
        manager.save(2, args=PyTreeSaveArgs(pytree_double))  # both
        manager.wait_until_finished()

        if not manager.in_primary_slice:
          test_utils.empty_directory(self.local_directory)
        restored = manager.restore(2)
        test_utils.assert_tree_equal(self, pytree_double, restored)

        # Test subsequent saves.

        # reload manager gives each process the correct view of existing local
        # checkpoints.
        manager.reload()

        manager.save(3, args=PyTreeSaveArgs(pytree))  # local
        manager.save(4, args=PyTreeSaveArgs(pytree))  # both
        manager.save(5, args=PyTreeSaveArgs(pytree_double))  # local
        manager.wait_until_finished()

        restored = manager.restore(5)
        test_utils.assert_tree_equal(self, pytree_double, restored)

    @parameterized.parameters(
        (False, 0),
        (True, 0),
        (True, 1),
    )
    def test_persistent_checkpoint_restore(
        self, enable_async_checkpointing: bool, replica_axis_index: int
    ):
      """Test case."""
      global_mesh, pytree = self.setup_pytree(
          self.make_global_mesh(replica_axis_index)
      )
      abstract_state = jax.tree.map(utils.to_shape_dtype_struct, pytree)
      pytree_double = test_utils.apply_function(pytree, lambda x: x * 2)
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(save_interval_steps=1, max_to_keep=2),
          persistent=PersistentCheckpointOptions(
              save_interval_steps=2, max_to_keep=2
          ),
          enable_async_checkpointing=enable_async_checkpointing,
          replica_axis_index=replica_axis_index,
      )

      with CheckpointManager(
          local_directory=self.local_directory,
          persistent_directory=self.persistent_directory,
          global_mesh=global_mesh,
          abstract_state=abstract_state,
          options=options,
      ) as manager:
        manager.save(0, args=PyTreeSaveArgs(pytree))
        manager.save(1, args=PyTreeSaveArgs(pytree))
        manager.save(2, args=PyTreeSaveArgs(pytree_double))
        manager.wait_until_finished()

        # remove all the local directories
        test_utils.empty_directory(self.local_directory)

        restored = manager.restore(2, args=PyTreeRestoreArgs())
        test_utils.assert_tree_equal(self, pytree_double, restored)

    def test_process_index_metadata(self):
      """Test case."""
      global_mesh, pytree = self.setup_pytree(self.make_global_mesh())
      abstract_state = jax.tree.map(utils.to_shape_dtype_struct, pytree)
      with CheckpointManager(
          local_directory=self.local_directory,
          persistent_directory=self.persistent_directory,
          global_mesh=global_mesh,
          abstract_state=abstract_state,
      ) as mngr:
        metadata_folder = mesh_consistency.process_metadata_folder(
            self.local_directory / '0' / 'process_metadata'
        )
        self.assertFalse(metadata_folder.exists())

        mngr.save(0, args=PyTreeSaveArgs(pytree))
        mngr.wait_until_finished()

        if mngr.in_primary_slice:
          self.assertFalse(metadata_folder.exists())
        else:
          metadata_path = (
              metadata_folder
              / mesh_consistency._GLOBAL_PROCESS_METADATA_FILE_NAME
          )
          self.assertTrue(metadata_path.exists())
          contents = json.loads(metadata_path.read_text())
          self.assertListEqual(
              multihost.distributed_to_device_ids(),
              contents,
          )

          metadata_path = (
              metadata_folder / mesh_consistency._MESH_METADATA_FILE_NAME
          )
          self.assertTrue(metadata_path.exists())
          device_ids = json.loads(metadata_path.read_text())
          self.assertListEqual(
              device_ids,
              [int(id) for id in global_mesh.device_ids.flatten()],
          )

    def test_should_save_fn(self):
      """Test case."""
      local_expectation = [1, 5, 7, 8]
      persistent_expectation = [1, 2]

      def should_save_local(step: int, _) -> bool:
        return step in local_expectation

      def should_save_persistent(step: int, _) -> bool:
        return step in persistent_expectation

      max_to_keep = 10
      global_mesh, pytree = self.setup_pytree(self.make_global_mesh())
      abstract_state = jax.tree.map(utils.to_shape_dtype_struct, pytree)
      options = CheckpointManagerOptions(
          local=LocalCheckpointOptions(
              should_save_fn=should_save_local, max_to_keep=max_to_keep
          ),
          persistent=PersistentCheckpointOptions(
              should_save_fn=should_save_persistent, max_to_keep=max_to_keep
          ),
      )
      manager = CheckpointManager(
          local_directory=self.local_directory,
          persistent_directory=self.persistent_directory,
          global_mesh=global_mesh,
          abstract_state=abstract_state,
          options=options,
      )

      for i in range(max_to_keep):
        manager.save(i, args=PyTreeSaveArgs(pytree))
        manager.wait_until_finished()

      self.assertSameElements(
          manager.all_steps(), set(local_expectation + persistent_expectation)
      )

    @parameterized.parameters(
        ((1, 2), 0, np.asarray([[0, 1]])),
        ((2, 2), 0, np.asarray([[2, 3], [0, 1]])),
        ((3, 2), 0, np.asarray([[4, 5], [2, 3], [0, 1]])),
        ((4, 2), 0, np.asarray([[6, 7], [4, 5], [2, 3], [0, 1]])),
        ((2, 1), 1, np.asarray([[0], [1]])),
        ((2, 2), 1, np.asarray([[1, 0], [3, 2]])),
        ((2, 3), 1, np.asarray([[2, 1, 0], [5, 4, 3]])),
        ((2, 4), 1, np.asarray([[3, 2, 1, 0], [7, 6, 5, 4]])),
    )
    def test_swap_slices_in_mesh(
        self, mesh_shape, replica_axis_index, expected_result
    ):
      """Test slice swapping utility."""

      class FakeMesh:

        def __init__(self, devices, axis_names):
          self.devices = devices
          self.axis_names = axis_names
          self.shape = mesh_shape
          self.shape_tuple = tuple(mesh_shape)

      devices = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
      with mock.patch.object(jax.sharding, 'Mesh', wraps=FakeMesh):
        swapped_mesh = swap_slices_in_mesh(
            jax.sharding.Mesh(devices, None),  # pytype: disable=wrong-arg-types
            replica_axis_index=replica_axis_index,
        )
        swapped_devices = swapped_mesh.devices
        np.testing.assert_array_equal(swapped_devices, expected_result)
