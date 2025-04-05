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

"""Common tests for AbstractCheckpointManager subclasses."""

# pylint: disable=protected-access
import copy
from typing import Any, Callable, Sequence
import unittest
from unittest import mock

from absl.testing import parameterized
from etils import epath
from flax import linen as nn
from flax.training import train_state
import jax
from jax.experimental import pjit
import jax.numpy as jnp
import numpy as np
import optax
from orbax.checkpoint import args
from orbax.checkpoint import test_utils
from orbax.checkpoint import utils
from orbax.checkpoint._src import asyncio_utils
from orbax.checkpoint._src.checkpointers import async_checkpointer
from orbax.checkpoint._src.checkpointers import checkpointer as checkpointer_lib
from orbax.checkpoint._src.handlers import async_checkpoint_handler
from orbax.checkpoint._src.handlers import composite_checkpoint_handler
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.logging import step_statistics
from orbax.checkpoint._src.metadata import array_metadata
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.metadata import checkpoint
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.path import atomicity
from orbax.checkpoint._src.path import step
from orbax.checkpoint._src.serialization import serialization
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.testing import test_tree_utils

from .pyglib import gfile


PyTreeCheckpointHandler = pytree_checkpoint_handler.PyTreeCheckpointHandler
AsyncCheckpointHandler = async_checkpoint_handler.AsyncCheckpointHandler
AsyncCheckpointer = async_checkpointer.AsyncCheckpointer
ArrayRestoreArgs = pytree_checkpoint_handler.ArrayRestoreArgs
TrainState = train_state.TrainState
Mesh = jax.sharding.Mesh
PyTree = Any


class CheckpointerTestBase:
  """Common tests for AbstractCheckpointer subclasses."""

  class Test(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):
    """Structure allows test to run as subclasses, not base class."""

    def checkpointer(self, handler, **kwargs) -> checkpointer_lib.Checkpointer:
      raise NotImplementedError

    def setUp(self):
      super().setUp()
      pytree, mesh_tree, axes_tree = test_utils.setup_sharded_pytree()
      doubled_pytree = test_utils.apply_function(pytree, lambda x: x * 2)

      self.empty_pytree = jax.tree.map(
          lambda x: object(), pytree, is_leaf=test_utils.is_leaf
      )
      self.pytree = pytree
      self.doubled_pytree = doubled_pytree
      self.mesh_tree = mesh_tree
      self.axes_tree = axes_tree
      self.pytree_restore_args = jax.tree.map(
          lambda mesh, axes: ArrayRestoreArgs(mesh=mesh, mesh_axes=axes),
          self.mesh_tree,
          self.axes_tree,
      )
      self.directory = (
          epath.Path(self.create_tempdir(name='checkpointing_test').full_path)
          / 'ckpt'
      )
      test_utils.set_tensorstore_driver_for_test()

      test_utils.sync_global_processes('CheckpointerTest:setup_complete')

    def tearDown(self):
      test_utils.sync_global_processes('CheckpointerTest:tests_complete')
      super().tearDown()

    def wait_if_async(self, checkpointer):
      if isinstance(checkpointer, AsyncCheckpointer):
        checkpointer.wait_until_finished()

    def is_world_readable(self, path):
      if not gfile.Exists(path):
        return False
      mode = gfile.Stat(path, stat_proto=True).mode
      return (mode & step.NON_WORLD_READABLE_MODE) != mode

    def test_save_restore(self):
      """Basic save and restore test."""
      with self.checkpointer(PyTreeCheckpointHandler()) as checkpointer:
        checkpointer.save(self.directory, self.pytree)
        self.wait_if_async(checkpointer)
        restored = checkpointer.restore(
            self.directory, restore_args=self.pytree_restore_args
        )
        test_utils.assert_tree_equal(self, self.pytree, restored)

    def test_save_restore_with_args(self):
      """Basic save and restore test."""
      with self.checkpointer(PyTreeCheckpointHandler()) as checkpointer:
        checkpointer.save(self.directory, args=args.PyTreeSave(self.pytree))
        self.wait_if_async(checkpointer)
        restored = checkpointer.restore(
            self.directory,
            args=args.PyTreeRestore(restore_args=self.pytree_restore_args),
        )
        test_utils.assert_tree_equal(self, self.pytree, restored)

    def test_save_restore_no_kwargs(self):
      """Restore with no GDA args."""
      with self.checkpointer(PyTreeCheckpointHandler()) as checkpointer:
        expected = jax.tree.map(test_utils.replicate_sharded_array, self.pytree)
        expected = jax.tree.map(
            lambda x: np.asarray(x.addressable_data(0)), expected
        )
        checkpointer.save(self.directory, expected)
        self.wait_if_async(checkpointer)
        restored = checkpointer.restore(self.directory)
        test_utils.assert_tree_equal(self, expected, restored)

    def test_restore_missing_path(self):
      """Restore with invalid item."""
      with self.checkpointer(PyTreeCheckpointHandler()) as checkpointer:
        with self.assertRaises(FileNotFoundError):
          checkpointer.restore('path/to/missing')

    def test_no_overwrite_existing(self):
      """Test same step does not overwrite."""
      with self.checkpointer(PyTreeCheckpointHandler()) as checkpointer:
        checkpointer.save(self.directory, self.pytree)
        self.wait_if_async(checkpointer)
        with self.assertRaises(ValueError):
          checkpointer.save(self.directory, self.doubled_pytree)
        self.wait_if_async(checkpointer)
        restored = checkpointer.restore(
            self.directory, restore_args=self.pytree_restore_args
        )
        expected = self.pytree
        test_utils.assert_tree_equal(self, expected, restored)
        self.assertFalse(self.is_world_readable(self.directory))

    def test_overwrite_existing(self):
      """Test overwrite existing path."""
      with self.checkpointer(PyTreeCheckpointHandler()) as checkpointer:
        checkpointer.save(self.directory, self.pytree)
        self.wait_if_async(checkpointer)
        checkpointer.save(self.directory, self.doubled_pytree, force=True)
        self.wait_if_async(checkpointer)
        restored = checkpointer.restore(
            self.directory, restore_args=self.pytree_restore_args
        )
        expected = self.doubled_pytree
        test_utils.assert_tree_equal(self, expected, restored)
        self.assertFalse(self.is_world_readable(self.directory))

    def test_flax_train_state(self):
      """Test using flax model."""

      class MLP(nn.Module):
        """A simple MLP model."""

        @nn.compact
        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
          x = x.reshape((x.shape[0], -1))  # flatten
          x = nn.Dense(features=8)(x)
          return x

      model = MLP()
      mesh = Mesh(np.asarray(jax.devices()), ('devices',))
      mesh_axes = jax.sharding.PartitionSpec()

      @jax.jit
      def init_state():
        params = model.init(jax.random.PRNGKey(0), jnp.ones([8, 8]))
        tx = optax.adamw(learning_rate=0.001)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        return state

      init = pjit.pjit(init_state, in_shardings=None, out_shardings=mesh_axes)

      with Mesh(mesh.devices, mesh.axis_names):
        state = init()
        state_shape = jax.eval_shape(init)

      restore_args = jax.tree.map(
          lambda _: ArrayRestoreArgs(mesh=mesh, mesh_axes=mesh_axes),
          state_shape,
      )

      with self.checkpointer(PyTreeCheckpointHandler()) as checkpointer:
        checkpointer.save(self.directory, state)
        self.wait_if_async(checkpointer)
        # Already fully replicated, don't need to provide args.
        restored = checkpointer.restore(
            self.directory, item=state_shape, restore_args=restore_args
        )

        test_utils.assert_tree_equal(self, state.params, restored.params)
        test_utils.assert_tree_equal(self, state.opt_state, restored.opt_state)

    @mock.patch.object(atomicity.AtomicRenameTemporaryPath, 'finalize')
    def test_save_preempted(self, _):
      """Simulate effects of preemption."""
      with self.checkpointer(PyTreeCheckpointHandler()) as checkpointer:
        checkpointer.save(self.directory, self.pytree)
        self.wait_if_async(checkpointer)
        paths = list(self.directory.parent.iterdir())
        self.assertLen(paths, 1)
        tmp_dir = paths[0]
        self.assertFalse(utils.is_checkpoint_finalized(tmp_dir))
        self.assertTrue(utils.is_tmp_checkpoint(tmp_dir))
        with self.assertRaisesRegex(ValueError, 'Found incomplete checkpoint'):
          checkpointer.restore(tmp_dir)

    @mock.patch.object(step, 'is_gcs_path', autospec=True, return_value=True)
    def test_gcs(self, is_gcs_path):
      """Test normal operation in simulated GCS environment."""
      del is_gcs_path
      with self.checkpointer(PyTreeCheckpointHandler()) as checkpointer:
        path = self.directory / '0' / 'params'
        checkpointer.save(path, self.pytree)
        self.wait_if_async(checkpointer)
        restored = checkpointer.restore(
            path, restore_args=self.pytree_restore_args
        )
        test_utils.assert_tree_equal(self, self.pytree, restored)
        self.assertTrue((path / step._COMMIT_SUCCESS_FILE).exists())

    @mock.patch.object(step, 'is_gcs_path', autospec=True, return_value=True)
    @mock.patch.object(atomicity.CommitFileTemporaryPath, 'finalize')
    def test_save_preempted_gcs(self, *mock_args):
      """Simulate effects of preemption."""
      del mock_args
      with self.checkpointer(PyTreeCheckpointHandler()) as checkpointer:
        checkpointer.save(self.directory, self.pytree)
        self.wait_if_async(checkpointer)
        paths = list(self.directory.parent.iterdir())
        self.assertLen(paths, 1)
        tmp_dir = paths[0]
        self.assertFalse(utils.is_checkpoint_finalized(tmp_dir))
        self.assertTrue(utils.is_tmp_checkpoint(tmp_dir))
        with self.assertRaisesRegex(ValueError, 'Found incomplete checkpoint'):
          checkpointer.restore(tmp_dir)
        self.assertFalse((tmp_dir / step._COMMIT_SUCCESS_FILE).exists())

    def test_configure_atomicity(self):
      """Test case."""
      with self.checkpointer(
          PyTreeCheckpointHandler(),
          temporary_path_class=atomicity.CommitFileTemporaryPath,
      ) as checkpointer:
        path = self.directory / '0' / 'params'
        checkpointer.save(path, self.pytree)
        self.wait_if_async(checkpointer)
        restored = checkpointer.restore(
            path, restore_args=self.pytree_restore_args
        )
        test_utils.assert_tree_equal(self, self.pytree, restored)
        self.assertTrue((path / step._COMMIT_SUCCESS_FILE).exists())  # pylint: disable=protected-access

    def test_legacy_api(self):
      """Test case."""

      # Skip CheckpointArgs registration.
      class MyCheckpointHandler(
          async_checkpoint_handler.AsyncCheckpointHandler
      ):

        async def async_save(self, directory, item, extra_args):
          return []

        def save(self, directory, item, extra_args):
          asyncio_utils.run_sync(self.async_save(directory, item, extra_args))

        def restore(self, directory, item=None, extra_args=None):
          return f'restored {item} with {extra_args} from {directory}.'

      with self.checkpointer(MyCheckpointHandler()) as ckptr:
        item = {'a': 1, 'b': 2}
        extra_args = {'special_args': True}
        ckptr.save(self.directory, item, extra_args)
        self.wait_if_async(ckptr)
        restored = ckptr.restore(
            self.directory, item=item, extra_args=extra_args
        )
        self.assertEqual(
            restored,
            f'restored {item} with {extra_args} from {self.directory}.',
        )

    def test_composite_handler(self):
      """Test case."""

      with self.checkpointer(
          composite_checkpoint_handler.CompositeCheckpointHandler()
      ) as ckptr:
        ckptr.save(
            self.directory,
            args.Composite(state=args.PyTreeSave(self.pytree)),
        )
        self.wait_if_async(ckptr)
        restored = ckptr.restore(
            self.directory,
            args=args.Composite(
                state=args.PyTreeRestore(restore_args=self.pytree_restore_args)
            ),
        )
        test_utils.assert_tree_equal(self, self.pytree, restored.state)

    @parameterized.parameters((5,), (9,))
    def test_memory_limiting(self, limit_bytes):
      """Test case."""
      sleep_time = 1.0
      handler, tree, restore_args = test_utils.concurrent_gb_test_setup()
      checkpointer = self.checkpointer(handler)

      byte_limiter = test_utils.get_byte_limiter(limit_bytes, sleep_time)
      with mock.patch.object(
          serialization,
          'get_byte_limiter',
          new=lambda _: byte_limiter,
      ):
        checkpointer.save(self.directory, tree)
      self.wait_if_async(checkpointer)
      completion_times = byte_limiter.completion_times
      # Replicated shards are handled within the _write_array_shard function.
      # Since shards are only saved once per replica, we only have to check
      # the primary process.
      if multihost.process_index() == 0:
        self.assertLen(completion_times, len(jax.tree.leaves(tree)))
        test_utils.assert_every_n_is_x_apart(
            self,
            completion_times,
            limit_bytes // np.int32().itemsize,
            sleep_time,
        )

      byte_limiter = test_utils.get_byte_limiter(limit_bytes, sleep_time)
      with mock.patch.object(
          serialization,
          'get_byte_limiter',
          new=lambda _: byte_limiter,
      ):
        checkpointer.restore(self.directory, restore_args=restore_args)
      completion_times = byte_limiter.completion_times
      self.assertLen(
          completion_times,
          len(jax.tree.leaves(tree)),
      )
      test_utils.assert_every_n_is_x_apart(
          self,
          completion_times,
          limit_bytes // np.int32().itemsize,
          sleep_time,
      )

    @parameterized.named_parameters(
        dict(
            testcase_name='default',
            save_input_provider=test_tree_utils.NamedTupleWithNestedAttributes,
            expected_restored_provider=(
                lambda: dict(
                    nested_mu_nu=None,
                    nested_dict=None,
                    nested_tuple=None,
                    nested_empty_named_tuple=None,
                    my_empty_chex=None,
                )
            ),
        ),
        dict(
            testcase_name='full',
            save_input_provider=(
                lambda: test_tree_utils.NamedTupleWithNestedAttributes(
                    nested_mu_nu=test_tree_utils.MuNu(
                        mu=test_utils.setup_sharded_pytree()[0],
                        nu=np.asarray([1, 2, 3]),
                    ),
                    nested_dict=dict(a=test_utils.setup_sharded_pytree()[0]),
                    nested_tuple=(
                        test_utils.setup_sharded_pytree()[0],
                        test_utils.setup_sharded_pytree()[0],
                    ),
                    nested_empty_named_tuple=test_tree_utils.EmptyNamedTuple(),
                    my_empty_chex=test_tree_utils.MyEmptyChex(),
                )
            ),
            expected_restored_provider=(
                lambda: dict(
                    nested_mu_nu=dict(
                        mu=test_utils.setup_sharded_pytree()[0],
                        nu=np.asarray([1, 2, 3]),
                    ),
                    nested_dict=dict(a=test_utils.setup_sharded_pytree()[0]),
                    nested_tuple=[
                        test_utils.setup_sharded_pytree()[0],
                        test_utils.setup_sharded_pytree()[0],
                    ],  # b/365169723
                    nested_empty_named_tuple={},
                    my_empty_chex={},
                )
            ),
        ),
        dict(
            testcase_name='partial',
            save_input_provider=(
                lambda: test_tree_utils.NamedTupleWithNestedAttributes(
                    nested_mu_nu=test_tree_utils.MuNu(
                        mu=None,
                        nu=np.asarray([1, 2, 3]),
                    ),
                    nested_dict=None,
                    nested_tuple=None,
                    nested_empty_named_tuple=test_tree_utils.EmptyNamedTuple(),
                )
            ),
            expected_restored_provider=(
                lambda: dict(
                    nested_mu_nu=dict(
                        mu=None,
                        nu=np.asarray([1, 2, 3]),
                    ),
                    nested_dict=None,
                    nested_tuple=None,
                    nested_empty_named_tuple={},
                    my_empty_chex=None,
                )
            ),
        ),
    )
    def test_save_restore_named_tuple(
        self,
        save_input_provider: Callable[[], PyTree],
        expected_restored_provider: Callable[[], PyTree],
    ):
      """Basic save and restore test."""
      with self.subTest('legacy_metadata'):
        with self.checkpointer(
            PyTreeCheckpointHandler(
                pytree_metadata_options=tree_metadata.PyTreeMetadataOptions(
                    support_rich_types=False
                )
            )
        ) as checkpointer:
          directory = self.directory / 'legacy_metadata'
          checkpointer.save(directory, save_input_provider())
          self.wait_if_async(checkpointer)
          restored = checkpointer.restore(directory)
          test_utils.assert_tree_equal(
              self, restored, expected_restored_provider()
          )

      with self.subTest('rich_typed_metadata'):
        with self.checkpointer(
            PyTreeCheckpointHandler(
                pytree_metadata_options=tree_metadata.PyTreeMetadataOptions(
                    support_rich_types=True
                )
            )
        ) as checkpointer:
          directory = self.directory / 'rich_typed_metadata'
          checkpointer.save(directory, save_input_provider())
          self.wait_if_async(checkpointer)
          # TODO: b/365169723 - Update this test when restore is ready.
          with self.assertRaises(NotImplementedError):
            _ = checkpointer.restore(directory)

    def test_save_step_metadata(self):
      """Basic save and restore test."""
      with self.checkpointer(PyTreeCheckpointHandler()) as checkpointer:
        directory = self.directory / 'step_metadata'
        checkpointer.save(
            directory,
            args.PyTreeSave(self.pytree),
            custom_metadata={'a': 1, 'b': 2}
        )
        self.wait_if_async(checkpointer)

        serialized_metadata = checkpoint.metadata_store(
            enable_write=False
        ).read(checkpoint.step_metadata_file_path(directory))
        self.assertIsNotNone(serialized_metadata)
        step_metadata = step_metadata_serialization.deserialize(
            serialized_metadata
        )

        self.assertEqual(
            step_metadata.item_handlers, checkpointer._handler.typestr()
        )
        self.assertIsNone(step_metadata.item_metadata)
        self.assertEqual(step_metadata.metrics, {})
        self.assertEqual(
            step_metadata.performance_metrics,
            step_statistics.SaveStepStatistics(),
        )
        self.assertGreater(step_metadata.init_timestamp_nsecs, 0)
        self.assertGreater(step_metadata.commit_timestamp_nsecs, 0)
        self.assertEqual(step_metadata.custom_metadata, {'a': 1, 'b': 2})

    async def test_save_with_failing_array_metadata_store_in_finalize(self):
      """ArrayMetadata validation in CheckpointHandler.finalize()."""

      class IncompleteArrayMetadataSerDeserializer(
          array_metadata_store_lib.SerDeserializer
      ):
        """process 0 metadata should be complete, others should be incomplete.

        Valid data in process 0 will enable successful pytree metadata write.
        Incomplete data in other processes will fail ArrayMetadata validation in
        CheckpointHandler.finalize().
        """

        def serialize(
            self, array_metadatas: Sequence[array_metadata.ArrayMetadata]
        ) -> str:
          if multihost.process_index() == 0:
            return super().serialize(array_metadatas)
          else:
            return super().serialize([array_metadatas[0]])  # incomplete

      array_metadata_store = array_metadata_store_lib.Store(
          ser_deser=IncompleteArrayMetadataSerDeserializer()
      )
      type_handler_registry = copy.deepcopy(
          type_handlers.GLOBAL_TYPE_HANDLER_REGISTRY
      )
      array_handler = type_handlers.ArrayHandler(
          array_metadata_store=array_metadata_store
      )
      type_handler_registry.add(jax.Array, array_handler, override=True)

      class InterceptingValidator(array_metadata_store_lib.Validator):
        """Intercepts ArrayMetadata validation in CheckpointHandler.finalize()."""

        def __init__(self):
          self.error = None

        def validate_all_array_metadatas(self, array_metadatas):
          try:
            super().validate_all_array_metadatas(array_metadatas)
          except ValueError as e:
            self.error = e

        def maybe_raise_error(self):
          if self.error is not None:
            raise self.error

      array_metadata_validator = InterceptingValidator()
      with self.checkpointer(
          PyTreeCheckpointHandler(
              type_handler_registry=type_handler_registry,
              array_metadata_validator=array_metadata_validator,
          )
      ) as checkpointer:
        checkpointer.save(self.directory, self.pytree)

      if multihost.process_index() == 0:
        with self.assertRaisesRegex(
            ValueError,
            'ArrayMetadata Store contains different number of params',
        ):
          array_metadata_validator.maybe_raise_error()
      self.assertIsNotNone(await array_metadata_store.read(self.directory))
