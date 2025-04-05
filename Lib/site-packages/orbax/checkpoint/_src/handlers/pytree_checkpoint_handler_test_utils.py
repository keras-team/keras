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

"""Common test cases for PyTreeCheckpointHandler.

Note: Do not add tests here unless they are not applicable to Pathways.
Prefer adding to the base test library.
"""

# pylint: disable=protected-access, missing-function-docstring

import contextlib
import copy
import dataclasses
import datetime
import functools
import json
import re
from typing import Any, Iterator, List, Optional, Sequence
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
from etils import epath
import flax
import flax.linen as nn
import flax.training.train_state
import jax
from jax import numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental import pjit
import numpy as np
import optax
from orbax.checkpoint import msgpack_utils
from orbax.checkpoint import test_utils
from orbax.checkpoint import transform_utils
from orbax.checkpoint import utils
from orbax.checkpoint._src.futures import future
from orbax.checkpoint._src.handlers import base_pytree_checkpoint_handler
from orbax.checkpoint._src.handlers import proto_checkpoint_handler
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.metadata import array_metadata
from orbax.checkpoint._src.metadata import array_metadata_store as array_metadata_store_lib
from orbax.checkpoint._src.metadata import empty_values
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import replica_slices
from orbax.checkpoint._src.serialization import serialization
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.tree import utils as tree_utils


PyTree = Any
ParamInfo = pytree_checkpoint_handler.ParamInfo
SaveArgs = pytree_checkpoint_handler.SaveArgs
RestoreArgs = pytree_checkpoint_handler.RestoreArgs
ArrayRestoreArgs = pytree_checkpoint_handler.ArrayRestoreArgs
PyTreeSaveArgs = pytree_checkpoint_handler.PyTreeSaveArgs
PyTreeRestoreArgs = pytree_checkpoint_handler.PyTreeRestoreArgs
Transform = transform_utils.Transform
RestoreTransform = transform_utils.RestoreTransform
PyTreeCheckpointHandler = test_utils.PyTreeCheckpointHandler
_SHARDING = '_sharding'
PYTREE_METADATA_FILE = pytree_checkpoint_handler.PYTREE_METADATA_FILE


ARRAY_METADATA_STORE = array_metadata_store_lib.Store()


def _raise_file_not_found_error(*args, **kwargs):
  del args, kwargs
  raise FileNotFoundError()


# Not in common util because we need to eliminate OSS dependency on flax.
def init_flax_model(model):
  params = model.init(jax.random.PRNGKey(0), jnp.ones([8, 8]))
  tx = optax.adamw(learning_rate=0.001)
  state = flax.training.train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx
  )
  return jax.tree.map(np.asarray, state)


class PyTreeCheckpointHandlerTestBase:
  """Base test cases for PyTreeCheckpointHandler."""

  class Test(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):
    """Test class."""

    def setUp(self):
      super().setUp()

      pytree, mesh_tree, axes_tree = test_utils.setup_sharded_pytree()
      self.numpy_pytree = test_utils.setup_pytree()
      self.numpy_pytree.update({'x': 4.5, 'y': 3})
      self.empty_pytree = jax.tree.map(
          lambda x: object(), pytree, is_leaf=test_utils.is_leaf
      )
      self.pytree = pytree
      self.mesh_tree = mesh_tree
      self.axes_tree = axes_tree

      def _create_restore_args(arr, mesh, axes):
        return ArrayRestoreArgs(
            restore_type=type(arr), mesh=mesh, mesh_axes=axes
        )

      self.restore_args = jax.tree.map(
          _create_restore_args, pytree, mesh_tree, axes_tree
      )
      self.directory = epath.Path(
          self.create_tempdir(name='checkpointing_test').full_path
      )
      # TODO: b/365169723 - Add tests for support_rich_types=True.
      self.pytree_metadata_options = tree_metadata.PyTreeMetadataOptions(
          support_rich_types=False
      )

      # default to use_ocdbt=False, so we can test non-ocdbt handler first
      self.handler = self.enter_context(
          self.ocdbt_checkpoint_handler(
              use_ocdbt=False, array_metadata_store=ARRAY_METADATA_STORE
          )
      )
      test_utils.set_tensorstore_driver_for_test()

      test_utils.sync_global_processes(
          'PyTreeCheckpointHandlerTest:setup_complete'
      )

    def tearDown(self):
      test_utils.sync_global_processes(
          'PyTreeCheckpointHandlerTest:tests_complete'
      )
      super().tearDown()

    @contextlib.contextmanager
    def ocdbt_checkpoint_handler(
        self,  # pylint: disable=unused-argument
        use_ocdbt: bool,
        use_zarr3: bool = False,
        pytree_metadata_options: tree_metadata.PyTreeMetadataOptions = (
            tree_metadata.PYTREE_METADATA_OPTIONS
        ),
        array_metadata_store: array_metadata_store_lib.Store | None = (
            ARRAY_METADATA_STORE
        ),
    ):
      """Registers handlers with OCDBT support and resets when done."""
      type_handler_registry = copy.deepcopy(
          type_handlers.GLOBAL_TYPE_HANDLER_REGISTRY
      )
      type_handler_registry.get(jax.Array)._array_metadata_store = (
          array_metadata_store
      )

      handler = PyTreeCheckpointHandler(
          use_ocdbt=use_ocdbt,
          use_zarr3=use_zarr3,
          type_handler_registry=type_handler_registry,
          pytree_metadata_options=pytree_metadata_options,
      )
      try:
        yield handler
      finally:
        handler.close()

    def create_mixed_format_pytree(
        self,
        add: int = 0,
        strings: bool = False,
        key_name: str = 'new_key',
    ) -> PyTree:
      """Creates a PyTree with different leaf types for testing.

      Args:
        add: Adds the specified value to numeric leafs.
        strings: If true, adds string leaves to the tree.
        key_name: Name of the pytree leaf that can be modified.

      Returns:
        PyTree
      """
      pytree = dict(test_utils.setup_pytree(add=add))
      pytree[key_name] = self.pytree
      meshes = jax.tree.map(lambda x: None, pytree, is_leaf=test_utils.is_leaf)
      meshes[key_name] = dict(self.mesh_tree)
      mesh_axes = jax.tree.map(
          lambda x: None, pytree, is_leaf=test_utils.is_leaf
      )
      mesh_axes[key_name] = dict(self.axes_tree)
      if strings:
        pytree['foo'] = 'foo_val'
        pytree['bar'] = 'bar_val'
        meshes['foo'] = None
        meshes['bar'] = None
        mesh_axes['foo'] = None
        mesh_axes['bar'] = None

      def _save_args(arr):
        del arr
        return SaveArgs()

      def _restore_args(arr, mesh, axes):
        if isinstance(arr, jax.Array):
          return ArrayRestoreArgs(
              restore_type=type(arr), mesh=mesh, mesh_axes=axes
          )
        else:
          return RestoreArgs(restore_type=type(arr))

      save_args = jax.tree.map(_save_args, pytree, is_leaf=test_utils.is_leaf)
      restore_args = jax.tree.map(
          _restore_args, pytree, meshes, mesh_axes, is_leaf=test_utils.is_leaf
      )

      return pytree, save_args, restore_args

    def validate_save(
        self,
        path: epath.Path,
        expected: PyTree,
        checkpoint_handler: PyTreeCheckpointHandler,
        save_args: Optional[PyTree] = None,
        restore_args: Optional[PyTree] = None,
    ):
      """Validate save was performed correctly."""
      del save_args
      if restore_args is None:
        restore_args = jax.tree.map(lambda _: RestoreArgs(), expected)
      actual = checkpoint_handler.restore(
          path, args=PyTreeRestoreArgs(restore_args=restore_args)
      )
      test_utils.assert_tree_equal(self, expected, actual)

    def validate_restore(self, expected, actual):
      test_utils.assert_tree_equal(self, expected, actual)

    # TODO(b/301122724) Remove after b/301122724 is implemented.
    def should_validate_metadata(self) -> bool:
      return True

    def validate_metadata(
        self,
        *,
        expected_reference_metadata_tree: PyTree,
        actual_metadata: PyTree,
        pytree_metadata_options: tree_metadata.PyTreeMetadataOptions,
        save_args=None,
        array_metadata_store: array_metadata_store_lib.Store | None,
    ):
      """Validate metadata, provided the original tree that was saved."""
      del save_args
      expected_reference_metadata_tree = tree_metadata.serialize_tree(
          expected_reference_metadata_tree, pytree_metadata_options
      )

      def _metadata(value):
        if empty_values.is_supported_empty_value(
            value, pytree_metadata_options
        ):
          return value
        if isinstance(value, np.ndarray):
          return value_metadata.ArrayMetadata(
              name='',
              directory=None,
              shape=value.shape,
              sharding=None,
              dtype=value.dtype,
              storage=value_metadata.StorageMetadata(
                  chunk_shape=value.shape,
              ),
          )
        if isinstance(value, jax.Array):
          expected_chunk_shape = test_utils.get_expected_chunk_shape(value)
          return value_metadata.ArrayMetadata(
              name='',
              directory=None,
              shape=value.shape,
              sharding=sharding_metadata.from_jax_sharding(value.sharding),
              dtype=value.dtype,
              storage=value_metadata.StorageMetadata(
                  chunk_shape=expected_chunk_shape,
                  write_shape=(
                      expected_chunk_shape
                      if array_metadata_store is not None
                      else None
                  ),
              ),
          )
        if isinstance(value, (float, int)):
          dtype = np.float64 if isinstance(value, float) else np.int64
          return value_metadata.ScalarMetadata(
              name='', directory=None, dtype=dtype
          )  # pytype: disable=wrong-arg-types  # jnp-type
        if isinstance(value, str):
          return value_metadata.StringMetadata(name='', directory=None)
        if isinstance(value, optax.EmptyState):
          return None
        raise ValueError(f'Unrecognized type: {type(value)}.')

      expected_metadata = jax.tree.map(
          _metadata,
          expected_reference_metadata_tree,
          is_leaf=tree_utils.is_empty_or_leaf,
      )
      test_utils.assert_tree_equal(
          self, expected_metadata, actual_metadata.tree
      )

    def test_get_param_names(self):
      param_names = pytree_checkpoint_handler.get_param_names(self.pytree)
      expected = {
          'a': 'a',
          'b': 'b',
          'c': {
              'a': 'c.a',
              'e': 'c.e',
          },
      }
      test_utils.assert_tree_equal(self, expected, param_names)

    def test_save_format(self):
      pytree = {'a': 0, 'c': {'d': np.arange(3), 'e': {'f': 5}}, 'g': 10}
      save_args = jax.tree.map(lambda x: SaveArgs(), pytree)
      self.handler.save(self.directory, args=PyTreeSaveArgs(pytree, save_args))
      fnames = ['a', 'c.d', 'c.e.f', 'g']
      paths = [self.directory / name for name in fnames]
      for p in paths:
        self.assertTrue(p.exists())
        self.assertTrue((p / '.zarray').exists())

    @parameterized.product(use_ocdbt=(True, False))
    def test_save_sharding(self, use_ocdbt: bool):
      if utils.is_pathways_backend():
        self.skipTest('Sharding metadata not present on Pathways.')
      with self.ocdbt_checkpoint_handler(use_ocdbt) as checkpoint_handler:
        pytree, save_args, restore_args = self.create_mixed_format_pytree(
            key_name='mlp/~/linear_0'
        )

        checkpoint_handler.save(
            self.directory, args=PyTreeSaveArgs(pytree, save_args)
        )

        self.validate_save(
            self.directory,
            pytree,
            checkpoint_handler,
            save_args=save_args,
            restore_args=restore_args,
        )

      path = self.directory

      self.assertTrue((path / _SHARDING).exists())
      with open(path / _SHARDING, 'r') as file:
        data = json.load(file)
        self.assertCountEqual(
            data.keys(),
            {
                'bWxwL34vbGluZWFyXzAuYQ==',  # mlp/~/linear_0.a
                'bWxwL34vbGluZWFyXzAuYg==',  # mlp/~/linear_0.b
                'bWxwL34vbGluZWFyXzAuYy5h',  # mlp/~/linear_0.c.a
                'bWxwL34vbGluZWFyXzAuYy5l',  # mlp/~/linear_0.c.e
            },
            None,
        )
        # mlp/~/linear_0.a
        self.assertEqual(
            sharding_metadata.NamedShardingMetadata.from_deserialized_dict(
                json.loads(data['bWxwL34vbGluZWFyXzAuYQ=='])
            ),
            sharding_metadata.NamedShardingMetadata.from_jax_sharding(
                pytree['mlp/~/linear_0']['a'].sharding
            ),
        )
        # mlp/~/linear_0.b
        self.assertEqual(
            sharding_metadata.NamedShardingMetadata.from_deserialized_dict(
                json.loads(data['bWxwL34vbGluZWFyXzAuYg=='])
            ),
            sharding_metadata.NamedShardingMetadata.from_jax_sharding(
                pytree['mlp/~/linear_0']['b'].sharding
            ),
        )
        # mlp/~/linear_0.c.a
        self.assertEqual(
            sharding_metadata.NamedShardingMetadata.from_deserialized_dict(
                json.loads(data['bWxwL34vbGluZWFyXzAuYy5h'])
            ),
            sharding_metadata.NamedShardingMetadata.from_jax_sharding(
                pytree['mlp/~/linear_0']['c']['a'].sharding
            ),
        )
        # mlp/~/linear_0.c.e
        self.assertEqual(
            sharding_metadata.NamedShardingMetadata.from_deserialized_dict(
                json.loads(data['bWxwL34vbGluZWFyXzAuYy5l'])
            ),
            sharding_metadata.NamedShardingMetadata.from_jax_sharding(
                pytree['mlp/~/linear_0']['c']['e'].sharding
            ),
        )

    @parameterized.product(
        use_ocdbt=(True, False),
        array_metadata_store=(None, ARRAY_METADATA_STORE),
    )
    def test_disable_write_sharding_file(
        self,
        use_ocdbt: bool,
        array_metadata_store: array_metadata_store_lib.Store | None,
    ):
      """Test case."""
      array_handler = type_handlers.ArrayHandler(
          enable_write_sharding_file=False,
          array_metadata_store=array_metadata_store,
      )
      ty = jax.Array
      fn = lambda ty: issubclass(ty, jax.Array)
      with test_utils.register_type_handler(ty, array_handler, fn):
        pytree, save_args, restore_args = self.create_mixed_format_pytree()
        with self.ocdbt_checkpoint_handler(
            use_ocdbt=use_ocdbt, array_metadata_store=array_metadata_store
        ) as checkpoint_handler:
          checkpoint_handler.save(
              self.directory, args=PyTreeSaveArgs(pytree, save_args)
          )
          self.validate_save(
              self.directory,
              pytree,
              checkpoint_handler,
              save_args=save_args,
              restore_args=restore_args,
          )
      self.assertFalse((self.directory / _SHARDING).exists())

    def test_sharding_variable_devices(self):
      if utils.is_pathways_backend():
        self.skipTest('Sharding metadata not present on Pathways.')
      mesh_axes = jax.sharding.PartitionSpec(
          'x',
      )
      devices_subset = []
      for idx in range(jax.process_count()):
        for d in jax.devices():
          if d.process_index == idx:
            devices_subset.append(d)
            break
      pytree = {
          'a': test_utils.create_sharded_array(
              np.arange(16),
              jax.sharding.Mesh(devices_subset, ('x',)),
              mesh_axes,
          ),
          'b': test_utils.create_sharded_array(
              np.arange(16), jax.sharding.Mesh(jax.devices(), ('x',)), mesh_axes
          ),
      }

      self.handler.save(self.directory, args=PyTreeSaveArgs(pytree))
      self.assertTrue((self.directory / _SHARDING).exists())
      a_sharding_metadata = sharding_metadata.NamedShardingMetadata(
          shape=np.array([2]),
          axis_names=['x'],
          partition_spec=('x',),
          device_mesh=sharding_metadata.DeviceMetadataMesh.from_jax_mesh(
              jax.sharding.Mesh(devices_subset, ('x',))
          ),
      )
      b_sharding_metadata = sharding_metadata.NamedShardingMetadata(
          shape=np.array([8]),
          axis_names=['x'],
          partition_spec=('x',),
          device_mesh=sharding_metadata.DeviceMetadataMesh.from_jax_mesh(
              jax.sharding.Mesh(jax.devices(), ('x',))
          ),
      )
      self.assertEqual(
          a_sharding_metadata,
          self.handler.metadata(self.directory)['a'].sharding,
      )
      self.assertEqual(
          b_sharding_metadata,
          self.handler.metadata(self.directory)['b'].sharding,
      )

    @parameterized.product(use_ocdbt=(True, False))
    def test_save_main(self, use_ocdbt: bool):
      with self.ocdbt_checkpoint_handler(use_ocdbt) as checkpoint_handler:
        checkpoint_handler.save(
            self.directory, args=PyTreeSaveArgs(self.pytree)
        )
        self.validate_save(
            self.directory,
            self.pytree,
            checkpoint_handler,
            restore_args=self.restore_args,
        )
        self.assertEqual(
            type_handlers.is_ocdbt_checkpoint(self.directory), use_ocdbt
        )

    @parameterized.product(use_ocdbt=(True, False))
    def test_save_keys_with_slashes(self, use_ocdbt: bool):
      with self.ocdbt_checkpoint_handler(use_ocdbt) as checkpoint_handler:
        pytree = {
            'a': np.arange(2),
            'b/c': np.arange(4),
        }
        checkpoint_handler.save(self.directory, args=PyTreeSaveArgs(pytree))
        self.validate_save(
            self.directory,
            pytree,
            checkpoint_handler,
        )

    def test_save_non_sharded(self):

      def _save_args(arr):
        del arr
        return SaveArgs()

      save_args = jax.tree.map(
          _save_args, self.numpy_pytree, is_leaf=test_utils.is_leaf
      )
      restore_args = jax.tree.map(
          lambda arr: RestoreArgs(restore_type=type(arr)),
          self.numpy_pytree,
          is_leaf=test_utils.is_leaf,
      )

      self.handler.save(
          self.directory, args=PyTreeSaveArgs(self.numpy_pytree, save_args)
      )
      self.validate_save(
          self.directory,
          self.numpy_pytree,
          self.handler,
          save_args=save_args,
          restore_args=restore_args,
      )

    @parameterized.product(
        use_ocdbt=(True, False),
        array_metadata_store=(None, ARRAY_METADATA_STORE),
    )
    def test_save_mixed(
        self,
        use_ocdbt: bool,
        array_metadata_store: array_metadata_store_lib.Store | None,
    ):
      """Test case."""
      with self.ocdbt_checkpoint_handler(
          use_ocdbt, array_metadata_store=array_metadata_store
      ) as checkpoint_handler:
        pytree, save_args, restore_args = self.create_mixed_format_pytree(
            strings=True
        )
        checkpoint_handler.save(
            self.directory, args=PyTreeSaveArgs(pytree, save_args)
        )
        self.validate_save(
            self.directory,
            pytree,
            checkpoint_handler,
            save_args=save_args,
            restore_args=restore_args,
        )
        if use_ocdbt:
          self.assertContainsSubset(
              [
                  '_strings.json',
                  'ocdbt.process_0',
                  'ocdbt.process_1',
              ],
              [f.name for f in self.directory.iterdir()],
          )
        else:
          self.assertIn(
              '_strings.json',
              [f.name for f in self.directory.iterdir()],
          )
        if self.should_validate_metadata():
          self.validate_metadata(
              expected_reference_metadata_tree=pytree,
              actual_metadata=checkpoint_handler.metadata(self.directory),
              pytree_metadata_options=self.pytree_metadata_options,
              save_args=save_args,
              array_metadata_store=array_metadata_store,
          )

    @parameterized.product(
        use_ocdbt=(True, False),
        array_metadata_store=(None, ARRAY_METADATA_STORE),
    )
    def test_save_strings(
        self,
        use_ocdbt: bool,
        array_metadata_store: array_metadata_store_lib.Store | None,
    ):
      """Test case."""
      if use_ocdbt and utils.is_pathways_backend():
        self.skipTest('Pathways + OCDBT not supported.')

      with self.ocdbt_checkpoint_handler(
          use_ocdbt, array_metadata_store=array_metadata_store
      ) as checkpoint_handler:
        pytree, _, restore_args = self.create_mixed_format_pytree(strings=True)

        save_args = jax.tree.map(
            lambda _: SaveArgs(), pytree, is_leaf=test_utils.is_leaf
        )
        checkpoint_handler.save(
            self.directory, args=PyTreeSaveArgs(pytree, save_args)
        )
        self.validate_save(
            self.directory,
            pytree,
            checkpoint_handler,
            save_args=save_args,
            restore_args=restore_args,
        )
        if self.should_validate_metadata():
          self.validate_metadata(
              expected_reference_metadata_tree=pytree,
              actual_metadata=checkpoint_handler.metadata(self.directory),
              pytree_metadata_options=self.pytree_metadata_options,
              save_args=save_args,
              array_metadata_store=array_metadata_store,
          )
        self.assertTrue((self.directory / '_strings.json').exists())
        with open(self.directory / '_strings.json') as file:
          data = json.load(file)
          self.assertCountEqual(
              data.keys(),
              {'foo', 'bar'},
              None,
          )
          self.assertEqual(data['foo'], 'foo_val')
          self.assertEqual(data['bar'], 'bar_val')

    def test_cast(self):
      pytree, save_args, restore_args = self.create_mixed_format_pytree()

      def set_dtype(args, dtype):
        args.dtype = dtype
        return args

      save_args = jax.tree.map(
          functools.partial(set_dtype, dtype=np.int16), save_args
      )
      self.handler.save(self.directory, args=PyTreeSaveArgs(pytree, save_args))

      def check_dtype(x, dtype):
        if not utils.is_scalar(x):
          self.assertEqual(x.dtype, dtype)

      # Restore without casting.
      restored = self.handler.restore(
          self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
      )
      jax.tree.map(lambda x: check_dtype(x, np.int16), restored)

      restore_args = jax.tree.map(
          functools.partial(set_dtype, dtype=np.uint32), restore_args
      )
      restored = self.handler.restore(
          self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
      )
      jax.tree.map(lambda x: check_dtype(x, np.uint32), restored)

    def test_cast_scalar(self):
      pytree = {'a': 5, 'b': 1.2}
      restore_args = {
          'a': RestoreArgs(
              restore_type=float
          ),  # pytype: disable=wrong-arg-types  # jnp-type
          'b': RestoreArgs(
              restore_type=int
          ),  # pytype: disable=wrong-arg-types  # jnp-type
      }

      self.handler.save(self.directory, args=PyTreeSaveArgs(pytree))
      restored = self.handler.restore(
          self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
      )
      self.assertIsInstance(restored['a'], float)
      self.assertIsInstance(restored['b'], int)

    def test_restore_type(self):
      pytree = {'a': 5, 'b': 6.1}
      restore_args = {
          'a': RestoreArgs(restore_type=np.ndarray),
          'b': RestoreArgs(restore_type=np.ndarray),
      }

      self.handler.save(self.directory, args=PyTreeSaveArgs(pytree))
      restored = self.handler.restore(
          self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
      )
      self.assertIsInstance(restored['a'], np.ndarray)
      self.assertIsInstance(restored['b'], np.ndarray)

    @parameterized.product(
        use_ocdbt=(True, False),
        use_zarr3=(True, False),
        array_metadata_store=(None, ARRAY_METADATA_STORE),
    )
    def test_restore(
        self,
        use_ocdbt: bool,
        use_zarr3: bool,
        array_metadata_store: array_metadata_store_lib.Store | None,
    ):
      """Test case."""
      with self.ocdbt_checkpoint_handler(
          use_ocdbt,
          use_zarr3=use_zarr3,
          array_metadata_store=array_metadata_store,
      ) as checkpoint_handler:
        checkpoint_handler.save(
            self.directory, args=PyTreeSaveArgs(self.pytree)
        )
        restored = checkpoint_handler.restore(
            self.directory,
            args=PyTreeRestoreArgs(restore_args=self.restore_args),
        )
        self.validate_restore(self.pytree, restored)
        if self.should_validate_metadata():
          self.validate_metadata(
              expected_reference_metadata_tree=self.pytree,
              actual_metadata=checkpoint_handler.metadata(self.directory),
              pytree_metadata_options=self.pytree_metadata_options,
              array_metadata_store=array_metadata_store,
          )

    @parameterized.product(use_ocdbt=(True, False))
    def test_restore_reverse_mesh(self, use_ocdbt: bool):
      if use_ocdbt and utils.is_pathways_backend():
        self.skipTest('Pathways + OCDBT not supported.')
      with self.ocdbt_checkpoint_handler(use_ocdbt) as checkpoint_handler:
        pytree, mesh_tree, axes_tree = test_utils.setup_sharded_pytree(
            reverse_devices=True
        )

        def _create_restore_args(arr, mesh, axes):
          return ArrayRestoreArgs(
              restore_type=type(arr), mesh=mesh, mesh_axes=axes
          )

        restore_args = jax.tree.map(
            _create_restore_args, pytree, mesh_tree, axes_tree
        )

        checkpoint_handler.save(self.directory, args=PyTreeSaveArgs(pytree))
        restored = checkpoint_handler.restore(
            self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
        )
        self.validate_restore(pytree, restored)

    def test_restore_with_sharding(self):
      if utils.is_pathways_backend():
        self.skipTest('Sharding metadata not present on Pathways.')

      jitted_pytree = jax.tree.map(
          jax.experimental.pjit.pjit(lambda x: x * 2), self.pytree
      )
      self.handler.save(self.directory, args=PyTreeSaveArgs(jitted_pytree))

      restore_args = jax.tree.map(
          lambda arr: ArrayRestoreArgs(sharding=arr.sharding), jitted_pytree
      )
      restored = self.handler.restore(
          self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
      )
      self.validate_restore(jitted_pytree, restored)

    def test_restore_with_sharding_metadata(self):
      if utils.is_pathways_backend():
        self.skipTest('Sharding metadata not present on Pathways.')

      jitted_pytree = jax.tree.map(
          jax.experimental.pjit.pjit(lambda x: x * 2), self.pytree
      )
      self.handler.save(self.directory, args=PyTreeSaveArgs(jitted_pytree))

      restore_args = jax.tree.map(
          lambda arr: ArrayRestoreArgs(
              sharding=sharding_metadata.from_jax_sharding(arr.sharding)
          ),
          jitted_pytree,
      )
      restored = self.handler.restore(
          self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
      )
      self.validate_restore(jitted_pytree, restored)

    def test_restore_with_sharding_without_sharding_arg(self):
      if utils.is_pathways_backend():
        self.skipTest('Sharding metadata not present on Pathways.')

      self.handler.save(self.directory, args=PyTreeSaveArgs(self.pytree))

      restore_args = jax.tree.map(lambda arr: ArrayRestoreArgs(), self.pytree)

      self.assertTrue((self.directory / _SHARDING).exists())
      restored_without_sharding_args = self.handler.restore(
          self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
      )
      self.validate_restore(self.pytree, restored_without_sharding_args)

      restored_without_restore_args = self.handler.restore(self.directory)
      self.validate_restore(self.pytree, restored_without_restore_args)

    def test_restore_different(self):
      for step in [0, 1]:
        directory = self.directory / str(step)
        if multihost.process_index() == 0:
          directory.mkdir()
        test_utils.sync_global_processes(
            'PyTreeCheckpointHandlerTest:test_restore_different_mkdir'
        )

        pytree, save_args, restore_args = self.create_mixed_format_pytree(
            add=step
        )
        self.handler.save(directory, args=PyTreeSaveArgs(pytree, save_args))

        restored = self.handler.restore(
            directory, args=PyTreeRestoreArgs(restore_args=restore_args)
        )
        self.validate_restore(pytree, restored)

    def test_restore_missing_checkpoint(self):
      directory = self.directory / 'nothing'
      with self.assertRaises(FileNotFoundError):
        self.handler.restore(directory)

    @parameterized.product(
        use_ocdbt=(True, False),
        array_metadata_store=(None, ARRAY_METADATA_STORE),
    )
    def test_flax_model(
        self,
        use_ocdbt: bool,
        array_metadata_store: array_metadata_store_lib.Store | None,
    ):
      """Test case."""

      @flax.struct.dataclass
      class Params(flax.struct.PyTreeNode):
        params: Any
        opt_state: Any

      @jax.jit
      def make_params():
        return Params(
            params=self.numpy_pytree,
            opt_state=(optax.EmptyState(), optax.EmptyState()),
        )

      params = make_params()
      empty_params = jax.eval_shape(make_params)
      mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('devices',))
      mesh_axes = jax.sharding.PartitionSpec()
      params = jax.tree.map(
          lambda arr: test_utils.create_sharded_array(arr, mesh, mesh_axes),
          params,
      )
      restore_args = jax.tree.map(
          lambda _: ArrayRestoreArgs(mesh=mesh, mesh_axes=mesh_axes), params
      )

      save_args = jax.tree.map(lambda _: SaveArgs(), params)
      with self.ocdbt_checkpoint_handler(
          use_ocdbt, array_metadata_store=array_metadata_store
      ) as checkpoint_handler:
        checkpoint_handler.save(
            self.directory, args=PyTreeSaveArgs(params, save_args)
        )
        restored = checkpoint_handler.restore(
            self.directory,
            args=PyTreeRestoreArgs(
                item=empty_params, restore_args=restore_args
            ),
        )
        self.validate_restore(params, restored)
        if self.should_validate_metadata():
          self.validate_metadata(
              expected_reference_metadata_tree=params,
              actual_metadata=checkpoint_handler.metadata(self.directory),
              pytree_metadata_options=self.pytree_metadata_options,
              save_args=save_args,
              array_metadata_store=array_metadata_store,
          )

    @parameterized.product(
        use_ocdbt=(
            True,
            False,
        ),
        data=(
            {},
            {'a': {}, 'b': 3},
            [1, {}, 2],
            None,
            {'a': None, 'b': 3},
            [1, None, 2],
            [],
            [1, [], 2],
            {'a': [], 'b': 3},
        ),
        save_args=(
            None,
            SaveArgs(),
        ),
        array_metadata_store=(None, ARRAY_METADATA_STORE),
    )
    def test_empty_data(
        self,
        use_ocdbt: bool,
        data: Any,
        save_args: SaveArgs | None,
        array_metadata_store: array_metadata_store_lib.Store | None,
    ):
      """Test case."""
      if save_args is None:
        save_args_tree = None
      else:
        save_args_tree = jax.tree.map(
            lambda _: save_args, data, is_leaf=tree_utils.is_empty_or_leaf
        )
      with self.ocdbt_checkpoint_handler(
          use_ocdbt, array_metadata_store=array_metadata_store
      ) as checkpoint_handler:
        if not data:
          with self.assertRaisesRegex(ValueError, 'Found empty item'):
            checkpoint_handler.save(
                self.directory,
                args=PyTreeSaveArgs(data, save_args=save_args_tree),
            )
          return

        checkpoint_handler.save(
            self.directory, args=PyTreeSaveArgs(data, save_args=save_args_tree)
        )
        restored = checkpoint_handler.restore(self.directory)
        self.assertEqual(restored, data)

        self.validate_metadata(
            expected_reference_metadata_tree=data,
            actual_metadata=checkpoint_handler.metadata(self.directory),
            pytree_metadata_options=self.pytree_metadata_options,
            save_args=save_args_tree,
            array_metadata_store=array_metadata_store,
        )

    @parameterized.product(
        use_ocdbt=(True, False),
        array_metadata_store=(None, ARRAY_METADATA_STORE),
    )
    def test_list(
        self,
        use_ocdbt: bool,
        array_metadata_store: array_metadata_store_lib.Store | None,
    ):
      """Test case."""
      item = [1, 2, 5, 6]
      with self.ocdbt_checkpoint_handler(
          use_ocdbt=use_ocdbt, array_metadata_store=array_metadata_store
      ) as checkpoint_handler:
        checkpoint_handler.save(self.directory, args=PyTreeSaveArgs(item))
        restore_args = jax.tree.map(
            lambda _: RestoreArgs(restore_type=int), item
        )
        restored = checkpoint_handler.restore(
            self.directory,
            args=PyTreeRestoreArgs(
                item=[0, 0, 0, 0], restore_args=restore_args
            ),
        )
        self.assertListEqual(restored, item)
        self.validate_metadata(
            expected_reference_metadata_tree=[0, 0, 0, 0],
            actual_metadata=checkpoint_handler.metadata(self.directory),
            pytree_metadata_options=self.pytree_metadata_options,
            array_metadata_store=array_metadata_store,
        )

        restored = checkpoint_handler.restore(self.directory)
        self.assertListEqual(
            restored,
            [
                np.asarray([1]),
                np.asarray([2]),
                np.asarray([5]),
                np.asarray([6]),
            ],
        )

    def test_only_aggregation(self):
      tree = {
          'a': 1,
          'b': 2,
          'c': {
              'd': np.arange(3),
          },
      }
      msgpack = msgpack_utils.msgpack_serialize(tree)
      if multihost.process_index() == 0:
        (self.directory / 'checkpoint').write_bytes(msgpack)
      test_utils.sync_global_processes(
          'PyTreeCheckpointHandlerTest:write_flax_checkpoint'
      )

      restore_args = jax.tree.map(
          lambda arr: RestoreArgs(restore_type=type(arr)),
          tree,
          is_leaf=test_utils.is_leaf,
      )
      restored = self.handler.restore(
          self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
      )
      self.validate_restore(tree, restored)

    def test_transform(self):
      pytree = self.pytree
      pytree['int_key'] = 5
      pytree['float_key'] = 2.5
      self.handler.save(self.directory, args=PyTreeSaveArgs(pytree))

      def _pjitted_value_fn(x):
        return test_utils.apply_function([x], lambda y: y * 2 + 3)[0]

      replicated_sharding = jax.sharding.NamedSharding(
          jax.sharding.Mesh(jax.devices(), ('x',)),
          jax.sharding.PartitionSpec(
              None,
          ),
      )

      def _pjitted_add(key, tree, args):
        del key
        del args
        return pjit.pjit(lambda a, b: a + b, out_shardings=replicated_sharding)(
            tree['a'], tree['b']
        )

      def _split(key, tree, args):
        if key == 'split1':
          result = np.asarray(tree['float_key'] * 2)
          result = jax.make_array_from_callback(
              result.shape, args.sharding, lambda idx: result[idx]
          )
        else:
          self.assertEqual(args.restore_type, np.ndarray)
          result = np.asarray(tree['float_key'] * 4)
        return result

      reference_item = {
          'x': 0,
          'y': 0,
          'c': {
              'a': 0,
          },
          'z': 100,  # All values in this tree are unused except 'z'.
          'int_key': 0,
          'added': 0,
          'split1': 0,
          'split2': 0,
          'fallback': np.arange(4),
      }
      restore_args = {
          'x': self.restore_args['a'],
          'y': self.restore_args['c']['e'],
          'c': {
              'a': self.restore_args['c']['a'],
          },
          'z': RestoreArgs(restore_type=int),
          'int_key': RestoreArgs(restore_type=int),
          'split1': ArrayRestoreArgs(
              sharding=jax.sharding.NamedSharding(
                  jax.sharding.Mesh(jax.devices(), ('x',)),
                  jax.sharding.PartitionSpec(None),
              )
          ),
          'split2': RestoreArgs(restore_type=np.ndarray),
          'added': RestoreArgs(restore_type=None),
          'fallback': RestoreArgs(restore_type=None),
      }
      expected = {
          'x': _pjitted_value_fn(pytree['a']),
          'y': pytree['c']['e'],
          'c': {
              'a': _pjitted_value_fn(pytree['c']['a']),
          },
          'z': 100,
          'int_key': 7,
          'added': test_utils.create_sharded_array(
              np.arange(8) + np.arange(8) * 2,
              replicated_sharding.mesh,
              replicated_sharding.spec,
          ),
          'split1': jax.make_array_from_callback(
              (),
              restore_args['split1'].sharding,
              lambda idx: np.asarray(5.0)[idx],
          ),
          'split2': np.asarray(10.0),
          'fallback': np.arange(4),
      }

      transforms = {
          'x': Transform(original_key='a', value_fn=_pjitted_value_fn),
          'y': Transform(original_key='c/e'),
          'c': {'a': Transform(value_fn=_pjitted_value_fn)},
          'int_key': Transform(value_fn=lambda x: x + 2),
          'added': RestoreTransform(
              multi_value_fn=_pjitted_add,
              multi_value_fn_input_args={
                  'a': ArrayRestoreArgs(
                      sharding=replicated_sharding, strict=False
                  ),
                  'b': ArrayRestoreArgs(
                      sharding=replicated_sharding,
                      global_shape=(8,),
                      strict=False,
                  ),
              },
          ),
          'split1': RestoreTransform(
              multi_value_fn=_split,
              multi_value_fn_input_args={
                  'float_key': RestoreArgs(restore_type=float)
              },
          ),
          'split2': RestoreTransform(
              multi_value_fn=_split,
              multi_value_fn_input_args={
                  'float_key': RestoreArgs(restore_type=float)
              },
          ),
          'fallback': Transform(use_fallback=True),
      }

      restored = self.handler.restore(
          self.directory,
          args=PyTreeRestoreArgs(
              item=reference_item,
              restore_args=restore_args,
              transforms=transforms,
          ),
      )
      self.validate_restore(expected, restored)

    @parameterized.product(
        use_ocdbt=(True, False),
        array_metadata_store=(None, ARRAY_METADATA_STORE),
    )
    def test_partial_restore(
        self,
        use_ocdbt: bool,
        array_metadata_store: array_metadata_store_lib.Store | None,
    ):
      """Test case."""
      checkpoint_handler = self.enter_context(
          self.ocdbt_checkpoint_handler(
              use_ocdbt=use_ocdbt, array_metadata_store=array_metadata_store
          )
      )
      checkpoint_handler.save(self.directory, args=PyTreeSaveArgs(self.pytree))

      reference_item = {
          'a': 0,
          'c': {
              'a': 0,
          },
      }
      restore_args = {
          'a': self.restore_args['a'],
          'c': {
              'a': self.restore_args['c']['a'],
          },
      }
      expected = {
          'a': self.pytree['a'],
          'c': {
              'a': self.pytree['c']['a'],
          },
      }
      transforms = {}

      # Ensure that no more parameters are being restored than the ones that are
      # strictly needed.
      with mock.patch.object(
          type_handlers.ArrayHandler, 'deserialize', autospec=True
      ) as mock_deserialize:
        checkpoint_handler.restore(
            self.directory,
            args=PyTreeRestoreArgs(
                item=reference_item,
                restore_args=restore_args,
                transforms=transforms,
            ),
        )
        mock_deserialize.assert_called_once()
        mock_args, _ = mock_deserialize.call_args
        _, infos, args = mock_args
        self.assertLen(infos, 2)
        self.assertLen(args, 2)

      restored = checkpoint_handler.restore(
          self.directory,
          args=PyTreeRestoreArgs(
              item=reference_item,
              restore_args=restore_args,
              transforms=transforms,
          ),
      )
      self.validate_restore(expected, restored)
      self.validate_metadata(
          expected_reference_metadata_tree=self.pytree,
          actual_metadata=checkpoint_handler.metadata(self.directory),
          pytree_metadata_options=self.pytree_metadata_options,
          array_metadata_store=array_metadata_store,
      )

    @parameterized.product(use_ocdbt=(True, False))
    def test_flax_transform(self, use_ocdbt: bool):

      with self.ocdbt_checkpoint_handler(use_ocdbt) as checkpoint_handler:

        class SmallModel(nn.Module):
          """Small Flax model."""

          @nn.compact
          def __call__(self, x):
            x = x.reshape((x.shape[0], -1))
            x = nn.Dense(features=8)(x)
            x = nn.sigmoid(x)
            x = nn.Dense(features=8)(x)
            return x

        old_state = init_flax_model(SmallModel())
        checkpoint_handler.save(self.directory, args=PyTreeSaveArgs(old_state))

        class LargeModel(nn.Module):
          """Large Flax model."""

          @nn.compact
          def __call__(self, x):
            x = x.reshape((x.shape[0], -1))  # flatten
            x = nn.Dense(features=16)(x)
            x = nn.sigmoid(x)
            x = nn.Dense(features=8)(x)
            x = nn.sigmoid(x)
            x = nn.Dense(features=8)(x)
            x = nn.sigmoid(x)
            x = nn.Dense(features=4)(x)
            return x

        new_state = init_flax_model(LargeModel())

        value_fn = lambda x: x * 2 + 1

        def _add(key, tree, args):
          del args
          flat_tree = tree_utils.to_flat_dict(tree, sep='/')
          match = re.fullmatch('(.*)Dense_2(.*)', key)
          assert match
          k0 = match.expand(r'\1Dense_0\2')
          k1 = match.expand(r'\1Dense_1\2')
          return flat_tree[k0] + flat_tree[k1]

        transformations = {
            # LargeModel layer 0 is a newly inserted layer, thus
            # use_fallback=True.
            r'(.*)Dense_0(.*)': Transform(use_fallback=True),
            # SmallModel layer 0 maps to LargeModel layer 1
            r'(.*)Dense_1(.*)': Transform(
                original_key=r'\1Dense_0\2', value_fn=value_fn
            ),
            # SmallModel layer 0, 1 maps to LargeModel layer 2
            r'(.*)Dense_2(.*)': RestoreTransform(
                multi_value_fn=_add,
                multi_value_fn_input_args={r'.*Dense_[0,1].*': RestoreArgs()},
            ),
        }  # Note: LargeModel layer 3 is newly added.

        restore_args = jax.tree.map(
            lambda arr: RestoreArgs(restore_type=type(arr)), new_state
        )
        restored_state = checkpoint_handler.restore(
            self.directory,
            args=PyTreeRestoreArgs(
                item=new_state,
                restore_args=restore_args,
                transforms=transformations,
            ),
        )

        # Construct expected tree
        old_flat_dict = tree_utils.to_flat_dict(old_state, sep='/')
        new_flat_dict = tree_utils.to_flat_dict(new_state, sep='/')
        expected_flat_dict = {}
        for k, v in new_flat_dict.items():
          if 'Dense_1' in k:
            expected_flat_dict[k] = value_fn(
                old_flat_dict[k.replace('Dense_1', 'Dense_0')]
            )
          elif 'Dense_2' in k:
            expected_flat_dict[k] = (
                old_flat_dict[k.replace('Dense_2', 'Dense_1')]
                + old_flat_dict[k.replace('Dense_2', 'Dense_0')]
            )
          elif 'Dense_' in k:  # layers in new, but not old.
            expected_flat_dict[k] = v
          else:  # extra keys in both, expected is the old value
            expected_flat_dict[k] = old_flat_dict[k]

        expected_state = tree_utils.from_flat_dict(
            expected_flat_dict, target=new_state, sep='/'
        )
        restored_state = jax.tree.map(np.asarray, restored_state)

        test_utils.assert_tree_equal(self, expected_state, restored_state)

    def test_no_metadata_file(self):
      expected = jax.tree.map(test_utils.replicate_sharded_array, self.pytree)
      expected = jax.tree.map(
          lambda x: np.asarray(x.addressable_data(0)), expected
      )
      self.handler.save(self.directory, args=PyTreeSaveArgs(expected))
      metadata_file = self.directory / PYTREE_METADATA_FILE
      if multihost.process_index() == 0:
        self.assertTrue(metadata_file.exists())
        metadata_file.unlink()
      test_utils.sync_global_processes('delete_metadata_file')
      self.assertFalse(metadata_file.exists())
      with self.assertRaises(FileNotFoundError):
        self.validate_restore(expected, self.handler.metadata(self.directory))

    @parameterized.product(
        use_ocdbt=(False,),
        array_metadata_store=(None, ARRAY_METADATA_STORE),
    )
    def test_override_tensorstore_metadata_name_sharded_arrays(
        self,
        use_ocdbt: bool,
        array_metadata_store: array_metadata_store_lib.Store | None,
    ):
      """Test case."""
      metadata_key = 'custom_zarray'
      ty = jax.Array
      array_handler = type_handlers.ArrayHandler(
          metadata_key=metadata_key, array_metadata_store=array_metadata_store
      )
      fn = lambda ty: issubclass(ty, jax.Array)
      pytree = self.pytree

      with test_utils.register_type_handler(ty, array_handler, fn):
        with self.ocdbt_checkpoint_handler(
            use_ocdbt=use_ocdbt, array_metadata_store=array_metadata_store
        ) as checkpoint_handler:
          checkpoint_handler.save(self.directory, args=PyTreeSaveArgs(pytree))
          param_names = pytree_checkpoint_handler.get_param_names(pytree)
          paths = jax.tree.map(lambda n: self.directory / n, param_names)

          def check_path(p):
            self.assertTrue(
                (p / metadata_key).exists(),
                msg=f'{p / metadata_key} should exist.',
            )
            self.assertFalse(
                (p / '.zarray').exists(),
                msg=f'{p / ".zarray"} should not exist.',
            )

          jax.tree.map(check_path, paths)

    @parameterized.product(
        use_ocdbt=(False,),
        array_metadata_store=(None, ARRAY_METADATA_STORE),
    )
    def test_override_tensorstore_metadata_name(
        self,
        use_ocdbt: bool,
        array_metadata_store: array_metadata_store_lib.Store | None,
    ):
      """Test case."""
      metadata_key = 'custom_zarray'
      ty = np.ndarray
      np_array_handler = type_handlers.NumpyHandler(metadata_key=metadata_key)
      fn = lambda ty: issubclass(ty, np.ndarray)
      pytree = self.numpy_pytree
      del pytree['x']
      del pytree['y']

      with test_utils.register_type_handler(ty, np_array_handler, fn):
        with self.ocdbt_checkpoint_handler(
            use_ocdbt, array_metadata_store=array_metadata_store
        ) as checkpoint_handler:
          checkpoint_handler.save(self.directory, args=PyTreeSaveArgs(pytree))
          param_names = pytree_checkpoint_handler.get_param_names(pytree)
          paths = jax.tree.map(lambda n: self.directory / n, param_names)

          def check_path(p):
            self.assertTrue(
                (p / metadata_key).exists(),
                msg=f'{p / metadata_key} should exist.',
            )
            self.assertFalse(
                (p / '.zarray').exists(),
                msg=f'{p / ".zarray"} should not exist.',
            )

          jax.tree.map(check_path, paths)

    @parameterized.product(with_metadata=(True, False))
    def test_aggregate_all(self, with_metadata: bool):
      path = epath.Path(absltest.get_default_test_srcdir()) / epath.Path(
          '/orbax/checkpoint/testing/aggregate_all.checkpoint'
      )
      files = [p.name for p in path.iterdir()]
      self.assertSameElements(
          [
              PYTREE_METADATA_FILE,
              pytree_checkpoint_handler._CHECKPOINT_FILE,
          ],
          files,
      )

      handler = PyTreeCheckpointHandler()
      if not with_metadata:
        handler._read_metadata_file = _raise_file_not_found_error
      restored = handler.restore(path)
      self.validate_restore(test_utils.setup_pytree(), restored)
      if with_metadata:
        metadata = handler.metadata(path)
        jax.tree.map(
            lambda x: self.assertIsInstance(x, value_metadata.Metadata),
            metadata,
        )

    @parameterized.product(with_metadata=(True, False))
    def test_aggregate_some(self, with_metadata: bool):
      path = epath.Path(absltest.get_default_test_srcdir()) / epath.Path(
          '/orbax/checkpoint/testing/aggregate_some.checkpoint'
      )
      files = [p.name for p in path.iterdir()]
      self.assertSameElements(
          [
              PYTREE_METADATA_FILE,
              pytree_checkpoint_handler._CHECKPOINT_FILE,
              'a',
              'b',
          ],
          files,
      )
      handler = PyTreeCheckpointHandler()
      if not with_metadata:
        handler._read_metadata_file = _raise_file_not_found_error
      restored = handler.restore(path)
      self.validate_restore(test_utils.setup_pytree(), restored)
      if with_metadata:
        metadata = handler.metadata(path)
        jax.tree.map(
            lambda x: self.assertIsInstance(x, value_metadata.Metadata),
            metadata,
        )

    def test_restore_aggregated_as_jax_array(self):
      path = epath.Path(absltest.get_default_test_srcdir()) / epath.Path(
          '/orbax/checkpoint/testing/aggregate_some.checkpoint'
      )
      expected, mesh, axes = test_utils.setup_sharded_pytree()
      restore_args = jax.tree.map(
          lambda v, m, a: ArrayRestoreArgs(
              dtype=v.dtype,
              global_shape=v.shape,
              sharding=jax.sharding.NamedSharding(m, a),
          ),
          expected,
          mesh,
          axes,
      )
      restored = self.handler.restore(
          path, args=PyTreeRestoreArgs(restore_args=restore_args)
      )
      self.validate_restore(expected, restored)

    @parameterized.parameters((True,), (False,))
    def test_reshape_padding(self, strict: bool):
      mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('x',))
      axes = jax.sharding.PartitionSpec(
          'x',
      )
      dtype = np.float32
      tree = {
          'x': test_utils.create_sharded_array(
              np.arange(8, dtype=dtype), mesh, axes
          )
      }
      restore_args = {
          'x': ArrayRestoreArgs(
              mesh=mesh, mesh_axes=axes, global_shape=(16,), strict=strict
          )
      }
      self.handler.save(self.directory, args=PyTreeSaveArgs(tree))
      if strict:
        with self.assertRaises(BaseException):
          self.handler.restore(
              self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
          )
      else:
        restored = self.handler.restore(
            self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
        )
        expected = {
            'x': test_utils.create_sharded_array(
                np.concatenate(
                    (np.arange(8, dtype=dtype), np.zeros(8, dtype=dtype))
                ),
                mesh,
                axes,
            )
        }
        self.validate_restore(expected, restored)

    @parameterized.parameters((True,), (False,))
    def test_reshape_truncate(self, strict: bool):
      mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('x',))
      axes = jax.sharding.PartitionSpec(
          'x',
      )
      tree = {'x': test_utils.create_sharded_array(np.arange(16), mesh, axes)}
      restore_args = {
          'x': ArrayRestoreArgs(
              mesh=mesh, mesh_axes=axes, global_shape=(8,), strict=strict
          )
      }
      self.handler.save(self.directory, args=PyTreeSaveArgs(tree))

      if strict:
        with self.assertRaises(BaseException):
          self.handler.restore(
              self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
          )
      else:
        restored = self.handler.restore(
            self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
        )
        expected = {
            'x': test_utils.create_sharded_array(np.arange(8), mesh, axes)
        }
        self.validate_restore(expected, restored)

    @parameterized.parameters(
        (jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec(('x', 'y'))),
        (jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec(('y', 'x'))),
        (jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec(('x',))),
        (jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec(('y',))),
        (jax.sharding.PartitionSpec(('x', 'y')), jax.sharding.PartitionSpec()),
        (
            jax.sharding.PartitionSpec(('x', 'y')),
            jax.sharding.PartitionSpec(('x',)),
        ),
        (
            jax.sharding.PartitionSpec(('x', 'y')),
            jax.sharding.PartitionSpec(('y',)),
        ),
        (
            jax.sharding.PartitionSpec(('x', 'y')),
            jax.sharding.PartitionSpec(('y', 'x')),
        ),
        (
            jax.sharding.PartitionSpec(('x',)),
            jax.sharding.PartitionSpec(('y',)),
        ),
    )
    def test_reshard(self, save_spec, restore_spec):
      devices = jax.devices()
      len_devices = len(devices)
      if len_devices < 4:
        return

      mesh = jax.sharding.Mesh(
          mesh_utils.create_device_mesh((4, len_devices // 4)), ('x', 'y')
      )
      tree = {
          'x': test_utils.create_sharded_array(
              np.arange(len_devices), mesh, save_spec
          )
      }
      restore_args = {'x': ArrayRestoreArgs(mesh=mesh, mesh_axes=restore_spec)}

      self.handler.save(self.directory, args=PyTreeSaveArgs(tree))
      restored = self.handler.restore(
          self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
      )
      expected = {
          'x': test_utils.create_sharded_array(
              np.arange(len_devices), mesh, restore_spec
          )
      }
      self.validate_restore(expected, restored)

    def test_restore_non_ocdbt(self):
      with self.ocdbt_checkpoint_handler(False) as checkpoint_handler:
        checkpoint_handler.save(
            self.directory, args=PyTreeSaveArgs(self.pytree)
        )
        self.assertFalse(type_handlers.is_ocdbt_checkpoint(self.directory))
      with self.ocdbt_checkpoint_handler(True) as checkpoint_handler:
        restored = checkpoint_handler.restore(
            self.directory,
            args=PyTreeRestoreArgs(restore_args=self.restore_args),
        )
        self.validate_restore(self.pytree, restored)

    def test_restore_non_ocdbt_mixed(self):
      pytree, save_args, restore_args = self.create_mixed_format_pytree(
          strings=True
      )
      with self.ocdbt_checkpoint_handler(False) as checkpoint_handler:
        checkpoint_handler.save(
            self.directory, args=PyTreeSaveArgs(pytree, save_args)
        )
        self.assertFalse(type_handlers.is_ocdbt_checkpoint(self.directory))
      with self.ocdbt_checkpoint_handler(True) as checkpoint_handler:
        restored = checkpoint_handler.restore(
            self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
        )
        self.validate_restore(pytree, restored)

    def test_check_zarray(self):
      self.handler.save(self.directory, args=PyTreeSaveArgs(self.pytree))
      zarr_path = self.directory / 'a' / '.zarray'
      zarr_path.unlink(missing_ok=True)
      test_utils.sync_global_processes(
          'PyTreeCheckpointHandlerTest:delete_zarray'
      )
      self.assertFalse(zarr_path.exists())
      error_type = (
          ValueError if utils.is_pathways_backend() else FileNotFoundError
      )
      with self.assertRaises(error_type):
        self.handler.restore(
            self.directory,
            args=PyTreeRestoreArgs(restore_args=self.restore_args),
        )

    def test_without_restore_args(self):
      arr = test_utils.create_sharded_array(
          np.arange(8),
          jax.sharding.Mesh(jax.devices(), ('x',)),
          jax.sharding.PartitionSpec('x'),
      )
      pytree = [arr]
      self.handler.save(self.directory, args=PyTreeSaveArgs(pytree))
      restored = self.handler.restore(self.directory)
      self.validate_restore(pytree, restored)

    @parameterized.product(use_ocdbt=(True, False))
    def test_masked_shape_dtype_struct(self, use_ocdbt: bool):

      def _should_mask(keypath):
        return keypath[0].key == 'a' or (
            keypath[0].key == 'c' and keypath[1].key == 'e'
        )

      def _mask(keypath, x):
        return optax.MaskedNode() if _should_mask(keypath) else x

      def _none(keypath, x):
        return None if _should_mask(keypath) else x

      masked_tree = jax.tree_util.tree_map_with_path(_mask, self.pytree)
      expected = jax.tree_util.tree_map_with_path(_none, self.pytree)

      with self.ocdbt_checkpoint_handler(use_ocdbt) as handler:
        handler.save(self.directory, args=PyTreeSaveArgs(masked_tree))
        if use_ocdbt:
          self.assertTrue(type_handlers.is_ocdbt_checkpoint(self.directory))

        # Restore it with state which was given before applying masking.
        restored = handler.restore(
            self.directory,
            args=PyTreeRestoreArgs(
                jax.tree.map(utils.to_shape_dtype_struct, self.pytree),
                restore_args=self.restore_args,
            ),
        )
        test_utils.assert_tree_equal(self, expected, restored)

        # Restore it with state after applying masking to it.
        restored = handler.restore(
            self.directory,
            args=PyTreeRestoreArgs(
                jax.tree.map(utils.to_shape_dtype_struct, masked_tree),
                restore_args=self.restore_args,
            ),
        )
        test_utils.assert_tree_equal(self, expected, restored)

        # Restore it without any state.
        restored = handler.restore(
            self.directory,
            args=PyTreeRestoreArgs(restore_args=self.restore_args),
        )
        test_utils.assert_tree_equal(self, expected, restored)

    def test_finalize(self):
      with self.ocdbt_checkpoint_handler(True) as checkpoint_handler:
        checkpoint_handler.save(
            self.directory, args=PyTreeSaveArgs(self.pytree)
        )
        process_index = multihost.process_index()
        process_dir = (
            self.directory / f'{ts_utils.PROCESS_SUBDIR_PREFIX}{process_index}'
        )
        self.assertTrue(process_dir.exists())
        self.assertTrue(process_dir.is_dir())
        self.assertTrue(type_handlers.is_ocdbt_checkpoint(self.directory))

    @parameterized.product(use_ocdbt=(True, False))
    def test_unregistered_types(self, use_ocdbt: bool):
      data = {'uncheckpointable_field': datetime.timedelta(seconds=5)}
      with self.ocdbt_checkpoint_handler(use_ocdbt) as checkpoint_handler:
        with self.assertRaisesRegex(ValueError, 'TypeHandler lookup failed'):
          checkpoint_handler.save(
              self.directory,
              args=PyTreeSaveArgs(
                  data,
                  save_args={'uncheckpointable_field': SaveArgs()},
              ),
          )

    @parameterized.product(
        target_data_file_size=[
            50 * 1024,  # 50KB
            10 * 1024,  # 10KB
            0,
            None,
        ],
        chunk_byte_size=[
            None,  # unspecified
            5 * 1024,  # 5KB
            100 * 1024,  # greater than target_data_file_size
        ],
        use_zarr3=[True, False],
        patch_default_ocdbt_data_file_size=[True, False],
    )
    def test_ocdbt_target_data_file_size(
        self,
        target_data_file_size,
        chunk_byte_size,
        use_zarr3,
        patch_default_ocdbt_data_file_size,
    ):
      """Test ocdbt_target_data_file_size."""
      array_len = 16 * 1024  # ~ 64KB of float data
      custom_pytree = {
          'a': np.arange(array_len, dtype=np.int32),
          'b': np.arange(array_len * 2, dtype=np.float32),
          'c': {
              'a': np.arange(array_len, dtype=np.int32).reshape(
                  2, array_len // 2
              ),
              'e': np.arange(array_len * 2, dtype=np.float32).reshape(
                  2, array_len
              ),
          },
      }

      pytree, mesh_tree, axes_tree = test_utils.setup_sharded_pytree(
          custom_pytree
      )

      save_args = jax.tree.map(
          lambda x: type_handlers.SaveArgs(chunk_byte_size=chunk_byte_size),
          pytree,
      )

      new_ocdbt_target_data_file_size = (
          1024
          if patch_default_ocdbt_data_file_size
          else ts_utils._DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
      )
      with mock.patch.object(
          ts_utils,
          '_DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE',
          new_ocdbt_target_data_file_size,
      ):
        if patch_default_ocdbt_data_file_size:
          assert ts_utils._DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE == 1024
        with self.ocdbt_checkpoint_handler(
            use_ocdbt=True, use_zarr3=use_zarr3
        ) as checkpoint_handler:

          checkpoint_handler.save(
              directory=self.directory,
              args=pytree_checkpoint_handler.PyTreeSaveArgs(
                  pytree,
                  save_args=save_args,
                  ocdbt_target_data_file_size=target_data_file_size,
              ),
          )

          data_dir = self.directory / 'd'
          self.assertTrue(data_dir.exists())
          self.assertTrue(data_dir.is_dir())

          for f in data_dir.iterdir():
            if f.is_file():
              if target_data_file_size not in (0, None):
                # it's expected the resulting file sizes can be larger than
                # the target_data_file_size, so we give some buffer here
                self.assertLessEqual(
                    f.stat().length,
                    target_data_file_size * 2.0,
                )  # some room
                if patch_default_ocdbt_data_file_size:
                  self.assertLessEqual(
                      f.stat().length,
                      new_ocdbt_target_data_file_size * 2.0,
                  )

          def _create_restore_args(arr, mesh, axes):
            return ArrayRestoreArgs(
                restore_type=type(arr), mesh=mesh, mesh_axes=axes
            )

          restore_args = jax.tree.map(
              _create_restore_args, pytree, mesh_tree, axes_tree
          )

          restored = checkpoint_handler.restore(
              self.directory,
              args=pytree_checkpoint_handler.PyTreeRestoreArgs(
                  restore_args=restore_args
              ),
          )

          test_utils.assert_tree_equal(self, pytree, restored)

    def test_local_registry(self):

      if utils.is_pathways_backend():
        # This does not test anything on the pathways backend
        # TODO(b/333114195): add proper pathways testing.
        return

      class PlusOneHandler(type_handlers.ScalarHandler):

        async def serialize(
            self,
            values: Sequence[int],  # pytype: disable=signature-mismatch
            infos: Sequence[ParamInfo],
            args: Optional[Sequence[SaveArgs]] = None,
        ) -> Sequence[future.Future]:
          """See superclass documentation."""
          values = [v + 1 for v in values]
          return await super().serialize(values, infos, args)

      registry = type_handlers.create_type_handler_registry(
          (int, PlusOneHandler()),
      )
      handler = PyTreeCheckpointHandler(
          type_handler_registry=registry,
      )
      with self.assertRaisesRegex(
          ValueError, "TypeHandler lookup failed for: type=<class 'float'>"
      ):
        handler.save(self.directory, {'a': 3, 'b': 1.0})
      handler.save(self.directory, {'a': 3})

      restored = handler.restore(self.directory)
      expected = {'a': 4}
      self.assertEqual(restored, expected)


    def test_empty_custom_node(self):

      class PyTreeDict(dict):
        pass

      jax.tree_util.register_pytree_node(
          PyTreeDict,
          lambda d: (tuple(d.values()), tuple(d.keys())),
          lambda keys, values: PyTreeDict(dict(zip(keys, values))),
      )

      with self.assertRaisesRegex(ValueError, 'Found empty item'):
        self.handler.save(self.directory, args=PyTreeSaveArgs(PyTreeDict()))

      self.handler.save(
          self.directory, args=PyTreeSaveArgs({'a': PyTreeDict()})
      )
      restored = self.handler.restore(self.directory)
      self.assertDictEqual({'a': {}}, restored)

      restored = self.handler.restore(
          self.directory, args=PyTreeRestoreArgs({'a': PyTreeDict()})
      )
      test_utils.assert_tree_equal(self, {'a': PyTreeDict()}, restored)

    @parameterized.parameters((5,), (9,))
    def test_concurrent_gb_save(self, limit_bytes):
      # TODO(b/346811105): Enable for Pathways.
      if utils.is_pathways_backend():
        self.skipTest(
            'Disabled on Pathways because completion_times cannot updated by'
            ' reference outside remote Python.'
        )
      sleep_time = 1.0
      handler, tree, _ = test_utils.concurrent_gb_test_setup()

      byte_limiter = test_utils.get_byte_limiter(limit_bytes, sleep_time)
      with mock.patch.object(
          serialization,
          'get_byte_limiter',
          new=lambda _: byte_limiter,
      ):
        handler.save(self.directory, args=PyTreeSaveArgs(tree))
      # Replicated shards are handled within the _write_array_shard function.
      # Since shards are only saved once per replica, we only have to check
      # the primary process.
      completion_times = byte_limiter.completion_times
      if multihost.process_index() == 0:
        self.assertLen(completion_times, len(jax.tree.leaves(tree)))
        test_utils.assert_every_n_is_x_apart(
            self,
            completion_times,
            limit_bytes // np.int32().itemsize,
            sleep_time,
        )

    @parameterized.parameters((5,), (9,))
    def test_concurrent_gb_restore(self, limit_bytes):
      # TODO(b/346811105): Enable for Pathways.
      if utils.is_pathways_backend():
        self.skipTest(
            'Disabled on Pathways because completion_times cannot updated by'
            ' reference outside remote Python.'
        )
      sleep_time = 1.0
      handler, tree, restore_args = test_utils.concurrent_gb_test_setup()
      handler.save(self.directory, args=PyTreeSaveArgs(tree))

      byte_limiter = test_utils.get_byte_limiter(limit_bytes, sleep_time)
      with mock.patch.object(
          serialization,
          'get_byte_limiter',
          new=lambda _,: byte_limiter,
      ):
        restored = handler.restore(
            self.directory,
            args=PyTreeRestoreArgs(restore_args=restore_args),
        )
      self.validate_restore(tree, restored)
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

    def test_enable_pinned_host_transfer(self):
      if utils.is_pathways_backend():
        self.skipTest(
            'Disabled on Pathways because local variables cannot updated by'
            ' reference outside remote Python.'
        )
      true_count = 0
      false_count = 0

      original_transfer_arrays_to_host = replica_slices.transfer_arrays_to_host

      def _transfer_arrays_to_host(
          arrays, replica_id, use_replica_parallel, enable_pinned_host_transfer
      ):
        nonlocal true_count, false_count
        if enable_pinned_host_transfer:
          true_count += 1
        else:
          false_count += 1
        return original_transfer_arrays_to_host(
            arrays,
            replica_id,
            use_replica_parallel=use_replica_parallel,
            enable_pinned_host_transfer=enable_pinned_host_transfer,
        )

      with mock.patch.object(
          replica_slices,
          'transfer_arrays_to_host',
          new=_transfer_arrays_to_host,
      ):
        self.handler.save(
            self.directory,
            args=PyTreeSaveArgs(self.pytree, enable_pinned_host_transfer=False),
        )

      self.assertEqual(true_count, 0)
      self.assertGreater(false_count, 0)

    def test_custom_metadata(self):
      custom_metadata = {'foo': 1}
      self.handler.save(
          self.directory,
          args=PyTreeSaveArgs(self.pytree, custom_metadata=custom_metadata),
      )
      metadata = self.handler.metadata(self.directory)
      self.assertEqual(metadata.custom_metadata, custom_metadata)

    @parameterized.product(
        use_ocdbt=(True, False),
        pytree_metadata_options=(
            tree_metadata.PyTreeMetadataOptions(support_rich_types=False),
            tree_metadata.PyTreeMetadataOptions(support_rich_types=True),
        ),
    )
    def test_write_shape_metadata_missing_for_all_types_other_than_jax_array(
        self,
        use_ocdbt: bool,
        pytree_metadata_options: tree_metadata.PyTreeMetadataOptions,
    ):
      """Test case."""
      checkpoint = {
          'a': 1,
          'b': np.array([2]),
          'c': 'hello',
      }
      expected_metadata_without_directory = {
          'a': value_metadata.ScalarMetadata(
              name='a',
              directory=None,
              shape=(),
              sharding=None,
              dtype=np.dtype('int64'),
              storage=None,
          ),
          'b': value_metadata.ArrayMetadata(
              name='b',
              directory=None,
              shape=(1,),
              sharding=None,
              dtype=np.dtype('int64'),
              storage=value_metadata.StorageMetadata(
                  chunk_shape=(1,), write_shape=None
              ),
          ),
          'c': value_metadata.StringMetadata(name='c', directory=None),
      }
      with self.ocdbt_checkpoint_handler(
          use_ocdbt,
          pytree_metadata_options=pytree_metadata_options,
          array_metadata_store=ARRAY_METADATA_STORE,
      ) as checkpoint_handler:
        checkpoint_handler.save(self.directory, args=PyTreeSaveArgs(checkpoint))

        self.assertFalse((self.directory / 'array_metadatas').exists())
        metadata = checkpoint_handler.metadata(self.directory)
        metadata_without_directory = jax.tree.map(
            lambda m: dataclasses.replace(m, directory=None), metadata.tree
        )
        self.assertEqual(
            expected_metadata_without_directory,
            metadata_without_directory,
        )

    @parameterized.product(
        use_ocdbt=(True, False),
        pytree_metadata_options=(
            tree_metadata.PyTreeMetadataOptions(support_rich_types=False),
            tree_metadata.PyTreeMetadataOptions(support_rich_types=True),
        ),
    )
    def test_write_shape_in_metadata_disabled(
        self,
        use_ocdbt: bool,
        pytree_metadata_options: tree_metadata.PyTreeMetadataOptions,
    ):
      """Test case."""
      with self.ocdbt_checkpoint_handler(
          use_ocdbt,
          pytree_metadata_options=pytree_metadata_options,
          array_metadata_store=None,
      ) as checkpoint_handler:
        checkpoint_handler.save(
            self.directory, args=PyTreeSaveArgs(self.pytree)
        )
        expected_tree_with_write_shapes = {
            'a': {'write_shape': None},
            'b': {'write_shape': None},
            'c': {
                'a': {'write_shape': None},
                'e': {'write_shape': None},
            },
        }
        metadata = checkpoint_handler.metadata(self.directory)
        tree_with_write_shapes = jax.tree.map(
            lambda m: {'write_shape': m.storage.write_shape}, metadata.tree
        )
        chex.assert_trees_all_equal(
            expected_tree_with_write_shapes, tree_with_write_shapes
        )

    # TODO(b/382230550): Add test for chunk_shape != write_shape.
    @parameterized.product(
        use_ocdbt=(True, False),
        pytree_metadata_options=(
            tree_metadata.PyTreeMetadataOptions(support_rich_types=False),
            tree_metadata.PyTreeMetadataOptions(support_rich_types=True),
        ),
    )
    def test_write_shape_in_metadata(
        self,
        use_ocdbt: bool,
        pytree_metadata_options: tree_metadata.PyTreeMetadataOptions,
    ):
      """Test case."""
      with self.ocdbt_checkpoint_handler(
          use_ocdbt, pytree_metadata_options=pytree_metadata_options
      ) as checkpoint_handler:
        checkpoint_handler.save(
            self.directory, args=PyTreeSaveArgs(self.pytree)
        )

        expected_tree_with_write_shapes = {
            'a': {'write_shape': (1,)},
            'b': {'write_shape': (2,)},
            'c': {
                'a': {'write_shape': (1, 1)},
                'e': {'write_shape': (2, 1)},
            },
        }
        metadata = checkpoint_handler.metadata(self.directory)
        tree_with_write_shapes = jax.tree.map(
            lambda m: {'write_shape': m.storage.write_shape}, metadata.tree
        )
        chex.assert_trees_all_equal(
            expected_tree_with_write_shapes, tree_with_write_shapes
        )

    @parameterized.product(use_ocdbt=(True, False))
    def test_array_metadata_disabled(self, use_ocdbt: bool):
      """Test case."""
      with self.ocdbt_checkpoint_handler(
          use_ocdbt, array_metadata_store=None
      ) as checkpoint_handler:
        pytree, save_args, restore_args = self.create_mixed_format_pytree()

        checkpoint_handler.save(
            self.directory, args=PyTreeSaveArgs(pytree, save_args)
        )

        self.validate_save(
            self.directory,
            pytree,
            checkpoint_handler,
            save_args=save_args,
            restore_args=restore_args,
        )

      self.assertFalse((self.directory / 'array_metadatas').exists())

    @parameterized.product(use_ocdbt=(True, False))
    async def test_array_metadata(self, use_ocdbt: bool):
      """Test case."""
      with self.ocdbt_checkpoint_handler(use_ocdbt) as checkpoint_handler:
        pytree, save_args, restore_args = self.create_mixed_format_pytree()

        checkpoint_handler.save(
            self.directory, args=PyTreeSaveArgs(pytree, save_args)
        )

        self.validate_save(
            self.directory,
            pytree,
            checkpoint_handler,
            save_args=save_args,
            restore_args=restore_args,
        )

      self.assertTrue((self.directory / 'array_metadatas').exists())
      if multihost.process_index() == 0:
        array_metadatas = await ARRAY_METADATA_STORE.read(self.directory)
        expected_array_metadatas = {
            0: [
                array_metadata.SerializedArrayMetadata(
                    param_name='new_key.a',
                    write_shape=(1,),
                    chunk_shape=(1,),
                ),
                array_metadata.SerializedArrayMetadata(
                    param_name='new_key.b',
                    write_shape=(2,),
                    chunk_shape=(2,),
                ),
                array_metadata.SerializedArrayMetadata(
                    param_name='new_key.c.a',
                    write_shape=(1, 1),
                    chunk_shape=(1, 1),
                ),
                array_metadata.SerializedArrayMetadata(
                    param_name='new_key.c.e',
                    write_shape=(2, 1),
                    chunk_shape=(2, 1),
                ),
            ],
            1: [
                array_metadata.SerializedArrayMetadata(
                    param_name='new_key.a',
                    write_shape=(1,),
                    chunk_shape=(1,),
                ),
                array_metadata.SerializedArrayMetadata(
                    param_name='new_key.b',
                    write_shape=(2,),
                    chunk_shape=(2,),
                ),
                array_metadata.SerializedArrayMetadata(
                    param_name='new_key.c.a',
                    write_shape=(1, 1),
                    chunk_shape=(1, 1),
                ),
                array_metadata.SerializedArrayMetadata(
                    param_name='new_key.c.e',
                    write_shape=(2, 1),
                    chunk_shape=(2, 1),
                ),
            ],
        }
        self.assertEqual(expected_array_metadatas, array_metadatas)

    @parameterized.product(use_ocdbt=(True, False))
    def test_save_with_missing_array_metadata_file(self, use_ocdbt: bool):
      """Test case."""
      if multihost.process_index() != 0:  # only test on primary host
        self.skipTest('Test only for primary host to avoid barrier timeout.')

      class PathResolverReturningNoMetadataFiles(
          array_metadata_store_lib.PathResolver
      ):

        def get_read_file_paths(
            self, checkpoint_dir: epath.Path, process_index: int | None = None
        ) -> Iterator[epath.Path] | epath.Path | None:
          return None

      with self.ocdbt_checkpoint_handler(
          use_ocdbt,
          array_metadata_store=array_metadata_store_lib.Store(
              path_resolver=PathResolverReturningNoMetadataFiles()
          ),
      ) as checkpoint_handler:
        with self.assertRaisesRegex(
            ValueError, 'No ArrayMetadata found for process_index'
        ):
          checkpoint_handler.save(
              self.directory, args=PyTreeSaveArgs(self.pytree)
          )

    @parameterized.product(use_ocdbt=(True, False))
    def test_save_with_missing_array_metadata_for_params(self, use_ocdbt: bool):
      """Test case."""
      if multihost.process_index() != 0:  # only test on primary host
        self.skipTest('Test only for primary host to avoid barrier timeout.')

      class MissingArrayMetadataSerDeserializer(
          array_metadata_store_lib.SerDeserializer
      ):

        def deserialize(
            self, serialized: str
        ) -> List[array_metadata.SerializedArrayMetadata]:
          true_data = super().deserialize(serialized)
          return [true_data.pop(0)]  # Delete the rest and return partial data.

      with self.ocdbt_checkpoint_handler(
          use_ocdbt,
          array_metadata_store=array_metadata_store_lib.Store(
              ser_deser=MissingArrayMetadataSerDeserializer()
          ),
      ) as checkpoint_handler:
        with self.assertRaisesRegex(
            ValueError, 'No ArrayMetadata found for param_info'
        ):
          checkpoint_handler.save(
              self.directory, args=PyTreeSaveArgs(self.pytree)
          )

    @parameterized.parameters((True,), (False,))
    def test_zero_size_array(self, use_jax_array: bool):
      arr = np.ones(shape=(0,))
      mesh = jax.sharding.Mesh(np.array(jax.devices()), axis_names=('x',))
      pspec = jax.sharding.PartitionSpec()
      if use_jax_array:
        arr = test_utils.create_sharded_array(arr, mesh, pspec)
      tree = [arr]
      with self.assertRaisesRegex(ValueError, 'zero size'):
        self.handler.save(self.directory, args=PyTreeSaveArgs(tree))

    @parameterized.product(use_ocdbt=(True, False))
    def test_save_restore_random_keys(self, use_ocdbt: bool):
      """Test saving and restoring random keys within a pytree."""

      # TODO(b/393160483) investigate Pathways remote Python support for
      # random.keys.
      if utils.is_pathways_backend():
        self.skipTest(
            'Disabled on Pathways because random keys are not supported by'
            ' remote Python.'
        )

      mesh = jax.sharding.Mesh(jax.devices(), ('x',))
      sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

      pytree = {
          'keys': {
              'kone': jax.random.key(jnp.array(0, device=sharding)),
              'impl_key': {
                  'rbg': jax.random.key(
                      jnp.array(1, device=sharding), impl='rbg'
                  ),
                  'unsafe_rbg': jax.random.key(
                      jnp.array(2, device=sharding), impl='unsafe_rbg'
                  ),
              },
              'split_keys': jax.random.split(
                  jax.random.key(jnp.array(123, device=sharding)), num=10
              ),
          },
          'arrays': self.pytree,
      }

      restore_args = jax.tree.map(
          lambda x: ArrayRestoreArgs(sharding=x.sharding), pytree
      )

      with self.ocdbt_checkpoint_handler(
          use_ocdbt=use_ocdbt,
          array_metadata_store=array_metadata_store_lib.Store(),
      ) as save_handler:
        save_handler.save(self.directory, args=PyTreeSaveArgs(pytree))

      with self.ocdbt_checkpoint_handler(
          use_ocdbt=use_ocdbt,
          array_metadata_store=array_metadata_store_lib.Store(),
      ) as restore_handler:
        restored = restore_handler.restore(
            self.directory, args=PyTreeRestoreArgs(pytree, restore_args)
        )
        test_utils.assert_tree_equal(self, pytree, restored)

    def test_pinned_host_loading(self):
      if multihost.is_pathways_backend():
        # TODO(b/404915487): Reenable when possible.
        self.skipTest('Disabled due to b/404915487.')
      pytree = dict(arr=np.ones((1024, 512)))
      self.handler.save(self.directory, args=PyTreeSaveArgs(pytree))

      mesh = jax.sharding.Mesh(
          np.asarray(jax.devices()).reshape((1, len(jax.devices()))), ('x', 'y')
      )
      sharding = jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec('x', 'y')
      ).with_memory_kind('pinned_host')

      restore_args = dict(arr=ArrayRestoreArgs(sharding=sharding))
      restored = self.handler.restore(
          self.directory, args=PyTreeRestoreArgs(restore_args=restore_args)
      )
      expected = dict(arr=jax.device_put(np.ones((1024, 512)), sharding))
      self.validate_restore(expected, restored)
