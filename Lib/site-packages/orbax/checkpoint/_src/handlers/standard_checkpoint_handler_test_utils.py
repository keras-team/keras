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

"""Test for standard_checkpoint_handler.py."""

# pylint: disable=protected-access, missing-function-docstring

import functools
from typing import Any

from absl.testing import parameterized
from etils import epath
import flax
import flax.training.train_state
import jax
from jax import numpy as jnp
from jax.experimental import layout
import numpy as np
import optax
from orbax.checkpoint import test_utils
from orbax.checkpoint import utils
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.handlers import standard_checkpoint_handler
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.metadata import value as value_metadata
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.serialization import type_handlers

DLL = layout.DeviceLocalLayout
Layout = layout.Layout
PyTree = Any
SaveArgs = type_handlers.SaveArgs
StandardRestoreArgs = standard_checkpoint_handler.StandardRestoreArgs
StandardSaveArgs = standard_checkpoint_handler.StandardSaveArgs
ArrayRestoreArgs = pytree_checkpoint_handler.ArrayRestoreArgs
RestoreArgs = pytree_checkpoint_handler.RestoreArgs


class StandardCheckpointHandler(
    standard_checkpoint_handler.StandardCheckpointHandler
):

  def save(self, directory, *args, **kwargs):
    super().save(directory, *args, **kwargs)
    test_utils.sync_global_processes('StandardCheckpointHandler:save')
    if multihost.process_index() == 0:
      self.finalize(directory)
    test_utils.sync_global_processes('StandardCheckpointHandler:finalize')


# Not in common util because we need to eliminate OSS dependency on flax.
def init_flax_model(model):
  params = model.init(jax.random.PRNGKey(0), jnp.ones([8, 8]))
  tx = optax.adamw(learning_rate=0.001)
  state = flax.training.train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx
  )
  return jax.tree.map(np.asarray, state)


class StandardCheckpointHandlerTestBase:
  """Base test cases for StandardCheckpointHandler."""

  class Test(parameterized.TestCase):
    """Test class."""

    save_args_cls = StandardSaveArgs
    restore_args_cls = StandardRestoreArgs

    def setUp(self):
      super().setUp()

      self.numpy_pytree = test_utils.setup_pytree()
      pytree, _, _ = test_utils.setup_sharded_pytree(self.numpy_pytree)
      zeros_pytree = jax.tree.map(
          np.zeros_like,
          self.numpy_pytree,
          is_leaf=test_utils.is_leaf,
      )
      zeros_pytree, _, _ = test_utils.setup_sharded_pytree(zeros_pytree)
      self.zeros_pytree = zeros_pytree

      self.numpy_pytree.update({'x': 4.5, 'y': 3})
      self.pytree = pytree
      self.mixed_pytree = {
          'sharded': self.pytree,
          'numpy': self.numpy_pytree,
      }

      self.directory = epath.Path(
          self.create_tempdir(name='checkpointing_test').full_path
      )
      test_utils.set_tensorstore_driver_for_test()

      test_utils.sync_global_processes(
          'StandardCheckpointHandler:setup_complete'
      )

    def tearDown(self):
      test_utils.sync_global_processes(
          'StandardCheckpointHandler:tests_complete'
      )
      self.handler.close()
      super().tearDown()

    @property
    def handler(self) -> StandardCheckpointHandler:
      return StandardCheckpointHandler()

    def test_basic(self):
      self.handler.save(self.directory, args=self.save_args_cls(self.pytree))
      self.assertTrue(type_handlers.is_ocdbt_checkpoint(self.directory))
      self.assertTrue(type_handlers.is_ocdbt_checkpoint(self.directory))
      restored = self.handler.restore(
          self.directory, args=self.restore_args_cls(self.zeros_pytree)
      )
      test_utils.assert_tree_equal(self, self.pytree, restored)

    def test_basic_no_item_arg(self):
      self.handler.save(self.directory, args=self.save_args_cls(self.pytree))
      restored = self.handler.restore(self.directory)
      self.assertTrue(type_handlers.is_ocdbt_checkpoint(self.directory))
      test_utils.assert_tree_equal(self, self.pytree, restored)

    def test_shape_dtype_struct(self):
      self.handler.save(
          self.directory, args=self.save_args_cls(self.mixed_pytree)
      )
      restored = self.handler.restore(
          self.directory,
          args=self.restore_args_cls(
              jax.tree.map(utils.to_shape_dtype_struct, self.mixed_pytree)
          ),
      )
      test_utils.assert_tree_equal(self, self.mixed_pytree, restored)

    def test_custom_layout(self):
      """Test custom layout at restoration."""
      if not hasattr(self.restore_args_cls, 'support_layout'):
        self.skipTest('custom layout is not supported for this handler')

      inp = np.arange(32).reshape(8, 4)
      mesh = jax.sharding.Mesh(
          np.asarray(jax.devices()).reshape(4, 2), ('x', 'y')
      )
      pspec = jax.sharding.PartitionSpec('x', 'y')
      arr = test_utils.create_sharded_array(inp, mesh, pspec)
      pytree = {'x': arr}

      self.handler.save(self.directory, args=self.save_args_cls(pytree))
      restored_regular = self.handler.restore(
          self.directory, args=self.restore_args_cls(item=pytree)
      )
      test_utils.assert_tree_equal(self, pytree, restored_regular)

      # create a custom layout
      custom_layout = Layout(
          device_local_layout=DLL(
              major_to_minor=arr.layout.device_local_layout.major_to_minor[::-1],  # pytype: disable=attribute-error
              _tiling=arr.layout.device_local_layout._tiling,  # pytype: disable=attribute-error
          ),
          sharding=arr.sharding,
      )
      arr_new_layout = jax.device_put(arr, custom_layout)
      self.assertNotEqual(arr_new_layout.layout, arr.layout)

      # use a pytree example
      with self.subTest('test with item=pytree with custom layout'):
        pytree_new_layout = {'x': arr_new_layout}
        restored_new_layout = self.handler.restore(
            self.directory,
            args=self.restore_args_cls(
                item=pytree_new_layout, support_layout=True
            ),
        )
        test_utils.assert_tree_equal(
            self, pytree_new_layout, restored_new_layout
        )

      with self.subTest('test with item=ShapeDtypeStruct with custom layout'):
        # use shape_dtype_struct with custom layout
        shape_dtypes = jax.tree.map(utils.to_shape_dtype_struct, pytree)
        shape_dtypes = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(
                x.shape, x.dtype, sharding=custom_layout
            ),
            shape_dtypes,
        )

        restored_shape_dtypes = self.handler.restore(
            self.directory,
            args=self.restore_args_cls(item=shape_dtypes, support_layout=True),
        )
        test_utils.assert_tree_equal(
            self, pytree_new_layout, restored_shape_dtypes
        )

      with self.subTest('test with support_layout=False'):
        # with support_layout=False, it should equal to original pytree
        restored_without_layout = self.handler.restore(
            self.directory,
            args=self.restore_args_cls(item=pytree_new_layout),
        )
        test_utils.assert_tree_equal(self, pytree, restored_without_layout)

    @parameterized.parameters((True,), (False,))
    def test_change_shape(self, strict: bool):
      if not hasattr(self.restore_args_cls, 'strict'):
        self.skipTest('strict option not supported for this handler')
      mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('x',))
      axes = jax.sharding.PartitionSpec(None)
      pytree = {'x': test_utils.create_sharded_array(np.arange(8), mesh, axes)}
      self.handler.save(self.directory, args=self.save_args_cls(pytree))

      mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('x',))
      axes = jax.sharding.PartitionSpec(None)
      expected_pytree = {
          'x': test_utils.create_sharded_array(np.arange(4), mesh, axes)
      }

      if strict:
        with self.assertRaises(BaseException):
          self.handler.restore(
              self.directory,
              args=self.restore_args_cls(
                  jax.tree.map(utils.to_shape_dtype_struct, expected_pytree),
                  strict=strict,
              ),
          )
      else:
        restored = self.handler.restore(
            self.directory,
            args=self.restore_args_cls(
                jax.tree.map(utils.to_shape_dtype_struct, expected_pytree),
                strict=strict,
            ),
        )
        test_utils.assert_tree_equal(self, expected_pytree, restored)

    def test_save_unsupported_type(self):
      pytree = {'str_key': 'str_value', **self.pytree}
      with self.assertRaisesRegex(ValueError, 'Unsupported type'):
        self.handler.save(self.directory, args=self.save_args_cls(pytree))

    def test_restore_unsupported_type(self):
      pytree = {'str_key': 'str_value', **self.pytree}
      self.handler.save(self.directory, args=self.save_args_cls(self.pytree))
      with self.assertRaisesRegex(ValueError, 'Unsupported type'):
        self.handler.restore(self.directory, args=self.restore_args_cls(pytree))

    def test_cast(self):
      # TODO(dicentra): casting from int dtypes currently doesn't work
      # in the model surgery context.
      save_args = jax.tree.map(
          lambda _: SaveArgs(dtype=jnp.dtype(jnp.float32)), self.pytree
      )
      self.handler.save(
          self.directory,
          args=self.save_args_cls(self.pytree, save_args=save_args),
      )
      metadata = self.handler.metadata(self.directory)
      jax.tree.map(lambda m: self.assertEqual(m.dtype, jnp.float32), metadata)

      def check_dtype(x, dtype):
        if utils.is_scalar(x):
          self.assertIsInstance(x, int)
        else:
          self.assertEqual(x.dtype, dtype)

      pytree = jax.tree.map(
          functools.partial(
              utils.to_shape_dtype_struct,
              dtype=jnp.bfloat16,
              scalar_dtype=int,
          ),
          self.pytree,
      )
      restored = self.handler.restore(
          self.directory,
          args=self.restore_args_cls(pytree),
      )
      jax.tree.map(lambda x: check_dtype(x, jnp.bfloat16), restored)

    def test_flax_model(self):

      @flax.struct.dataclass
      class Params(flax.struct.PyTreeNode):
        params: Any
        opt_state: Any

      def make_params():
        return Params(
            params=self.numpy_pytree,
            opt_state=(optax.EmptyState(), optax.EmptyState()),
        )

      params = make_params()
      mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('devices',))
      mesh_axes = jax.sharding.PartitionSpec()
      params = jax.tree.map(
          lambda arr: test_utils.create_sharded_array(arr, mesh, mesh_axes),
          params,
      )
      target = jax.tree.map(utils.to_shape_dtype_struct, params)

      self.handler.save(self.directory, args=self.save_args_cls(params))
      restored = self.handler.restore(
          self.directory, args=self.restore_args_cls(target)
      )
      test_utils.assert_tree_equal(self, params, restored)

    def test_empty_error(self):
      with self.assertRaises(ValueError):
        self.handler.save(self.directory, args=self.save_args_cls({}))

    def test_empty_dict_node(self):
      item = {'a': {}, 'b': 3}
      self.handler.save(self.directory, args=self.save_args_cls(item))
      restored = self.handler.restore(
          self.directory, args=self.restore_args_cls(item)
      )
      self.assertDictEqual(restored, item)

    def test_empty_none_node(self):
      item = {'c': None, 'd': 2}
      self.handler.save(self.directory, args=self.save_args_cls(item))
      restored = self.handler.restore(
          self.directory, args=self.restore_args_cls(item)
      )
      self.assertDictEqual(restored, item)

    def test_none_node_in_restore_args(self):
      devices = np.asarray(jax.devices())
      mesh = jax.sharding.Mesh(devices, ('x',))
      mesh_axes = jax.sharding.PartitionSpec(
          'x',
      )
      arr = test_utils.create_sharded_array(np.arange(16), mesh, mesh_axes)
      item = {'b': arr}
      self.handler.save(self.directory, args=self.save_args_cls(item))

      restored = self.handler.restore(
          self.directory,
          args=self.restore_args_cls({'b': None}),
      )
      test_utils.assert_tree_equal(self, restored, {'b': None})

    def test_masked_shape_dtype_struct(self):

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

      self.handler.save(self.directory, args=self.save_args_cls(masked_tree))
      self.assertTrue(type_handlers.is_ocdbt_checkpoint(self.directory))

      # Restore it with item which was given before applying masking.
      restored = self.handler.restore(
          self.directory,
          args=self.restore_args_cls(
              jax.tree.map(utils.to_shape_dtype_struct, self.pytree)
          ),
      )
      test_utils.assert_tree_equal(self, expected, restored)

      # Restore it with item after applying masking to it.
      restored = self.handler.restore(
          self.directory,
          args=self.restore_args_cls(
              jax.tree.map(utils.to_shape_dtype_struct, masked_tree)
          ),
      )
      test_utils.assert_tree_equal(self, expected, restored)

      # Restore it without any item.
      restored = self.handler.restore(self.directory)
      test_utils.assert_tree_equal(self, expected, restored)

    def test_custom_metadata(self):
      custom_metadata = {'foo': 1}
      self.handler.save(
          self.directory,
          args=self.save_args_cls(self.pytree, custom_metadata=custom_metadata),
      )
      metadata = self.handler.metadata(self.directory)
      self.assertEqual(metadata.custom_metadata, custom_metadata)

    def test_construct_restore_args_with_fallback_sharding(self):
      invalid_sharding_metadata = sharding_metadata.NamedShardingMetadata(
          shape=np.array([2, 4]),
          axis_names=['x'],
          partition_spec=(jax.sharding.PartitionSpec('x'),),
          device_mesh=sharding_metadata.DeviceMetadataMesh.from_dict(
              {'mesh': {'id': 1000}}
          ),
      )
      fallback_sharding = jax.sharding.NamedSharding(
          mesh=jax.sharding.Mesh(jax.devices(), ('x',)),
          spec=jax.sharding.PartitionSpec('x'),
      )
      pytree = tree_metadata.build_default_tree_metadata({
          'a': value_metadata.ScalarMetadata(
              name='', directory=epath.Path(''), dtype=jnp.int32
          ),
          'b': value_metadata.ArrayMetadata(
              name='',
              directory=epath.Path(''),
              shape=(2, 4),
              sharding=None,
              dtype=jnp.float64,
          ),
          'c': value_metadata.ArrayMetadata(
              name='',
              directory=epath.Path(''),
              shape=(2, 4),
              sharding=invalid_sharding_metadata,
              dtype=jnp.float64,
          ),
      })
      expected_restore_args = {
          'a': RestoreArgs(restore_type=int, dtype=jnp.int32),
          'b': RestoreArgs(restore_type=np.ndarray, dtype=jnp.float64),
          'c': ArrayRestoreArgs(
              restore_type=jax.Array,
              sharding=fallback_sharding,
              global_shape=(2, 4),
              dtype=jnp.float64,
          ),
      }

      restore_args = standard_checkpoint_handler._construct_restore_args(
          pytree, fallback_sharding
      )

      self.assertSameElements(expected_restore_args.keys(), restore_args.keys())
      with self.subTest(name='array_restore_args'):
        self.assertEqual(expected_restore_args['c'], restore_args['c'])
      with self.subTest(name='restore_args'):
        self.assertEqual(expected_restore_args['a'], restore_args['a'])
        self.assertEqual(expected_restore_args['b'], restore_args['b'])
