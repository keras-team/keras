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

"""Base class for save/load tests."""

# pylint: disable=missing-class-docstring,protected-access,missing-function-docstring

from absl.testing import parameterized
from etils import epath
import flax
import jax
from jax import numpy as jnp
import numpy as np
import optax
from orbax.checkpoint import test_utils
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.tree import utils as tree_utils
import orbax.checkpoint.experimental.v1 as ocp
from orbax.checkpoint.experimental.v1._src.testing import array_utils as array_test_utils
from orbax.checkpoint.experimental.v1._src.tree import types as tree_types


PyTree = tree_types.PyTree

create_sharded_array = array_test_utils.create_sharded_array
create_numpy_pytree = array_test_utils.create_numpy_pytree
create_sharded_pytree = array_test_utils.create_sharded_pytree
as_abstract_type = array_test_utils.as_abstract_type


class SaveLoadTestBase:

  class Test(parameterized.TestCase):

    def setUp(self):
      super().setUp()

      self.directory = (
          epath.Path(self.create_tempdir(name='saving_test').full_path) / 'ckpt'
      )
      self.pytree, self.abstract_pytree = create_numpy_pytree()
      self.numpy_pytree, self.abstract_numpy_pytree = create_sharded_pytree()

      test_utils.set_tensorstore_driver_for_test()
      test_utils.sync_global_processes('SavingTest:setup_complete')

    def tearDown(self):
      super().tearDown()
      test_utils.sync_global_processes('SavingTest:teardown_complete')

    def save_and_wait(self, *args, use_async: bool, **kwargs):
      if use_async:
        response = ocp.save_pytree_async(*args, **kwargs)
        self.assertIsNotNone(response)
        self.assertIsNone(response.result())
      else:
        self.assertIsNone(ocp.save_pytree(*args, **kwargs))

    def load_and_wait(self, *args, use_async: bool, **kwargs):
      del use_async
      return ocp.load_pytree(*args, **kwargs)

    @parameterized.parameters((True,), (False,))
    def test_save_load_pytree(self, use_async):
      self.save_and_wait(self.directory, self.pytree, use_async=use_async)
      loaded = self.load_and_wait(
          self.directory, self.abstract_pytree, use_async=use_async
      )
      test_utils.assert_tree_equal(self, self.pytree, loaded)

    @parameterized.parameters((True,), (False,))
    def test_load_default(self, use_async):
      self.save_and_wait(self.directory, self.pytree, use_async=use_async)
      loaded = self.load_and_wait(self.directory, use_async=use_async)
      test_utils.assert_tree_equal(self, self.pytree, loaded)

    def test_save_pytree_async(self):
      response = ocp.save_pytree_async(self.directory, self.pytree)
      self.assertFalse(self.directory.exists())  # Not finalized yet.
      # But a tmp dir should have been created.
      self.assertNotEmpty(list(self.directory.parent.iterdir()))
      response.result()
      self.assertTrue(self.directory.exists())
      loaded = ocp.load_pytree(self.directory)
      test_utils.assert_tree_equal(self, self.pytree, loaded)

    @parameterized.parameters(
        (tuple([]),),
        (dict(),),
        (list(),),
        (None,),
        (optax.EmptyState(),),
    )
    def test_empty_tree(self, tree):
      with self.assertRaisesRegex(ValueError, 'empty'):
        ocp.save_pytree(self.directory, tree)

    # Note the ommission of jax.Array, since this is covered in
    # several other tests.
    @parameterized.parameters(
        (np.arange(8),),
        (2,),
        (2.2,),
        ('foo',),
        (np.asarray(3.14),),
    )
    def test_standard_leaf_types(self, value):
      ocp.save_pytree(self.directory, dict(k=value))
      loaded = ocp.load_pytree(self.directory)
      if isinstance(value, np.ndarray):
        np.testing.assert_array_equal(loaded['k'], value)
      else:
        self.assertEqual(loaded['k'], value)

    def test_jax_array_leaf_types(self):
      mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('devices',))
      # TODO(cpgaffney): Add support for missing arrays.
      values = {
          # 'simple_array': jnp.arange(16),
          # 'single_device_array': jnp.arange(8, device=jax.local_devices()[0]),
          'replicated_array': jnp.arange(
              12,
              device=jax.sharding.NamedSharding(
                  mesh, jax.sharding.PartitionSpec()
              ),
          ),
          'sharded_array': jnp.arange(
              32,
              device=jax.sharding.NamedSharding(
                  mesh, jax.sharding.PartitionSpec(('devices',))
              ),
          ),
          # 'single_device_cpu_array': jnp.arange(
          #     24, device=jax.local_devices(backend='cpu')[0]
          # ),
      }
      for k, v in values.items():
        with self.subTest(k):
          ocp.save_pytree(self.directory / k, [v])
          with self.subTest('with_abstract_pytree'):
            loaded = ocp.load_pytree(self.directory / k, [as_abstract_type(v)])
            test_utils.assert_tree_equal(self, [v], loaded)
          with self.subTest('without_abstract_pytree'):
            if multihost.is_pathways_backend():
              self.skipTest('Must provide abstract_pytree for Pathways.')
            loaded = ocp.load_pytree(self.directory / k)
            test_utils.assert_tree_equal(self, [v], loaded)

    def test_leaf_change_type(self):
      mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('devices',))
      sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
      numpy_arr = np.arange(16)
      jax_arr = create_sharded_array(numpy_arr, sharding)

      with self.subTest('numpy_to_jax'):
        subdir = 'numpy'
        ocp.save_pytree(self.directory / subdir, [numpy_arr])
        test_utils.assert_tree_equal(
            self,
            [jax_arr],
            ocp.load_pytree(
                self.directory / subdir, [as_abstract_type(jax_arr)]
            ),
        )
        if multihost.is_pathways_backend():
          self.skipTest('Must provide abstract_pytree for Pathways.')
        test_utils.assert_tree_equal(
            self, [numpy_arr], ocp.load_pytree(self.directory / subdir)
        )

      with self.subTest('jax_to_numpy'):
        subdir = 'jax'
        ocp.save_pytree(self.directory / subdir, [jax_arr])
        test_utils.assert_tree_equal(
            self,
            [numpy_arr],
            ocp.load_pytree(
                self.directory / subdir, [as_abstract_type(numpy_arr)]
            ),
        )
        if multihost.is_pathways_backend():
          self.skipTest('Must provide abstract_pytree for Pathways.')
        test_utils.assert_tree_equal(
            self, [jax_arr], ocp.load_pytree(self.directory / subdir)
        )

    def test_empty_array(self):
      value = np.ones(shape=(0,))
      with self.assertRaisesRegex(ValueError, 'zero size'):
        ocp.save_pytree(self.directory, dict(k=value))

    @parameterized.parameters(
        (complex(1.1, 2.2),),
        (2j,),
        # (b'foo',),  # TODO(cpgaffney): Unknown error in multihost.
        # (bytes(12),),  # TODO(cpgaffney): Unknown error in multihost.
    )
    def test_invalid_leaf_types(self, value):
      # TODO(cpgaffney): Consider improving the error message raised.
      with self.assertRaises(ValueError):
        ocp.save_pytree(self.directory, dict(k=value))

    # TODO(b/337105122): Add tests passing invalid abstract PyTrees.
    def test_flax_model(self):

      @flax.struct.dataclass
      class Params(flax.struct.PyTreeNode):
        params: PyTree
        opt_state: PyTree

      def make_state_with_optax():
        return Params(
            params=self.numpy_pytree,
            opt_state=(optax.EmptyState(), optax.EmptyState()),
        )

      def make_state_with_nones():
        return Params(
            params=self.numpy_pytree,
            opt_state=(None, None),
        )

      state = make_state_with_optax()
      ocp.save_pytree(self.directory, state)

      with self.subTest('with_abstract_state'):
        abstract_state = jax.tree.map(as_abstract_type, state)
        loaded = ocp.load_pytree(self.directory, abstract_state)
        test_utils.assert_tree_equal(self, state, loaded)

      with self.subTest('without_abstract_state'):
        if multihost.is_pathways_backend():
          self.skipTest('Must provide abstract_pytree for Pathways.')
        loaded = ocp.load_pytree(self.directory)
        expected_tree = tree_utils.serialize_tree(
            make_state_with_nones(),
            keep_empty_nodes=True,
        )
        test_utils.assert_tree_equal(self, expected_tree, loaded)

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
    def test_reshard(self, save_spec, load_spec):
      len_devices = len(jax.devices())
      self.assertGreaterEqual(len_devices, 4)
      mesh = jax.sharding.Mesh(
          np.asarray(jax.devices()).reshape((4, len_devices // 4)), ('x', 'y')
      )
      save_sharding = jax.sharding.NamedSharding(mesh, save_spec)
      tree = {'x': create_sharded_array(np.arange(len_devices), save_sharding)}
      ocp.save_pytree(self.directory, tree)

      load_sharding = jax.sharding.NamedSharding(mesh, load_spec)
      expected_tree = {
          'x': create_sharded_array(np.arange(len_devices), load_sharding)
      }
      abstract_tree = {'x': as_abstract_type(expected_tree['x'])}
      loaded = ocp.load_pytree(self.directory, abstract_pytree=abstract_tree)
      test_utils.assert_tree_equal(self, expected_tree, loaded)

    @parameterized.parameters(
        (np.int32, np.int16, np.int64),
        (np.int32, np.int64, np.int16),
        (np.int32, np.float32, np.int64),
        (np.int32, np.float32, np.float64),
        (np.float32, np.int16, np.int64),
        (np.float32, np.int64, np.int16),
        (np.float32, np.float32, np.int64),
        (np.float32, np.int32, np.int64),
        (np.int64, np.float32, jnp.bfloat16),
        (np.int64, jnp.bfloat16, np.int32),
        (jnp.bfloat16, np.int32, jnp.bfloat16),
    )
    def test_casting(self, original_dtype, save_dtype, load_dtype):
      sharding = jax.sharding.NamedSharding(
          jax.sharding.Mesh(np.asarray(jax.devices()), ('devices',)),
          jax.sharding.PartitionSpec(),
      )
      tree = {
          'jax_array': create_sharded_array(
              np.arange(len(jax.devices()), dtype=original_dtype),
              sharding,
          ),
          'numpy_array': np.arange(len(jax.devices()), dtype=original_dtype),
      }
      save_casted_tree = {
          'jax_array': create_sharded_array(
              np.arange(len(jax.devices()), dtype=save_dtype),
              sharding,
          ),
          'numpy_array': np.arange(len(jax.devices()), dtype=save_dtype),
      }
      load_casted_tree = {
          'jax_array': create_sharded_array(
              np.arange(len(jax.devices()), dtype=load_dtype),
              sharding,
          ),
          'numpy_array': np.arange(len(jax.devices()), dtype=load_dtype),
      }

      create_array_storage_options_fn = (
          lambda k, v: ocp.options.ArrayOptions.Saving.StorageOptions(
              dtype=save_dtype
          )
      )
      with ocp.Context(
          pytree_options=ocp.options.PyTreeOptions(
              saving=ocp.options.PyTreeOptions.Saving(
                  create_array_storage_options_fn=create_array_storage_options_fn
              )
          )
      ):
        ocp.save_pytree(self.directory, tree)

      with self.subTest('with_abstract_tree'):
        abstract_tree = jax.tree.map(as_abstract_type, load_casted_tree)
        loaded = ocp.load_pytree(self.directory, abstract_tree)
        test_utils.assert_tree_equal(self, load_casted_tree, loaded)

      with self.subTest('without_abstract_tree'):
        if multihost.is_pathways_backend():
          self.skipTest('Must provide abstract_pytree for Pathways.')
        loaded = ocp.load_pytree(self.directory)
        test_utils.assert_tree_equal(self, save_casted_tree, loaded)

    # TODO(b/295313820): Improve mismatched-tree error messages.
    def test_mismatched_abstract_tree(self):
      ocp.save_pytree(self.directory, self.pytree)

      with self.subTest('subset_of_keys'):
        abstract_pytree = dict(self.abstract_pytree)
        del abstract_pytree['a'], abstract_pytree['b']
        with self.assertRaisesRegex(ValueError, 'Dict key mismatch'):
          ocp.load_pytree(self.directory, abstract_pytree)

      with self.subTest('superset_of_keys'):
        abstract_pytree = dict(self.abstract_pytree)
        abstract_pytree['z'] = as_abstract_type(np.arange(16))
        with self.assertRaisesRegex(ValueError, 'Dict key mismatch'):
          ocp.load_pytree(self.directory, abstract_pytree)

      with self.subTest('renamed_key'):
        abstract_pytree = dict(self.abstract_pytree)
        abstract_pytree['z'] = abstract_pytree.pop('a')
        with self.assertRaisesRegex(ValueError, 'Dict key mismatch'):
          ocp.load_pytree(self.directory, abstract_pytree)
