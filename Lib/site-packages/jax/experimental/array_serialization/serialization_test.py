# Copyright 2021 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for serialization and deserialization of GDA."""

import asyncio
import math
from functools import partial
import os
import pathlib
import tracemalloc as tm

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
from jax._src import array
from jax.sharding import NamedSharding, GSPMDSharding, SingleDeviceSharding
from jax.sharding import PartitionSpec as P
from jax.experimental.array_serialization import serialization
from jax.experimental.layout import Layout, DeviceLocalLayout as DLL
import numpy as np
import tensorstore as ts

jax.config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


class CheckpointTest(jtu.JaxTestCase):

  def _on_commit_callback(self, temp_ckpt_dir, final_ckpt_dir):
    os.rename(temp_ckpt_dir, final_ckpt_dir)

  def test_deserialize_on_array_list(self):
    global_mesh = jtu.create_mesh((2, 4), ('x', 'y'))
    inp_shape = (16, 64)
    pspec = P('x', 'y')
    sharding = NamedSharding(global_mesh, pspec)
    inputs = []
    lambda_fn = lambda idx: src[idx]
    num_arrays = 5
    for _ in range(num_arrays):
      src = jax.random.normal(jax.random.key(0), inp_shape)
      inp = array.make_array_from_callback(inp_shape, sharding, lambda_fn)
      inputs.append(inp)
    ckpt_dir = pathlib.Path(self.create_tempdir().full_path)
    tspecs = [
        serialization.get_tensorstore_spec(f'{ckpt_dir}/array_{i}')
        for i in range(num_arrays)
    ]
    inputs = tuple(inputs)
    tspecs = tuple(tspecs)
    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize(
        inputs,
        tspecs,
        on_commit_callback=partial(
            self._on_commit_callback, ckpt_dir, ckpt_dir
        ),
    )
    manager.wait_until_finished()
    shardings = tuple([sharding] * num_arrays)
    restored_arrays = manager.deserialize(shardings, tspecs)
    self.assertLen(restored_arrays, num_arrays)
    for inp, deserialized_array in zip(inputs, restored_arrays):
      self.assertArraysEqual(deserialized_array, inp)

  @jtu.skip_on_devices('cpu')
  def test_memory_consumption(self):
    global_mesh = jtu.create_mesh((2, 4), ('x', 'y'))
    inp_shape = (2_048, 4_096)
    pspec = P('x', 'y')
    num = math.prod(inp_shape)
    sharding = NamedSharding(global_mesh, pspec)
    src = jnp.arange(num, dtype=np.int32).reshape(inp_shape)  # 8e9
    inp = array.make_array_from_callback(
        inp_shape, sharding,
        lambda idx: src[idx])
    ckpt_dir = pathlib.Path(self.create_tempdir('memprof').full_path)
    tspec = serialization.get_tensorstore_spec(str(ckpt_dir))

    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize(
        [inp], [tspec],
        on_commit_callback=partial(self._on_commit_callback, ckpt_dir, ckpt_dir))
    manager.wait_until_finished()

    async def deserialize_with_byte_limit():
      r = await serialization.async_deserialize(
        sharding, tspec, inp_shape,
        byte_limiter=serialization._LimitInFlightBytes(4_200_000))
      r.block_until_ready()

    tm.start()
    asyncio.run(deserialize_with_byte_limit())
    unused_current, peak = tm.get_traced_memory()
    # NB: some padding + tensorstore overhead. It should always be
    # less than array size (2048 * 4096 * 4 = 32M)
    self.assertLess(peak, 10_000_000)
    deserialize_wo_limit = serialization.async_deserialize(
        sharding, tspec, inp_shape)
    tm.clear_traces()
    # NB: call block_until_ready() is important here and above
    # because otherwise this leads to racing condition and segfault with
    # tensorstore attempting to dealloc using tracemalloc which is already
    # destroyed.
    asyncio.run(deserialize_wo_limit).block_until_ready()

    unused_current, peak = tm.get_traced_memory()
    # We load entire array in memory here.
    self.assertGreater(peak, 30_000_000)
    tm.stop()

  def test_memory_consumption_for_save(self):
    global_mesh = jtu.create_mesh((1, 1), ('x', 'y'))
    inp_shape = (16 * 1024, 16 * 1024)
    pspec = P('x', 'y')
    num = math.prod(inp_shape)
    sharding = NamedSharding(global_mesh, pspec)
    src = jnp.arange(num, dtype=np.int32).reshape(inp_shape)
    inp = array.make_array_from_callback(
        inp_shape, sharding, lambda idx: src[idx]
    )
    ckpt_dir = pathlib.Path(self.create_tempdir('memprofsave').full_path)
    tspec = serialization.get_tensorstore_spec(str(ckpt_dir))
    tspec['metadata'] = {
        'shape': inp.shape,
        'compressor': None,
        'chunks': inp.shape,
    }

    is_cpu = jtu.test_device_matches(['cpu'])
    tm.start()
    try:
      manager = serialization.GlobalAsyncCheckpointManager()
      manager.serialize(
          [inp],
          [tspec],
          on_commit_callback=partial(
              self._on_commit_callback, ckpt_dir, ckpt_dir
          ),
      )
      manager.wait_until_finished()
      unused_current, peak = tm.get_traced_memory()
      self.assertLess(peak, src.nbytes * (1 * (not is_cpu) + 0.5))
    finally:
      tm.stop()

  def test_checkpointing_with_path_variant(self):
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    inp_shape = (8, 2)
    pspec = P('x', 'y')
    num = math.prod(inp_shape)

    # First Array
    global_input_data1 = np.arange(num, dtype=np.int32).reshape(inp_shape)
    a1 = array.make_array_from_callback(
        inp_shape, NamedSharding(global_mesh, pspec),
        lambda idx: global_input_data1[idx])
    ckpt_dir = pathlib.Path(self.create_tempdir('ckpt_variant').full_path)
    ckpt_path1 = pathlib.Path(
        self.create_tempdir(f'{ckpt_dir}/first').full_path)

    ckpt_paths = [str(ckpt_path1)]
    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize_with_paths(
        [a1], ckpt_paths,
        on_commit_callback=partial(self._on_commit_callback, ckpt_dir, ckpt_dir))
    manager.wait_until_finished()

    m1, = manager.deserialize_with_paths(
        [NamedSharding(global_mesh, pspec)], ckpt_paths)
    self.assertIsInstance(m1, array.ArrayImpl)
    self.assertArraysEqual(np.asarray(m1.addressable_shards[0].data),
                           np.array([[0], [2]], dtype=np.int32))
    self.assertArraysEqual(np.asarray(m1.addressable_shards[1].data),
                           np.array([[1], [3]], dtype=np.int32))
    self.assertEqual(m1.addressable_shards[0].data.shape, (2, 1))
    self.assertEqual(m1.dtype, np.int32)

  def test_checkpointing_jax_array(self):
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    inp_shape = (8, 2)
    pspec = P('x', 'y')
    num = math.prod(inp_shape)

    # First Array
    global_input_data1 = np.arange(num, dtype=np.int32).reshape(inp_shape)
    a1 = array.make_array_from_callback(
        inp_shape, NamedSharding(global_mesh, pspec),
        lambda idx: global_input_data1[idx])
    ckpt_dir = pathlib.Path(self.create_tempdir('ckpt').full_path)
    ckpt_path1 = pathlib.Path(self.create_tempdir(f'{ckpt_dir}/first').full_path)

    # Second Array
    global_input_data2 = np.arange(
        num, num + num, dtype=np.int32).reshape(inp_shape)
    a2 = array.make_array_from_callback(
        inp_shape, NamedSharding(global_mesh, pspec),
        lambda idx: global_input_data2[idx])
    ckpt_path2 = pathlib.Path(self.create_tempdir(f'{ckpt_dir}/second').full_path)

    # Third Array
    def cb3(_):
      return np.array([], dtype=np.float32)
    global_mesh1d = jtu.create_mesh((8,), ('x',))
    a3 = array.make_array_from_callback(
        (0,), NamedSharding(global_mesh1d, P(None)), cb3)
    ckpt_path3 = pathlib.Path(self.create_tempdir(f'{ckpt_dir}/third').full_path)

    ckpt_paths = [str(ckpt_path1), str(ckpt_path2), str(ckpt_path3)]
    tspecs = jax.tree_util.tree_map(serialization.get_tensorstore_spec, ckpt_paths)

    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize(
        [a1, a2, a3], tspecs,
        on_commit_callback=partial(self._on_commit_callback, ckpt_dir, ckpt_dir))
    manager.wait_until_finished()

    m1, m2, m3 = serialization.run_deserialization(
        [NamedSharding(global_mesh, pspec),
         NamedSharding(global_mesh, P('x')),
         NamedSharding(global_mesh1d, P(None))],
        tspecs)

    self.assertIsInstance(m1, array.ArrayImpl)
    self.assertArraysEqual(np.asarray(m1.addressable_shards[0].data),
                           np.array([[0], [2]], dtype=np.int32))
    self.assertArraysEqual(np.asarray(m1.addressable_shards[1].data),
                           np.array([[1], [3]], dtype=np.int32))
    self.assertEqual(m1.addressable_shards[0].data.shape, (2, 1))
    self.assertEqual(m1.dtype, np.int32)

    self.assertIsInstance(m2, array.ArrayImpl)
    self.assertArraysEqual(np.asarray(m2.addressable_shards[0].data),
                           np.array([[16, 17], [18, 19]], dtype=np.int32))
    self.assertArraysEqual(np.asarray(m2.addressable_shards[1].data),
                           np.array([[16, 17], [18, 19]], dtype=np.int32))
    self.assertEqual(m2.addressable_shards[0].data.shape, (2, 2))
    self.assertEqual(m2.dtype, np.int32)

    self.assertIsInstance(m3, array.ArrayImpl)
    for i, s in enumerate(m3.addressable_shards):
      self.assertEqual(s.index, (slice(None),))
      self.assertEqual(s.replica_id, i)
      self.assertArraysEqual(np.asarray(s.data), np.array([], dtype=np.float32))
    self.assertEqual(m3.dtype, np.float32)

  def test_checkpointing_ocdbt_transaction(self):
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    inp_shape = (8, 2)
    pspec = P('x', 'y')
    num = math.prod(inp_shape)

    # First Array
    global_input_data1 = np.arange(num, dtype=np.int32).reshape(inp_shape)
    a1 = array.make_array_from_callback(
        inp_shape,
        NamedSharding(global_mesh, pspec),
        lambda idx: global_input_data1[idx],
    )
    ckpt_dir = pathlib.Path(self.create_tempdir('ckpt').full_path)
    ckpt_path1 = ckpt_dir / 'first'

    # Second Array
    global_input_data2 = np.arange(num, num + num, dtype=np.int32).reshape(
        inp_shape
    )
    a2 = array.make_array_from_callback(
        inp_shape,
        NamedSharding(global_mesh, pspec),
        lambda idx: global_input_data2[idx],
    )
    ckpt_path2 = ckpt_dir / 'second'

    # Third Array
    def cb3(_):
      return np.array([], dtype=np.float32)

    global_mesh1d = jtu.create_mesh((8,), ('x',))
    a3 = array.make_array_from_callback(
        (0,), NamedSharding(global_mesh1d, P(None)), cb3
    )
    ckpt_path3 = ckpt_dir / 'third'

    ckpt_paths = [str(ckpt_path1), str(ckpt_path2), str(ckpt_path3)]
    tspecs = jax.tree_util.tree_map(
        lambda p: serialization.get_tensorstore_spec(p, ocdbt=True), ckpt_paths
    )

    manager = serialization.GlobalAsyncCheckpointManager()
    with ts.Transaction(atomic=True) as transaction:
      manager.serialize(
          [a1, a2, a3],
          tspecs,
          on_commit_callback=partial(
              self._on_commit_callback, ckpt_dir, ckpt_dir
          ),
          transaction=transaction,
      )
    manager.wait_until_finished()

    m1, m2, m3 = serialization.run_deserialization(
        [
            NamedSharding(global_mesh, pspec),
            NamedSharding(global_mesh, P('x')),
            NamedSharding(global_mesh1d, P(None)),
        ],
        tspecs,
    )

    self.assertIsInstance(m1, array.ArrayImpl)
    self.assertArraysEqual(
        np.asarray(m1.addressable_shards[0].data),
        np.array([[0], [2]], dtype=np.int32),
    )
    self.assertArraysEqual(
        np.asarray(m1.addressable_shards[1].data),
        np.array([[1], [3]], dtype=np.int32),
    )
    self.assertEqual(m1.addressable_shards[0].data.shape, (2, 1))
    self.assertEqual(m1.dtype, np.int32)

    self.assertIsInstance(m2, array.ArrayImpl)
    self.assertArraysEqual(
        np.asarray(m2.addressable_shards[0].data),
        np.array([[16, 17], [18, 19]], dtype=np.int32),
    )
    self.assertArraysEqual(
        np.asarray(m2.addressable_shards[1].data),
        np.array([[16, 17], [18, 19]], dtype=np.int32),
    )
    self.assertEqual(m2.addressable_shards[0].data.shape, (2, 2))
    self.assertEqual(m2.dtype, np.int32)

    self.assertIsInstance(m3, array.ArrayImpl)
    for i, s in enumerate(m3.addressable_shards):
      self.assertEqual(s.index, (slice(None),))
      self.assertEqual(s.replica_id, i)
      self.assertArraysEqual(np.asarray(s.data), np.array([], dtype=np.float32))
    self.assertEqual(m3.dtype, np.float32)

  @parameterized.product(input_dtype=[np.int32, jnp.bfloat16])
  def test_checkpointing_with_bigger_shape_jax_array(self, input_dtype):
    global_mesh = jtu.create_mesh((2, 2), ('x', 'y'), iota_order=True)
    global_input_shape = (8, 2)
    num = math.prod(global_input_shape)

    global_input_data1 = np.arange(num, dtype=input_dtype).reshape(
        global_input_shape
    )
    def cb1(index):
      return global_input_data1[index]
    arr = array.make_array_from_callback(
        global_input_shape, NamedSharding(global_mesh, P('x', 'y')), cb1)
    ckpt_dir = pathlib.Path(self.create_tempdir('first').full_path)

    ckpt_paths = [str(ckpt_dir)]
    tspecs = jax.tree_util.tree_map(serialization.get_tensorstore_spec, ckpt_paths)

    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize(
        [arr], tspecs,
        on_commit_callback=partial(self._on_commit_callback, ckpt_dir, ckpt_dir))
    manager.wait_until_finished()

    ds = NamedSharding(jtu.create_mesh((4, 2), ('x', 'y'), iota_order=True),
                       P('x', 'y'))

    m1, = serialization.run_deserialization([ds], tspecs, [(12, 2)],
                                            [np.float32])

    expected_data = {
        0: np.array([[0], [2], [4]], dtype=np.float32),
        1: np.array([[1], [3], [5]], dtype=np.float32),
        2: np.array([[6], [8], [10]], dtype=np.float32),
        3: np.array([[7], [9], [11]], dtype=np.float32),
        4: np.array([[12], [14], [0]], dtype=np.float32),
        5: np.array([[13], [15], [0]], dtype=np.float32),
        6: np.array([[0], [0], [0]], dtype=np.float32),
        7: np.array([[0], [0], [0]], dtype=np.float32),
    }

    for l in m1.addressable_shards:
      self.assertArraysEqual(np.asarray(l.data), expected_data[l.device.id])

    new_ds = GSPMDSharding.get_replicated(list(global_mesh.devices.flat))
    m2, = serialization.run_deserialization([new_ds], tspecs, [(8, 2)], [np.float32])
    for l in m2.addressable_shards:
      self.assertArraysEqual(l.data, global_input_data1.astype('float32'))

  @parameterized.product(input_dtype=[jnp.int4, jnp.int8])
  def test_checkpointing_with_int4(self, input_dtype):
    if config.use_shardy_partitioner.value:
      self.skipTest("TODO(b/376077396): Fix XlaRuntimeError: INVALID_ARGUMENT")
    global_mesh = jtu.create_mesh((2, 2), ('x', 'y'), iota_order=True)
    global_input_shape = (8, 2)
    num = math.prod(global_input_shape)

    global_input_data = np.arange(num, dtype=input_dtype).reshape(
        global_input_shape
    )
    def cb(index):
      return global_input_data[index]
    arr = array.make_array_from_callback(
        global_input_shape, NamedSharding(global_mesh, P('x', 'y')), cb)
    ckpt_dir = pathlib.Path(self.create_tempdir('first').full_path)

    ckpt_paths = [str(ckpt_dir)]
    tspecs = jax.tree_util.tree_map(serialization.get_tensorstore_spec, ckpt_paths)

    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize(
        [arr], tspecs,
        on_commit_callback=partial(self._on_commit_callback, ckpt_dir, ckpt_dir))
    manager.wait_until_finished()

    ds = NamedSharding(jtu.create_mesh((4, 2), ('x', 'y'), iota_order=True),
                       P('x', 'y'))

    target_dtype = jnp.dtype('int4')
    m1, = serialization.run_deserialization([ds], tspecs, [(12, 2)],
                                            [target_dtype])

    # values bigger than 7 are converted properly.
    expected_data = {
        0: jnp.array([[0], [2], [4]], dtype=target_dtype),
        1: jnp.array([[1], [3], [5]], dtype=target_dtype),
        2: jnp.array([[6], [8], [10]], dtype=target_dtype),
        3: jnp.array([[7], [9], [11]], dtype=target_dtype),
        4: jnp.array([[12], [14], [0]], dtype=target_dtype),
        5: jnp.array([[13], [15], [0]], dtype=target_dtype),
        6: jnp.array([[0], [0], [0]], dtype=target_dtype),
        7: jnp.array([[0], [0], [0]], dtype=target_dtype),
    }

    for l in m1.addressable_shards:
      self.assertArraysEqual(np.asarray(l.data), expected_data[l.device.id])

    new_ds = GSPMDSharding.get_replicated(list(global_mesh.devices.flat))
    m2, = serialization.run_deserialization([new_ds], tspecs, [(8, 2)], [target_dtype])
    for l in m2.addressable_shards:
      self.assertArraysEqual(l.data, global_input_data.astype(target_dtype))

  def test_checkpointing_scalar_jax_array(self):
    global_mesh = jtu.create_mesh((2,), ('x'))
    global_input_shape = ()
    data = np.array(4)
    s = NamedSharding(global_mesh, P(None))
    array1 = array.make_array_from_callback(
        global_input_shape, s, lambda idx: data[idx])
    ckpt_dir = pathlib.Path(self.create_tempdir('first').full_path)

    ckpt_paths = [str(ckpt_dir)]
    tspecs = jax.tree_util.tree_map(serialization.get_tensorstore_spec, ckpt_paths)

    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize(
        [array1], tspecs,
        on_commit_callback=partial(self._on_commit_callback, ckpt_dir, ckpt_dir))
    manager.wait_until_finished()

    ds = NamedSharding(jtu.create_mesh((2,), ('x')), P(None))

    m1, = serialization.run_deserialization(
        [ds],
        tspecs,
        [()],
        [np.float32]
    )

    for l in m1.addressable_shards:
      self.assertArraysEqual(np.asarray(l.data), data.astype(np.float32))

  def test_deserialize_tensorstore_array_jax_array(self):
    global_mesh = jtu.create_mesh((2,), ('x'))
    data = np.arange(1024)
    tspec = ts.array(data).spec()
    m1, = serialization.run_deserialization(
        [NamedSharding(global_mesh, P(None))],
        [tspec]
    )
    for l in m1.addressable_shards:
      self.assertArraysEqual(np.asarray(l.data), data)

  def test_spec_has_metadata(self):
    spec = {
        'a': {
            'b': 1,
            'c': 2,
        },
        'd': 3,
        'e': {
            'a': 2,
            'metadata': 3
        },
        'f': 4
    }
    self.assertTrue(serialization._spec_has_metadata(spec))
    self.assertTrue(
        serialization._spec_has_metadata({
            'driver': 'zarr',
            'kvstore': 'gfile',
            'metadata': {
                'chunks': 4,
                'shape': (32, 64)
            },
            'one_more': 'thing'
        }))

  def test_spec_has_no_metadata(self):
    spec = {
        'a': {
            'b': 1,
            'c': 2,
        },
        'd': 3,
        'e': {
            'a': 2,
        },
        'f': 4
    }
    self.assertFalse(serialization._spec_has_metadata(spec))

  def test_empty_spec_has_no_metadata(self):
    spec = {}
    self.assertFalse(serialization._spec_has_metadata(spec))

  @parameterized.named_parameters(
      ('gcs', 'gs://my/ckpt/dir/path'),
      ('file', '/my/ckpt/dir/path')
  )
  def test_get_tensorstore_spec_ocdbt(self, path):
    spec = serialization.get_tensorstore_spec(path, ocdbt=True)
    is_gcs_path = path.startswith('gs://')
    if is_gcs_path:
      self.assertEqual(spec['kvstore']['base'], os.path.dirname(path))
    else:
      self.assertEqual(spec['kvstore']['base'],
                       f'{serialization._DEFAULT_DRIVER}://{os.path.dirname(path)}')
    self.assertEqual(spec['kvstore']['path'], 'path')

  def test_get_tensorstore_spec_not_absolute_path(self):
    path = 'my/ckpt/path'
    with self.assertRaisesRegex(ValueError,
                                "Checkpoint path should be absolute"):
      serialization.get_tensorstore_spec(path, ocdbt=True)

  def test_maybe_cloud_storage(self):
    gs_path = 'gs://some-buck/path'
    gs_spec = serialization.get_tensorstore_spec(gs_path, ocdbt=True)
    self.assertTrue(serialization.is_remote_storage(gs_spec))

    local_path = '/tmp/checkpoint'
    local_spec = serialization.get_tensorstore_spec(local_path, ocdbt=True)
    self.assertFalse(serialization.is_remote_storage(local_spec))

    nested_tspec = {
        'driver': 'cast',
        'dtype': 'int32',
        'base': {
            'driver': 'zarr',
            'kvstore': {'driver': 'ocdbt', 'base': 's3://some-bucket/path'},
        },
    }
    self.assertTrue(serialization.is_remote_storage(nested_tspec))

  def test_load_with_layout(self):
    if not jtu.test_device_matches(['tpu']):
      self.skipTest('Layouts are only supported on TPUs')

    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    np_inp = np.arange(32).reshape(8, 4)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    out_layout = jax.jit(lambda x: x.T, out_shardings=Layout(DLL.AUTO)).lower(
        arr).compile().output_layouts
    self.assertEqual(arr.layout.device_local_layout.major_to_minor,
                     out_layout.device_local_layout.major_to_minor[::-1])

    ckpt_dir = pathlib.Path(self.create_tempdir('ckpt').full_path)
    ckpt_path = pathlib.Path(self.create_tempdir(f'{ckpt_dir}/first').full_path)
    tspecs = jax.tree_util.tree_map(serialization.get_tensorstore_spec, [ckpt_path])

    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize(
        [arr], tspecs,
        on_commit_callback=partial(self._on_commit_callback, ckpt_dir, ckpt_dir))
    manager.wait_until_finished()

    out, = serialization.run_deserialization([out_layout], tspecs)

    self.assertEqual(out.layout, out_layout)
    self.assertIsInstance(out, array.ArrayImpl)
    self.assertArraysEqual(out, np_inp)
    for s in out.addressable_shards:
      self.assertArraysEqual(s.data, np_inp[s.index])

  def test_deserialization_with_int4(self):
    if config.use_shardy_partitioner.value:
      self.skipTest("TODO(b/376077396): Fix XlaRuntimeError: INVALID_ARGUMENT")
    if jtu.test_device_matches(['gpu']):
      self.skipTest("Fails on GPU. Enable after it's fixed")
    dtype = jnp.int4
    shape = (8, 2)
    arr = jnp.arange(np.prod(shape)).reshape(shape).astype(dtype)

    ckpt_dir = pathlib.Path(self.create_tempdir('test_ckpt').full_path)

    # Run serialization.
    sharding = jax.sharding.GSPMDSharding.get_replicated(jax.devices())
    tspecs = jax.tree_util.tree_map(
        serialization.get_tensorstore_spec, [ckpt_dir]
    )
    manager = serialization.GlobalAsyncCheckpointManager()
    manager.serialize(
        [arr],
        tspecs,
        on_commit_callback=lambda: None,
    )
    manager.wait_until_finished()

    # Run deserialization.
    deserialized_arr, = serialization.run_deserialization(
        shardings=[sharding],
        tensorstore_specs=tspecs,
        global_shapes=[shape],
        dtypes=[dtype],
    )

    out = deserialized_arr.astype(jnp.int8)  # doesn't crash
    self.assertEqual(out.dtype, jnp.int8)
    self.assertArraysEqual(out + out, out * 2)


class TransferShardTest(jtu.JaxTestCase):

  @jtu.skip_on_devices('cpu')
  def test_transfer_shard_to_host(self):
    np_inp = np.arange(16).reshape((4, 4))
    sharding = SingleDeviceSharding(jax.devices()[0], memory_kind="device")
    arr = jax.device_put(np_inp, sharding)
    shard = arr.addressable_shards[0]

    np_out = asyncio.run(serialization.transfer_shard_to_host(shard))

    self.assertArraysEqual(np_out, np_inp)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
