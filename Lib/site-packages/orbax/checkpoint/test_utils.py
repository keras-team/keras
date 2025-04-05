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

"""Utils for orbax tests."""

# pylint: disable=protected-access

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from concurrent import futures
import contextlib
import copy
import inspect
import time
import typing
from typing import Any, List, Optional, Set
from unittest import mock

from absl import logging
from etils import epath
import jax
from jax.experimental import pjit
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import checkpoint_args
from orbax.checkpoint._src.handlers import async_checkpoint_handler
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler
from orbax.checkpoint._src.metadata import checkpoint as checkpoint_metadata
from orbax.checkpoint._src.metadata import step_metadata_serialization
from orbax.checkpoint._src.multihost import multihost
from orbax.checkpoint._src.multihost import multislice
from orbax.checkpoint._src.path import atomicity
from orbax.checkpoint._src.path import step as step_lib
from orbax.checkpoint._src.serialization import replica_slices
from orbax.checkpoint._src.serialization import serialization
from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
from orbax.checkpoint._src.serialization import type_handlers
from orbax.checkpoint._src.tree import utils as tree_utils


PyTree = Any


def sync_global_processes(name: str):
  jax.experimental.multihost_utils.sync_global_devices(name)


def save_fake_tmp_dir(
    directory: epath.Path,
    step: int,
    item: str,
    subdirs: Optional[List[str]] = None,
    step_prefix: Optional[str] = None,
) -> epath.Path:
  """Saves a directory with a tmp folder to simulate preemption."""
  subdirs = subdirs or []
  if not step_prefix:
    step_prefix = ''
  step_final_directory = directory / (step_prefix + str(step))
  step_tmp_directory = (
      step_final_directory
      if step_lib.is_gcs_path(step_final_directory)
      else atomicity._get_tmp_directory(step_final_directory)
  )
  create_tmp_directory(step_tmp_directory, step_final_directory)

  item_final_directory = step_tmp_directory / item
  item_tmp_directory = (
      item_final_directory
      if step_lib.is_gcs_path(item_final_directory)
      else atomicity._get_tmp_directory(item_final_directory)
  )
  create_tmp_directory(item_tmp_directory, item_final_directory)

  if multihost.process_index() == 0:
    for sub in subdirs:
      (item_tmp_directory / sub).mkdir(parents=True)
  sync_global_processes('save_fake_tmp_dir')
  return item_tmp_directory


def create_tmp_directory(
    tmp_dir: epath.Path,
    final_dir: epath.Path,
    *,
    primary_host: Optional[int] = 0,
    active_processes: Optional[Set[int]] = None,
    barrier_sync_key_prefix: Optional[str] = None,
    path_permission_mode: int = step_lib.WORLD_READABLE_MODE,
    metadata_store: Optional[checkpoint_metadata.MetadataStore] = None,
) -> epath.Path:
  """Creates a non-deterministic tmp directory for saving for given `final_dir`.

  Also writes checkpoint metadata in the tmp directory.

  Args:
    tmp_dir: The temporary directory path.
    final_dir: The eventual directory path where checkpoint will be committed.
    primary_host: primary host id, default=0.
    active_processes: Ids of active processes. default=None
    barrier_sync_key_prefix: A prefix to use for the barrier sync key.
    path_permission_mode: Path permission mode for the temp directory. e.g.
      0o750. Please check
      https://github.com/google/etils/blob/main/etils/epath/backend.py if your
        path is supported.
    metadata_store: optional `MetadataStore` instance. If present then it is
      used to create `StepMetadata` with current timestamp.

  Returns:
    The tmp directory.

  Raises:
    FileExistsError: if tmp directory already exists.
  """
  # Sync before existence is checked and directory is created because there are
  # additional existence checks happening in the callers of this function.
  multihost.sync_global_processes(
      multihost.unique_barrier_key(
          'create_tmp_directory:pre',
          prefix=barrier_sync_key_prefix,
          suffix=f'{final_dir.name}',
      ),
      timeout=multihost.DIRECTORY_CREATION_TIMEOUT,
      processes=active_processes,
  )
  if multihost.is_primary_host(primary_host):
    if tmp_dir.exists():
      if step_lib.is_tmp_checkpoint(tmp_dir):
        logging.warning(
            'Attempted to create temporary directory %s which already exists.'
            ' Removing existing directory since it is not finalized.',
            tmp_dir,
        )
        tmp_dir.rmtree()
      else:
        raise FileExistsError(
            f'Attempted to create temporary directory {tmp_dir} which already'
            ' exists but appears to be a finalized checkpoint.'
        )
    logging.info('Creating tmp directory %s', tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False, mode=path_permission_mode)
    if metadata_store is not None:
      metadata = checkpoint_metadata.StepMetadata(
          init_timestamp_nsecs=time.time_ns(),
      )
      metadata_store.write(
          file_path=checkpoint_metadata.step_metadata_file_path(tmp_dir),
          metadata=step_metadata_serialization.serialize(metadata),
      )

  multihost.sync_global_processes(
      multihost.unique_barrier_key(
          'create_tmp_directory:post',
          prefix=barrier_sync_key_prefix,
          suffix=f'{final_dir.name}',
      ),
      timeout=multihost.DIRECTORY_CREATION_TIMEOUT,
      processes=active_processes,
  )
  return tmp_dir


def replicate_sharded_array(arr: jax.Array):
  """Returns the input array, but replicated across all devices."""
  mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ('x',))
  replicated_sharding = jax.sharding.NamedSharding(
      mesh,
      jax.sharding.PartitionSpec(
          None,
      ),
  )
  return pjit.pjit(lambda x: x, out_shardings=replicated_sharding)(arr)


def apply_function(tree, function):
  """Applies the given function to every leaf in tree.

  Args:
    tree: a nested dict where every leaf is a sharded jax.Array.
    function: a function accepting an array and returning jax.Array.

  Returns:
    a transformed sharded array tree.
  """

  def f(arr):
    return pjit.pjit(function)(arr)

  return jax.tree.map(f, tree)


def assert_array_equal(testclass, v_expected, v_actual):
  """Asserts that two arrays are equal."""
  if hasattr(v_expected, 'dtype'):
    testclass.assertEqual(v_expected.dtype, v_actual.dtype)
  testclass.assertIsInstance(v_actual, type(v_expected))
  if isinstance(v_expected, jax.Array):
    if jax.dtypes.issubdtype(v_expected.dtype, jax.dtypes.prng_key):
      # random key
      testclass.assertTrue(
          jax.dtypes.issubdtype(v_actual.dtype, jax.dtypes.prng_key)
      )
      expected_array = jax.random.key_data(v_expected)
      actual_array = jax.random.key_data(v_actual)
      assert_array_equal(testclass, expected_array, actual_array)
      testclass.assertEqual(
          str(jax.random.key_impl(v_expected)),
          str(jax.random.key_impl(v_actual)),
      )
    else:
      # regular array
      testclass.assertEqual(
          len(v_expected.addressable_shards), len(v_actual.addressable_shards)
      )
      for shard_expected, shard_actual in zip(
          v_expected.addressable_shards, v_actual.addressable_shards
      ):
        np.testing.assert_array_equal(shard_expected.data, shard_actual.data)
  elif isinstance(v_expected, (np.ndarray, jnp.ndarray)):
    np.testing.assert_array_equal(v_expected, v_actual)
  else:
    testclass.assertEqual(v_expected, v_actual)


def assert_tree_equal(testclass, expected, actual):
  """Asserts that two PyTrees are equal."""
  expected_flat = tree_utils.to_flat_dict(expected)
  actual_flat = tree_utils.to_flat_dict(actual)
  testclass.assertSameElements(expected_flat.keys(), actual_flat.keys())

  def _eq(x, y):
    if x is None:
      return
    assert_array_equal(testclass, x, y)

  jax.tree.map(_eq, expected, actual, is_leaf=lambda x: x is None)


def setup_pytree(add: int = 0):
  """Creates a numpy PyTree for testing."""
  pytree = {
      'a': np.arange(8) * 1,
      'b': np.arange(16) * 2,
      'c': {
          'a': np.arange(8).reshape((2, 4)) * 3,
          'e': np.arange(16).reshape((4, 4)) * 4,
      },
  }
  pytree = jax.tree.map(lambda x: x + add, pytree, is_leaf=is_leaf)
  return pytree


def setup_sharded_pytree(
    pytree: Optional[pytree_checkpoint_handler.PyTree] = None,
    reverse_devices: bool = False,
):
  """Creates a PyTree of sharded arrays for testing."""
  devices = jax.devices()
  num_devices = len(devices)
  if reverse_devices:
    devices = np.asarray(list(reversed(devices)))
  else:
    devices = np.asarray(devices)

  mesh_2d = jax.sharding.Mesh(
      devices.reshape((2, num_devices // 2)), ('x', 'y')
  )
  mesh_axes_2d = jax.sharding.PartitionSpec('x', 'y')
  mesh_1d = jax.sharding.Mesh(devices, ('x',))
  mesh_axes_1d = jax.sharding.PartitionSpec(
      'x',
  )
  mesh_0d = jax.sharding.Mesh(devices, ('x',))
  mesh_axes_0d = jax.sharding.PartitionSpec(
      None,
  )

  if pytree is None:
    pytree = setup_pytree()
  mesh_tree = {
      'a': mesh_0d,
      'b': mesh_1d,
      'c': {
          'a': mesh_2d,
          'e': mesh_2d,
      },
  }
  axes_tree = {
      'a': mesh_axes_0d,
      'b': mesh_axes_1d,
      'c': {
          'a': mesh_axes_2d,
          'e': mesh_axes_2d,
      },
  }

  pytree = jax.tree.map(
      create_sharded_array, pytree, mesh_tree, axes_tree, is_leaf=is_leaf
  )
  return pytree, mesh_tree, axes_tree


def get_fake_global_mesh_for_slices(
    slice_processes: List[Set[int]],
    replica_axis_index: int = 0,
) -> jax.sharding.Mesh:
  """Creates a "multi-slice" global mesh for testing.

  Args:
    slice_processes: List of sets of process indices, where each element in the
      list is a set of processes that are active in a single slice.
    replica_axis_index: The index of the replica axis in the global mesh.

  Returns:
    A global mesh.
  """
  assert replica_axis_index in [0, 1]
  devices = jax.devices()
  slice_devices = []
  devices_per_slices = None
  all_processes = set()
  for processes in slice_processes:
    all_processes |= processes
    slice_devices.append([
        d
        for d in devices
        if multihost.runtime_to_distributed_process_id(d.process_index)
        in processes
    ])
    devices_per_slices = devices_per_slices or len(slice_devices[-1])
    if len(slice_devices[-1]) != devices_per_slices:
      raise ValueError('All slices must have the same number of devices.')
  if len(all_processes) != jax.process_count():
    raise ValueError('All processes must be accounted for.')

  slice_devices = np.asarray(slice_devices)
  axis_names = ('replica', 'data')
  if replica_axis_index == 1:
    slice_devices = np.transpose(slice_devices, (1, 0))
    axis_names = ('data', 'replica')
  return jax.sharding.Mesh(slice_devices, axis_names)


def select_single_replica(
    arrays: List[jax.Array], global_mesh: jax.sharding.Mesh
) -> List[jax.Array]:
  """Returns arrays sharded over single slice."""
  slice_devices = multislice.local_slice_devices(
      global_mesh, replica_axis_index=0
  )
  single_slice_mesh = jax.sharding.Mesh(
      slice_devices, global_mesh.axis_names[1:]
  )

  def _get_single_slice_sharding(arr: jax.Array):
    sharding = typing.cast(jax.sharding.NamedSharding, arr.sharding)
    return jax.sharding.NamedSharding(
        single_slice_mesh,
        jax.sharding.PartitionSpec(*sharding.spec),  # exclude 'replica' axis
    )

  single_slice_shardings = jax.tree.map(
      _get_single_slice_sharding,
      arrays,
  )

  def _make_single_slice_array(
      arr: jax.Array, single_slice_sharding: jax.sharding.NamedSharding
  ):
    data = [
        arr.addressable_data(idx) for idx in range(len(arr.addressable_shards))
    ]
    return jax.make_array_from_single_device_arrays(
        arr.shape, single_slice_sharding, data
    )

  return jax.tree.map(_make_single_slice_array, arrays, single_slice_shardings)


def is_leaf(x):
  return (
      isinstance(x, np.ndarray)
      or isinstance(x, jax.sharding.Mesh)
      or isinstance(x, jax.sharding.PartitionSpec)
      or isinstance(x, pytree_checkpoint_handler.ParamInfo)
  )


def create_sharded_array(arr, mesh, mesh_axes):
  """Create sharded jax.Array."""
  if isinstance(arr, (int, float)):
    arr = np.asarray(arr)
  return jax.make_array_from_callback(
      arr.shape,
      jax.sharding.NamedSharding(mesh, mesh_axes),
      lambda idx: np.asarray(arr[idx], dtype=arr.dtype),
  )


def print_directory(directory: epath.PathLike, level: int = 0):
  """Prints a directory tree for debugging purposes."""
  directory = epath.Path(directory)
  assert directory.is_dir()
  level_str = '..' * level
  if level == 0:
    logging.info('Printing directory tree: %s/', directory)
  else:
    logging.info('%s%s/', level_str, directory.name)

  level_str = '..' * (level + 1)
  for p in directory.iterdir():
    if p.is_dir():
      print_directory(p, level=level + 1)
    else:
      logging.info('%s%s', level_str, p.name)


def erase_and_create_empty(directory: epath.PathLike) -> epath.Path:
  """Creates an empty directory, erasing existing if present.

  Generally should be avoided in production code, as the heedless deletion is
  not very safe.

  Args:
    directory: directory to create.

  Returns:
    The created directory.
  """
  directory = epath.Path(directory)
  if directory.exists():
    directory.rmtree()
  directory.mkdir()
  return directory


def empty_directory(directory: epath.PathLike) -> bool:
  directory = epath.Path(directory)
  if not directory.exists():
    return False
  for p in directory.iterdir():
    if p.is_dir():
      p.rmtree()
    else:
      p.unlink()
  return True


def set_tensorstore_driver_for_test():
  # Sets TS driver for testing. Within Google, this defaults to `gfile`, which
  # results in issues writing to the OCDBT manifest. When using `gfile` on the
  # local filesystem, write operations are not atomic.
  ts_utils.DEFAULT_DRIVER = 'file'


class PyTreeCheckpointHandler(
    pytree_checkpoint_handler.PyTreeCheckpointHandler
):

  def save(self, directory, *args, **kwargs):
    super().save(directory, *args, **kwargs)
    sync_global_processes('PyTreeCheckpointHandler:save')
    if multihost.process_index() == 0:
      self.finalize(directory)
    sync_global_processes('PyTreeCheckpointHandler:finalize')


class ErrorCheckpointHandler(async_checkpoint_handler.AsyncCheckpointHandler):
  """Wrapper for PyTreeCheckpointHandler that has an error during save."""

  def __init__(
      self,
      handler: pytree_checkpoint_handler.PyTreeCheckpointHandler,
      executor: futures.ThreadPoolExecutor,
  ):
    self._handler = handler
    self._executor = executor

  def save(self, directory: epath.Path, args: ErrorSaveArgs):
    return self._handler.save(directory, args=args)

  def restore(self, directory: epath.Path, args: ErrorRestoreArgs):
    return self._handler.restore(directory, args=args)

  async def async_save(self, directory: epath.Path, args: ErrorSaveArgs):
    commit_futures = await self._handler.async_save(directory, args=args)

    def error_commit():
      time.sleep(3)  # Pretend to write data.
      a = 1
      b = 2
      if a != b:
        raise SystemError()
      return 42

    return commit_futures + [self._executor.submit(error_commit)]

  def finalize(self, directory: epath.Path):
    self._handler.finalize(directory)


@checkpoint_args.register_with_handler(ErrorCheckpointHandler, for_save=True)
class ErrorSaveArgs(pytree_checkpoint_handler.PyTreeSaveArgs):
  pass


@checkpoint_args.register_with_handler(ErrorCheckpointHandler, for_restore=True)
class ErrorRestoreArgs(pytree_checkpoint_handler.PyTreeSaveArgs):
  pass


@contextlib.contextmanager
def register_type_handler(ty, handler, func):
  """Registers new func for type, and restores original handler when done."""
  original_handler = type_handlers.get_type_handler(ty)
  type_handlers.register_type_handler(ty, handler, func=func, override=True)
  try:
    yield
  finally:
    type_handlers.register_type_handler(
        ty, original_handler, func=func, override=True
    )


@contextlib.contextmanager
def global_type_handler_registry_context():
  """Context manager for changing the GLOBAL_TYPE_HANDLER_REGISTRY."""
  original_type_handlers = copy.deepcopy(type_handlers._DEFAULT_TYPE_HANDLERS)
  try:
    yield
  finally:
    for original_type, original_handler in original_type_handlers:
      type_handlers.GLOBAL_TYPE_HANDLER_REGISTRY.add(
          original_type, original_handler, override=True
      )


@contextlib.contextmanager
def ocdbt_checkpoint_context(use_ocdbt: bool, ts_context: Any):
  """Use OCDBT driver within context."""
  original_type_handlers = copy.deepcopy(type_handlers._DEFAULT_TYPE_HANDLERS)
  if use_ocdbt:
    type_handlers.register_standard_handlers_with_options(
        use_ocdbt=use_ocdbt, ts_context=ts_context
    )
  try:
    yield
  finally:
    for original_type, original_handler in original_type_handlers:
      type_handlers.GLOBAL_TYPE_HANDLER_REGISTRY.add(
          original_type, original_handler, override=True
      )


def _get_test_wrapper_function(test_func):
  """Creates a function to wrap a test method with custom patches."""

  def test_wrapper(self, *args, **kwargs):

    def _get_unique_barrier_key(key: str) -> str:
      return f'{key}.{self.id()}'

    with mock.patch.object(
        multihost,
        '_unique_barrier_key',
        new=_get_unique_barrier_key,
    ):
      return test_func(self, *args, **kwargs)

  return test_wrapper


def barrier_compatible_test(cls):
  """A decorator to be used with a test class.

  This is primarily needed when different processes in a multihost test may be
  executing different code. This will cause operation IDs to get out of sync. If
  all processes always execute the same code, this decorator is not needed.

  E.g.

  @barrier_compatible_test
  class MyTest(googletest.TestCase):
    def test_foo(self):
      ...

  The point of this decorator is to modify all functions to mock the private
  method `multihost._unique_barrier_key`, to append the test case name.
  This allows multiple test cases to reuse barrier names that would otherwise
  conflict.

  Args:
    cls: the test class to decorate.

  Returns:
    The decorated class.
  """
  for name, func in inspect.getmembers(cls, predicate=inspect.isfunction):
    if name.startswith('test'):
      setattr(cls, name, _get_test_wrapper_function(func))
  return cls


def ensure_atomic_save(
    temp_ckpt_dir: epath.Path,
    final_ckpt_dir: epath.Path,
    metadata_store: Optional[checkpoint_metadata.MetadataStore] = None,
):
  """Wrapper around TemporaryPath.finalize for testing."""
  if temp_ckpt_dir == final_ckpt_dir:
    atomicity.CommitFileTemporaryPath(
        temp_ckpt_dir,
        final_ckpt_dir,
        checkpoint_metadata_store=metadata_store,
    ).finalize(
    )
  else:
    atomicity.AtomicRenameTemporaryPath(
        temp_ckpt_dir,
        final_ckpt_dir,
        checkpoint_metadata_store=metadata_store,
    ).finalize(
    )


def create_single_replica_restore_args(
    arr: jax.Array,
    mesh: jax.sharding.Mesh,
    pspec: jax.sharding.PartitionSpec,
    replica_axis_index: int,
):
  replica_devices = _replica_devices(mesh.devices, replica_axis_index)
  replica_mesh = jax.sharding.Mesh(replica_devices, mesh.axis_names)
  ss_sharding = jax.sharding.NamedSharding(replica_mesh, pspec)

  return type_handlers.SingleReplicaArrayRestoreArgs(
      sharding=jax.sharding.NamedSharding(mesh, pspec),
      single_replica_sharding=ss_sharding,
      global_shape=arr.shape,
      dtype=arr.dtype,
  )


def _find_idx(array: np.ndarray, replica_axis_idx: int):
  """Returns the index along given dimension that the current host belongs to."""
  idx = None
  for idx, val in np.ndenumerate(array):
    if val.process_index == multihost.process_index():
      break
  return idx[replica_axis_idx]


def _replica_devices(device_array: np.ndarray, replica_axis_idx: int):
  """Returns the devices from the replica that current host belongs to.

  Replicas are assumed to be restricted to the first axis.

  Args:
    device_array: devices of the mesh that can be obtained by mesh.devices()
    replica_axis_idx: axis dimension along which replica is taken

  Returns:
    devices inside the replica that current host is in
  """
  idx = _find_idx(device_array, replica_axis_idx)
  replica_result = np.take(device_array, idx, axis=replica_axis_idx)
  return np.expand_dims(replica_result, axis=replica_axis_idx)


class TestLimitInFlightBytes(serialization.LimitInFlightBytes):
  """Limits in-flight bytes when reading/writing checkpoints per process."""

  def __init__(self, limit_bytes: int, sleep_time: float):
    super().__init__(limit_bytes)
    self.sleep_time = sleep_time
    self.completion_times = []

  async def wait_for_bytes(self, requested_bytes: int):
    await super().wait_for_bytes(requested_bytes)
    await asyncio.sleep(self.sleep_time / 2)

  async def release_bytes(self, requested_bytes: int):
    await asyncio.sleep(self.sleep_time / 2)
    await super().release_bytes(requested_bytes)
    self.completion_times.append(time.time())


def get_byte_limiter(
    concurrent_bytes: int, sleep_time: float
) -> TestLimitInFlightBytes:
  return TestLimitInFlightBytes(concurrent_bytes, sleep_time)


def concurrent_gb_test_setup():
  """Setup for tests exercising concurrent_gb setting."""
  # Need to override later so we can use a small number of bytes.
  handler = PyTreeCheckpointHandler(
      save_concurrent_gb=1, restore_concurrent_gb=1, use_ocdbt=False
  )

  mesh = jax.sharding.Mesh(
      jax.devices(),
      ('x',),
  )
  pspec = jax.sharding.PartitionSpec(
      None,
  )

  def _create_sharded_array(arr):
    return create_sharded_array(arr, mesh, pspec)

  # 4 arrays, each has a single chunk, with 4 bytes each.
  tree = jax.tree.map(
      _create_sharded_array,
      {
          'a': np.arange(1, dtype=np.int32),
          'b': np.arange(1, dtype=np.int32),
          'c': np.arange(1, dtype=np.int32),
          'd': np.arange(1, dtype=np.int32),
      },
  )
  restore_args = jax.tree.map(
      lambda _: type_handlers.ArrayRestoreArgs(
          sharding=jax.sharding.NamedSharding(mesh, pspec)
      ),
      tree,
  )
  return handler, tree, restore_args


def assert_every_n_is_x_apart(testclass, values, n, x):
  # For an array of values which is divided into sub-arrays of size n,
  # asserts that the first element of every group is at least x greater
  # than the last element of the previous group.
  values = sorted(values)
  assert len(values) % n == 0
  for i in range(n, len(values), n):
    testclass.assertGreaterEqual(values[i], values[i - 1] + x)


def get_expected_chunk_shape(
    arr: jax.Array, *, use_replica_parallel: bool = True
) -> tuple[int, ...]:
  """Expected chunk shape for an array, accounting for replica-parallel."""
  if not use_replica_parallel:
    return arr.sharding.shard_shape(arr.shape)
  _, local_shape = (
      replica_slices.calculate_replica_parallel_axis_and_local_shape(arr)
  )
  if local_shape is None:
    local_shape = arr.sharding.shard_shape(arr.shape)
  return local_shape


def filter_metadata_fields(
    pytree: PyTree, include_fields: Sequence[str]
) -> PyTree:
  """Returns a PyTree of dicts with keys in `include_fields`."""

  def _include(metadata):
    result = {}
    for f in include_fields:
      if hasattr(metadata, f):
        result[f] = getattr(metadata, f)
    return result

  return jax.tree.map(_include, pytree)
