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
"""Array serialization and deserialization."""

from __future__ import annotations

import abc
import asyncio
from collections.abc import Awaitable, Callable, Sequence
from functools import partial
import itertools
import logging
import os
import re
import threading
import time
from typing import Any, Optional

import jax
from jax._src import array
from jax._src import distributed
from jax._src import sharding
from jax._src.layout import Layout
from jax._src import typing
from jax._src import util
from jax._src.lib import xla_extension as xe
import jax.numpy as jnp
import numpy as np
import tensorstore as ts


TS_CONTEXT = ts.Context({'file_io_concurrency': {'limit': 128}})
_REMOVED_VALUE = 'Value removed'
_CHECKPOINT_SUCCESS = 'checkpoint_write_success'
_module_unique_count = itertools.count()
_DEFAULT_DRIVER = 'file'
_DISTRIBUTED_SYSTEM_MSG = (
    'Please initialize the distributed system via '
    '`jax.distributed.initialize()` at the start of your program.')
_REMOTE_URL_PREFIXES = ['gs://', 's3://']
_REMOTE_DRIVER_VALIDATIONS = [
    {'driver': 'gcs', 'path_regex': None},
    {'driver': 's3', 'path_regex': None},
]

class BarrierTimeoutException(Exception):
  pass

_BARRIER_TIMED_OUT_MSG = (
    "Suggestions for possible fixes:\n"
    "* Check the logs to see if one or more processes failed.\n"
    "* Make sure the training and checkpointing endpoints are close geographically.\n"
    "* Try increasing the timeout you pass to GlobalAsyncCheckpointManager.")

logger = logging.getLogger(__name__)


async def create_async_array_from_callback(
    global_shape: array.Shape,
    inp_sharding: jax.sharding.Sharding,
    data_callback: Callable[[array.Index, jax.Device], Awaitable[jax.Array]],
):
  device_to_index_map = inp_sharding.devices_indices_map(global_shape)
  addressable_da = inp_sharding._addressable_device_assignment
  future_arrays = [data_callback(device_to_index_map[d], d)
                   for d in addressable_da]
  dbs = await asyncio.gather(*future_arrays)
  return array.make_array_from_single_device_arrays(
      global_shape, inp_sharding, dbs)


def _get_metadata(arr):
  local_shape = arr.addressable_data(0).shape
  return {
      'compressor': {'id': 'zstd'},
      'shape': arr.shape,
      'chunks': np.array(np.maximum(1, local_shape)),
  }


def _spec_has_metadata(tree):
  if not isinstance(tree, dict):
    return False
  return 'metadata' in tree or any(
      _spec_has_metadata(subtree) for _, subtree in tree.items())

def _get_kvstore_for_gcs(ckpt_path: str):
  m = re.fullmatch('^gs://([^/]*)/(.*)$', ckpt_path, re.DOTALL)
  if m is None:
    raise ValueError('The ckpt_path should contain the bucket name and the '
                      f'file path inside the bucket. Got: {ckpt_path}')
  gcs_bucket = m.group(1)
  path_without_bucket = m.group(2)
  return {'driver': 'gcs', 'bucket': gcs_bucket, 'path': path_without_bucket}

def get_tensorstore_spec(ckpt_path: str, ocdbt: bool = False):
  # Normalize path to exclude trailing '/'. In GCS path case, we will need to
  # fix the path prefix to add back the stripped '/'.
  ckpt_path = os.path.normpath(ckpt_path).replace('gs:/', 'gs://')
  is_gcs_path = ckpt_path.startswith('gs://')
  spec = {'driver': 'zarr', 'kvstore': {}}
  if ocdbt:
    if not is_gcs_path and not os.path.isabs(ckpt_path):
      raise ValueError(f'Checkpoint path should be absolute. Got {ckpt_path}')
    base_path = os.path.dirname(ckpt_path)
    spec['kvstore'] = {
        'driver': 'ocdbt',
        'base': base_path if is_gcs_path else f'{_DEFAULT_DRIVER}://{base_path}',
        'path': os.path.basename(ckpt_path),
    }
  else:
    if is_gcs_path:
      spec['kvstore'] = _get_kvstore_for_gcs(ckpt_path)
    else:
      spec['kvstore'] = {'driver': _DEFAULT_DRIVER, 'path': ckpt_path}

  return spec


def is_remote_storage(tspec: dict[str, Any] | str) -> bool:
  """Detect if user is using cloud storages.

  This can detect common defines and unable to detect some corner cases such as
  using gcsfuse.
  """
  if isinstance(tspec, str):
    # KvStoreUrl
    if re.match(rf'^({"|".join(_REMOTE_URL_PREFIXES)})', tspec):
      return True
    else:
      return False

  for key in ('base', 'kvstore'):
    if key in tspec:
      return is_remote_storage(tspec[key])

  if 'driver' in tspec:
    for rule in _REMOTE_DRIVER_VALIDATIONS:
      if tspec['driver'] == rule['driver']:
        if rule['path_regex'] is None:
          return True

        # check if path matches the regex.
        if re.match(rule['path_regex'], tspec['path']):
          return True

  return False


# Lifted from T5X.
class _LimitInFlightBytes:
  """Limits in-flight bytes when reading/writing checkpoints per process."""

  def __init__(self, num_bytes):
    self._max_bytes = num_bytes
    self._available_bytes = num_bytes
    self._cv = asyncio.Condition(lock=asyncio.Lock())

  async def wait_for_bytes(self, requested_bytes):
    if requested_bytes > self._max_bytes:
      raise ValueError('Requested more bytes than we reserved space for: '
                       f'{requested_bytes} > {self._max_bytes}')
    async with self._cv:
      await self._cv.wait_for(lambda: self._available_bytes > requested_bytes)
      self._available_bytes -= requested_bytes
      assert self._available_bytes >= 0

  async def release_bytes(self, requested_bytes):
    async with self._cv:
      self._available_bytes += requested_bytes
      assert self._available_bytes <= self._max_bytes
      self._cv.notify_all()


async def transfer_shard_to_host(shard: array.Shard) -> np.ndarray:
  data = shard.data
  has_pinned_host = any(
      m.kind == "pinned_host" for m in shard.device.addressable_memories())
  if has_pinned_host:
    # If available, transfer to pinned host memory
    sharding = jax.sharding.SingleDeviceSharding(shard.device,
        memory_kind="pinned_host")
    data = jax.device_put(data, sharding)
  else:
    data.copy_to_host_async()
  # Allow other transfers to be scheduled simultaneously
  await asyncio.sleep(0)
  # Ensure that jax.Array's internal numpy array can be zero-copied. Tensorstore
  # implicitly converts the written data to a numpy array, and would otherwise
  # silently copy host-to-host.
  return np.array(data, copy=False)


async def async_serialize(
    arr_inp,
    tensorstore_spec,
    commit_future=None,
    context=TS_CONTEXT,
    primary_host: int | None = 0,
    replica_id: int = 0,
    transaction: Optional[ts.Transaction] = None,
):
  """Serialize an array using TensorStore.

  Args:
    arr_inp: The array to serialize.
    tensorstore_spec: The tensorstore spec to use.
    commit_future: A list of futures that will be appended to. The futures can
      be awaited asynchronously. If None, the futures will be awaited
      synchronously by this method.
    context: ts.Context instance.
    primary_host: Primary host, which indicates the host that will be treated as
      the "leader". If None, all hosts are treated as the primary. DO NOT USE
      unless you are sure you know what you are doing.
    replica_id: Allows overriding the shard replica id that will be saved. DO
      NOT USE unless you are sure you know what you are doing.
    transaction: TensorStore transaction to use for opening and writing the
      array.  If not specified, a non-transactional write will be used.
  """
  if (isinstance(arr_inp, array.ArrayImpl) and jax.process_count() > 1 and
      arr_inp.is_fully_addressable):
    raise ValueError(
        f'Passing fully addressable arrays to a multiprocess '
        f'serialization is not allowed, as this may lead to a race condition '
        f'between processes. Serialization have failed for the array with '
        f'the path "{tensorstore_spec["kvstore"]["path"]}".')

  # 'metadata' may not be present at the top level (for example, if we are using
  # a 'cast' driver).
  if not _spec_has_metadata(tensorstore_spec):
    tensorstore_spec['metadata'] = _get_metadata(arr_inp)

  # Set dtype if it's not in spec
  if 'dtype' not in tensorstore_spec:
    tensorstore_spec['dtype'] = jnp.dtype(arr_inp.dtype).name

  # If primary_host is None, all hosts will checkpoint. This is used
  # for checkpointing to local filesystem.
  if primary_host is None or jax.process_index() == primary_host:
    open_future = ts.open(
        ts.Spec(tensorstore_spec),
        create=True,
        open=True,
        context=context,
        transaction=transaction,
    )
    # Asynchronous case.
    if commit_future is not None:
      assert isinstance(commit_future, list)
      commit_future.append(open_future)
    else:
      await open_future

  # `ts.open` runs twice for process `primary_host` because for the first time,
  # we just get the future to be awaited upon in the background thread. The
  # second one runs with `assume_metadata=True` which does no I/O operation and
  # returns the tensorstore object.
  # For every process other than `primary_host`, we open with
  # `assume_metadata=True`.
  t = await ts.open(
      ts.Spec(tensorstore_spec),
      open=True,
      assume_metadata=True,
      context=context,
      transaction=transaction,
  )

  async def _write_array(shard):
    if shard.replica_id == replica_id:
      data = await transfer_shard_to_host(shard)
      write_future = t[shard.index].write(
          data,
          # Avoid additional copy of input array into the TensorStore chunk
          # cache.  If `arr_inp` is a jax.Array, the result of converting
          # it to a NumPy array, as is done internally by TensorStore, is
          # guaranteed to be immutable and therefore it is safe to retain a
          # reference indefinitely.
          can_reference_source_data_indefinitely=isinstance(
              arr_inp, array.ArrayImpl
          ),
      )
      if commit_future is not None:
        assert isinstance(commit_future, list)
        commit_future.append(write_future.commit)
        await write_future.copy
      else:
        await write_future.commit

  local_shards = arr_inp.addressable_shards
  future_write_state = jax.tree_util.tree_map(_write_array, local_shards)
  return await asyncio.gather(*future_write_state)


def run_serialization(arrays, tensorstore_specs):
  async def _run_serializer():
    future_writer = jax.tree_util.tree_map(async_serialize, arrays, tensorstore_specs)
    return await asyncio.gather(*future_writer)
  asyncio.run(_run_serializer())


def estimate_read_memory_footprint(t: ts.TensorStore,
                                   domain: ts.IndexDomain) -> int:
  rank = t.rank
  num_bytes = t.dtype.numpy_dtype.itemsize
  chunk_template = t.chunk_layout.read_chunk_template
  if domain is None:
    domain = t.domain
  origin = domain.origin
  shape = domain.shape
  chunk_origin = chunk_template.origin
  chunk_shape = chunk_template.shape

  # Some TensorStore drivers are not chunked, e.g. the inline 'array' driver.
  # For those, instead of returning a near-infinite memory footprint, estimate
  # the footprint as the entire shape.
  for i in range(rank):
    if not chunk_template[i].finite:
      return domain.size * num_bytes

  # Otherwise, if we have a chunked driver, estimate based on chunk size.
  for i in range(rank):
    origin_value = origin[i]
    chunk_origin_value = chunk_origin[i]
    chunk_size = chunk_shape[i]
    lower = origin_value - chunk_origin_value
    upper = origin_value + shape[i] - chunk_origin_value
    lower_aligned = lower // chunk_size * chunk_size
    upper_aligned = -(-upper // chunk_size) * chunk_size
    num_bytes *= (upper_aligned - lower_aligned)

  return num_bytes


async def async_deserialize(
    user_in_sharding: jax.sharding.Sharding | Layout,
    tensorstore_spec: ts.Spec | dict[str, Any],
    global_shape: Sequence[int] | None = None,
    dtype=None,
    byte_limiter: _LimitInFlightBytes | None = None,
    context=TS_CONTEXT,
    assume_metadata: bool = False,
):
  in_sharding = (user_in_sharding.sharding
                 if isinstance(user_in_sharding, Layout) else user_in_sharding)
  if not isinstance(in_sharding, jax.sharding.Sharding):
    raise ValueError(
        'sharding passed to deserialization should be specified, concrete and'
        f' an instance of `jax.sharding.Sharding`. Got {in_sharding}')
  dll = (user_in_sharding.device_local_layout
         if isinstance(user_in_sharding, Layout) else None)
  t = await ts.open(
      tensorstore_spec,
      open=True,
      assume_metadata=assume_metadata,
      context=context,
  )
  shape = t.shape if global_shape is None else global_shape
  new_shard_shape = in_sharding.shard_shape(tuple(shape))

  async def cb(index: array.Index, device: jax.Device):
    requested_domain = ts.IndexTransform(input_shape=shape)[index].domain
    restricted_domain = t.domain.intersect(requested_domain)
    requested_bytes = estimate_read_memory_footprint(t, restricted_domain)
    # Limit the bytes read for every shard.
    if byte_limiter is not None:
      await byte_limiter.wait_for_bytes(requested_bytes)
    # This maybe needed because the shape the array was saved with is smaller
    # than the requested shape of the array in which it will be reloaded. So
    # the extra values will be filled with 0s.
    out = np.zeros(new_shard_shape, dtype=t.dtype.numpy_dtype)
    await ts.array(out)[ts.d[:].translate_to[requested_domain.origin]][
        restricted_domain].write(t[restricted_domain])
    if dtype is not None:
      # Cast while reloading on process to avoid 2 copies on device if the
      # casting is done on device.
      out = out.astype(dtype)
    # Convert to jnp array so that layouts are initialized properly for
    # sub-byte dtypes.
    # TODO(yashkatariya): This is a band-aid fix. Figure out a better way to
    # make this work.
    if out.dtype == jnp.int4:
      out = jnp.asarray(out)  # type: ignore
    result = jax.device_put(
        out, Layout(dll, jax.sharding.SingleDeviceSharding(device)))
    if byte_limiter is not None:
      # NB: `out` actually might not be ready for garbage collection by the
      # time we call release_bytes . Thus peak memory usage still might grow
      # beyond what byte_limiter limit suggests it should. The simplest option
      # would be to call  `result.block_until_ready()`` here. However it
      # also comes with ~15-20% perf penalty as we would be waiting for CPU->GPU
      # transfer instead of loading data. In the future, if memory pressure
      # becomes a problem, we can instead instrument  bytelimiter to
      # keep track of all in-flight tensors and only block_until_ready, if byte
      # limiter hits the limit to get reduced memory usage, without losing
      # performance in common use cases.
      await byte_limiter.release_bytes(requested_bytes)
    return result

  return await create_async_array_from_callback(tuple(shape), in_sharding, cb)


def run_deserialization(shardings: Sequence[sharding.Sharding | Layout],
                        tensorstore_specs: Sequence[dict[str, Any]],
                        global_shapes: Sequence[array.Shape] | None = None,
                        dtypes: Sequence[typing.DTypeLike] | None = None,
                        concurrent_gb: int = 32):
  concurrent_bytes = concurrent_gb * 10**9

  async def _run_deserializer():
    # Object should be created once per process.
    byte_limiter = _LimitInFlightBytes(concurrent_bytes)
    future_arrays = jax.tree_util.tree_map(
        partial(async_deserialize, byte_limiter=byte_limiter),
        list(shardings), list(tensorstore_specs),
        [None] * len(tensorstore_specs) if global_shapes is None else global_shapes,
        [None] * len(tensorstore_specs) if dtypes is None else dtypes)
    return await asyncio.gather(*future_arrays)
  return asyncio.run(_run_deserializer())


def _get_key(key: int):
  return f'tensorstore_checkpoint_{key}'


class GlobalAsyncCheckpointManagerBase(util.StrictABC):
  """Interface for checkpointing GDAs asynchronously.

  This class manages the state of an ongoing asynchronous checkpoint.

  For example, say a checkpoint happens on every step. If you checkpoint on
  step 1 and after some computation the model is on checkpoint 2. But step 1's
  checkpoint hasn't finished committing to the storage layer yet. So until that
  is finished, checkpoint for step 2 will need to be blocked. Maintaining a
  class allows to maintain that state.

  Examples:

  Below is a simplified training loop:

  ```
  # Call this at the start of your program.
  jax.distributed.initialize()

  manager = GlobalAsyncCheckpointManager()

  # Restore checkpoint if available or initialize the train_state from
  # init_fn().
  train_state = manager.deserialize(...)

  while ...:
    if step % num_steps_between_checkpoints == 0:
      manager.serialize(train_state, temp_checkpoint_dir=...,
                        final_checkpoint_dir=...)
      train_state = train_step(train_state, input)
      # This is a non-blocking call.
      manager.check_for_errors()

  manager.serialize(train_state, temp_checkpoint_dir=...,
                    final_checkpoint_dir=...)
  # Wait before the end of the program for the checkpoint to finish. This is a
  # blocking call.
  manager.wait_until_finished()
  ```
  """

  @abc.abstractmethod
  def check_for_errors(self):
    """Checks if any errors have been raised in the child thread.

    This is a non-blocking call that can be called in the main thread.
    """

  @abc.abstractmethod
  def wait_until_finished(self):
    """Blocks until serialization has finished."""

  @abc.abstractmethod
  def serialize(self, arrays, tensorstore_specs, *,
                on_commit_callback: Callable[[], None]):
    """Serializes GDAs to TensorStore."""

  @abc.abstractmethod
  def deserialize(self, shardings: Sequence[sharding.Sharding],
                  tensorstore_specs: Sequence[dict[str, Any]],
                  global_shapes: Sequence[array.Shape] | None = None,
                  dtypes: Sequence[typing.DTypeLike] | None = None):
    """Deserializes GDAs from TensorStore."""


class AsyncManager:

  def __init__(self, timeout_secs=300):
    self._timeout_secs = timeout_secs
    self._timeout_in_ms = self._timeout_secs * 1000

    self._commit_futures = None
    self._thread = None
    self._exception = None

    if jax.process_count() > 1 and distributed.global_state.client is None:
      raise ValueError(_DISTRIBUTED_SYSTEM_MSG)
    if jax.process_count() > 1:
      self._client = distributed.global_state.client
    self._count = None

  def __del__(self):
    if self._thread is not None and self._thread.is_alive():
      logger.warning('Please add `.wait_until_finished()` in the main thread '
                      'before your program finishes because there is a '
                      'possibility of losing errors raised if the '
                      'this class is deleted before writing is completed.')

  def _thread_func(self):
    try:
      current_process = jax.process_index()
      process_count = jax.process_count()
      logger.info('Starting commit to storage layer by process: %s',
                   current_process)
      thread_start_time = time.time()
      for future in self._commit_futures:
        future.result()
      logger.info('Finished committing to storage layer by process: %s',
                   current_process)

      if process_count > 1:
        # All processes will wait at the barrier. When all processes are at the
        # barrier, the barrier will be satisfied. If not, then it will timeout.
        key_for_barrier = _get_key(self._count)
        logger.info('Key used for barrier is %s for process %s',
                    key_for_barrier, current_process)
        self._client.wait_at_barrier(key_for_barrier, self._timeout_in_ms)
        logger.info('Finished waiting at barrier for process %s',
                    current_process)

      if current_process == 0:
        self._on_commit_callback()
        logger.info('on_commit_callback successfully ran!')
        if process_count > 1:
          self._client.key_value_set(key_for_barrier, _CHECKPOINT_SUCCESS)
          logger.info('Process 0 successfully set key %s in the kv store',
                      key_for_barrier)

      jax.monitoring.record_event_duration_secs(
          '/jax/checkpoint/write/async/thread_duration_sec',
          time.time() - thread_start_time)

    except Exception as e:
      self._exception = e

  def _start_async_commit(self, on_commit_callback):
    self._count = next(_module_unique_count)

    self._on_commit_callback = on_commit_callback
    self._thread = threading.Thread(target=self._thread_func)
    self._thread.start()

  def check_for_errors(self):
    if self._exception is not None:
      # Clears self._exception so it is only raised once.
      exception = self._exception
      self._exception = None
      if (isinstance(exception, xe.XlaRuntimeError) and
          'DEADLINE_EXCEEDED: Barrier timed out' in str(exception)):
        raise BarrierTimeoutException(
            '\n'.join([str(exception), _BARRIER_TIMED_OUT_MSG]))
      raise exception  # pylint: disable=raising-bad-type

  def wait_until_finished(self):
    if self._thread is not None:
      self._thread.join()
      self._thread = None
      logger.info('Thread joined successfully')

    self.check_for_errors()
    logger.info('Error check finished successfully')

    if jax.process_count() > 1 and self._count is not None:
      # Block until process 0 writes success value to the key value store.
      # If it fails to write it, then `blocking_key_value_get` will time out.
      get_key = _get_key(self._count)
      self._client.blocking_key_value_get(get_key, self._timeout_in_ms)
      logger.info('blocking_key_value_get on key %s was successfully '
                  'completed.', get_key)

  def _add_futures(self, futures: Sequence[asyncio.Future]):
    self._commit_futures = futures


class GlobalAsyncCheckpointManager(AsyncManager, GlobalAsyncCheckpointManagerBase):
  """Responsible for serializing GDAs via TensorStore."""

  def serialize(
      self,
      arrays,
      tensorstore_specs,
      *,
      on_commit_callback,
      transaction: Optional[ts.Transaction] = None,
  ):
    """Serializes Arrays or Arrays via TensorStore asynchronously.

    TensorStore writes to a storage layer in 2 steps:
    *  Reading/copying from the source after which the source can be modified.
         * Returns a copy future.
    *  Writing/committing to the storage layer.
         * Returns a commit future.

    In asynchronous mode, the serialization waits for the commit future to
    finish in a separate thread allowing other computation to proceed.

    Args:
      arrays: Arrays or Arrays that should be serialized.
      tensorstore_specs: TensorStore specs that are used to serialize GDAs or
        Arrays.
      on_commit_callback: This callback will be executed after all processes
        have finished writing their checkpoints to disk. Filesystems where
        atomic rename operations are supported, you can rename from the
        temporary directory to the final directory. On GCS, you write to the
        final directory directly and in `on_commit_callback` you write a success
        file indicating that the serialization was successful because GCS does
        not support atomic rename operations.
      transaction: Optional TensorStore transaction to use.
    """
    logger.info('Waiting for previous serialization to finish.')
    self.wait_until_finished()

    commit_futures: list[ts.Future] = []

    async def _run_serializer():
      future_writer = jax.tree_util.tree_map(
          lambda arr_inp, tensorstore_spec: async_serialize(
              arr_inp,
              tensorstore_spec,
              commit_future=commit_futures,
              transaction=transaction,
          ),
          arrays,
          tensorstore_specs,
      )
      return await asyncio.gather(*future_writer)

    asyncio.run(_run_serializer())

    self._add_futures(commit_futures)

    # Used in wait_until_finished to check on process != 0, if the checkpoint
    # has finished writing.
    self._start_async_commit(on_commit_callback)

  def serialize_with_paths(
      self,
      arrays: Sequence[jax.Array],
      paths: Sequence[str],
      *,
      on_commit_callback,
      transaction: Optional[ts.Transaction] = None,
  ):
    tspecs = jax.tree.map(get_tensorstore_spec, paths)
    self.serialize(
        arrays,
        tspecs,
        on_commit_callback=on_commit_callback,
        transaction=transaction,
    )

  def deserialize(self, shardings: Sequence[sharding.Sharding | Layout],
                  tensorstore_specs: Sequence[dict[str, Any]],
                  global_shapes: Sequence[array.Shape] | None = None,
                  dtypes: Sequence[typing.DTypeLike] | None = None,
                  concurrent_gb: int = 32):
    self.wait_until_finished()
    return run_deserialization(shardings, tensorstore_specs,
                               global_shapes, dtypes, concurrent_gb)

  def deserialize_with_paths(
      self, shardings: Sequence[sharding.Sharding],
      paths: Sequence[str],
      global_shapes: Sequence[array.Shape] | None = None,
      dtypes: Sequence[typing.DTypeLike] | None = None,
      concurrent_gb: int = 32):
    tspecs = jax.tree.map(get_tensorstore_spec, paths)
    return self.deserialize(shardings, tspecs, global_shapes, dtypes,
                            concurrent_gb)
