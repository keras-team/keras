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

"""Handles replica slices of jax.Arrays and host transfers."""

import collections
import dataclasses
import functools
import math
from typing import Optional, Sequence

from absl import logging
import jax
import numpy as np
from orbax.checkpoint._src.arrays import fragments
from orbax.checkpoint._src.arrays import numpy_utils
from orbax.checkpoint._src.arrays import types
from orbax.checkpoint._src.multihost import multihost


Shape = types.Shape
Index = types.Index
OptionalAxisAndShape = tuple[int | None, Shape| None]

HashableIndex = types.HashableIndex
HashableSlice = types.HashableSlice


@dataclasses.dataclass(frozen=True)
class SliceArgs:
  """Arguments for slicing a jax.Array.

  Intended to be passed to `ReplicaSlice` in order to select a slice of
  the `unsliced_data` array.
  """
  start_index: int
  limit_index: int
  axis: int


@dataclasses.dataclass(frozen=True)
class ReplicaSlice:
  """ReplicaSlice.

  ReplicaSlice represents the part of a jax.Shard that a replica is uniquely
  responsible for. A replica slice can be either on-device (backed by a slice of
  a single-sharding array) or on-host (backed by a numpy ndarray).

  With single-replica checkpointing the entirety of each jax.Shard is owned by
  exactly one replica. With replica-parallel checkpointing ownership of each
  jax.Shard is split evenly across replicas, hence each of the R replicas will
  be responsible for saving 1/R of each shard.

  `unsliced_data` refers to the corresponding jax.Shard's single-device array.
  The part of `unsliced_data` actually owned is given by `slice_args`.
  """

  index: Index
  unsliced_data: jax.Array | np.ndarray
  slice_args: Optional[SliceArgs]

  def __post_init__(self):
    if self.is_on_host:
      assert not self.slice_args, f'On-host {self!r} must not have slice args'

  @property
  def is_on_host(self):
    return isinstance(self.unsliced_data, np.ndarray)

  def data(self):
    if self.slice_args is None:
      return self.unsliced_data
    else:
      return jax.lax.slice_in_dim(
          self.unsliced_data,
          start_index=self.slice_args.start_index,
          limit_index=self.slice_args.limit_index,
          axis=self.slice_args.axis,
      )


@dataclasses.dataclass(frozen=True)
class ReplicaSlices:
  """ReplicaSlices.

  ReplicaSlices groups all the sliced data of one jax.Array that a replica is
  uniquely responsible for. Slices are either all on-device or all on-host.
  """

  global_shape: Shape
  local_shape: Shape
  sharding: jax.sharding.Sharding
  dtype: np.dtype | jax.numpy.dtype
  is_on_host: bool
  replica_slices: list[ReplicaSlice]

  def __post_init__(self):
    if not all(
        rslice.is_on_host == self.is_on_host for rslice in self.replica_slices
    ):
      raise ValueError(f'Inconsistent is_on_host in {self!r}')

  @property
  def nbytes(self) -> int:
    slice_nbytes = math.prod(self.local_shape) * self.dtype.itemsize
    return slice_nbytes * len(self.replica_slices)

  def to_fragments(self) -> fragments.Fragments:
    """Converts replica slices to fragments."""
    assert self.is_on_host
    result = fragments.Fragments(
        shape=self.global_shape,
        dtype=self.dtype,
        fragments=[
            fragments.Fragment(
                index=numpy_utils.resolve_slice(
                    rslice.index, self.global_shape
                ),
                value=rslice.data(),
            )
            for rslice in self.replica_slices
        ],
    )
    if result.fragments:
      fragments.validate_fragments_can_be_stacked(result)
    if not result.is_degenerate():
      assert self.local_shape == result.fragments[0].shape
    return result


@functools.lru_cache(maxsize=4096)
def _sharding_num_replicas(
    sharding: jax.sharding.Sharding, global_shape: Shape
) -> int:
  """Get the number of unique replicas for a sharding/shape.

  Uses the devices_indices_map to get the mapping of devices to the slice of the
  global array. This gives us the domains of every shard, which may be
  non-unique. For any index (domain), we increment the count by one. When `n`
  devices have the same index, this results in the replica count for that index
  being `n`. We can assert that the number of replicas for each index should be
  the same.

  We can cache results because we typically expect `save` to be called
  repeatedly on the same model (with changing array values).
  The model shardings and shapes do not change during the course of a typical
  training run.

  Training typically occurs with stacked layers, so we expect the number of
  model parameters to be significantly less than the cache size. Checkpoints
  with unstacked layers may have thousands of parameters, but these are
  typically used for inference, so saving is less relevant.

  Args:
    sharding: Array sharding.
    global_shape: The global shape of the array.

  Returns:
    The number of unique replicas for the sharding/shape.
  """
  counts = collections.defaultdict(int)
  for index in sharding.devices_indices_map(global_shape).values():
    counts[numpy_utils.to_hashable_index(index, shape=global_shape)] += 1
  num_replicas = next(iter(counts.values()))
  assert all(count == num_replicas for count in counts.values())
  return num_replicas


def calculate_replica_parallel_axis_and_local_shape(
    arr: jax.Array,
) -> OptionalAxisAndShape:
  """Calculates a local shape for replica-parallel serialization."""
  shard0 = arr.addressable_shards[0]
  replica_count = _sharding_num_replicas(arr.sharding, arr.shape)
  if shard0.data.size == 0 or replica_count <= 1:
    return None, None
  try:
    axis = next(
        axis_index
        for axis_index, axis_size in enumerate(shard0.data.shape)
        if axis_size % replica_count == 0
    )
  except StopIteration:
    return None, None
  local_shape = tuple(
      axis_size // (replica_count if axis_index == axis else 1)
      for axis_index, axis_size in enumerate(shard0.data.shape)
  )
  return axis, local_shape


def get_replica_slices(
    arr: jax.Array,
    replica_id: Optional[int],
    use_replica_parallel: bool,
) -> ReplicaSlices:
  """Returns the replica slices a given replica is responsible for.

  Does not transfer allocate or transfer any data.

  Args:
    arr: The jax.Array to get replica slices for.
    replica_id: Configured replica_id. Omitting the replica id just picks the
      first addressable shard's replica id so that the process writes each of
      its addressable shards exactly once. (This is the desired behavior for
      local checkpointing.)
    use_replica_parallel: Whether to use replica-parallel serialization to allow
      arrays with replicated shards to be written cooperatively by different
      hosts.

  Returns:
    ReplicaSlices object.
  """
  Result = tuple[list[ReplicaSlice], Shape]
  shard0 = arr.addressable_shards[0]

  # single-replica: a single replica saves an entire shard.
  def pick_single_replica() -> Result:
    target_replica_id = shard0.replica_id if replica_id is None else replica_id
    rslices = [
        ReplicaSlice(
            index=shard.index,
            unsliced_data=shard.data,
            slice_args=None,
        )
        for shard in arr.addressable_shards
        if shard.replica_id == target_replica_id
    ]
    local_shape = shard0.data.shape
    return rslices, local_shape

  # replica-parallel: every replica saves part of a shard.
  # Logic based on axlearn:
  # https://github.com/apple/axlearn/blob/226d27ab7569668f2c38a35cf32d5dc5190ebdbb/axlearn/common/array_serialization.py#L75
  # TODO(gspschmid): Support replica-parallel in arrays without any evenly-
  # divisible dimension. The last replica would transfer a smaller slice.
  def maybe_pick_replica_parallel() -> Optional[Result]:
    if replica_id is None:
      raise ValueError(
          '`use_replica_parallel` is incompatible with local checkpointing'
      )

    # Check whether replica-parallel applies: we are dealing with non-empty
    # shards, we have more than one replica, and some dimension of the shards
    # is evenly divisible across replicas.
    axis, local_shape = calculate_replica_parallel_axis_and_local_shape(arr)
    if axis is None or local_shape is None:
      return None

    rslices: list[ReplicaSlice] = []
    for shard in arr.addressable_shards:
      # Sanity check that all shards have the same shape.
      assert shard.data.shape == shard0.data.shape

      size = local_shape[axis]
      slize = shard.index[axis]
      start = slize.start or 0
      assert slize.step is None
      assert slize.stop is None or slize.stop == start + shard.data.shape[axis]

      start_offset = shard.replica_id * size
      end_offset = start_offset + size
      new_slice = slice(start + start_offset, start + end_offset)

      rslices.append(
          ReplicaSlice(
              index=shard.index[:axis] + (new_slice,) + shard.index[axis + 1 :],
              unsliced_data=shard.data,
              slice_args=SliceArgs(start_offset, end_offset, axis),
          )
      )

    return rslices, local_shape

  if logging.vlog_is_on(1):
    logging.vlog(
        1,
        '[process=%d] get_replica_slices: replica_id=%s, shards=[%s]',
        multihost.process_index(),
        replica_id,  # note: may be None
        ', '.join([
            f'Shard(index={shard.index}, replica_id={shard.replica_id})'
            for shard in arr.addressable_shards
        ]),
    )

  # In order for all processes to agree on the right serialization metadata
  # we want to compute the correct local shape regardless of whether there
  # are any replica slices to save locally.
  rslices, local_shape = (
      use_replica_parallel
      and maybe_pick_replica_parallel()
      or pick_single_replica()
  )
  return ReplicaSlices(
      global_shape=arr.shape,
      local_shape=local_shape,
      sharding=arr.sharding,
      dtype=arr.dtype,
      is_on_host=False,
      replica_slices=rslices,
  )


def transfer_arrays_to_host(
    arrays: Sequence[jax.Array],
    replica_id: Optional[int],
    use_replica_parallel: bool,
    *,
    enable_pinned_host_transfer: bool = False,
) -> Sequence[ReplicaSlices]:
  """Transfers arrays to host memory.

  Transfers jax.Arrays to host memory and returns all the fragments to be
  serialized by the given replica, along with local shape. Blocks until
  completion.

  Args:
    arrays: The jax.Arrays to transfer.
    replica_id: Configured replica_id.
    use_replica_parallel: Whether to use replica-parallel serialization to allow
      arrays with replicated shards to be written cooperatively by different
      hosts.
    enable_pinned_host_transfer: Whether to allow transfer to pinned host
      memory. Pinned memory is closely associated with a TPU device and can

  Returns:
    ReplicaSlices objects, in host memory.
  """
  logging.info(
      'Transferring arrays to host memory with options:'
      ' use_replica_parallel=%s, enable_pinned_host_transfer=%s',
      use_replica_parallel,
      enable_pinned_host_transfer,
  )

  def use_pinned_host_transfer(device: jax.Device):
    has_pinned_host = any(
        m.kind == 'pinned_host' for m in device.addressable_memories()
    )
    return (
        enable_pinned_host_transfer
        and has_pinned_host
    )

  def async_transfer_slice(
      rslice: ReplicaSlice,
  ) -> tuple[ReplicaSlice, jax.Array]:
    assert not rslice.is_on_host
    data = rslice.data()
    assert isinstance(data, jax.Array)
    device = data.device
    # Start the asynchronous device-to-host copy
    if use_pinned_host_transfer(device):
      # If available, transfer to pinned host memory
      data = jax.device_put(
          data,
          jax.sharding.SingleDeviceSharding(device, memory_kind='pinned_host'),
      )
    else:
      data.copy_to_host_async()
    return rslice, data

  # Gather the replica slices to be saved for each array.
  rslices_per_array = [
      get_replica_slices(arr, replica_id, use_replica_parallel)
      for arr in arrays
  ]
  # Kick off transfers for all replica slices to be saved.
  transfers_per_array = [
      [async_transfer_slice(rslice) for rslice in rslices.replica_slices]
      for rslices in rslices_per_array
  ]
  # Wait for all the transferred data to be ready.
  return [
      dataclasses.replace(
          rslices,
          is_on_host=True,
          replica_slices=[
              dataclasses.replace(
                  rslice_on_device,
                  # Conversion to numpy arrays forces block_until_ready.
                  unsliced_data=np.asarray(data),
                  slice_args=None,
              )
              for rslice_on_device, data in transfers
          ],
      )
      for rslices, transfers in zip(rslices_per_array, transfers_per_array)
  ]
