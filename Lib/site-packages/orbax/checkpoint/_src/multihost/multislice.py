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

"""Multislice utilities."""

import functools
from typing import Any, Optional, Set, Tuple, Union

from absl import logging
import jax
from jax import numpy as jnp
import numpy as np
from orbax.checkpoint._src.multihost import multihost

PyTree = Any

# When using broadcasting from single replica to others, 3 copies of the data
# are stored in memory.
MEMORY_FACTOR = 3


def process_slice_id(
    process_index: int,
    global_mesh: jax.sharding.Mesh,
    *,
    replica_axis_index: int = 0,
) -> int:
  """Returns the slice id that the process_index belongs to."""
  for slice_id in range(
      slice_count(global_mesh, replica_axis_index=replica_axis_index)
  ):
    device_slice = slice_devices(
        global_mesh, replica_id=slice_id, replica_axis_index=replica_axis_index
    )
    if process_index in multihost.unique_processes_from_devices(device_slice):
      return slice_id
  return -1


def _process_in_device_slice(
    process_index: int, device_slice: np.ndarray
) -> bool:
  return process_index in multihost.unique_processes_from_devices(device_slice)


def slice_devices(
    global_mesh: jax.sharding.Mesh,
    *,
    replica_id: int = 0,
    replica_axis_index: int = 0,
) -> np.ndarray:
  devices = global_mesh.devices
  if hasattr(jax.devices()[0], 'slice_index'):
    get_slice_id = np.vectorize(lambda x: x.slice_index)
    return devices[get_slice_id(devices) == replica_id]
  else:
    return np.take(
        global_mesh.devices,
        replica_id,
        axis=replica_axis_index,
    )


def slice_count(
    global_mesh: jax.sharding.Mesh, *, replica_axis_index: int = 0
) -> int:
  """Number of slices implied by the mesh's replica dimension."""
  if len(global_mesh.shape_tuple) == 1:
    return 1
  return global_mesh.devices.shape[replica_axis_index]


def local_slice_devices(
    global_mesh: jax.sharding.Mesh, *, replica_axis_index: int = 0
) -> np.ndarray:
  """Get devices in the host-local slice."""
  for replica_id in range(
      slice_count(global_mesh, replica_axis_index=replica_axis_index)
  ):
    if in_slice(
        multihost.process_index(),
        global_mesh,
        replica_id=replica_id,
        replica_axis_index=replica_axis_index,
    ):
      return slice_devices(
          global_mesh,
          replica_id=replica_id,
          replica_axis_index=replica_axis_index,
      )
  raise ValueError(
      f'process_index {multihost.process_index()} does not exist in provided'
      ' `global_mesh`'
  )


def primary_process_in_slice(
    global_mesh: jax.sharding.Mesh,
    *,
    replica_id: int = 0,
    replica_axis_index: int = 0,
) -> int:
  """Returns an arbitrary process in the requested slice to serve as primary."""
  device_slice = slice_devices(
      global_mesh,
      replica_axis_index=replica_axis_index,
      replica_id=replica_id,
  )
  processes = multihost.unique_processes_from_devices(device_slice)
  return next(iter(processes))


def in_slice(
    process_index: int,
    global_mesh: jax.sharding.Mesh,
    *,
    replica_id: int = 0,
    replica_axis_index: int = 0,
) -> bool:
  """Returns if the process belongs to the indicated slice ID."""
  return _process_in_device_slice(
      process_index,
      slice_devices(
          global_mesh,
          replica_id=replica_id,
          replica_axis_index=replica_axis_index,
      ),
  )


@functools.partial(jax.jit, static_argnums=0)
def fake_zero_data(sharding, x):
  x = jnp.zeros_like(x)
  return jax.lax.with_sharding_constraint(x, sharding)


def get_device_memory() -> int:
  """Returns HBM capacity of the device on which the code is running(in bytes)."""
  device = jax.devices()[0]
  if device.platform not in ('tpu', 'gpu'):
    raise ValueError('Only select TPU and GPU devices are supported.')
  hbm_memory = {
      'TPU v3': int(16e9),  # two cores per chip each with 16 GB HBM
      'TPU v4': int(32e9),  # one megacore per chip with 32 GB HBM
      'TPU v5 lite': int(16e9),  # one core per chip with 16 GB HBM
      'TPU v5': int(96e9),  # one megacore per chip with 96 GB HBM
      'TPU v6 lite': int(32e9),  # one core per chip with 32 GB HBM
      'NVIDIA H100': int(144e9),
      'NVIDIA H200': int(80e9),
  }
  memory = hbm_memory.get(device.device_kind, None)
  if memory is None:
    raise ValueError(
        f'get_device_memory is not supported for {device.device_kind}.'
    )
  return memory


def get_leaf_memory_per_device(arr: jax.Array) -> int:
  """Returns the memory usage of a sharded array per device (in bytes)."""
  shard = arr.addressable_shards[0]
  return shard.data.size * shard.data.itemsize


def tree_memory_per_device(tree: Tuple[PyTree, ...]) -> int:
  """Returns the memory usage of a PyTree on each device (in bytes)."""
  leaf_memory_per_device = jax.tree_util.tree_map(
      get_leaf_memory_per_device, tree
  )
  return jax.tree.reduce(lambda x, y: x + y, leaf_memory_per_device)


def get_available_memory(
    in_tree: Tuple[PyTree, ...], scaling_factor: float
) -> int:
  """Returns estimated available memory for broadcasting (in bytes).

  After computing the available memory, we scale it by a factor of 0.75 to
  account for the fact that the actual memory usage could be different than the
  estimated memory usage. This will help us to avoid OOM errors for edge cases.

  Args:
    in_tree: pytree that occupies the memory.
    scaling_factor: indicates the frunction of the estimated available memory to
      be used when broadcustind data.
  """
  if scaling_factor > 1:
    raise ValueError('scaling_factorshould be less than 1.')
  total_device_memory = get_device_memory()
  used_device_memory = tree_memory_per_device(in_tree)
  available_memory = total_device_memory - used_device_memory
  return int(available_memory * scaling_factor / MEMORY_FACTOR)


def broadcast_one_replica_to_all(
    in_tree: Tuple[PyTree, ...],
    global_mesh: jax.sharding.Mesh,
    replica_axis_index: int,
    is_source: bool,
    memory_limit_bytes: Optional[Union[int, None]] = None,
    memory_scaling_factor: Optional[float] = 0.75,
) -> Tuple[Tuple[PyTree, ...], int]:
  """One replica reads the data and broadcasts to others.

  Args:
    in_tree: pytree to be broadcast. Shardings should correspond to the origin
      replica.
    global_mesh: global mesh.
    replica_axis_index: axis index along which the data is replicated.
    is_source: indicates if the current host is in origin replica.
    memory_limit_bytes: memory limit for broadcasting in bytes.
    memory_scaling_factor: indicates the fraction of the estimated available
      memory to be used when broadcasting data.

  Returns:
     Tuple containing:
      - pytree with broadcasted data
      - number of broadcasts performed.
  """
  num_replicas = global_mesh.devices.shape[replica_axis_index]
  replica_axis_name = global_mesh.axis_names[replica_axis_index]

  if memory_limit_bytes is None:
    memory_limit_bytes = get_available_memory(in_tree, memory_scaling_factor)
    logging.info('Using available memory of %d bytes.', memory_limit_bytes)

  # Set replica_axis to be 0, regardless of its actual value.
  def globalize_single_replica_arrays(inp):
    sharding = inp.sharding
    if not isinstance(sharding, jax.sharding.NamedSharding):
      raise ValueError(
          'Must provide input arrays with NamedSharding. '
          f'Got {type(sharding)} instead.'
      )
    if not is_source:
      inp = fake_zero_data(sharding, inp)
    inp = jnp.expand_dims(inp, axis=0)
    in_spec = jax.sharding.PartitionSpec(
        replica_axis_name,
        *sharding.spec,
    )
    global_shape = (num_replicas,) + inp.shape[1:]
    global_sharding = jax.sharding.NamedSharding(global_mesh, in_spec)
    return jax.make_array_from_single_device_arrays(
        global_shape, global_sharding, [s.data for s in inp.addressable_shards]
    )

  tree_len = len(in_tree)
  start = 0
  out_tree = []
  num_broadcasts = 0
  while start < tree_len:
    subtree = []
    current_memory = 0
    end = start
    if tree_memory_per_device(in_tree[start]) > memory_limit_bytes:
      logging.warning(
          'in_tree leaf size exceeds memory limit for broadcasting. '
          'Leaf size: %d bytes. Allowed memory limit: %d bytes. Proceeding.',
          tree_memory_per_device(in_tree[start]),
          memory_limit_bytes,
      )
      subtree.append(in_tree[end])
      end += 1
    else:
      while end < tree_len and (
          current_memory + tree_memory_per_device(in_tree[end])
          <= memory_limit_bytes
      ):
        subtree.append(in_tree[end])
        current_memory += tree_memory_per_device(in_tree[end])
        end += 1
    subtree = tuple(subtree)
    num_broadcasts += 1
    out_sharding = jax.tree.map(
        lambda x: jax.sharding.NamedSharding(
            global_mesh, jax.sharding.PartitionSpec(*x.sharding.spec)
        ),
        subtree,
    )
    in_tree_sharded = jax.tree.map(globalize_single_replica_arrays, subtree)
    # Delete immediately to conserve memory.
    jax.tree.map(lambda x: x.delete(), subtree)

    out_subtree = jax.jit(
        lambda tree: jax.tree.map(functools.partial(jnp.sum, axis=0), tree),
        out_shardings=out_sharding,
    )(in_tree_sharded)
    out_tree.extend(out_subtree)
    jax.block_until_ready(out_subtree)
    start = end

  if is_source:
    logging.info('Total number of broadcasts: %d', num_broadcasts)
  return tuple(out_tree), num_broadcasts


def get_primary_replica_ids_and_pids(
    replica_axis_idx: int,
    mesh: jax.sharding.Mesh,
    primary_replica_id: int,
) -> Tuple[Set[int], Set[int]]:
  """Returns the primary replica ids and process ids."""
  replica_devices = slice_devices(
      mesh,
      replica_id=primary_replica_id,
      replica_axis_index=replica_axis_idx,
  ).flatten()
  ids = set([d.id for d in replica_devices])
  pids = set([d.process_index for d in replica_devices])
  return ids, pids
