# Copyright 2018 The JAX Authors.
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

# A ShardingSpec describes at a high level how a logical array is sharded across
# devices (each array sharded with a `PmapSharding` has a ShardingSpec, and
# ShardingSpecs also describe how to shard inputs to a parallel computation).
# spec_to_indices() encodes exactly how a given ShardingSpec is translated to
# device buffers, i.e. how the sharded array is "laid out" across devices. Given
# a sequence of devices, we shard the data across the devices in row-major
# order, with replication treated as an extra inner dimension.
#
# For example, given the logical data array [1, 2, 3, 4], if we were to
# partition this array 4 ways with a replication factor of 2, for a total of 8
# devices, the data on each device would be: [1, 1], [2, 2], [3, 3], [4, 4].
#
# This encoding is assumed by various parts of the system, e.g. generating
# replica groups for collective operations.

from __future__ import annotations

from collections.abc import Sequence
import itertools
import math
from typing import Union

import numpy as np

from jax._src import config
from jax._src import util
from jax._src.lib import pmap_lib

unsafe_map, map = map, util.safe_map

NoSharding = pmap_lib.NoSharding
Chunked = pmap_lib.Chunked
Unstacked = pmap_lib.Unstacked

_UNSHARDED_INSTANCE = NoSharding()

ShardedAxis = pmap_lib.ShardedAxis
Replicated = pmap_lib.Replicated
MeshDimAssignment = Union[ShardedAxis, Replicated]

ShardingSpec = pmap_lib.ShardingSpec


def _sharding_spec_indices(self, shape: tuple[int, ...]) -> np.ndarray:
  """Returns NumPy-style indices corresponding to a sharding spec.

  Args:
    shape: The shape of the logical array being sharded.

  Returns:
    An ndarray with the same shape as the logical mesh (as derived form
    `mesh_mapping`). Each entry is a NumPy-style index selecting the subset of
    the data array to be placed on a corresponding device. The indices can be
    ints, slice objects with step=1, or tuples of those.
  """
  assert len(shape) == len(self.sharding), (shape, self.sharding)

  axis_indices: list[Sequence[Index]] = []
  shard_indices_shape = []
  for dim, sharding in enumerate(self.sharding):
    axis_size = shape[dim]
    if isinstance(sharding, NoSharding):
      axis_indices.append([slice(None)])
      # NOTE: We don't append unsharded dimensions to shard_indices_shape here,
      #       because they do not appear in the mesh mapping.
    elif isinstance(sharding, Unstacked):
      assert axis_size == sharding.size, f'{axis_size} != {sharding.size}'
      axis_indices.append(range(axis_size))
      shard_indices_shape.append(axis_size)
    elif isinstance(sharding, Chunked):
      total_chunks = math.prod(sharding.chunks)
      shard_size, ragged = divmod(axis_size, total_chunks)
      assert not ragged, (axis_size, total_chunks, dim)
      axis_indices.append([slice(i * shard_size, (i + 1) * shard_size)
                           for i in range(total_chunks)])
      shard_indices_shape.extend(sharding.chunks)
    else:
      util.assert_unreachable(sharding)

  # shard_indices is an ndarray representing the sharded axes of the logical array,
  # with each dimension having size equal to the number of shards across the corresponding
  # logical array dimension, and each element containing the multi-dimensional index that
  # is used to extract the corresponding shard of the logical array.
  shard_indices = np.empty([math.prod(shard_indices_shape)], dtype=np.object_)
  for i, idxs in enumerate(itertools.product(*axis_indices)):
    shard_indices[i] = idxs  # type: ignore  # numpy 2.2
  shard_indices = shard_indices.reshape(shard_indices_shape)

  # Ensure that each sharded axis is used exactly once in the mesh mapping
  num_sharded_dim = len(shard_indices_shape)
  sharded_dim_perm = [a.axis for a in self.mesh_mapping if isinstance(a, ShardedAxis)]
  assert (set(sharded_dim_perm) == set(range(num_sharded_dim)) and
          len(sharded_dim_perm) == num_sharded_dim)
  # Replicate/reorder the indices according to the mesh mapping
  replica_sizes = tuple(a.replicas for a in self.mesh_mapping if isinstance(a, Replicated))
  replica_dim, sharded_dim = itertools.count(0), iter(sharded_dim_perm)
  perm = [next(replica_dim) if isinstance(a, Replicated) else
          len(replica_sizes) + next(sharded_dim)
          for a in self.mesh_mapping]
  return (np.broadcast_to(shard_indices, replica_sizes + shard_indices.shape)
            .transpose(perm))

def _sharding_spec_repr(self):
  return f'ShardingSpec({self.sharding}, {self.mesh_mapping})'


ShardingSpec.indices = _sharding_spec_indices
# mypy raises: error: Cannot assign to a method  [assignment]
ShardingSpec.__repr__ = _sharding_spec_repr  # type: ignore


Index = Union[int, slice, tuple[Union[int, slice], ...]]

def spec_to_indices(shape: Sequence[int],
                    spec: ShardingSpec) -> tuple[Index, ...]:
  """Returns numpy-style indices corresponding to a sharding spec.

  Each index describes a shard of the array. The order of the indices is the
  same as the device_buffers of a Array sharded using PmapSharding (i.e. the
  data is laid out row-major).

  Args:
    shape: The shape of the logical array being sharded.
    spec: Describes how the array is sharded and how the shards are assigned to
      the logical mesh.

  Returns:
    A tuple of length equal to the size of the mesh (inferred as the product of
    sharded dimension sizes and all replication factors).  Each element is an
    int, a slice object with step=1, or a tuple thereof, to be treated as an
    index into the full logical array.
  """
  return tuple(spec.indices(shape).flat)  # type: ignore


def pmap_sharding_spec(nrep, axis_size, sharded_shape: Sequence[int],
                       map_axis: int | None) -> ShardingSpec:
  """Sharding spec for arguments or results of a pmap.
  Args:
    nrep: number of local XLA replicas (product of local axis sizes)
    axis_size: local axis size for outer pmap
    sharded_aval: the aval of the value inside the outer pmap, an instance of
      a ShapedArray.
    map_axis: the axis along which the value is mapped in the outer pmap
  Returns:
    A ShardingSpec.
  """
  replication_factor, ragged = divmod(nrep, axis_size)
  assert not ragged
  pspec = ShardingSpec(sharding=[_UNSHARDED_INSTANCE] * len(sharded_shape),
                       mesh_mapping=())
  maybe_replicate = () if replication_factor == 1 else (Replicated(replication_factor),)
  if map_axis is not None:
    sharded_in_axis = sum(not isinstance(s, NoSharding) for s in pspec.sharding[:map_axis])
    def shift_sharded_axis(a: MeshDimAssignment):
      if isinstance(a, ShardedAxis) and a.axis >= sharded_in_axis:
        return ShardedAxis(a.axis + 1)
      return a
    # replication_factor represents the product of inner pmaps, so it goes
    # after the outer pmapped axis at index 0
    if config.pmap_no_rank_reduction.value:
      sharding = util.tuple_update(
          pspec.sharding, map_axis, Chunked([axis_size]))
    else:
      sharding = util.tuple_insert(
          pspec.sharding, map_axis, Unstacked(axis_size))
    return ShardingSpec(
      sharding=sharding,
      mesh_mapping=itertools.chain(
          [ShardedAxis(sharded_in_axis)], maybe_replicate,
          map(shift_sharded_axis, pspec.mesh_mapping)))
  else:
    return ShardingSpec(
      sharding=pspec.sharding,
      mesh_mapping=(Replicated(axis_size),) + maybe_replicate + pspec.mesh_mapping)


def create_pmap_sharding_spec(shape: tuple[int, ...], sharded_dim: int = 0,
                              sharded_dim_size: int | None = None):
  if sharded_dim is not None:
    if config.pmap_no_rank_reduction.value:
      sharded_shape = util.tuple_update(shape, sharded_dim, 1)
    else:
      sharded_shape = util.tuple_delete(shape, sharded_dim)
    if sharded_dim_size is None:
      sharded_dim_size = shape[sharded_dim]
  else:
    assert sharded_dim_size is not None
    sharded_shape = shape

  return pmap_sharding_spec(sharded_dim_size, sharded_dim_size, sharded_shape,
                            sharded_dim)
