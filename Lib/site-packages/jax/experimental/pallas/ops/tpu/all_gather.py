# Copyright 2023 The JAX Authors.
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

"""Simple all-gather kernel.

This is meant to be a pedagogical example of how to write a custom collective
using Pallas. It doesn't have all possible performance optimizations and doesn't
currently handle more diverse topologies.

The kernel assumes a ring structure on a single mesh axis. It takes the local
chunk, splits it in two, and sends each of the half-chunks in each direction
(left and right) until every device has received the half chunks.
"""
from __future__ import annotations
import functools

from collections.abc import Sequence

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


P = jax.sharding.PartitionSpec


def get_neighbor(
    idx: jax.Array, mesh: jax.sharding.Mesh, axis_name: str, *, direction: str
) -> tuple[jax.Array, ...]:
  """Helper function that computes the mesh indices of a neighbor."""
  axis_names = mesh.axis_names
  which_axis = axis_names.index(axis_name)
  mesh_index = [
      idx if i == which_axis else lax.axis_index(a)
      for i, a in enumerate(axis_names)
  ]
  axis_size = lax.psum(1, axis_name)
  if direction == "right":
    next_idx = lax.rem(idx + 1, axis_size)
  else:
    left = idx - 1
    next_idx = jnp.where(left < 0, left + axis_size, left)
  mesh_index[which_axis] = next_idx
  return tuple(mesh_index)


def ag_kernel(x_ref, o_ref, send_sem, recv_sem, *, axis_name: str,
              mesh: jax.sharding.Mesh):
  my_id = lax.axis_index(axis_name)
  # TODO(sharadmv): could speed this up having the first remote DMA go from
  # x_ref->o_ref immediately instead of a blocking HBM copy.
  with jax.named_scope("initial_copy"):
    pltpu.async_copy(x_ref, o_ref.at[my_id], recv_sem[0]).wait()

  with jax.named_scope("neighbour_lookup"):
    axis_size = lax.psum(1, axis_name)
    left_neighbor = get_neighbor(my_id, mesh, axis_name, direction="left")
    right_neighbor = get_neighbor(my_id, mesh, axis_name, direction="right")

  with jax.named_scope("main_barrier"):
    sem = pltpu.get_barrier_semaphore()
    pltpu.semaphore_signal(sem, 1, device_id=left_neighbor)
    pltpu.semaphore_signal(sem, 1, device_id=right_neighbor)
    pltpu.semaphore_wait(sem, 2)

  shard_size = x_ref.shape[0]
  right_dma, left_dma = None, None
  # Main strategy for this AG: carve up our input into two slices. Send
  # each slice along each direction until they reach every device.
  for i in range(axis_size - 1):
    right_slot = my_id - i
    right_slice = pl.ds(shard_size // 2, shard_size // 2)
    slot = jnp.where(right_slot < 0, axis_size + right_slot, right_slot)
    if right_dma:
      with jax.named_scope("wait_right_dma"):
        right_dma.wait()
    right_dma = pltpu.async_remote_copy(
        o_ref.at[slot, right_slice],
        o_ref.at[slot, right_slice],
        send_sem[1],
        recv_sem[1],
        device_id=right_neighbor,
    )

    left_slot = my_id + i
    left_slice = pl.ds(0, shard_size // 2)
    slot = lax.rem(left_slot, axis_size)
    if left_dma:
      with jax.named_scope("wait_left_dma"):
        left_dma.wait()
    left_dma = pltpu.async_remote_copy(
        o_ref.at[slot, left_slice],
        o_ref.at[slot, left_slice],
        send_sem[0],
        recv_sem[0],
        device_id=left_neighbor,
    )
  with jax.named_scope("wait_all_dma"):
    assert right_dma is not None
    assert left_dma is not None
    right_dma.wait()
    left_dma.wait()


@functools.partial(
    jax.jit, static_argnames=["mesh", "axis_name", "memory_space"]
)
def all_gather(x, *, mesh: jax.sharding.Mesh, axis_name: str | Sequence[str],
               memory_space: pltpu.TPUMemorySpace = pltpu.VMEM):
  if isinstance(axis_name, str):
    axis_name = (axis_name,)
  # TODO(sharadmv): enable all gather over multiple axes
  if len(axis_name) > 1:
    raise NotImplementedError("Only one axis supported.")
  axis_name, = axis_name
  if mesh.shape[axis_name] == 1:
    # We can short-circuit here if our axis size is 1
    return x
  def ag_local(x_shard):
    axis_size = lax.psum(1, axis_name)
    out_shape = jax.ShapeDtypeStruct((axis_size, *x_shard.shape), x_shard.dtype)
    out = pl.pallas_call(
        functools.partial(ag_kernel, axis_name=axis_name, mesh=mesh),
        out_shape=out_shape,
        compiler_params=pltpu.TPUCompilerParams(collective_id=0),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            scratch_shapes=(
                (pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA),
                (pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA),
            ),
            in_specs=[pl.BlockSpec(memory_space=memory_space)],
            out_specs=pl.BlockSpec(memory_space=memory_space),
        ),
    )(x_shard)
    return out.reshape((axis_size * x_shard.shape[0], *x_shard.shape[1:]))

  return shard_map.shard_map(
      ag_local, mesh=mesh, in_specs=P(axis_name), out_specs=P(None),
      check_rep=False
  )(x)
