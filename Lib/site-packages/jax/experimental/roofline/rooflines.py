# Copyright 2024 The JAX Authors.
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
from collections import defaultdict
from dataclasses import replace
import itertools as it
import numpy as np

from jax._src import ad_util
from jax._src import core, util
from jax._src import ops
from jax._src import prng
from jax._src import random
from jax._src.lax import (
  ann,
  convolution,
  fft,
  lax,
  linalg,
  parallel as lax_parallel,
  slicing,
  special,
  windowed_reductions,
)
from jax.experimental import roofline
from jax.experimental import shard_map


for prim in it.chain(
  ad_util.__dict__.values(),
  ann.__dict__.values(),
  convolution.__dict__.values(),
  fft.__dict__.values(),
  lax.__dict__.values(),
  linalg.__dict__.values(),
  ops.__dict__.values(),
  prng.__dict__.values(),
  random.__dict__.values(),
  shard_map.__dict__.values(),
  slicing.__dict__.values(),
  special.__dict__.values(),
  windowed_reductions.__dict__.values(),
):
  if isinstance(prim, core.Primitive):
    roofline.register_standard_roofline(prim)


@roofline.register_roofline(lax.dot_general_p)
def _dot_general_roofline(
  ctx: roofline.RooflineRuleContext,
  *args,
  dimension_numbers: lax.DotDimensionNumbers,
  **kw,
) -> roofline.RooflineResult:
  lhs, rhs = (roofline.RooflineShape.from_aval(aval) for aval in ctx.avals_in)
  out = roofline.RooflineShape.from_aval(ctx.avals_out[0])
  (lhs_contract, _), (lhs_batch, _) = dimension_numbers

  flops = (
    2
    * lhs.size
    * rhs.size
    / np.prod([lhs.shape[i] for i in lhs_contract])
    / np.prod([lhs.shape[i] for i in lhs_batch])
  )

  hbm_bytes = 0
  if not ctx.pin_lhs_in_vmem:
    hbm_bytes += lhs.bytes
    hbm_bytes += out.bytes
  if not ctx.pin_rhs_in_vmem:
    hbm_bytes += rhs.bytes

  return roofline.RooflineResult(flops=int(flops), hbm_bytes=hbm_bytes)


def _return_zeros_if_one_sized_axis(
  ctx: roofline.RooflineRuleContext, axes: tuple[str, ...]
) -> roofline.RooflineResult | None:
  axes_size = np.prod([ctx.mesh.shape[axis] for axis in axes])
  if axes_size > 1:
    return None
  return roofline.RooflineResult(
    ici_bytes={axis: 0 for axis in axes},
    ici_latency={axis: 0 for axis in axes},
  )


def _ring_collective_roofline(
  ctx: roofline.RooflineRuleContext,
  *args,
  axes: tuple[str, ...],
  is_reduce: bool = True,
  **kw,
) -> roofline.RooflineResult:
  if zeros_result := _return_zeros_if_one_sized_axis(ctx, axes):
    return zeros_result

  mesh = ctx.mesh.shape
  current_shard_size = roofline.RooflineShape.total_bytes(ctx.avals_in)
  if is_reduce:
    current_shard_size /= np.prod([mesh[axis] for axis in axes])

  # We model the slowest color as the bottleneck.
  sorted_axes = sorted(axes, key=lambda x: mesh[x], reverse=True)
  num_axes = len(sorted_axes)

  ici_bytes = 0
  # Phase split.
  current_shard_size //= num_axes
  for axis in sorted_axes:
    axis_size = mesh[axis]
    # Do phase.
    ici_bytes += current_shard_size * (axis_size - 1)
    # Increase shard size.
    current_shard_size *= axis_size

  # Bottleneck is the longest axis.
  ici_latency = mesh[sorted_axes[0]] * num_axes

  return roofline.RooflineResult(
    ici_bytes={axis: int(ici_bytes) for axis in sorted_axes},
    ici_latency={axis: int(ici_latency) for axis in sorted_axes},
  )


roofline.register_roofline(lax_parallel.reduce_scatter_p)(
  lambda *args, axis_name, **kw: _ring_collective_roofline(*args, axes=axis_name, **kw)
)
roofline.register_roofline(lax_parallel.all_gather_p)(
  lambda *args, axis_name, **kw: _ring_collective_roofline(
    *args, axes=axis_name, is_reduce=False, **kw
  )
)


def _scalar_collective_roofline(
  ctx: roofline.RooflineRuleContext,
  *args,
  axes: tuple[str, ...],
  **kw,
) -> roofline.RooflineResult:
  shapes = [roofline.RooflineShape.from_aval(aval) for aval in ctx.avals_in]
  ctx = replace(ctx, avals_in=[core.ShapedArray((1,), shape.dtype) for shape in shapes])
  return _ring_collective_roofline(ctx, *args, axes=axes, is_reduce=False, **kw)


roofline.register_roofline(lax_parallel.pmin_p)(_scalar_collective_roofline)
roofline.register_roofline(lax_parallel.pmax_p)(_scalar_collective_roofline)


@roofline.register_roofline(shard_map.psum2_p)
def _psum2_roofline(
  ctx: roofline.RooflineRuleContext,
  *args,
  axes: tuple[str, ...],
  **kw,
) -> roofline.RooflineResult:
  ring_roofline = _ring_collective_roofline(ctx, *args, axes=axes, **kw)

  def double_dict(d: dict[str, int]) -> dict[str, int]:
    return {k: v * 2 for k, v in d.items()}

  return roofline.RooflineResult(
    ici_bytes=double_dict(ring_roofline.ici_bytes),
    ici_latency=double_dict(ring_roofline.ici_latency),
  )


@roofline.register_roofline(lax_parallel.all_to_all_p)
def _all_to_all_roofline(
  ctx: roofline.RooflineRuleContext,
  *args,
  axis_name: tuple[str, ...],
  **kw,
) -> roofline.RooflineResult:
  if zeros_result := _return_zeros_if_one_sized_axis(ctx, axis_name):
    return zeros_result

  mesh = ctx.mesh.shape
  size = roofline.RooflineShape.total_bytes(ctx.avals_in) * np.prod([
    mesh[axis] for axis in axis_name
  ])

  smallest_axis = sorted(axis_name, key=lambda x: mesh[x])[0]
  num_axes = len(axis_name)
  bisection_bw = mesh[smallest_axis] ** (num_axes - 1)
  if mesh[smallest_axis] > 2:
    # Times 2 because of wraparound.
    bisection_bw *= 2

  # Half the data needs to cross the bisection on average.
  ici_bytes = size / 2 / bisection_bw

  # The latency is the max number of hops across the mesh.
  ici_latency = sum(mesh[axis] / 2 for axis in axis_name)

  return roofline.RooflineResult(
    ici_bytes={axis: int(ici_bytes) for axis in axis_name},
    ici_latency={axis: int(ici_latency) for axis in axis_name},
  )


@roofline.register_roofline(lax_parallel.ppermute_p)
def _ppermute_roofline(
  ctx: roofline.RooflineRuleContext,
  *args,
  axis_name: tuple[str, ...],
  perm: tuple[tuple[int, int], ...],
  **kw,
) -> roofline.RooflineResult:
  if zeros_result := _return_zeros_if_one_sized_axis(ctx, axis_name):
    return zeros_result

  mesh = ctx.mesh.shape
  mesh_dims: list[int] = [mesh.get(axis, 1) for axis in axis_name]
  shard_size = roofline.RooflineShape.total_bytes(ctx.avals_in)

  ici_contention: dict[tuple[tuple[int, ...], ...], float] = defaultdict(float)
  ici_latency = 0

  for src, dst in perm:
    if src == dst:
      continue
    # Perms are linearized.
    src_coords = tuple(int(i) for i in np.unravel_index(src, mesh_dims))
    dst_coords = tuple(int(i) for i in np.unravel_index(dst, mesh_dims))

    ici_latency_for_perm = 0

    # For each dimension.
    for i in range(len(axis_name)):
      dim_size = mesh_dims[i]
      src_pos = src_coords[i]
      dst_pos = dst_coords[i]

      if src_pos != dst_pos:
        # Calculate distance with wraparound.
        clockwise_dist = (dst_pos - src_pos) % dim_size
        counter_dist = (src_pos - dst_pos) % dim_size
        direction = 1 if clockwise_dist <= counter_dist else -1

        curr_pos = src_pos
        while curr_pos != dst_pos:
          curr_coords = util.tuple_update(src_coords, i, curr_pos)
          next_pos = (curr_pos + direction) % dim_size
          next_coords = util.tuple_update(curr_coords, i, next_pos)
          ici_contention[tuple(sorted([curr_coords, next_coords]))] += 1
          curr_pos = next_pos

        distance = min(clockwise_dist, counter_dist)
        ici_latency_for_perm += distance

    ici_latency = max(ici_latency, ici_latency_for_perm)

  ici_bytes = shard_size * max(ici_contention.values(), default=0)
  return roofline.RooflineResult(
    ici_bytes={axis: int(ici_bytes) for axis in axis_name},
    ici_latency={axis: int(ici_latency) for axis in axis_name},
  )
