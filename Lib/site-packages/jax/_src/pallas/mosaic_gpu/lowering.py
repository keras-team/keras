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

"""Module for lowering JAX primitives to Mosaic GPU."""

from __future__ import annotations

import collections
from collections.abc import Hashable, MutableMapping, MutableSequence, Sequence
import contextlib
import dataclasses
import functools
import itertools as it
import math
from typing import Any, Protocol, cast

import jax
from jax import lax
from jax._src import core as jax_core
from jax._src import pjit
from jax._src import util
from jax._src import source_info_util
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith as arith_dialect
from jax._src.lib.mlir.dialects import gpu as gpu_dialect
from jax._src.lib.mlir.dialects import memref as memref_dialect
from jax._src.lib.mlir.dialects import nvvm as nvvm_dialect
from jax._src.lib.mlir.dialects import scf as scf_dialect
from jax._src.pallas import core as pallas_core
from jax._src.pallas import pallas_call
from jax._src.pallas import primitives
from jax._src.pallas import utils as pallas_utils
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.state import discharge
from jax._src.state import indexing
from jax._src.state import types as state_types
from jax._src.state import primitives as sp
from jax._src.state.types import RefReshaper
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.mosaic.gpu import core as mgpu_core
from jax.experimental.mosaic.gpu import utils as mgpu_utils
from jax.experimental.mosaic.gpu import profiler as mgpu_profiler
import jax.numpy as jnp
import numpy as np


# TODO(slebedev): Enable type checking.
# mypy: ignore-errors
# pytype: skip-file

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip

partial = functools.partial
SMEM = gpu_core.SMEM
# We align all our SMEM allocations to 1024 bytes. TMA and WGMMA are very
# sensitive to alignment and while this is quite conservative, it gets the job
# done. We should make this more refined in the future.
_SMEM_ALIGNMENT = 1024
WARPGROUP_SIZE = 128

def _align_to(x: int, alignment: int):
  if (rem := x % alignment):
    return x + alignment - rem
  return x


@dataclasses.dataclass(kw_only=True, frozen=True)
class Resources:
  smem_scratch_bytes: int = 0
  barrier_counts: collections.Counter[mgpu.Barrier] = dataclasses.field(
      default_factory=collections.Counter
  )

  def __post_init__(self):
    object.__setattr__(
        self,
        "smem_scratch_bytes",
        _align_to(self.smem_scratch_bytes, _SMEM_ALIGNMENT),
    )

  @property
  def barriers(self) -> Sequence[mgpu.Barrier]:
    return list(self.barrier_counts.elements())

  def __add__(self, other: Resources) -> Resources:
    # TODO(slebedev): Optimize this.
    #
    # At the moment, if we have run_scoped(b1) followed by run_scoped(b2)
    # we will allocate two barriers, even though one would be enough.
    return Resources(
        smem_scratch_bytes=self.smem_scratch_bytes + other.smem_scratch_bytes,
        barrier_counts=self.barrier_counts + other.barrier_counts,
    )

  def __or__(self, other: Resources) -> Resources:
    return Resources(
        smem_scratch_bytes=max(
            self.smem_scratch_bytes, other.smem_scratch_bytes
        ),
        barrier_counts=self.barrier_counts | other.barrier_counts,
    )


class ResourceEstimator(Protocol):

  def __call__(self, *args: Any, **params: Any) -> Resources:
    ...


_resource_estimators: dict[jax_core.Primitive, ResourceEstimator] = {}


def _register_resource_estimator(primitive: jax_core.Primitive):
  def deco(fn):
    _resource_estimators[primitive] = fn
    return fn

  return deco


def _estimate_resources(jaxpr: jax_core.Jaxpr) -> Resources:
  """Estimates the resources required by the kernel."""
  rs = Resources(smem_scratch_bytes=0)
  for eqn in jaxpr.eqns:
    # TODO(slebedev): Add support for other primitives, notably control flow.
    rule = _resource_estimators.get(eqn.primitive)
    if rule is None:
      # Assume that unsupported primitives are neutral wrt resource usage.
      continue
    rs |= rule(*(invar.aval for invar in eqn.invars), **eqn.params)

  return rs


@_register_resource_estimator(lax.cond_p)
def _cond_resource_estimator(*args, branches) -> int:
  del args  # Unused.
  return functools.reduce(
      lambda a, b: a | b,
      (_estimate_resources(branch.jaxpr) for branch in branches),
  )


@_register_resource_estimator(lax.scan_p)
def _scan_resource_estimator(*args, jaxpr: jax_core.ClosedJaxpr, **params) -> int:
  del args, params  # Unused.
  return _estimate_resources(jaxpr)


@_register_resource_estimator(lax.while_p)
def _while_resource_estimator(*args, cond_jaxpr: jax_core.ClosedJaxpr, body_jaxpr: jax_core.ClosedJaxpr, **params) -> int:
  del args, params  # Unused.
  return _estimate_resources(cond_jaxpr) | _estimate_resources(body_jaxpr)


@_register_resource_estimator(primitives.run_scoped_p)
def _run_scoped_resource_estimator(*consts, jaxpr: jax_core.Jaxpr) -> int:
  del consts  # Unused.
  rs = Resources()
  for v in jaxpr.invars:
    aval = v.aval
    if isinstance(aval.dtype, gpu_core.BarrierType):
      rs += Resources(
          barrier_counts=collections.Counter([
              mgpu.Barrier(
                  aval.dtype.num_arrivals * WARPGROUP_SIZE, *aval.shape
              )
          ])
      )
    else:
      rs += Resources(
          smem_scratch_bytes=math.prod(aval.shape) * aval.dtype.itemsize
      )
  return rs + _estimate_resources(jaxpr)


@_register_resource_estimator(lax.reduce_sum_p)
def _reduce_sum_resource_estimator(x_aval: jax_core.ShapedArray, *, axes) -> int:
  # We don't need shmem for some reductons, but it depends on the layout, so we
  # conservatively request some scratch space.
  return Resources(smem_scratch_bytes=4 * x_aval.dtype.itemsize)


@dataclasses.dataclass
class ModuleContext:
  name: str
  grid_names: Sequence[Hashable] | None
  program_ids: Sequence[ir.Value] | None
  approx_math: bool
  runtime_smem: ir.Value  # ir.MemRefType
  smem_used_bytes: int
  runtime_barriers: MutableMapping[
      mgpu.Barrier, MutableSequence[mgpu.BarrierRef]
  ]
  name_stack: source_info_util.NameStack
  traceback_caches: mlir.TracebackCaches
  squashed_dims: tuple[int, ...]

  def reserve_barrier(self, barrier: mgpu.Barrier) -> mgpu.BarrierRef:
    """Reserves a barrier.

    Raises:
      RuntimeError: If the barrier is already reserved.
    """
    available = self.runtime_barriers.get(barrier, [])
    if not available:
      raise RuntimeError(f"Barrier {barrier} is already reserved")
    return available.pop()

  # TODO(cperivol): Only return the shapes and figure out the sizes when freeing.
  @contextlib.contextmanager
  def scratch_view(
      self, structs: Sequence[jax.ShapeDtypeStruct]
  ) -> Sequence[ir.Value]:
    """Creates a view into the runtime scratch buffer for each struct.

    This is a low-level API. Use it only if you know what you are doing.

    The function allocates bytes at the top of a stack, which need to be
    deallocated in a FIFO fashion with :meth:`ModuleContext.stack_free_smem`.
    After deallocation, the view is invalid and cannot be used.

    Args:
      structus: The shapes and dtypes of the views to create.

    Returns:
      A tuple, where the first element is the number of bytes allocated,
      and the second element is a sequence of memref views into the
      runtime scratch buffer.
    """
    smem_scratch_bytes = math.prod(ir.MemRefType(self.runtime_smem.type).shape)

    views = []
    off = initial_used_bytes = self.smem_used_bytes
    assert off % _SMEM_ALIGNMENT == 0
    smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
    for s in structs:
      scratch_ty = ir.MemRefType.get(
          s.shape,
          mgpu_utils.dtype_to_ir_type(s.dtype),
          memory_space=smem,
      )
      views.append(
          memref_dialect.view(scratch_ty, self.runtime_smem, _as_index(off), [])
      )
      off += _align_to(
          math.prod(s.shape) * jnp.dtype(s.dtype).itemsize, _SMEM_ALIGNMENT
      )
    assert off <= smem_scratch_bytes, "Ran out of scoped SMEM"
    assert off % _SMEM_ALIGNMENT == 0

    self.smem_used_bytes = off
    yield views
    self.smem_used_bytes = initial_used_bytes


@dataclasses.dataclass(frozen=True)
class LoweringRuleContext:
  module_ctx: ModuleContext
  launch_ctx: mgpu.LaunchContext
  predicate: ir.Value
  avals_in: Sequence[jax_core.ShapedArray]
  avals_out: Sequence[jax_core.ShapedArray]

  replace = dataclasses.replace


@dataclasses.dataclass(frozen=True)
class LoweringResult:
  module: ir.Module
  grid: tuple[int, ...]
  block: tuple[int, ...]
  out_structs: tuple[jax.ShapeDtypeStruct, ...]
  profiler_context: ProfilerContext | None


@dataclasses.dataclass(frozen=True)
class ProfilerContext:
  dump_path: str
  spec: mgpu_profiler.ProfilerSpec


class LoweringError(Exception):  # pylint: disable=g-bad-exception-name
  pass


def _eval_index_map(
    module_ctx: ModuleContext,
    launch_ctx: mgpu.LaunchContext,
    idx: Sequence[ir.Value],
    block_mapping: pallas_core.BlockMapping,
) -> Sequence[ir.Value]:
  block_indices = lower_jaxpr_to_mosaic_gpu(
      module_ctx, launch_ctx, block_mapping.index_map_jaxpr.jaxpr, idx
  )
  result = []
  for i, b in zip(block_indices, block_mapping.block_shape):
    if b is pallas_core.mapped:
      result.append(i)
    else:
      # TODO(slebedev): Use a type-agnostic multiplication wrapper.
      result.append(arith_dialect.muli(_as_index(i), _as_index(b)))
  return tuple(result)


def _uses_arguments(cjaxpr: jax_core.ClosedJaxpr) -> list[bool]:
  jaxpr = cjaxpr.jaxpr
  return pe.dce_jaxpr(jaxpr, used_outputs=[True] * len(jaxpr.outvars))[1]


def _check_block_mappings(
    block_mappings: Sequence[pallas_core.BlockMapping],
    name_and_src_info: pallas_core.NameAndSrcInfo,
) -> None:
  def err_details(bm: pallas_core.BlockMapping) -> str:
    return (
        f"Block spec for {bm.origin} in pallas_call {name_and_src_info}"
        f" has block shape {bm.block_shape}, array shape"
        f" {bm.array_shape_dtype.shape},"
        # TODO(necula): add index_map source location info
        f" and index_map {bm.index_map_jaxpr.jaxpr} in"
        f" memory space {bm.transformed_block_aval.memory_space}."
        " See details at"
        " https://jax.readthedocs.io/en/latest/pallas/grid_blockspec.html#pallas-blockspec."
    )

  for bm in block_mappings:
    if (
        bm.transformed_block_aval.memory_space == gpu_core.GMEM
        and not bm.has_trivial_window()
    ):
      raise NotImplementedError(
          "Mosaic GPU lowering currently requires blocks in GMEM memory space "
          "to have same block shape as the array shape "
          "and a trivial index_map (returning all 0s).\n\n"
          + err_details(bm)
      )

    if not isinstance(bm.indexing_mode, pallas_core.Blocked):
      raise NotImplementedError(
          "Only Blocked indexing mode is supported in Mosaic GPU lowering.\n\n"
          + err_details(bm)
      )


def lower_jaxpr_to_module(
    grid_mapping: pallas_core.GridMapping,
    jaxpr: jax_core.Jaxpr,
    name_and_src_info: pallas_core.NameAndSrcInfo,
    compiler_params: dict[str, Any],
    cost_estimate: pallas_core.CostEstimate | None,
) -> LoweringResult:
  del cost_estimate  # Unused.

  assert len(jaxpr.outvars) == 0
  assert not grid_mapping.vmapped_dims
  if grid_mapping.num_dynamic_grid_bounds:
    raise NotImplementedError(
        "Dynamic grid bounds not supported in the Mosaic GPU lowering."
    )
  if grid_mapping.num_index_operands:
    raise NotImplementedError(
        "Scalar prefetch not supported in Mosaic GPU lowering."
    )

  block_mappings = grid_mapping.block_mappings
  _check_block_mappings(block_mappings, name_and_src_info)

  params = compiler_params.get("mosaic_gpu", {})
  approx_math = params.get("approx_math", False)
  max_concurrent_steps = params.get("max_concurrent_steps", 1)
  delay_release = params.get("delay_release", 0)
  dimension_semantics = params.get("dimension_semantics")
  if dimension_semantics is None:
    dimension_semantics = ["parallel"] * len(grid_mapping.grid)
  elif len(dimension_semantics) != len(grid_mapping.grid):
    raise ValueError(
        "dimension_semantics must have an entry for each grid dimension:"
        f" {len(dimension_semantics)=}, but len(grid) is {grid_mapping.grid})."
    )
  sequential_axes = tuple(
      i for i, s in enumerate(dimension_semantics) if s == "sequential"
  )
  if max_concurrent_steps <= delay_release:
    raise ValueError(
        "max_concurrent_steps must be greater than delay_release, but"
        f" {max_concurrent_steps=}, {delay_release=}"
    )

  if grid_mapping.grid_names:  # Last dim corresponds to the warpgroup count
    block = (128 * grid_mapping.grid[-1], 1, 1)
    logical_grid = grid_mapping.grid[:-1]
  else:
    block = (128, 1, 1)
    logical_grid = grid_mapping.grid

  parallel_grid = [
      d for i, d in enumerate(logical_grid) if i not in sequential_axes
  ]
  if len(parallel_grid) <= 3:
    squashed_dims = ()
    parallel_grid += (1,) * (3 - len(parallel_grid))
  else:
    # If we have >3 parallel dimensions, we merge all leading dimensions
    # into the first (Dimension.x) CUDA grid dimension.
    squashed_dims = parallel_grid[:-2]
    parallel_grid = [math.prod(parallel_grid[:-2]), *parallel_grid[-2:]]

  if sequential_axes:
    # TODO(slebedev): Support multiple sequential axes.
    if len(sequential_axes) > 1:
      raise NotImplementedError(
          "Multiple sequential axes are not supported in Mosaic GPU lowering."
      )
    [sequential_axis] = sequential_axes
    num_steps = grid_mapping.grid[sequential_axis]
    out_sequential_invariant = [
        not _uses_arguments(bm.index_map_jaxpr)[sequential_axis]
        for bm in grid_mapping.block_mappings_output
    ]
  else:
    num_steps = 1
    out_sequential_invariant = [True] * len(grid_mapping.out_shapes)

  # Shrink ``max_concurrent_steps`` if the total number of steps is lower to
  # reduce the size of the allocated buffers below.
  if max_concurrent_steps > num_steps:
    max_concurrent_steps = num_steps
    delay_release = 0  # No need to delay anything

  in_in_smem, out_in_smem = util.split_list(
      [
          bm.transformed_block_aval.memory_space in (None, gpu_core.SMEM)
          for bm in block_mappings
      ],
      [grid_mapping.num_inputs],
  )

  in_block_mappings, out_block_mappings = util.split_list(
      block_mappings, [grid_mapping.num_inputs]
  )
  in_structs_gmem = [*grid_mapping.in_shapes]
  # We allocate the fully transformed shapes here. All primitives have seen the
  # inverse transformation stack and will understand how to handle it.
  in_structs_smem = [
      jax.ShapeDtypeStruct(
          [max_concurrent_steps, *bm.transformed_block_aval.shape],
          bm.transformed_block_aval.dtype,
      )
      if in_smem
      else None
      for bm, in_smem in zip(
          block_mappings[: grid_mapping.num_inputs], in_in_smem
      )
  ]
  in_gmem_transforms = [
      cast(gpu_core.MemoryRefTransform, bm.transforms)
      for bm in in_block_mappings
  ]
  out_structs_gmem = [*grid_mapping.out_shapes]
  out_structs_smem = [
      jax.ShapeDtypeStruct(
          [max_concurrent_steps, *bm.transformed_block_aval.shape], s.dtype
      )
      if in_smem
      else None
      for bm, in_smem, s in zip(
          block_mappings[grid_mapping.num_inputs :],
          out_in_smem,
          grid_mapping.out_shapes,
      )
  ]
  out_gmem_transforms = [
      cast(gpu_core.MemoryRefTransform, bm.transforms)
      for bm in out_block_mappings
  ]

  def body(launch_ctx: mgpu.LaunchContext, *buffers: ir.Value):
    *buffers_gmem, (
        buffers_smem,
        *scratch_buffers_smem,
        runtime_smem,
        barriers,
    ) = buffers
    assert len(buffers_gmem) == len(buffers_smem)
    in_buffers_gmem, out_buffers_gmem = util.split_list(
        buffers_gmem, [grid_mapping.num_inputs]
    )
    in_buffers_smem, out_buffers_smem = util.split_list(
        buffers_smem, [grid_mapping.num_inputs]
    )
    barriers, runtime_barriers, extra_barriers = barriers

    parallel_count = it.count()
    program_ids_template = [
        _program_id(next(parallel_count), squashed_dims=squashed_dims)
        if axis not in sequential_axes
        else None
        for axis in range(len(logical_grid))
    ]

    def make_program_ids(step: ir.Value):
      assert ir.IndexType.isinstance(step.type)
      step = arith_dialect.index_cast(ir.IntegerType.get_signless(32), step)
      return [step if pid is None else pid for pid in program_ids_template]

    grouped_barriers = collections.defaultdict(list)
    for barrier, barrier_ref in zip(rs.barriers, runtime_barriers):
      grouped_barriers[barrier].append(barrier_ref)
    module_ctx = ModuleContext(
        name_and_src_info.name,
        grid_mapping.grid_names,
        None,
        approx_math,
        runtime_smem,
        smem_used_bytes=0,
        runtime_barriers=grouped_barriers,
        name_stack=source_info_util.NameStack(),
        traceback_caches=mlir.TracebackCaches(),
        squashed_dims=squashed_dims,
    )
    del runtime_smem, grouped_barriers, runtime_barriers

    smem_scratch_it = iter(scratch_buffers_smem)
    scratch_buffers_template = []
    should_discharge = []
    accs = []
    for aval in scratch_avals:
      match aval:
        case gpu_core.WGMMAAbstractAccumulatorRef():
          scratch_buffers_template.append(None)
          should_discharge.append(True)
          accs.append(
              mgpu.WGMMAAccumulator.zero(
                  *aval.shape, dtype=mgpu_utils.dtype_to_ir_type(aval.dtype)
              )
          )
        case gpu_core.AbstractMemoryRef() if isinstance(
            aval.dtype, gpu_core.BarrierType
        ):
          pass
        case gpu_core.AbstractMemoryRef() if aval.memory_space == SMEM:
          scratch_buffers_template.append(next(smem_scratch_it))
          should_discharge.append(False)
        case _:
          raise NotImplementedError(
              f"Unsupported scratch operand type: {aval}"
          )
    assert not jaxpr.outvars
    if any(should_discharge):
      # User-visible WGMMA APIs use the effectful accumulator references, but we
      # can't lower that directly to Mosaic GPU that uses pure dataflow for
      # accumulators. So we have to discharge the effects first.
      assert not jaxpr.constvars
      should_discharge = (
          [False] * len(grid_mapping.block_mappings)
          + should_discharge
          + [False] * len(extra_barriers)
      )
      with grid_mapping.trace_env():
        lowered_jaxpr, _ = discharge.discharge_state(
            jaxpr, (), should_discharge=should_discharge
        )
    else:
      lowered_jaxpr = jaxpr

    # Precompute the total number of bytes transferred from GMEM to SMEM,
    # so that we can do a single arrive instruction for all of the inputs.
    in_transfer_bytes = 0
    for in_smem, b_smem in zip(in_in_smem, in_buffers_smem):
      if not in_smem:
        continue
      b_smem_type = ir.MemRefType(b_smem.type)
      in_transfer_bytes += math.prod(b_smem_type.shape[1:]) * mgpu.bytewidth(
          b_smem_type.element_type
      )

    def gmem_slice(
        step: ir.Value,
        block_mapping: pallas_core.BlockMapping,
    ) -> Sequence[mgpu.DynamicSlice]:
      assert len(sequential_axes) <= 1
      program_ids = make_program_ids(step)
      idxs = _eval_index_map(module_ctx, launch_ctx, program_ids, block_mapping)
      return tuple(
          mgpu.ds(idx, dim) for idx, dim in zip(idxs, block_mapping.block_shape)
      )

    is_memory_thread = mgpu.single_thread_predicate(per_block=True)

    def fetch(idx: int, step: ir.Value, slot: ir.Value) -> None:
      if not in_in_smem[idx]:
        return

      swizzle = None
      pl_transforms = in_gmem_transforms[idx]
      if pl_transforms and isinstance(
          pl_transforms[-1], gpu_core.SwizzleTransform
      ):
        swizzle = pl_transforms[-1].swizzle
        pl_transforms = pl_transforms[:-1]
      gmem_transforms = tuple(x.to_gpu_transform() for x in pl_transforms)
      launch_ctx.async_copy(
          src_ref=in_buffers_gmem[idx],
          dst_ref=mgpu.memref_slice(in_buffers_smem[idx], slot),
          gmem_slice=gmem_slice(step, in_block_mappings[idx]),
          barrier=barriers[slot],
          gmem_transform=gmem_transforms,
          swizzle=swizzle,
          arrive=False,  # The caller must do ``arrive_expect_tx`` manually!
          predicate=is_memory_thread,
      )

    def store(
        idx: int, step: ir.Value, slot: ir.Value, prev_base_offset: ir.Value | None
    ) -> ir.Value | None:
      if not out_in_smem[idx]:
        return _as_index(-1)

      store_slice = gmem_slice(step, out_block_mappings[idx])
      if out_sequential_invariant[idx]:
        assert prev_base_offset is None
        do_store = None  # Lack of predicate defaults to True.
        base_offset = None
      else:
        assert prev_base_offset is not None
        # We have to do some work to make sure that consecutive stores are not
        # going to be writing to the same location, or else we'll end up with
        # multiple concurrent writes and a racy program.
        # TODO(apaszke,slebedev): This still diverges significantly from the TPU
        # semantics in that it will move on to the next SMEM output slice even if
        # it's not storing the previous one.
        strides, _ = ir.MemRefType(out_buffers_gmem[idx].type).get_strides_and_offset()
        base_offset = _as_index(0)
        for stride, slc in zip(strides, store_slice):
          base_offset = arith_dialect.addi(
              base_offset, arith_dialect.muli(slc.base, _as_index(stride))
          )
        base_offset_changed = arith_dialect.cmpi(
            arith_dialect.CmpIPredicate.ne, base_offset, prev_base_offset
        )
        is_last_step = arith_dialect.cmpi(
            arith_dialect.CmpIPredicate.eq, step, _as_index(num_steps - 1)
        )
        do_store = arith_dialect.andi(
            is_memory_thread, arith_dialect.ori(base_offset_changed, is_last_step)
        )

      swizzle = None
      pl_transforms = out_gmem_transforms[idx]
      if pl_transforms and isinstance(
          pl_transforms[-1], gpu_core.SwizzleTransform
      ):
        swizzle = pl_transforms[-1].swizzle
        pl_transforms = pl_transforms[:-1]
      gmem_transforms = tuple(x.to_gpu_transform() for x in pl_transforms)
      launch_ctx.async_copy(
          src_ref=mgpu.memref_slice(out_buffers_smem[idx], slot),
          dst_ref=out_buffers_gmem[idx],
          gmem_slice=store_slice,
          gmem_transform=gmem_transforms,
          swizzle=swizzle,
          predicate=do_store,
      )
      return base_offset

    for slot in range(min(max_concurrent_steps, num_steps)):
      barriers[slot].arrive_expect_tx(in_transfer_bytes, predicate=is_memory_thread)
      for idx in range(grid_mapping.num_inputs):
        fetch(idx, _as_index(slot), _as_index(slot))

    last_store_offsets = [None if inv else _as_index(-1) for inv in out_sequential_invariant]

    @mgpu.fori(_as_index(num_steps), (accs, last_store_offsets))
    def _(step, carry):
      accs, last_store_offsets = carry
      slot = arith_dialect.remui(step, _as_index(max_concurrent_steps))
      if grid_mapping.num_inputs:
        # Only wait if async copies were issued.
        barriers[slot].wait()
      # We need to make sure the output copy is complete before the kernel starts
      # writing to the output window.
      launch_ctx.await_async_copy(
          max_concurrent_steps - (1 + delay_release), await_read_only=True
      )

      args = [
          mgpu.memref_slice(buffers_smem[idx], slot)
          if in_smem
          else buffers_gmem[idx]
          for idx, in_smem in enumerate(it.chain(in_in_smem, out_in_smem))
      ]
      accs_it = iter(accs)
      scratch_buffers = [
          b if b is not None else next(accs_it)
          for b in scratch_buffers_template
      ]
      args.extend(scratch_buffers)
      # TODO(apaszke): This assumes barriers come after buffers in scratch args,
      # but that's not necessarily true.
      args.extend(extra_barriers)
      new_accs = lower_jaxpr_to_mosaic_gpu(
          dataclasses.replace(module_ctx, program_ids=make_program_ids(step)),
          launch_ctx,
          lowered_jaxpr,
          args,
      )

      if not all(out_sequential_invariant):
        mgpu.commit_shared()
      new_store_offsets = []
      for idx in range(grid_mapping.num_outputs):
        last_offset = last_store_offsets[idx]
        new_store_offsets.append(
            store(idx, step, slot, last_offset)
            if not out_sequential_invariant[idx]
            else last_offset  # Only store if the output can depend on the step.
        )

      del slot  # Just to make sure we don't accidentally use it.
      fetch_step = arith_dialect.addi(
          step, _as_index(max_concurrent_steps - delay_release)
      )
      fetch_step_in_bounds = arith_dialect.cmpi(
          arith_dialect.CmpIPredicate.ult, fetch_step, _as_index(num_steps)
      )
      not_initial_step = arith_dialect.cmpi(
          arith_dialect.CmpIPredicate.uge, step, _as_index(delay_release)
      )
      fetch_slot = arith_dialect.remui(fetch_step, _as_index(max_concurrent_steps))
      with mgpu.when(arith_dialect.andi(fetch_step_in_bounds, not_initial_step)):
        barriers[fetch_slot].arrive_expect_tx(in_transfer_bytes, predicate=is_memory_thread)
        for idx in range(grid_mapping.num_inputs):
          fetch(idx, fetch_step, fetch_slot)

      return list(new_accs), new_store_offsets

    # Outputs invariant to the sequential axis are never written from inside the
    # loop. This is the only place where we store them.
    if all(out_sequential_invariant):
      mgpu.commit_shared()
    last_slot = _as_index((num_steps - 1) % max_concurrent_steps)
    for idx in range(grid_mapping.num_outputs):
      if out_sequential_invariant[idx]:
        store(idx, _as_index(0), last_slot, None)

    launch_ctx.await_async_copy(0)

  scratch_avals = [
      var.aval for var in jaxpr.invars[grid_mapping.slice_scratch_ops]
  ]
  local_spaces = (gpu_core.SMEM, gpu_core.REGS)
  if not all(
      isinstance(aval, pallas_core.AbstractMemoryRef)
      and aval.memory_space in local_spaces
      for aval in scratch_avals
  ):
    raise TypeError(
        "All scratch operands must be SMEM references or accumulators (ACC),"
        f" but got: {scratch_avals}"
    )
  rs = _estimate_resources(jaxpr)
  extra_barriers = [
      mgpu.Barrier(aval.dtype.num_arrivals * WARPGROUP_SIZE, *aval.shape)
      for aval in scratch_avals
      if isinstance(aval.dtype, gpu_core.BarrierType)
  ]
  extra_smem_scratch = [
      jax.ShapeDtypeStruct(aval.shape, aval.dtype)
      for aval in scratch_avals
      if not isinstance(aval.dtype, gpu_core.BarrierType)
      and aval.memory_space == gpu_core.SMEM
  ]
  smem_scratch_bytes = params.get("smem_scratch_bytes")
  if smem_scratch_bytes is None:
    smem_scratch_bytes = rs.smem_scratch_bytes
  extra_smem_scratch.append(
      jax.ShapeDtypeStruct(shape=[smem_scratch_bytes], dtype=np.int8)
  )

  prof_ctx = prof_spec = None
  if prof_space := params.get("profile_space", 0):
    # Each range is 2 events, each event is 4 bytes.
    prof_spec = mgpu_profiler.ProfilerSpec(prof_space * 2 * 4)
    prof_ctx = ProfilerContext(params["profile_dir"], prof_spec)
  module, out_structs_gmem, _, launch_ctx, scratch_arr = (
      mgpu_core._lower_as_gpu_kernel(
          body,
          grid=parallel_grid,
          cluster=(),
          block=block,
          in_shapes=in_structs_gmem,
          out_shape=out_structs_gmem,
          smem_scratch_shape=(
              (*in_structs_smem, *out_structs_smem),
              *extra_smem_scratch,
              (
                  mgpu.Barrier(
                      arrival_count=1, num_barriers=max_concurrent_steps
                  ),
                  rs.barriers,
                  extra_barriers,
              ),
          ),
          module_name=name_and_src_info.name,
          prof_spec=prof_spec,
      )
  )
  mgpu_core._initialize_scratch(launch_ctx, scratch_arr)

  return LoweringResult(
      module, parallel_grid, block, out_structs_gmem, prof_ctx
  )


mosaic_lowering_rules = {}


def register_lowering_rule(primitive: jax_core.Primitive):
  def deco(fn):
    mosaic_lowering_rules[primitive] = fn
    return fn

  return deco


def _compute_name_stack_updates(
    old_name_stack: list[str],
    new_name_stack: list[str]
) -> tuple[list[str], list[str]]:
  common_prefix_idx = 0
  for i, (old, new) in enumerate(unsafe_zip(old_name_stack, new_name_stack)):
    if old == new:
      common_prefix_idx = i+1
    else:
      break
  return old_name_stack[common_prefix_idx:], new_name_stack[common_prefix_idx:]


def lower_jaxpr_to_mosaic_gpu(
    module_ctx: ModuleContext,
    launch_ctx: mgpu.LaunchContext,
    jaxpr: jax_core.Jaxpr,
    args: Sequence[ir.Value],
    consts=(),
) -> Sequence[ir.Value]:
  env = {}

  def read_env(atom: jax_core.Atom):
    return atom.val if isinstance(atom, jax_core.Literal) else env[atom]

  def write_env(var: jax_core.Var, val):
    env[var] = val

  map(write_env, jaxpr.constvars, consts)
  map(write_env, jaxpr.invars, args)
  # TODO(justinfu): Handle transform scopes.
  last_local_name_stack: list[str] = []
  named_regions = []
  for eqn in jaxpr.eqns:
    invals = map(read_env, eqn.invars)
    source_info = eqn.source_info.replace(
        name_stack=module_ctx.name_stack + eqn.source_info.name_stack
    )
    loc = mlir._source_info_to_location(module_ctx, eqn.primitive, source_info)
    with source_info_util.user_context(eqn.source_info.traceback), loc:
      if eqn.primitive not in mosaic_lowering_rules:
        raise NotImplementedError(
            "Unimplemented primitive in Pallas Mosaic GPU lowering: "
            f"{eqn.primitive.name}. "
            "Please file an issue on https://github.com/jax-ml/jax/issues."
        )
      new_local_name_stack = [scope.name for scope in eqn.source_info.name_stack.stack]
      popped, pushed = _compute_name_stack_updates(last_local_name_stack, new_local_name_stack)
      last_local_name_stack = new_local_name_stack
      for _ in popped:
        named_regions.pop().close()
      for name in pushed:
        wrapper_stack = contextlib.ExitStack()
        wrapper_stack.enter_context(launch_ctx.named_region(name))
        named_regions.append(wrapper_stack)
      rule = mosaic_lowering_rules[eqn.primitive]
      rule_ctx = LoweringRuleContext(
          module_ctx,
          launch_ctx,
          predicate=mgpu.single_thread_predicate(per_block=False),
          avals_in=[cast(jax_core.ShapedArray, v.aval) for v in eqn.invars],
          avals_out=[cast(jax_core.ShapedArray, v.aval) for v in eqn.outvars],
      )
      try:
        outvals = rule(rule_ctx, *invals, **eqn.params)
      except LoweringError:
        raise  # We only add the extra info to the innermost exception.
      except Exception as e:
        if not pallas_call._verbose_errors_enabled():
          raise
        inval_types = map(lambda t: getattr(t, "type", None), invals)
        raise LoweringError(
            f"Exception while lowering eqn:\n  {eqn}\nWith context:\n "
            f" {rule_ctx}\nWith inval types={inval_types}\nIn jaxpr:\n{jaxpr}"
        ) from e
      if eqn.primitive.multiple_results:
        map(write_env, eqn.outvars, outvals)
      else:
        write_env(eqn.outvars[0], outvals)
  while named_regions:  # Drain the name stack.
    named_regions.pop().close()
  return map(read_env, jaxpr.outvars)


@register_lowering_rule(primitives.program_id_p)
def _program_id_lowering_rule(ctx: LoweringRuleContext, axis):
  if ctx.module_ctx.program_ids is None:
    raise NotImplementedError("pl.program_id() is not supported in this context")
  return ctx.module_ctx.program_ids[axis]

def _unravel_program_id(
    block_id: ir.Value,
    axis: int,
    dimensions: tuple[int, ...],
    row_major: bool = False
) -> ir.Value:
  """Computes the program ID for axes compressed into one block dimension."""
  if row_major:
    div_value = math.prod(dimensions[axis+1:])
  else:
    div_value = math.prod(dimensions[:axis])
  div_value = _as_index(_i32_constant(div_value))
  pid = arith_dialect.divui(block_id, div_value)
  axis_size = _as_index(_i32_constant(dimensions[axis]))
  pid = arith_dialect.remui(pid, axis_size)
  return arith_dialect.index_cast(ir.IntegerType.get_signless(32), pid)


def _program_id(parallel_axis: int, squashed_dims: tuple[int, ...]) -> ir.Value:
  if squashed_dims:
    if parallel_axis < len(squashed_dims):
      # All squashed dimensions are mapped to Dimension.x.
      block_id = gpu_dialect.block_id(gpu_dialect.Dimension.x)
      return _unravel_program_id(block_id, parallel_axis, squashed_dims)
    else:
      # Handle unsquashed axes.
      return arith_dialect.index_cast(
          ir.IntegerType.get_signless(32),
          gpu_dialect.block_id(gpu_dialect.Dimension(
              parallel_axis - len(squashed_dims) + 1)),
      )
  else:
    return arith_dialect.index_cast(
        ir.IntegerType.get_signless(32),
        gpu_dialect.block_id(gpu_dialect.Dimension(parallel_axis)),
    )


@register_lowering_rule(primitives.num_programs_p)
def _num_programs_lowering_rule(ctx: LoweringRuleContext, axis):
  del ctx  # Unused.
  return arith_dialect.index_cast(
      ir.IntegerType.get_signless(32),
      gpu_dialect.block_dim(gpu_dialect.Dimension(axis)),
  )


def _handle_reshaping(
    ref: ir.Value, transforms: Sequence[gpu_core.Transform]
) -> tuple[ir.Value, Sequence[gpu_core.Transform]]:
  is_trivial_indexer = lambda t: isinstance(
      t, indexing.NDIndexer
  ) and gpu_core.is_trivial_index(t.indices, t.shape)

  last_reshaper_idx = next(
      reversed([i for i, t in enumerate(transforms) if isinstance(t, RefReshaper)]),
      None,
  )
  if last_reshaper_idx is None:
    return ref, transforms
  # Check that before the reshape are only trivial indexes and or
  # other reshapes.
  # TODO(cperivol): Reshapes should bubble up  rather than being
  # expected to effectively be the first ref transform.
  if not all(isinstance(t, RefReshaper) or is_trivial_indexer(t) for t in transforms[:last_reshaper_idx]):
    raise NotImplementedError(
        "Reshapes do not compose with other transforms and indexers must be"
        f" trivial (transforms: {transforms})"
    )
  reshaper = cast(RefReshaper, transforms[last_reshaper_idx])
  # Skip all the reshapes and trivial indexes.
  return mgpu.memref_reshape(ref, reshaper.shape), transforms[last_reshaper_idx + 1:]


def _handle_indexing(
    ref: ir.Value, transforms: Sequence[gpu_core.Transform]
) -> tuple[ir.Value, Sequence[gpu_core.Transform]]:
  if not transforms:
    pass
  indexer_idxs = [
      i for i, t in enumerate(transforms) if isinstance(t, indexing.NDIndexer)
  ]
  if not indexer_idxs:
    return ref, transforms
  sliced_ref = ref
  new_transforms = []
  for t in transforms:
    if not isinstance(t, indexing.NDIndexer):
      new_transforms.append(t)
      continue
    indexer = cast(indexing.NDIndexer, t)
    if indexer.int_indexer_shape:
      raise NotImplementedError("int_indexer_shape non-empty")
    indices = _ndindexer_indices(indexer)
    new_transforms_rev = []
    for t in reversed(new_transforms):
      indices, new_t = t.untransform_index(indices)
      new_transforms_rev.append(new_t)
    sliced_ref = mgpu.memref_slice(sliced_ref, indices)
    new_transforms = list(reversed(new_transforms_rev))
  return sliced_ref, new_transforms


def _ndindexer_indices(indexer: indexing.NDIndexer) -> tuple[gpu_core.Index, ...]:
  indices = []
  for idx in indexer.indices:
    if not isinstance(idx, indexing.Slice):
      indices.append(_as_index(idx))
    elif not idx.is_dynamic_start and not idx.is_dynamic_size:
      indices.append(slice(idx.start, idx.start + idx.size, idx.stride))
    elif idx.stride == 1:
      indices.append(
          mgpu.DynamicSlice(
              _as_index(idx.start) if idx.is_dynamic_start else idx.start,
              _as_index(idx.size) if idx.is_dynamic_size else idx.size,
          )
      )
    else:
      raise NotImplementedError(f"Unsupported slice: {idx}")
  return tuple(indices)


@register_lowering_rule(sp.get_p)
def _get_lowering_rule(ctx: LoweringRuleContext, x_smem, *leaves, tree):
  if not isinstance(x_smem, ir.Value) and ir.MemRefType.isinstance(x_smem):
    raise TypeError(f"Can only load from references (got {x_smem}).")

  x_aval = ctx.avals_in[0]

  transforms = jax.tree.unflatten(tree, leaves)
  x_smem, transforms = _handle_reshaping(x_smem, transforms)
  x_smem, transforms = _handle_indexing(x_smem, transforms)

  match transforms:
    case (gpu_core.UnswizzleRef(swizzle), gpu_core.UntileRef(tiling)):
      if tiling != (64, swizzle // x_aval.dtype.itemsize):
        raise NotImplementedError("Tiling does not fit swizzle")
      return mgpu.FragmentedArray.load_tiled(
          x_smem, is_signed=mgpu_utils.is_signed(x_aval.dtype), swizzle=swizzle
      )
    case ():
      # Handle scalar indexing.
      if not ctx.avals_out[0].shape:
        is_signed = mgpu_utils.is_signed(x_aval.dtype)
        val = memref_dialect.load(x_smem, [])
        return mgpu.FragmentedArray.splat(val, shape=(), is_signed=is_signed)

      return mgpu.FragmentedArray.load_strided(
          x_smem, is_signed=mgpu_utils.is_signed(x_aval.dtype)
      )
    case _:
      raise NotImplementedError(f"Unsupported transforms: {transforms}")


@register_lowering_rule(sp.swap_p)
def _swap_lowering_rule(
    ctx: LoweringRuleContext, x_smem, value, *leaves, tree
):
  if not isinstance(value, mgpu.FragmentedArray):
    raise TypeError(f"Can only store arrays (got {value}).")
  if not isinstance(x_smem, ir.Value) and ir.MemRefType.isinstance(x_smem):
    raise TypeError(f"Can only store to references (got {x_smem}).")
  x_aval = ctx.avals_in[0]
  transforms = jax.tree.unflatten(tree, leaves)
  x_smem, transforms = _handle_reshaping(x_smem, transforms)
  x_smem, transforms = _handle_indexing(x_smem, transforms)
  match transforms:
    case (gpu_core.UnswizzleRef(swizzle), gpu_core.UntileRef(tiling)):
      if tiling != (64, swizzle // x_aval.dtype.itemsize):
        raise NotImplementedError("Tiling does not fit swizzle")
      old_value = mgpu.FragmentedArray.load_tiled(
          x_smem, is_signed=mgpu_utils.is_signed(x_aval.dtype), swizzle=swizzle
      )
      value.store_tiled(x_smem, swizzle=swizzle)
      return old_value
    case ():
      old_value = mgpu.FragmentedArray.load_strided(
          x_smem, is_signed=mgpu_utils.is_signed(x_aval.dtype)
      )
      value.store_untiled(x_smem)
      return old_value
    case _:
      raise NotImplementedError(f"Unsupported transforms: {transforms}")


@register_lowering_rule(pjit.pjit_p)
def _pjit_lowering_rule(ctx: LoweringRuleContext, *args, jaxpr, **_):
  if jaxpr.consts:
    raise NotImplementedError
  return lower_jaxpr_to_mosaic_gpu(
      ctx.module_ctx, ctx.launch_ctx, jaxpr.jaxpr, args
  )


@register_lowering_rule(lax.slice_p)
def _slice_lowering_rule(
    ctx: LoweringRuleContext, x, limit_indices, start_indices, strides
):
  if strides is not None:
    raise NotImplementedError("Strides are not supported.")

  return x[tuple(slice(b, e) for b, e in zip(start_indices, limit_indices))]


@register_lowering_rule(lax.select_n_p)
def _select_n_lowering_rule(ctx: LoweringRuleContext, pred, *cases):
  if len(cases) != 2:
    raise NotImplementedError(
        "Mosaic GPU lowering only supports select_n with 2 cases, got"
        f" {len(cases)}"
    )
  pred_aval, *cases_avals = ctx.avals_in
  [out_aval] = ctx.avals_out
  pred = _ensure_fa(pred, pred_aval.dtype)
  cases = _bcast(*cases, *cases_avals, out_aval)
  # ``select`` expects the first case to be the true branch, but ``select_n``
  # orders the cases in reverse.
  return pred.select(*reversed(cases))


@register_lowering_rule(lax.broadcast_in_dim_p)
def _broadcast_in_dim_lowering_rule(
    ctx: LoweringRuleContext,
    x: mgpu.FragmentedArray,
    *,
    broadcast_dimensions,
    shape,
    sharding,
):
  del sharding
  [x_aval] = ctx.avals_in
  [y_aval] = ctx.avals_out
  x = _ensure_fa(x, x_aval.dtype)
  if (
      broadcast_dimensions == tuple(range(x_aval.ndim))
      and y_aval.ndim == x_aval.ndim + 1
      and x.layout == mgpu.WGMMA_ROW_LAYOUT
  ):
    return x.broadcast_minor(y_aval.shape[-1])
  if broadcast_dimensions:
    raise NotImplementedError
  return x.broadcast(shape)


@register_lowering_rule(lax.convert_element_type_p)
def _convert_element_type_lowering_rule(
    ctx: LoweringRuleContext, x, *, new_dtype, weak_type, sharding
):
  del weak_type, sharding
  [x_aval] = ctx.avals_in
  return _ensure_fa(x, x_aval.dtype).astype(
      mgpu_utils.dtype_to_ir_type(new_dtype), is_signed=mgpu_utils.is_signed(new_dtype)
  )


mosaic_lowering_rules.update({
    lax.neg_p: lambda ctx, x: -x,
    lax.not_p: lambda ctx, x: ~x,
})


def _binary_op_lowering_rule(ctx: LoweringRuleContext, x, y, *, impl):
  x, y = _bcast(x, y, *ctx.avals_in, *ctx.avals_out)
  return impl(x, y)


mosaic_lowering_rules.update({
    lax.add_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x + y),
    lax.sub_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x - y),
    lax.mul_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x * y),
    lax.rem_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x % y),
    lax.and_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x & y),
    lax.or_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x | y),
    lax.xor_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x ^ y),
    lax.gt_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x > y),
    lax.lt_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x < y),
    lax.ge_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x >= y),
    lax.le_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x <= y),
    lax.eq_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x == y),
    lax.ne_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x != y),
    lax.max_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x.max(y)),
    lax.min_p: partial(_binary_op_lowering_rule, impl=lambda x, y: x.min(y)),
})


@register_lowering_rule(lax.div_p)
def _div_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, *ctx.avals_in, *ctx.avals_out)
  if ir.FloatType.isinstance(x.mlir_dtype):
    return x / y
  return x // y


@register_lowering_rule(lax.integer_pow_p)
def _integer_pow_lowering_rule(ctx: LoweringRuleContext, x, y):
  [x_aval] = ctx.avals_in
  x = _ensure_fa(x, x_aval.dtype)
  if y == 2:
    return x * x
  return NotImplementedError

@register_lowering_rule(lax.square_p)
def _square_lowering_rule(ctx: LoweringRuleContext, x):
  [x_aval] = ctx.avals_in
  x = _ensure_fa(x, x_aval.dtype)
  return x * x

@register_lowering_rule(lax.rsqrt_p)
def _rsqrt_lowering_rule(ctx: LoweringRuleContext, x):
  [x_aval] = ctx.avals_in
  return _ensure_fa(x, x_aval.dtype).rsqrt(approx=ctx.module_ctx.approx_math)

@register_lowering_rule(lax.tanh_p)
def _tanh_lowering_rule(ctx: LoweringRuleContext, x):
  [x_aval] = ctx.avals_in
  return _ensure_fa(x, x_aval.dtype).tanh(approx=ctx.module_ctx.approx_math)


@register_lowering_rule(lax.logistic_p)
def _logistic_lowering_rule(ctx: LoweringRuleContext, x):
  [x_aval] = ctx.avals_in
  a = _ensure_fa(x, x_aval.dtype)
  return 1. / (1. + (-a).exp(approx=ctx.module_ctx.approx_math))

@register_lowering_rule(lax.exp_p)
def _exp_lowering_rule(ctx: LoweringRuleContext, x):
  [x_aval] = ctx.avals_in
  a = _ensure_fa(x, x_aval.dtype)
  return a.exp(approx=ctx.module_ctx.approx_math)


@register_lowering_rule(lax.exp2_p)
def _exp2_lowering_rule(ctx: LoweringRuleContext, x):
  [x_aval] = ctx.avals_in
  a = _ensure_fa(x, x_aval.dtype)
  return a.exp2(approx=ctx.module_ctx.approx_math)


@register_lowering_rule(lax.reduce_sum_p)
def _reduce_sum_lowering_rule(ctx: LoweringRuleContext, x, *, axes):
  [x_aval] = ctx.avals_in
  match x.layout:
    case mgpu.WGStridedFragLayout():
      if set(axes) != set(range(x_aval.ndim)):
        raise NotImplementedError("No support for axes yet")
      scratch_ty = jax.ShapeDtypeStruct(shape=(4,), dtype=x_aval.dtype)
      with ctx.module_ctx.scratch_view([scratch_ty]) as [scratch]:
        return x.reduce_sum(scratch)
    case mgpu.WGMMA_LAYOUT:
      if axes != (x_aval.ndim - 1,):
        raise NotImplementedError
      if not jnp.issubdtype(x_aval.dtype, jnp.floating):
        raise NotImplementedError
      return x.reduce("add", axes[0])
    case _:
      raise NotImplementedError(f"Unsupported layout {x.layout}")


@register_lowering_rule(lax.reduce_max_p)
def _reduce_max_lowering_rule(ctx: LoweringRuleContext, x, *, axes):
  [x_aval] = ctx.avals_in
  match x.layout:
    case mgpu.WGMMA_LAYOUT:
      if axes != (x_aval.ndim - 1,):
        raise NotImplementedError
      if not jnp.issubdtype(x_aval.dtype, jnp.floating):
        raise NotImplementedError
      return x.reduce("max", axes[0])
    case _:
      raise NotImplementedError(f"Unsupported layout {x.layout}")


@register_lowering_rule(lax.axis_index_p)
def _axis_index_rule(ctx: LoweringRuleContext, *, axis_name: Hashable):
  i32 = ir.IntegerType.get_signless(32)
  grid_names = ctx.module_ctx.grid_names
  squashed_dims = ctx.module_ctx.squashed_dims
  if squashed_dims:
    unsquashed_names = grid_names[-3:]
    squashed_names = grid_names[:-3]
  else:
    # These are unused but initialized for type checkers.
    unsquashed_names = ()
    squashed_names = ()
  if grid_names and axis_name in grid_names:
    if axis_name == grid_names[-1]:
      return mgpu.warpgroup_idx(sync=True)
    else:
      if squashed_dims:
        if axis_name in unsquashed_names:
          # We add 1 to the index because the first dimension is the
          # squashed dimension.
          # e.g. for the grid (a, b, c, d, wg)
          # squashed = (a, b)  Mapped to Dimension.x (0)
          # unsquashed = (c, d)  Mapped to Dimension.y (1) and Dimension.z (2)
          idx = unsquashed_names.index(axis_name) + 1
          return arith_dialect.index_cast(
            i32,
            gpu_dialect.block_id(gpu_dialect.Dimension(idx)),
          )
        elif axis_name in squashed_names:
          # All squashed dimensions are mapped to Dimension.x.
          block_id = gpu_dialect.block_id(gpu_dialect.Dimension.x)
          axis = squashed_names.index(axis_name)
          return _unravel_program_id(block_id, axis, squashed_dims)
      else:
        if axis_name in grid_names:
          idx = grid_names.index(axis_name)
          return arith_dialect.index_cast(
            i32,
            gpu_dialect.block_id(gpu_dialect.Dimension(idx)),
          )
  raise ValueError(
      "Named axes can only refer to GPUMesh axes in Mosaic GPU kernels"
  )


@register_lowering_rule(primitives.debug_print_p)
def _debug_print_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    fmt,
    has_placeholders: bool,
):
  del has_placeholders  # Unused.
  primitives.check_debug_print_format(fmt, *args)
  if not any(aval.shape for aval in ctx.avals_in):
    mgpu.debug_print(
        fmt,
        *(
            _ensure_ir_value(arg, aval.dtype)
            for arg, aval in zip(args, ctx.avals_in)
        ),
    )
  elif len(ctx.avals_in) == 1:
    [arg] = args
    @arg.foreach
    def _(val, idx):
      idx_fmt = ", ".join(["{}"] * len(idx))
      fmt_str = fmt.format(f"[{idx_fmt}]/{list(arg.shape)}: {{}}")
      mgpu.debug_print(fmt_str, *idx, val, uniform=False)
  else:
    raise NotImplementedError(
        "debug_print only supports printing of scalar values, or a single array"
        " value when using the Mosaic GPU backend."
    )

  return ()


@register_lowering_rule(primitives.run_scoped_p)
def _run_scoped_lowering_rule(
    ctx: LoweringRuleContext, *consts, jaxpr: jax_core.Jaxpr
):
  input_refs = []
  should_discharge = []
  alloc_stack = contextlib.ExitStack()
  for v in jaxpr.invars:
    aval = v.aval
    if isinstance(aval, gpu_core.WGMMAAbstractAccumulatorRef):
      mlir_dtype = mlir.dtype_to_ir_type(aval.dtype)
      input_refs.append(mgpu.WGMMAAccumulator.zero(*aval.shape, mlir_dtype))
      should_discharge.append(True)
    elif isinstance(aval.dtype, gpu_core.BarrierType):
      input_refs.append(
          ctx.module_ctx.reserve_barrier(
              mgpu.Barrier(
                  aval.dtype.num_arrivals * WARPGROUP_SIZE, *aval.shape
              )
          )
      )
      should_discharge.append(False)
    elif aval.memory_space == gpu_core.SMEM:
      [input_ref] = alloc_stack.enter_context(
          ctx.module_ctx.scratch_view(
              [jax.ShapeDtypeStruct(shape=aval.shape, dtype=aval.dtype)]
          )
      )
      input_refs.append(input_ref)
      should_discharge.append(False)
    else:
      raise ValueError(f"Can't convert to ref: {aval}")

  if any(should_discharge):
    # We convert consts to args, because we only have ir.Values and
    # not JAX values during lowering. discharge_state() produces JAX
    # valiues for the aguments but expects them to be provided for the
    # consts. We also don't want to wrap the values in refs.
    no_const_jaxpr = pe.convert_constvars_jaxpr(jaxpr)
    should_discharge = [False] * len(consts) + should_discharge
    discharged_jaxpr, _ = discharge.discharge_state(no_const_jaxpr, (), should_discharge=should_discharge)
    new_input_vals = consts + tuple(input_refs)
    outs = lower_jaxpr_to_mosaic_gpu(
        ctx.module_ctx, ctx.launch_ctx, discharged_jaxpr, new_input_vals, ()
    )
    # Discharge appends to the output the refs that got discharged.
    outs = outs[:-sum(should_discharge)]
  else:
    outs = lower_jaxpr_to_mosaic_gpu(
        ctx.module_ctx, ctx.launch_ctx, jaxpr, input_refs, consts
    )

  for o in outs:
    # This is definitely one of the accumulators we produced. Each
    # run_scoped call is responsible for dereferencing its own
    # accumulators.
    if isinstance(o, mgpu.WGMMAAccumulator) or (
        isinstance(o, ir.Value) and ir.MemRefType.isinstance(o.type)
    ):
      raise ValueError(f"No references are allowed to escape a scope. (got {o})")

  assert len(outs) == len(jaxpr.outvars), (jaxpr, outs)
  return outs


@register_lowering_rule(discharge.run_state_p)
def _run_state_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    jaxpr: jax_core.Jaxpr,
    which_linear: tuple[bool, ...],
    is_initialized: tuple[bool, ...],
):
  del which_linear
  # TODO(apaszke): This should be unified with run_scoped.
  if not all(is_initialized):
    raise NotImplementedError("Uninitialized Refs are not supported in lowering of run_state.")

  should_discharge = []
  new_input_vals = []
  for arg, v, out_aval in zip(args, jaxpr.invars, ctx.avals_out):
    aval = v.aval
    if isinstance(aval, gpu_core.WGMMAAbstractAccumulatorRef):
      new_input_vals.append(mgpu.WGMMAAccumulator.from_registers(arg))
      should_discharge.append(True)
      assert isinstance(out_aval, jax_core.ShapedArray)
    else:
      new_input_vals.append(arg)
      should_discharge.append(not isinstance(out_aval, state_types.AbstractRef))
  if not any(should_discharge):
    raise NotImplementedError(
        "Expected at least one accumulator to in run_state."
    )

  discharged_jaxpr, new_consts = discharge.discharge_state(
      jaxpr, (), should_discharge=should_discharge
  )
  assert not new_consts
  outs = lower_jaxpr_to_mosaic_gpu(
      ctx.module_ctx, ctx.launch_ctx, discharged_jaxpr, new_input_vals, ()
  )
  # Await the accumulators and extract their final values.
  nvvm_dialect.wgmma_wait_group_sync_aligned(0)
  outs = [
      out.value if isinstance(out, mgpu.WGMMAAccumulator) else out
      for out in outs
  ]
  # Blend the discharge results with refs we closed over. I don't fully
  # understand the reasons behind this calling convention, but sharadmv@ has
  # assured me that this is ok.
  outs_it = iter(outs)
  return [next(outs_it) if d else a for d, a in zip(should_discharge, args)]


def _lower_jaxpr_to_for_loop(
    ctx: LoweringRuleContext,
    jaxpr: jax_core.Jaxpr,
    start: ir.Value,
    length: ir.Value,
    consts,
    *args,
    has_loop_index: bool,
):

  _consts_avals, arg_avals = util.split_list(ctx.avals_in, [len(consts)])
  arg_avals = arg_avals[has_loop_index:]
  out_avals = []
  if arg_avals:
    out_avals = ctx.avals_out[-len(arg_avals):]

  @mgpu.fori(length, [*map(_ensure_fa, args, arg_avals)])
  def loop(loop_index, body_args):
    if has_loop_index:
      loop_index = arith_dialect.addi(loop_index, start)
      jaxpr_args = [*consts, loop_index, *body_args]
    else:
      jaxpr_args = [*consts, *body_args]
    outs = lower_jaxpr_to_mosaic_gpu(
        ctx.module_ctx, ctx.launch_ctx, jaxpr, jaxpr_args
    )
    return map(_ensure_fa, outs, out_avals)

  return loop.results


@register_lowering_rule(lax.scan_p)
def _scan_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    jaxpr: jax_core.ClosedJaxpr,
    linear: tuple[bool, ...],
    length: int,
    reverse: bool,
    unroll: bool | int,
    num_consts: int,
    num_carry: int,
    _split_transpose: bool,
):
  # Can only handle fori_loop-like scans.
  if (
      (num_extensive := len(args) - num_consts - num_carry)
      or reverse
      or unroll != 1
  ):
    raise NotImplementedError
  del linear, num_extensive, reverse, unroll

  jaxpr, jaxpr_consts = jaxpr.jaxpr, jaxpr.consts
  if jaxpr_consts:
    raise NotImplementedError
  del jaxpr_consts

  jaxpr, has_loop_index = pallas_utils.pattern_match_scan_to_fori_loop(
      jaxpr, num_consts, num_carry
  )
  consts, args = util.split_list(args, [num_consts])
  _consts_avals, arg_avals = util.split_list(ctx.avals_in, [num_consts])
  if has_loop_index:
    start, *args = args
    index_aval, *_ = arg_avals
    start: ir.Value = _ensure_ir_value(start, index_aval.dtype)
    length = _ir_constant(length, start.type)
  else:
    start = _i32_constant(0)
    length = _i32_constant(length)
  for_out = _lower_jaxpr_to_for_loop(
      ctx, jaxpr, start, length, consts, *args, has_loop_index=has_loop_index
  )
  if has_loop_index:
    # Need to return the final loop index value if the outer scan expects
    # it as an output.
    return [length, *for_out]
  return for_out


def _lower_while_via_fori(
    ctx: LoweringRuleContext,
    *args,
    fori_jaxpr,
    cond_nconsts,
    body_nconsts,
):
  assert not fori_jaxpr.constvars
  # The pattern matcher looks for conditions with no constants.
  assert cond_nconsts == 0

  # Reflect the changes of the pattern matcher to the context.
  lb_aval, ub_aval, *_ = ctx.avals_in[cond_nconsts + body_nconsts:]
  ctx = ctx.replace(
      avals_in=(
          *ctx.avals_in[cond_nconsts:body_nconsts],
          ctx.avals_in[body_nconsts],  # the index
          *ctx.avals_in[body_nconsts + 2 :],
      ),
      avals_out=tuple(ctx.avals_out[2:]),
  )
  _, consts, (lb, ub, *args) = util.split_list(
      args, [cond_nconsts, body_nconsts]
  )
  lb = _ensure_ir_value(lb, lb_aval.dtype)
  ub = _ensure_ir_value(ub, ub_aval.dtype)
  for_out = _lower_jaxpr_to_for_loop(
      ctx,
      fori_jaxpr,
      lb,
      arith_dialect.subi(ub, lb),
      consts,
      *args,
      has_loop_index=True,
  )
  return ub, ub, *for_out


@register_lowering_rule(lax.while_p)
def _while_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    cond_jaxpr,
    body_jaxpr,
    cond_nconsts,
    body_nconsts,
):
  # First try to lower via a simpler fori loop, which may optimize better.
  fori_jaxpr, _ = pallas_utils.pattern_match_while_to_fori_loop(
      cond_jaxpr, cond_nconsts, body_jaxpr, body_nconsts
  )
  if fori_jaxpr is not None:
    return _lower_while_via_fori(
        ctx,
        *args,
        fori_jaxpr=fori_jaxpr,
        cond_nconsts=cond_nconsts,
        body_nconsts=body_nconsts,
    )

  # If we fail conversion to fori, fallback to an ordinary while loop.
  cond_consts, body_consts, carry = util.split_list(
      args, [cond_nconsts, body_nconsts]
  )
  _cond_avals, body_avals, carry_avals = util.split_list(
      ctx.avals_in, [cond_nconsts, body_nconsts]
  )
  carry = map(_ensure_fa, carry, carry_avals)
  # Flatten the carry to get a concatenated list of registers from each FA.
  # Note that the treedef is also used below to unflatten the body results.
  flat_carry, carry_treedef = jax.tree.flatten(carry)
  flat_carry_types = [a.type for a in flat_carry]
  while_op = scf_dialect.WhileOp(flat_carry_types, flat_carry)

  before_block = while_op.before.blocks.append(*flat_carry_types)
  with ir.InsertionPoint.at_block_begin(before_block):
    cond_args = [*cond_consts, *carry_treedef.unflatten(before_block.arguments)]
    [cond] = lower_jaxpr_to_mosaic_gpu(
        ctx.module_ctx, ctx.launch_ctx, cond_jaxpr.jaxpr, cond_args
    )
    scf_dialect.condition(
        _ensure_ir_value(cond, *cond_jaxpr.out_avals), before_block.arguments
    )

  after_block = while_op.after.blocks.append(*flat_carry_types)
  with ir.InsertionPoint.at_block_begin(after_block):
    body_args = [*body_consts, *carry_treedef.unflatten(after_block.arguments)]
    loop_out = lower_jaxpr_to_mosaic_gpu(
        ctx.module_ctx, ctx.launch_ctx, body_jaxpr.jaxpr, body_args
    )
    loop_out = map(_ensure_fa, loop_out, carry_avals)
    for idx, (carry_fa, out_fa) in enumerate(zip(carry, loop_out)):
      if carry_fa.layout != out_fa.layout:
        raise ValueError(
            f"The loop body output has unexpected layout: output[{idx}] has"
            f" layout {out_fa.layout}, when it should be {carry_fa.layout}."
        )
    scf_dialect.yield_(
        carry_treedef.flatten_up_to(loop_out) if loop_out else []
    )
  return carry_treedef.unflatten(list(while_op.results))


@register_lowering_rule(lax.cond_p)
def _cond_lowering_rule(ctx: LoweringRuleContext, index, *args, branches):
  index_aval, *_arg_avals = ctx.avals_in

  def _yielded_values(outs, avals):
    ret = []
    for out, aval in zip(outs, avals):
      if isinstance(out, mgpu.FragmentedArray):
        ret.append(out)
      else:
        ret.append(_ensure_ir_value(out, aval.dtype))
    return ret

  # We need the branch return mlir types in order to construct the
  # switch operation. To avoid leaking information about what kind of
  # mlir types are internal to FragmentedArrays and other mgpu types,
  # we run one of the branches in a dummy module that we throw away to
  # extract the return types
  with ir.InsertionPoint(ir.Module.create().body):
    outs = lower_jaxpr_to_mosaic_gpu(
        ctx.module_ctx, ctx.launch_ctx, branches[0].jaxpr, args
    )
    yielded_types = [v.type for v in jax.tree.leaves(_yielded_values(outs, ctx.avals_out))]
    del outs

  switch_op = scf_dialect.IndexSwitchOp(
      yielded_types,
      _as_index(_ensure_ir_value(index, index_aval.dtype)),
      ir.DenseI64ArrayAttr.get(range(len(branches) - 1)),
      num_caseRegions=len(branches) - 1,
  )

  # ``RegionSequence`` in MLIR does not support slicing, so the
  # auto-generated Python bindings for ``caseRegions`` fail at runtime!
  # We convert it to a list to work around that.
  regions = list(switch_op.regions)
  # Move the default region to the back.
  regions = regions[1:] + regions[:1]
  treedef = None
  for branch, region in zip(branches, regions):
    with ir.InsertionPoint(region.blocks.append()):
      outs = lower_jaxpr_to_mosaic_gpu(
          ctx.module_ctx, ctx.launch_ctx, branch.jaxpr, args, consts=branch.consts
      )

      yielded_leaves, yielded_treedef = jax.tree.flatten(_yielded_values(outs, ctx.avals_out))
      if treedef is None:
        treedef = yielded_treedef
      else:
        assert treedef == yielded_treedef

      scf_dialect.yield_(yielded_leaves)

  assert treedef is not None
  return treedef.unflatten(list(switch_op.results))


@register_lowering_rule(lax.bitcast_convert_type_p)
def _bitcast_convert_type_lowering_rule(
    ctx: LoweringRuleContext, operand, *, new_dtype
):
  # TODO(petebu) Handle case where src and dst types have different bitwidths
  [operand_aval] = ctx.avals_in
  operand = _ensure_fa(operand, operand_aval.dtype)
  src_elem_type = mgpu_utils.dtype_to_ir_type(operand_aval.dtype)
  dst_elem_type = mgpu_utils.dtype_to_ir_type(new_dtype)
  assert isinstance(src_elem_type, (ir.IntegerType, ir.FloatType))
  assert isinstance(dst_elem_type, (ir.IntegerType, ir.FloatType))
  if src_elem_type.width != dst_elem_type.width:
    raise NotImplementedError(
        f"Can't bitcast from {operand_aval.dtype} to {new_dtype} because they"
        " have different widths"
    )
  if ir.IntegerType.isinstance(dst_elem_type):
    output_is_signed = mgpu_utils.is_signed(new_dtype)
  else:
    output_is_signed = None
  return mgpu.FragmentedArray.bitcast(
      operand, dst_elem_type, output_is_signed=output_is_signed
  )


@register_lowering_rule(lax.optimization_barrier_p)
def _optimization_barrier_lowering(ctx: LoweringRuleContext, *args):
  args = (_ensure_fa(arg, aval.dtype) for arg, aval in zip(args, ctx.avals_in))
  return mgpu.optimization_barrier(*args)


def _bcast(
    x: ir.Value,
    y: ir.Value,
    x_aval: jax_core.ShapedArray,
    y_aval: jax_core.ShapedArray,
    out_aval: jax_core.ShapedArray,
) -> tuple[mgpu.FragmentedArray, mgpu.FragmentedArray]:
  if not isinstance(x, mgpu.FragmentedArray):
    x_dtype = x_aval.dtype
    if x_aval.weak_type:
      x_dtype = y_aval.dtype
    x = _ensure_fa(x, x_dtype)
  if not isinstance(y, mgpu.FragmentedArray):
    y_dtype = y_aval.dtype
    if y_aval.weak_type:
      y_dtype = x_aval.dtype
    y = _ensure_fa(y, y_dtype)
  if x_aval.shape != out_aval.shape:
    x = x.broadcast(out_aval.shape)
  if y_aval.shape != out_aval.shape:
    y = y.broadcast(out_aval.shape)
  return x, y


def _ensure_fa(x: object, dtype: jnp.dtype) -> mgpu.FragmentedArray:
  if isinstance(x, mgpu.FragmentedArray):
    assert x.mlir_dtype == mgpu_utils.dtype_to_ir_type(dtype)
    return x
  return mgpu.FragmentedArray.splat(
      _ensure_ir_value(x, dtype), (), is_signed=mgpu_utils.is_signed(dtype)
  )


def _ensure_ir_value(x: object, dtype: jnp.dtype) -> ir.Value:
  if isinstance(x, ir.Value):
    assert x.type == mgpu_utils.dtype_to_ir_type(dtype)
    return x
  elif isinstance(x, mgpu.FragmentedArray):
    assert x.mlir_dtype == mgpu_utils.dtype_to_ir_type(dtype)
    if isinstance(x.layout, mgpu.WGSplatFragLayout):
      return x.registers.item()
    raise NotImplementedError(f"Unsupported layout: {x.layout}")
  return _ir_constant(x, mgpu_utils.dtype_to_ir_type(dtype))


def _ir_constant(v: object, t: ir.Type) -> ir.Value:
  if isinstance(v, (np.number, np.ndarray, int, float)):
    if isinstance(t, (ir.IntegerType, ir.IndexType)):
      v = int(v)
    else:
      assert isinstance(t, ir.FloatType)
      v = float(v)
    return arith_dialect.constant(t, v)
  raise NotImplementedError(f"Unsupported constant: {v!r}")


def _i32_constant(v: int) -> ir.Value:
  if v < jnp.iinfo(jnp.int32).min or v > jnp.iinfo(jnp.int32).max:
    raise ValueError(f"Integer constant out of range for i32: {v}")
  return arith_dialect.constant(ir.IntegerType.get_signless(32), v)


def _i64_constant(v: int) -> ir.Value:
  if v < jnp.iinfo(jnp.int64).min or v > jnp.iinfo(jnp.int64).max:
    raise ValueError(f"Integer constant out of range for i64: {v}")
  return arith_dialect.constant(ir.IntegerType.get_signless(64), v)


def _as_index(v: object) -> ir.Value:
  match v:
    case int():
      return arith_dialect.constant(ir.IndexType.get(), v)
    case ir.Value() if ir.IndexType.isinstance(v.type):
      return v
    case ir.Value() if ir.IntegerType.isinstance(v.type):
      return arith_dialect.index_cast(ir.IndexType.get(), v)
    case mgpu.FragmentedArray(layout=mgpu.WGSplatFragLayout()):
      return _as_index(v.registers.item())
    case _:
      raise ValueError(f"Unsupported index: {v} of type {type(v)}")


def merge_indexers(
    indexers: Sequence[indexing.NDIndexer]) -> indexing.NDIndexer:
  """Merges multiple indexers into a single indexer.

  This function computes a new indexer such that applying the
  new indexer produces the same result as applying the sequence
  of input indexers in order from first-to-last.
  """
  if len(indexers) == 0:
    raise ValueError("Cannot merge empty list of indexers")
  if len(indexers) == 1:
    return indexers[0]
  root_shape = indexers[0].shape
  current_indices = [indexing.Slice(0, size, 1) for size in root_shape]
  removed_dimensions = set()
  for indexer in indexers:
    if indexer.int_indexer_shape:
      raise NotImplementedError()

    num_skipped = 0
    for i in range(len(current_indices)):
      # Integer indexers remove dimensions which should be
      # skipped by following indexers.
      if i in removed_dimensions:
        num_skipped += 1
        continue
      dim_indexer = indexer.indices[i - num_skipped]
      current_index = current_indices[i]
      assert isinstance(current_index, indexing.Slice)

      current_start_index = _ensure_fa(current_index.start, jnp.int32)
      if isinstance(dim_indexer, indexing.Slice):
        if dim_indexer.stride != 1:
          raise NotImplementedError("Non-unit strides not implemented.")
        current_indices[i] = indexing.Slice(
            current_start_index + _ensure_fa(dim_indexer.start, jnp.int32),
            dim_indexer.size,
            1,
        )
      else:
        current_indices[i] = current_start_index + _ensure_fa(
              dim_indexer, dtype=jnp.int32)
        removed_dimensions.add(i)
  return indexing.NDIndexer(
      indices=tuple(current_indices),
      shape=root_shape,
      int_indexer_shape=(),
  )
