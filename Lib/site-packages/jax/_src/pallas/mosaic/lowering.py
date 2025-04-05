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

"""Module for lowering JAX to Mosaic-compatible MLIR dialects."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import contextlib
import dataclasses
import functools
import string
from typing import Any, Hashable

import jax
from jax import lax
from jax import tree_util
from jax._src import ad_util
from jax._src import checkify
from jax._src import core as jax_core
from jax._src import custom_derivatives
from jax._src import debugging
from jax._src import dtypes
from jax._src import linear_util as lu
from jax._src import mesh as mesh_lib
from jax._src import pjit
from jax._src import prng
from jax._src import source_info_util
from jax._src import state
from jax._src import traceback_util
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import lax as lax_internal
from jax._src.lax.control_flow import for_loop
from jax._src.lib import version as jaxlib_version
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import func
from jax._src.lib.mlir.dialects import math
from jax._src.lib.mlir.dialects import memref
from jax._src.lib.mlir.dialects import scf
from jax._src.lib.mlir.dialects import vector
from jax._src.pallas import core as pallas_core
from jax._src.pallas import pallas_call
from jax._src.pallas import primitives
from jax._src.pallas import utils as pallas_utils
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import error_handling
from jax._src.pallas.mosaic import primitives as tpu_primitives
from jax._src.pallas.mosaic import random as pl_random
from jax._src.state import discharge as state_discharge
from jax._src.state import indexing
from jax._src.state import primitives as state_primitives
from jax._src.state.types import RefBitcaster, RefReshaper
from jax._src.state.utils import dtype_bitwidth
from jax._src.typing import Array, DTypeLike
from jax._src.util import safe_map
from jax._src.util import safe_zip
from jax._src.util import split_list
from jax._src.util import unzip2
from jax.experimental.mosaic.dialects import tpu
import jax.numpy as jnp
from jaxlib.mlir.ir import Module
import numpy as np

# TODO(sharadmv): enable type checking
# mypy: ignore-errors

NDIndexer = indexing.NDIndexer
TPUMemorySpace = tpu_core.TPUMemorySpace
MemorySpace = pallas_core.MemorySpace | TPUMemorySpace
VMEM = tpu_core.TPUMemorySpace.VMEM
SMEM = tpu_core.TPUMemorySpace.SMEM
# Booleans are stored as the following type in memrefs.
BOOL_MEMREF_TYPE = np.dtype('int32')

# The value interpreted as a dynamic dimension by MLIR.
MLIR_DYNAMIC = -9223372036854775808

partial = functools.partial
map, unsafe_map = safe_map, map  # pylint: disable=redefined-builtin
zip, unsafe_zip = safe_zip, zip  # pylint: disable=redefined-builtin


@dataclasses.dataclass
class MeshContext:
  mesh_shape: tuple[int, ...]
  axis_names: tuple[str, ...]
  mesh_strides: tuple[int, ...]


class LoweringDynamicShapeEnv:
  dim_expr_to_placeholder: dict[Any, ir.Value] = {}

  def to_placeholder(self, dim_expr: Any) -> ir.Value:
    if dim_expr not in self.dim_expr_to_placeholder:
      next_val = np.iinfo(np.int32).max - len(self.dim_expr_to_placeholder)
      self.dim_expr_to_placeholder[dim_expr] = next_val
    return self.dim_expr_to_placeholder[dim_expr]


@dataclasses.dataclass
class LoweringContext:
  ir_context: ir.Context
  grid_sizes: tuple[int, ...]  # Includes both user and vmap axes.
  grid_names: tuple[Hashable, ...] | None
  mapped_dims: tuple[int, ...]  # Indices of vmapped grid dimensions.
  user_grid_indices: Sequence[ir.Value] | None
  block_shapes: list[tuple[int | pallas_core.Mapped, ...]]
  name_stack: source_info_util.NameStack
  mesh_context: MeshContext | None
  replace = dataclasses.replace
  traceback_caches: mlir.TracebackCaches
  for_verification: bool
  forward_compatible: bool
  dynamic_shape_replacement_fn: Callable[
      [tuple[jax.DimSize, ...]], tuple[int, ...]
  ]

  @property
  def grid_rank(self):
    return len(self.grid_sizes)

  @contextlib.contextmanager
  def grid_name_context(self):
    # TODO(b/355036977): generalize this across other platforms
    if not self.grid_names:
      yield
      return
    grid_names = self.grid_names
    valid_grid_sizes = tuple(
        d for i, d in enumerate(self.grid_sizes) if i not in self.mapped_dims
    )
    grid_env = zip(grid_names, valid_grid_sizes)
    with jax_core.extend_axis_env_nd(grid_env):
      yield


@dataclasses.dataclass
class LoweringRuleContext:
  lowering_context: LoweringContext
  avals_in: Sequence[jax_core.AbstractValue]
  avals_out: Sequence[jax_core.AbstractValue]
  block_shapes: Sequence[tuple[int | pallas_core.Mapped, ...] | None]

  replace = dataclasses.replace

  @property
  def forward_compatible(self):
    return self.lowering_context.forward_compatible


def _memory_space_to_tpu_memory_space(memory_space: MemorySpace | None
                                     ) -> TPUMemorySpace:
  match memory_space:
    case None:
      # We pick VMEM as the default one when no memory space is
      # specified
      return TPUMemorySpace.VMEM
    case pallas_core.MemorySpace.ANY:
      # Map the general ANY memory space to TPU ANY memory space
      return TPUMemorySpace.ANY
    case pallas_core.MemorySpace.ERROR | pallas_core.MemorySpace.INDEX:
      return TPUMemorySpace.SMEM
    case TPUMemorySpace():
      # Leave the memory space unchanged
      return memory_space
    case _:
      raise ValueError("Invalid memory space: {memory_space}")


def _memory_space_to_mosaic_attribute(memory_space: MemorySpace | None
                                      ) -> ir.Attribute:
  tpu_memory_space = _memory_space_to_tpu_memory_space(memory_space)
  return ir.Attribute.parse(f"#tpu.memory_space<{tpu_memory_space}>")

def _dtype_to_ir_type(dtype: jnp.dtype,
                      is_kernel_boundary: bool = False) -> ir.Type:
  if jnp.issubdtype(dtype, tpu_core.semaphore_dtype):
    if jnp.issubdtype(dtype, tpu_core.dma_semaphore):
      return ir.Type.parse("!tpu.dma_semaphore")
    elif jnp.issubdtype(dtype, tpu_core.semaphore):
      return ir.Type.parse("!tpu.semaphore")
    elif jnp.issubdtype(dtype, tpu_core.barrier_semaphore):
      return ir.Type.parse("!tpu.semaphore")
    else:
      raise NotImplementedError
  if is_kernel_boundary and jnp.issubdtype(dtype, jnp.dtype('bool')):
    dtype = BOOL_MEMREF_TYPE
  # TODO(justinfu): Remove after mosaic supports unsigned types.
  # This conversion makes mosaic interpret all unsigned types as signed types.
  type =  mlir.dtype_to_ir_type(dtype)
  if isinstance(type, ir.IntegerType):
    return ir.IntegerType.get_signless(type.width)
  else:
    return type


def aval_to_ir_type(
    dynamic_shape_replacement_fn,
    aval,
    shape=None,
    memory_space: MemorySpace | None = None,
    is_kernel_boundary: bool = False,
):
  if isinstance(aval, tpu_core.AbstractSemaphore):
    if aval.sem_type is tpu_core.SemaphoreType.DMA:
      sem_type = ir.Type.parse("!tpu.dma_semaphore")
    elif aval.sem_type is tpu_core.SemaphoreType.REGULAR:
      sem_type = ir.Type.parse("!tpu.semaphore")
    elif aval.sem_type is tpu_core.SemaphoreType.BARRIER:
      sem_type = ir.Type.parse("!tpu.semaphore")
    else:
      raise ValueError(f"Cannot allocate {aval.sem_type}.")
    memspace = _memory_space_to_mosaic_attribute(TPUMemorySpace.SEMAPHORE)
    return ir.MemRefType.get((), sem_type, memory_space=memspace)
  if dtypes.issubdtype(aval.dtype, dtypes.prng_key):
    shape = aval.dtype._impl.key_shape
    if pl_random.is_pallas_impl(aval.dtype._impl):
      if memory_space is None:
        memory_space = TPUMemorySpace.SMEM
      if memory_space != TPUMemorySpace.SMEM:
        raise ValueError(
            f"PRNG keys must be stored in SMEM. Got {memory_space}"
        )
    memspace = _memory_space_to_mosaic_attribute(memory_space)
    return ir.MemRefType.get(shape, _dtype_to_ir_type(np.dtype(np.uint32)),
                             memory_space=memspace)
  if isinstance(aval, state.AbstractRef):
    if shape is None:
      shape = aval.shape
    memspace = _memory_space_to_mosaic_attribute(memory_space)
    shape = dynamic_shape_replacement_fn(shape)
    return ir.MemRefType.get(shape,
      _dtype_to_ir_type(aval.dtype, is_kernel_boundary=True),
      memory_space=memspace)
  if isinstance(aval, jax_core.ShapedArray):
    if shape is None:
      shape = aval.shape
    if not shape:
      return _dtype_to_ir_type(
          aval.dtype, is_kernel_boundary=is_kernel_boundary)
    shape = dynamic_shape_replacement_fn(shape)
    return ir.VectorType.get(
        shape,
        _dtype_to_ir_type(aval.dtype, is_kernel_boundary=is_kernel_boundary))
  raise NotImplementedError(aval)


def ir_constant(x, mlir_type=None):
  if not hasattr(x, "dtype"):
    if isinstance(x, int):
      x = np.array(x, np.int32)
    elif isinstance(x, float):
      x = np.array(x, np.float32)
  if not mlir_type:
    mlir_type = _dtype_to_ir_type(x.dtype)
  if isinstance(x, int) or np.issubdtype(x.dtype, np.integer):
    return arith.constant(mlir_type, ir.IntegerAttr.get(mlir_type, int(x)))
  elif isinstance(x, float) or x.dtype == np.float32:
    return arith.constant(mlir_type, ir.FloatAttr.get(mlir_type, float(x)))
  elif x.dtype == jnp.bfloat16:
    return arith.constant(mlir_type, ir.FloatAttr.get(mlir_type, float(x)))
  elif x.dtype == jnp.bool_:
    return arith.constant(mlir_type, ir.BoolAttr.get(bool(x)))
  raise NotImplementedError(x.dtype)


lowering_rules = {}
skip_mlir_conversions = set()

def _get_aval_physical_dtype_shape(aval):
  dtype_physical_shape = jax_core.physical_aval(aval).shape[
      len(aval.shape) :
  ]
  return dtype_physical_shape


def _get_arg_type(
    dynamic_shape_replacement_fn: Callable[
        [tuple[jax.DimSize, ...]], tuple[jax.DimSize, ...]
    ],
    aval,
    block_mapping: pallas_core.BlockMapping | None,
):
  memory_space = None
  if isinstance(aval, pallas_core.AbstractMemoryRef):
    memory_space = aval.memory_space
    # We assume unannotated memory refs are in VMEM
    if memory_space is None:
      memory_space = TPUMemorySpace.VMEM
  if isinstance(aval, tpu_core.AbstractSemaphore):
    return aval_to_ir_type(dynamic_shape_replacement_fn, aval), None
  # TODO(necula): clean this None block_mapping
  if block_mapping is None:
    return (
        aval_to_ir_type(
            dynamic_shape_replacement_fn, aval, memory_space=memory_space
        ),
        aval.shape,
    )
  shape = tuple(1 if b is pallas_core.mapped else b for b in block_mapping.block_shape)
  return (
      aval_to_ir_type(
          dynamic_shape_replacement_fn,
          aval,
          shape=shape,
          memory_space=memory_space,
      ),
      block_mapping.block_shape,
  )


@dataclasses.dataclass(init=False)
class MosaicGridMapping:
  grid: tuple[int, ...] | None
  grid_names: tuple[Hashable, ...] | None
  jaxpr: jax_core.Jaxpr
  block_mappings: tuple[pallas_core.BlockMapping | None, ...]
  mapped_dims: tuple[int, ...]
  scalar_prefetch_types: tuple[ir.Type, ...]
  operand_types: tuple[ir.Type, ...]
  scratch_types: tuple[ir.Type, ...]
  grid_types: tuple[ir.Type, ...]
  scalar_prefetch_block_shapes: tuple[tuple[int, ...], ...]
  operand_block_shapes: tuple[tuple[int, ...], ...]
  scratch_block_shapes: tuple[tuple[int, ...], ...]
  mesh_info: MeshInfo | None
  get_grid_indices: Callable | None

  def __init__(
      self,
      jaxpr: jax_core.Jaxpr,
      grid_mapping: pallas_core.GridMapping,
      dimension_semantics: tuple[str, ...] | None,
      mesh: mesh_lib.Mesh | None,
      dynamic_shape_replacement_fn: Callable[
          [tuple[jax.DimSize, ...]], tuple[int, ...]
      ],
  ):
    self.grid = grid_mapping.grid
    self.grid_names = grid_mapping.grid_names
    self.jaxpr = jaxpr
    self.block_mappings = grid_mapping.block_mappings
    self.mapped_dims = grid_mapping.vmapped_dims
    # TODO(mvoz): Generalize to not need this
    user_grid = tuple(
        g for i, g in enumerate(self.grid) if i not in self.mapped_dims
    )
    if dimension_semantics is None:
      dimension_semantics = ("arbitrary",) * len(user_grid)
    if len(user_grid) != len(dimension_semantics):
      raise ValueError(
          "Must have dimension semantics for each dimension of the grid."
      )
    assert len(self.mapped_dims) + len(dimension_semantics) == len(
        self.grid
    ), (
        f"Misconfigured grid: {self.mapped_dims=}, {dimension_semantics=},"
        f" {self.grid=}"
    )
    # dimension_semantics is user provided and won't take into account vmap
    # dimensions. Here we add in parallel dimensions for the vmaps.
    semantics_iter = iter(dimension_semantics)
    self._dimension_semantics = tuple(
        next(semantics_iter) if i not in self.mapped_dims else "parallel"
        for i in range(len(self.grid))
    )

    in_avals = [invar.aval for invar in self.jaxpr.invars]
    # jaxpr has signature [*scalar_prefetch, *consts, *in_ops, *out_ops, *scratch]
    scalar_prefetch_avals = in_avals[grid_mapping.slice_index_ops]
    operand_avals = in_avals[grid_mapping.slice_block_ops]
    scratch_avals = in_avals[grid_mapping.slice_scratch_ops]
    self.scalar_prefetch_types, _ = unzip2([
        _get_arg_type(dynamic_shape_replacement_fn, aval, None)
        for aval in scalar_prefetch_avals
    ])
    self.scalar_prefetch_block_shapes = tuple(
        aval.shape for aval in scalar_prefetch_avals)
    self.operand_types, self.operand_block_shapes = unzip2([
        _get_arg_type(dynamic_shape_replacement_fn, aval, block_mapping)
        for aval, block_mapping in zip(operand_avals, self.block_mappings)
    ])
    self.scratch_types, _ = unzip2([
        _get_arg_type(dynamic_shape_replacement_fn, aval, None)
        for aval in scratch_avals
    ])
    self.scratch_block_shapes = tuple(
        aval.shape if not isinstance(aval, tpu_core.AbstractSemaphore) else None
        for aval in scratch_avals
    )
    self.grid_types, _ = unzip2([
        _get_arg_type(
            dynamic_shape_replacement_fn,
            pallas_core.index_map_grid_aval,
            None,
        )
        for _ in range(len(self.grid))
    ])
    self._prepare_mesh_info(mesh)

    if grid_mapping.get_grid_indices is None:

      # Avoid using self.mapped_dims within the function, since doing so will
      # introduce a self->_get_grid_indices->self reference cycle that means
      # MosaicGridMapping instances can only ever be deleted by GC, rather than
      # by their reference counts going to 0.
      mapped_dims = self.mapped_dims
      def _get_grid_indices(indices, maybe_include_mapped_dims: bool):
        if maybe_include_mapped_dims:
          return indices
        return tuple(
            idx for i, idx in enumerate(indices) if i not in mapped_dims
        )

      self.get_grid_indices = _get_grid_indices
    else:
      self.get_grid_indices = grid_mapping.get_grid_indices

  def _prepare_mesh_info(self, mesh: mesh_lib.Mesh | None):
    if not self.has_communication:
      self.mesh_info = None
      return
    if mesh is None:
      raise ValueError(
          "Cannot use communication in pallas_call without shard_map."
      )
    axis_names = mesh.axis_names
    if self.grid_names is not None:
      if any(a in self.grid_names for a in axis_names):
        raise ValueError(
            "Cannot shadow axis mesh axis names with grid names. mesh axis"
            f" names: {mesh.axis_names}, grid names: {self.grid_names}"
        )
    # We need mesh <-> logical translation tables. Since the logical IDs are
    # just linearized versions of the mesh IDs, we create those tables.
    mesh_strides = pallas_utils.strides_from_shape(tuple(
        mesh.shape[a] for a in axis_names
    ))
    mesh_shape = tuple(mesh.shape.values())
    self.mesh_info = MeshInfo(mesh_shape, axis_names, mesh_strides)

  def maybe_compress_grid(self):
    # If we have many leading parallel dimensions, we should "compress" them
    # into one so we can load balance across cores as best as we can.
    # TODO(sharadmv): implement this optimization
    pass

  @functools.cached_property
  def has_communication(self) -> bool:
    nonlocal_axis_names = set()
    def _get_nonlocal_axis_names(jaxpr: jax_core.Jaxpr):
      return {
          e.name
          for e in jaxpr.effects
          if isinstance(e, jax_core.NamedAxisEffect)
          and (not self.grid_names or e.name not in self.grid_names)
      }
    nonlocal_axis_names.update(_get_nonlocal_axis_names(self.jaxpr))
    for bm in self.block_mappings:
      if bm is not None:
        nonlocal_axis_names.update(_get_nonlocal_axis_names(bm.index_map_jaxpr))
    return bool(nonlocal_axis_names)

  def get_extra_args(self) -> tuple[Any, ...]:
    return ()

  def get_dimension_semantics(self) -> ir.ArrayAttr:

    def _get_semantics(s: str | None) -> str:
      if s is None:
        return "#tpu.dimension_semantics<arbitrary>"
      return f"#tpu.dimension_semantics<{s}>"

    return ir.ArrayAttr.get(
        map(
            ir.Attribute.parse,
            map(_get_semantics, self._dimension_semantics),
        )
    )

@dataclasses.dataclass
class MeshInfo:
  mesh_shape: tuple[int, ...]
  axis_names: list[str]
  mesh_strides: tuple[int, ...]


def _check_block_mappings(
    block_mappings: tuple[pallas_core.BlockMapping, ...],
    lowering_context: mlir.LoweringRuleContext,
    name_and_src_info: pallas_core.NameAndSrcInfo,
) -> None:
  del lowering_context  # originally needed for forward compat
  for bm in block_mappings:
    rank = len(bm.block_shape)
    # TODO(necula): add tests for SMEM blocks with trivial windowing
    # We support scalars too
    if (bm.block_aval.memory_space == tpu_core.TPUMemorySpace.SMEM and
        bm.has_trivial_window()):
      continue
    if bm.block_aval.memory_space == tpu_core.TPUMemorySpace.SEMAPHORE:
      continue

    def err_details():
      return (f"Block spec for {bm.origin} in pallas_call {name_and_src_info} "
              "has block shape "
              f"{bm.block_shape}, array shape {bm.array_shape_dtype.shape}, "
              # TODO(necula): add index_map source location info
              f"and index_map {bm.index_map_jaxpr.jaxpr}, in "
              f"memory space {bm.block_aval.memory_space}."
              "\nSee details at https://jax.readthedocs.io/en/latest/pallas/grid_blockspec.html#pallas-blockspec")
    if rank < 1:
      raise ValueError(
          "The Pallas TPU lowering currently supports only blocks of "
          "rank >= 1. " + err_details())

    if (bm.block_aval.memory_space == tpu_core.TPUMemorySpace.ANY and
        not bm.has_trivial_window()):
      raise ValueError(
          "The Pallas TPU lowering currently supports in memory space ANY "
          "only blocks having the same block shape as the array shape "
          "and a trivial index_map (returning all 0s)." + err_details())

    unmapped_bs = [
        1 if bs is pallas_core.mapped else bs for bs in bm.block_shape]
    bs0, as0 = unmapped_bs[-1], bm.array_shape_dtype.shape[-1]
    if rank >= 2:
      bs1, as1 = unmapped_bs[-2], bm.array_shape_dtype.shape[-2]
    else:
      bs1, as1 = 1, 1

    if rank >= 2:
      evenly_divisible = (
          (bs0 == as0 or bs0 % 128 == 0) and
          (bs1 == as1 or bs1 % 8 == 0)
      )
      if not evenly_divisible:
        extra_msg = ""
        if pallas_core.dynamic_shapes_export_enabled():
          extra_msg = (
              " In dynamic shape export - your kernel symbolic args must be"
              " annotated with constraints where the computation *after*"
              " applying any grid mapping is divisible by 8 and 128"
              " respectively. Ex: (mod(floordiv(m_dim, grid_size), 8) == 0))"
          )
        raise ValueError(
            "The Pallas TPU lowering currently requires that the last two "
            "dimensions of your block shape are divisible by 8 and 128 "
            "respectively, or be equal to the respective dimensions of the "
            "overall array. "
            + extra_msg
            + err_details()
        )
    else:
      assert rank == 1
      # bools get a bitwidth of 32 due to how mosaic handles them
      if bm.array_shape_dtype.dtype == jnp.bool_:
        bitwidth = 32
      else:
        bitwidth = lax_internal._bit_width(bm.array_shape_dtype.dtype)
      packing = 32 // bitwidth
      tiling_size = 128 * packing
      evenly_divisible = (bs0 == as0 or bs0 % tiling_size == 0)
      if not evenly_divisible:
        raise ValueError(
            "The Pallas TPU lowering currently requires that rank 1 block"
            " shapes, either 1) the first (and only) dimension of the block"
            " shape is equal to the first (and only) dimension of the array"
            " shape, or 2) the first (and only) dimension of the block shape"
            f" is a multiple of the tiling size ({tiling_size} = 128 * (32 //"
            f" {lax_internal._bit_width(bm.array_shape_dtype.dtype)})) of the"
            " array shape. "
            + err_details()
        )


def lower_jaxpr_to_module(
    lowering_context: mlir.LoweringRuleContext,
    ctx: ir.Context,
    grid_mapping: pallas_core.GridMapping,
    jaxpr: jax_core.Jaxpr,
    *,
    dimension_semantics: tuple[str | None, ...] | None,
    name_and_src_info: pallas_core.NameAndSrcInfo,
    mesh: mesh_lib.Mesh | None = None,
    for_verification: bool = False,
    dynamic_shape_replacement_enabled: bool = False,
) -> tuple[Module, tuple[Any, ...]]:
  if dynamic_shape_replacement_enabled:
    _mosaic_lowering_dynamic_shape_env = LoweringDynamicShapeEnv()

    def dynamic_shape_replacement_fn(
        shape: jax_core.Shape,
    ) -> tuple[int, ...]:
      return tuple(
          _mosaic_lowering_dynamic_shape_env.to_placeholder(dim_expr)
          if jax_core.is_dim(dim_expr)
          else dim_expr
          for dim_expr in shape
      )

  else:
    dynamic_shape_replacement_fn = lambda x: x

  # Verify that we have legal block mappings to catch errors early.
  _check_block_mappings(grid_mapping.block_mappings, lowering_context,
                        name_and_src_info)

  mosaic_grid_mapping = MosaicGridMapping(
      jaxpr,
      grid_mapping,
      dimension_semantics,
      mesh,
      dynamic_shape_replacement_fn,
  )
  mosaic_grid_mapping.maybe_compress_grid()
  m = ir.Module.create()
  attrs = m.operation.attributes
  module_name = name_and_src_info.name
  attrs["sym_name"] = ir.StringAttr.get(module_name)
  sym_tab = ir.SymbolTable(m.operation)

  func_op = lower_jaxpr_to_func(
      ctx,
      jaxpr,
      mosaic_grid_mapping=mosaic_grid_mapping,
      name="main",
      for_verification=for_verification,
      forward_compatible=lowering_context.is_forward_compat(),
      dynamic_shape_replacement_fn=dynamic_shape_replacement_fn,
  )
  m.body.append(func_op)
  sym_tab.insert(func_op)
  window_params = []
  grid = mosaic_grid_mapping.grid
  if grid:
    for i, bm in enumerate(grid_mapping.block_mappings):
      func_name = f"transform_{i}"
      # ANY and SEMAPHORE operands don't support windowing and require empty window_params.
      tpu_memory_space = _memory_space_to_tpu_memory_space(
          bm.block_aval.memory_space)
      if (
          tpu_memory_space == tpu_core.TPUMemorySpace.ANY
          or tpu_memory_space == tpu_core.TPUMemorySpace.SEMAPHORE
      ):
        # We checked above that the block does not require windowing.
        window_params.append(ir.DictAttr.get())
        continue

      mlir_func = lower_jaxpr_to_transform_func(
          ctx,
          bm.index_map_jaxpr.jaxpr,
          bm.block_aval,
          name=func_name,
          mosaic_grid_mapping=mosaic_grid_mapping,
          for_verification=for_verification,
          forward_compatible=lowering_context.is_forward_compat(),
          dynamic_shape_replacement_fn=dynamic_shape_replacement_fn,
      )
      assert mlir_func.verify(), mlir_func
      block_shape = [
          1 if b is pallas_core.mapped else b for b in bm.block_shape
      ]
      # If we have an extended dtype, we need to add the block shape for the
      # remaining physical dtype.
      block_shape += list(_get_aval_physical_dtype_shape(bm.block_aval.inner_aval))
      block_shape = dynamic_shape_replacement_fn(block_shape)
      window_shape = ir.DenseI64ArrayAttr.get(block_shape)
      block_params = dict(
          window_bounds=window_shape,
          transform_indices=ir.FlatSymbolRefAttr.get(func_name),
      )
      if isinstance(bm.indexing_mode, pallas_core.Unblocked):
        if bm.indexing_mode.padding is None:
          pad_low = pad_high = [0] * len(bm.block_shape)
        else:
          pad_low, pad_high = map(list, zip(*bm.indexing_mode.padding))
        block_params["window_kind"] = ir.Attribute.parse(
            f"#tpu.element_window<{pad_low},{pad_high}>"
        )
      window_params.append(ir.DictAttr.get(block_params))
      m.body.append(mlir_func)
      sym_tab.insert(mlir_func)
    func_op.attributes["window_params"] = ir.ArrayAttr.get(window_params)

    static_grid = [
        MLIR_DYNAMIC if b is pallas_core.dynamic_grid_dim else b for b in grid
    ]
    static_grid = dynamic_shape_replacement_fn(static_grid)
    func_op.attributes["iteration_bounds"] = ir.DenseI64ArrayAttr.get(static_grid)

  func_op.attributes["scalar_prefetch"] = ir.IntegerAttr.get(
      ir.IntegerType.get_signless(64), len(mosaic_grid_mapping.scalar_prefetch_types))
  func_op.attributes["scratch_operands"] = ir.IntegerAttr.get(
      ir.IntegerType.get_signless(64), len(mosaic_grid_mapping.scratch_types))
  func_op.attributes["dimension_semantics"] = (
      mosaic_grid_mapping.get_dimension_semantics()
  )
  return m, mosaic_grid_mapping.get_extra_args()


def lower_jaxpr_to_transform_func(
    ctx: ir.Context,
    jaxpr: jax_core.Jaxpr,
    aval: jax_core.AbstractValue,
    *,
    name: str,
    mosaic_grid_mapping: MosaicGridMapping,
    for_verification: bool,
     forward_compatible: bool,
    dynamic_shape_replacement_fn: (
        Callable[[tuple[jax.DimSize, ...]], tuple[int, ...]] | None
    ) = None,
) -> func.FuncOp:
  num_grid = len(mosaic_grid_mapping.grid_types)
  arg_types = [
      *mosaic_grid_mapping.grid_types,
      *mosaic_grid_mapping.scalar_prefetch_types,
  ]
  def body_func(*args):
    grid_indices, scalar_prefetch = split_list(args, [num_grid])
    jaxpr_indices = mosaic_grid_mapping.get_grid_indices(
        grid_indices, maybe_include_mapped_dims=True
    )
    arg_block_shapes = [
        *[()] * len(jaxpr_indices),
        *mosaic_grid_mapping.scalar_prefetch_block_shapes,
    ]

    mesh_info = mosaic_grid_mapping.mesh_info
    if mesh_info is not None:
      mesh_context = MeshContext(
          mesh_info.mesh_shape, mesh_info.axis_names, mesh_info.mesh_strides
      )
    else:
      mesh_context = None
    lowering_context = LoweringContext(
        ctx,
        mosaic_grid_mapping.grid,
        mosaic_grid_mapping.grid_names,
        mosaic_grid_mapping.mapped_dims,
        None,
        arg_block_shapes,
        source_info_util.NameStack(),
        mesh_context=mesh_context,
        traceback_caches=mlir.TracebackCaches(),
        for_verification=for_verification,
        forward_compatible=forward_compatible,
        dynamic_shape_replacement_fn=dynamic_shape_replacement_fn,
    )
    out = jaxpr_subcomp(lowering_context, jaxpr, *jaxpr_indices,
                        *scalar_prefetch)
    assert isinstance(aval, state.AbstractRef), aval
    # If we have an extended dtype, we need to add 0s for the block indices
    # for the remaining physical dtype.
    out += [
        ir_constant(0, mlir_type=_dtype_to_ir_type(jnp.dtype("int32")))
    ] * len(_get_aval_physical_dtype_shape(aval.inner_aval))
    return out

  body_func.__name__ = name
  body = func.FuncOp.from_py_func(*arg_types, name=name)(body_func)
  try:
    body.func_op.verify()
  except ir.MLIRError as e:
    raise error_handling.mlir_error_to_verification_error(e) from e
  return body.func_op


def lower_jaxpr_to_func(
    ctx: ir.Context,
    jaxpr: jax_core.Jaxpr,
    *,
    mosaic_grid_mapping: MosaicGridMapping,
    name: str,
    for_verification: bool,
    forward_compatible: bool,
    dynamic_shape_replacement_fn: (
        Callable[[tuple[jax.DimSize, ...]], tuple[int, ...]] | None
    ) = None,
) -> func.FuncOp:
  num_grid = len(mosaic_grid_mapping.grid_types)
  num_scalar_prefetch = len(mosaic_grid_mapping.scalar_prefetch_types)
  arg_types = [
      *mosaic_grid_mapping.grid_types,
      *mosaic_grid_mapping.scalar_prefetch_types,
      *mosaic_grid_mapping.operand_types,
      *mosaic_grid_mapping.scratch_types,
  ]
  arg_block_shapes = [
      *mosaic_grid_mapping.scalar_prefetch_block_shapes,
      *mosaic_grid_mapping.operand_block_shapes,
      *mosaic_grid_mapping.scratch_block_shapes,
  ]
  def body_func(*args):
    grid_indices, scalar_prefetch, operands_and_scratch = split_list(
        args, [num_grid, num_scalar_prefetch])
    jaxpr_indices = mosaic_grid_mapping.get_grid_indices(
        grid_indices, maybe_include_mapped_dims=False
    )
    mesh_info = mosaic_grid_mapping.mesh_info
    if mesh_info is not None:
      mesh_context = MeshContext(
          mesh_info.mesh_shape, mesh_info.axis_names, mesh_info.mesh_strides
      )
    else:
      mesh_context = None
    lowering_context = LoweringContext(
        ctx,
        mosaic_grid_mapping.grid,
        mosaic_grid_mapping.grid_names,
        mosaic_grid_mapping.mapped_dims,
        jaxpr_indices,
        arg_block_shapes,
        source_info_util.NameStack(),
        mesh_context=mesh_context,
        traceback_caches=mlir.TracebackCaches(),
        for_verification=for_verification,
        forward_compatible=forward_compatible,
        dynamic_shape_replacement_fn=dynamic_shape_replacement_fn,
    )
    return jaxpr_subcomp(
        lowering_context, jaxpr, *scalar_prefetch, *operands_and_scratch
    )
  body_func.__name__ = name
  body = func.FuncOp.from_py_func(*arg_types, name=name)(body_func)
  try:
    body.func_op.verify()
  except ir.MLIRError as e:
    raise error_handling.mlir_error_to_verification_error(e) from e
  return body.func_op


def lower_fun(fun: Callable, *, multiple_results: bool) -> Callable:
  def f_lowered(ctx: LoweringRuleContext, *args, **params):
    f = fun if multiple_results else lambda *args, **kw: (fun(*args, **kw),)
    wrapped_fun = lu.wrap_init(f, params)
    jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(wrapped_fun, ctx.avals_in)
    if consts:
      raise NotImplementedError
    jaxpr = pe.convert_constvars_jaxpr(jaxpr)
    lowering_context = ctx.lowering_context.replace(
        block_shapes=ctx.block_shapes)
    out = jaxpr_subcomp(lowering_context, jaxpr, *consts, *args)
    if not multiple_results:
      return out[0]
    return out

  return f_lowered


class LoweringException(Exception):
  pass


def _compute_name_stack_updates(
    old_name_stack: list[str],
    new_name_stack: list[str]
) -> tuple[list[str], list[str]]:
  """Computes the popped/pushed items to the name stack after an update.

  Args:
    old_name_stack: The name stack prior to the update.
    new_name_stack: The name stack after the update.

  Returns:
    popped: A list of names popped from the name stack as part of the update.
    pushed: A list of names pushed to the name stack as part of the update.
  """
  common_prefix_idx = 0
  for i, (old, new) in enumerate(unsafe_zip(old_name_stack, new_name_stack)):
    if old == new:
      common_prefix_idx = i+1
    else:
      break
  return old_name_stack[common_prefix_idx:], new_name_stack[common_prefix_idx:]


def jaxpr_subcomp(
    ctx: LoweringContext, jaxpr: jax_core.Jaxpr, *args: ir.Value
) -> Sequence[ir.Value]:
  assert not jaxpr.constvars
  env = {}
  block_shape_env = {}

  def read_block_shape(atom: jax_core.Atom):
    if isinstance(atom, jax_core.Literal):
      return None
    return block_shape_env.get(atom, None)

  def read_env(atom: jax_core.Atom):
    return atom.val if isinstance(atom, jax_core.Literal) else env[atom]

  def write_env(var: jax_core.Var, val):
    is_valid_type = isinstance(val, (ir.Value, KeyScalarBundle))
    assert is_valid_type, type(val)
    env[var] = val

  for invar, bs in zip(jaxpr.invars, ctx.block_shapes):
    block_shape_env[invar] = bs
  map(write_env, jaxpr.invars, args)

  initial_name_stack = [scope.name for scope in ctx.name_stack.stack]
  current_name_stack: list[str] = []
  # TODO(justinfu): Handle transform scopes.
  current_name_stack.extend(initial_name_stack)
  for eqn in jaxpr.eqns:
    invals = map(read_env, eqn.invars)
    source_info = eqn.source_info.replace(
        name_stack=ctx.name_stack + eqn.source_info.name_stack
    )
    loc = mlir._source_info_to_location(ctx, eqn.primitive, source_info)
    with source_info_util.user_context(eqn.source_info.traceback), loc:
      if eqn.primitive in lowering_rules:
        if eqn.primitive not in skip_mlir_conversions:
          invals = [_ensure_mlir_value(x, v.aval)
                    for x, v in zip(invals, eqn.invars)]
        block_shapes = map(read_block_shape, eqn.invars)
        rule_context = LoweringRuleContext(
            ctx,
            [v.aval for v in eqn.invars],
            [v.aval for v in eqn.outvars],
            block_shapes,
        )

        # Insert trace_start and trace_stop ops on named_scope boundaries.
        name_stack = [scope.name for scope in source_info.name_stack.stack]
        popped, pushed = _compute_name_stack_updates(
            current_name_stack, name_stack)
        current_name_stack = name_stack
        for _ in popped:
          tpu.TraceStopOp()
        for name in pushed:
          tpu.TraceStartOp(message=name, level=10)

        try:
          ans = lowering_rules[eqn.primitive](
              rule_context, *invals, **eqn.params
          )
        except LoweringException:
          raise  # We only add the extra info to the innermost exception.
        except Exception as e:
          if not pallas_call._verbose_errors_enabled():
            raise
          msg = (f"{type(e).__name__}: {e}\n" +
                "Additional diagnostics: \n" +
                f"Failing jaxpr equation: {eqn}\n")
          new_error = LoweringException(msg)
          # We insert the traceback here so that the user code shows
          # up in the traceback for the post-transform error.
          if source_info.traceback is not None:
            tb = source_info.traceback.as_python_traceback()
            new_error.__traceback__ = traceback_util.filter_traceback(tb)
          raise new_error from e
      else:
        raise NotImplementedError(
            "Unimplemented primitive in Pallas TPU lowering: "
            f"{eqn.primitive.name}. "
            "Please file an issue on https://github.com/jax-ml/jax/issues.")
      if eqn.primitive.multiple_results:
        map(write_env, eqn.outvars, ans)
      else:
        write_env(eqn.outvars[0], ans)

  # Drain the name stack at the end of a jaxpr and insert trace_stop ops.
  popped, pushed = _compute_name_stack_updates(
      current_name_stack, initial_name_stack)
  for _ in popped:
    tpu.TraceStopOp()
  assert len(pushed) == 0

  outvals = map(read_env, jaxpr.outvars)
  outvals = [
      ir_constant(x) if isinstance(var, jax_core.Literal) else x
      for x, var in zip(outvals, jaxpr.outvars)
  ]
  return outvals


def _ensure_mlir_value(val, aval):
  if isinstance(val, ir.Value):
    return val
  if isinstance(val, KeyScalarBundle):
    return val
  elif isinstance(val, (np.generic, np.ndarray, int, float)):
    return ir_constant(val, _dtype_to_ir_type(aval.dtype))
  else:
    raise RuntimeError(
        f"Unsupported argument to a JAX primitive of type: {type(val)}"
    )


def _get_lowering_rule(
    ctx: LoweringRuleContext, ref, *idx, tree,
):
  indexers = tree_util.tree_unflatten(tree, idx)
  indexers_avals = tree_util.tree_unflatten(tree, ctx.avals_in[1:])
  # Call _load_lowering_rule (since it's more general)
  ref_aval, *_ = ctx.avals_in
  args_flat, args_tree = tree_util.tree_flatten((ref, indexers, None, None))
  avals_flat = tree_util.tree_leaves((ref_aval, indexers_avals, None, None))
  ctx = ctx.replace(
      avals_in=avals_flat,
      block_shapes=[ctx.block_shapes[0], *[None] * (len(avals_flat) - 1)],
  )
  return _load_lowering_rule(ctx, *args_flat, args_tree=args_tree)


lowering_rules[state_primitives.get_p] = _get_lowering_rule
skip_mlir_conversions.add(state_primitives.get_p)


def _swap_lowering_rule(
    ctx: LoweringRuleContext,
    ref,
    val,
    *idx,
    tree
):
  indexers = tree_util.tree_unflatten(tree, idx)
  indexers_avals = tree_util.tree_unflatten(tree, ctx.avals_in[2:])
  # Call _masked_swap_lowering_rule (since it's more general)
  ref_aval, val_aval, *_ = ctx.avals_in
  args_flat, args_tree = tree_util.tree_flatten((ref, indexers, val, None))
  avals_flat = tree_util.tree_leaves(
      (ref_aval, indexers_avals, val_aval, None)
  )
  ctx = ctx.replace(
      avals_in=avals_flat,
      block_shapes=[ctx.block_shapes[0], *[None] * (len(avals_flat) - 1)],
  )
  return _masked_swap_lowering_rule(ctx, *args_flat, args_tree=args_tree)

lowering_rules[state_primitives.swap_p] = _swap_lowering_rule
skip_mlir_conversions.add(state_primitives.swap_p)


def _make_index(s):
  if isinstance(s, (int, np.ndarray)):
    return ir_constant(s, ir.IndexType.get())
  if s.type == ir.IndexType.get():
    return s
  return arith.index_cast(ir.IndexType.get(), s)


def _maybe_cast_to_index(cast_to_index, x):
  if cast_to_index:
    return _make_index(x)
  return _ensure_mlir_value(x, aval=pallas_core.index_map_grid_aval)


def _index_to_start_size_stride(
    idx: tuple[indexing.Slice | int | ir.Value, ...], cast_to_index: bool
) -> tuple[ir.Value, int | ir.Value, int, bool]:
  assert not isinstance(idx, slice)
  if isinstance(idx, indexing.Slice):
    start = _maybe_cast_to_index(cast_to_index, idx.start)
    size = idx.size
    stride = idx.stride
    squeeze = False
  elif isinstance(idx, int):
    start = _maybe_cast_to_index(cast_to_index, idx)
    size = 1
    stride = 1
    squeeze = True
  else:
    if np.shape(idx):
      raise ValueError(f"Can only use ()-shaped and slice indexing: {idx}")
    start = _maybe_cast_to_index(cast_to_index, idx)
    size = 1
    stride = 1
    squeeze = True
  return start, size, stride, squeeze


def _indexer_to_start_size_stride(
    indexer: NDIndexer,
    ref_block_shape: tuple[int | pallas_core.Mapped, ...],
    *,
    cast_to_index: bool,
) -> tuple[
    tuple[ir.Value, ...],
    tuple[int | ir.Value, ...],
    tuple[int, ...],
    tuple[bool, ...],
    tuple[int | pallas_core.Mapped, ...],
]:
  indices_iter = iter(indexer.indices)
  starts, sizes, strides, squeeze_dims = [], [], [], []
  for s in ref_block_shape:
    start, size, stride, squeeze_dim = (
        (
            _maybe_cast_to_index(cast_to_index, 0),
            1,
            1,
            True,
        )
        if s is pallas_core.mapped
        else _index_to_start_size_stride(next(indices_iter), cast_to_index)
    )
    starts.append(start)
    sizes.append(size)
    strides.append(stride)
    squeeze_dims.append(squeeze_dim)
  next_index = next(indices_iter, None)
  assert next_index is None, (indexer.indices, ref_block_shape)
  new_ref_block_shape = tuple(s for s, squeeze in zip(sizes, squeeze_dims)
                              if not squeeze)
  return (
      tuple(starts),
      tuple(sizes),
      tuple(strides),
      tuple(squeeze_dims),
      new_ref_block_shape,
  )


def _slice_memref(
    ref: ir.Value,
    indexer: NDIndexer,
    ref_dtype: DTypeLike,
    ref_block_shape: tuple[int | pallas_core.Mapped, ...],
) -> tuple[ir.Value, tuple[int | pallas_core.Mapped, ...]]:
  assert ref_block_shape is not None
  target_shape = indexer.get_indexer_shape()
  starts, sizes, strides, squeeze_dims, ref_block_shape = (
      _indexer_to_start_size_stride(
          indexer,
          ref_block_shape,
          cast_to_index=False,
      )
  )
  if not all((s is None or s == 1) for s in strides):
    raise NotImplementedError("Strided slices of references are unsupported.")
  dynamic_sizes = tuple(s for s in sizes if isinstance(s, ir.Value))
  ir_dynamic_size = ir.ShapedType.get_dynamic_size()
  static_sizes = tuple(s if not isinstance(s, ir.Value)
                       else ir_dynamic_size for s in sizes)
  target_ref_ty = ir.MemRefType.get(
      static_sizes,
      _dtype_to_ir_type(ref_dtype),
      memory_space=ref.type.memory_space,
  )
  out = tpu.memref_slice(target_ref_ty, ref, starts, dynamic_sizes)
  if any(squeeze_dims):
    # We need to squeeze out some dimensions
    static_sizes = tuple(s if not isinstance(s, ir.Value)
                         else ir_dynamic_size for s in target_shape)
    squeezed_ref_ty = ir.MemRefType.get(
        static_sizes,
        _dtype_to_ir_type(ref_dtype),
        memory_space=ref.type.memory_space,
    )
    out = tpu.memref_squeeze(squeezed_ref_ty, out)
  return out, ref_block_shape


def _bitcast_memref(
    ref: ir.Value,
    bitcaster: RefBitcaster,
    ref_dtype: DTypeLike,
    ref_block_shape: tuple[int | pallas_core.Mapped, ...],
) -> tuple[ir.Value, DTypeLike, tuple[int | pallas_core.Mapped, ...]]:
  src_bitwidth = dtype_bitwidth(ref_dtype)
  dst_bitwidth = dtype_bitwidth(bitcaster.dtype)
  if src_bitwidth != dst_bitwidth:
    if len(ref_block_shape) < 2:
      raise NotImplementedError(
          "Bitcast 1D ref with bitwidth change is not supported."
      )
    if ref_block_shape[-2] is pallas_core.mapped:
      raise NotImplementedError(
          "Bitcast a ref whose 2nd minormost dimension is squeezed when"
          " bitwidth changes."
      )
  new_ref_dtype = bitcaster.dtype
  target_ref_ty = ir.MemRefType.get(
      bitcaster.shape,
      _dtype_to_ir_type(new_ref_dtype),
      memory_space=ref.type.memory_space,
  )
  new_ref_block_shape = list(ref_block_shape)
  if (
      len(new_ref_block_shape) >= 2
      and new_ref_block_shape[-2] is not pallas_core.mapped
  ):
    new_ref_block_shape[-2] = (
        new_ref_block_shape[-2] * src_bitwidth // dst_bitwidth
    )
  return (
      tpu.memref_bitcast(target_ref_ty, ref),
      new_ref_dtype,
      tuple(new_ref_block_shape),
  )


def _reshape_memref(
    ref: ir.Value,
    reshaper: RefReshaper,
    ref_dtype: DTypeLike,
    ref_block_shape: tuple[int | pallas_core.Mapped, ...],
) -> tuple[ir.Value, DTypeLike, tuple[int | pallas_core.Mapped, ...]]:
  if ref_dtype != reshaper.dtype:
    raise ValueError(
        f"Reshape a ref with dtype change: {reshaper.dtype} vs {ref_dtype}"
    )
  if len(ref_block_shape) < 2:
    raise NotImplementedError("Reshape 1D ref is not supported.")
  if (
      ref_block_shape[-2] is pallas_core.mapped
      or ref_block_shape[-1] is pallas_core.mapped
  ):
    raise NotImplementedError(
        "Reshape a ref with squeezed dimension on last two dimensions."
    )
  if np.prod(ref_block_shape) != np.prod(reshaper.shape):
    raise ValueError(
        f"Reshape a ref with different number of elements: {ref_block_shape} "
        f"vs {reshaper.shape}"
    )
  target_ref_ty = ir.MemRefType.get(
      reshaper.shape,
      _dtype_to_ir_type(reshaper.dtype),
      memory_space=ref.type.memory_space,
  )
  return (
      tpu.memref_reshape(target_ref_ty, ref),
      reshaper.shape,
  )


def _transform_ref(ref, ref_dtype, ref_block_shape, transforms):
  for transform in transforms:
    match transform:
      case NDIndexer():
        ref, ref_block_shape = _slice_memref(
            ref, transform, ref_dtype, ref_block_shape
        )
      case RefBitcaster():
        ref, ref_dtype, ref_block_shape = _bitcast_memref(
            ref, transform, ref_dtype, ref_block_shape
        )
      case RefReshaper():
        ref, ref_block_shape = _reshape_memref(
            ref, transform, ref_dtype, ref_block_shape
        )
      case _:
        raise NotImplementedError(f"Unsupported transform: {transform}")
  return ref, ref_block_shape


@dataclasses.dataclass(frozen=True)
class KeyScalarBundle:
  """A container class for PRNG key data.

  We pass around keys as a KeyScalarBundle in the lowering pass rather than
  as a vector, since we want the key data to live in scalar registers rather
  than vector registers. This special dataclass exists so we can return
  multiple scalar values from load_op, because the load_op primitive does
  not allow multiple results.

  Attributes:
    scalars: A list of OpResults representing scalar key data during the
      lowering pass.
  """
  key_shape: tuple[int, ...]
  scalars: list[ir.OpResult]

def _load_lowering_rule(ctx: LoweringRuleContext, *args_flat, args_tree, **_):
  ref, transforms, mask, _ = args_tree.unflatten(args_flat)
  ref_aval, transforms_avals, _, _ = args_tree.unflatten(ctx.avals_in)
  (*prev_transforms, idx) = transforms
  # Select last aval, which is the one that will be used for the load.
  (*_, idx_aval) = transforms_avals

  if mask is not None:
    raise NotImplementedError

  ref_block_shape, *_ = ctx.block_shapes
  ref, ref_block_shape = _transform_ref(
      ref, ref_aval.dtype, ref_block_shape, prev_transforms
  )
  ref_type = ir.MemRefType(ref.type)
  is_smem_load = str(ref_type.memory_space) == "#tpu.memory_space<smem>"
  (aval_out,) = ctx.avals_out
  if isinstance(aval_out.dtype, prng.KeyTy) and pl_random.is_pallas_impl(
      aval_out.dtype._impl
  ):
    if not is_smem_load:
      raise ValueError("PRNG keys must be loaded from SMEM. Did you set "
                       "the memory space to TPUMemorySpace.SMEM in the "
                       "BlockSpec for the PRNG key input?")
    return _prng_key_load_lowering_rule(ctx, *args_flat, args_tree=args_tree)
  if not is_smem_load and not ref_block_shape:
    raise NotImplementedError(
        "Indexing into a ()-shaped Ref not yet supported on TPU.")
  if any(
      (not isinstance(a, primitives.Slice) and a.shape)
      for a in idx_aval.indices
  ):
    raise ValueError("Cannot do int indexing on TPU")
  starts, sizes, strides, _, _ = _indexer_to_start_size_stride(
      idx,
      ref_block_shape,
      cast_to_index=True,
  )
  need_stride = not all((s is None or s == 1) for s in strides)
  if is_smem_load:
    if ctx.avals_out[0].shape:
      raise ValueError("Can only load scalars from SMEM")
    return _maybe_cast_load_to_bool(ctx, aval_out, memref.load(ref, starts))
  elif str(ref_type.memory_space) != "#tpu.memory_space<vmem>":
    extra = ""
    if str(ref_type.memory_space) == "#tpu.memory_space<any>":
      extra = " ANY memory space can only be accessed using async_copy."
    raise ValueError(
        "Loads are only allowed on VMEM and SMEM references." + extra
    )
  load_aval = jax_core.ShapedArray(sizes, dtype=aval_out.dtype)
  if need_stride:
    load_val = tpu.strided_load(
        aval_to_ir_type(
            ctx.lowering_context.dynamic_shape_replacement_fn,
            load_aval,
            is_kernel_boundary=True,
        ),
        ref,
        starts,
        strides,
    )
  else:
    load_val = vector.load(
        aval_to_ir_type(
            ctx.lowering_context.dynamic_shape_replacement_fn,
            load_aval,
            is_kernel_boundary=True,
        ),
        ref,
        starts,
    )
  if load_aval != aval_out:
    vec_type = ir.VectorType.get(aval_out.shape,
                                _dtype_to_ir_type(aval_out.dtype,
                                                  is_kernel_boundary=True))
    load_val = vector.shape_cast(vec_type, load_val)
  return _maybe_cast_load_to_bool(ctx, aval_out, load_val)

def _prng_key_load_lowering_rule(ctx: LoweringRuleContext, *args_flat, args_tree) -> KeyScalarBundle:
  """Lowering rule for loading PRNG keys from SMEM.

  PRNG key loads are currently lowered as a list of scalar loads from SMEM,
  rather than a single vector load.
  We store these scalars in a bundle type called KeyScalarBundle, which has
  special case handling for functions that consume the key such as set_seed.
  """
  ref, _, _, _ = args_tree.unflatten(args_flat)
  (aval_out,) = ctx.avals_out
  assert isinstance(aval_out.dtype, prng.KeyTy)
  ref_block_shape = aval_out.dtype._impl.key_shape

  if len(ref_block_shape) != 2:
    raise NotImplementedError("Seed key_data must be 2D.")
  if tuple(ref_block_shape) != (1, 1):
    raise NotImplementedError(
      f"Seed key_data of shape != (1, 1) not supported. Got: {ref_block_shape}")

  load_ops = []
  for i in range(ref_block_shape[0]):
    idx = NDIndexer(indices=(0, i), shape=ref_block_shape,
                    int_indexer_shape=tuple())
    starts, _, _, _, _ = _indexer_to_start_size_stride(
        idx,
        ref_block_shape,
        cast_to_index=True,
    )
    load_ops.append(memref.load(ref, starts))
  return KeyScalarBundle(scalars=load_ops, key_shape=tuple(ref_block_shape))


lowering_rules[primitives.load_p] = _load_lowering_rule
skip_mlir_conversions.add(primitives.load_p)


def _maybe_cast_load_to_bool(
    ctx, out_aval, val: ir.Value
) -> tuple[ir.Value, jnp.dtype]:
  """Casts a memref load value to bool if the requested value is a bool.

  Mosaic does not support boolean-type memrefs, since booleans
  typically live in mask registers. We instead load booleans as integers from
  memrefs and move them to mask registers on load using this function.

  Args:
    out_aval: The output aval of the load.
    val: The input value.

  Returns:
    The loaded value, and the JAX dtype of the input value.
  """
  if out_aval.dtype != jnp.bool_:
    return val
  load_scalar_type = _dtype_to_ir_type(BOOL_MEMREF_TYPE)
  pred = _cmpsi_lowering_types[lax.ne_p]
  predicate = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), pred)
  const_zero = ir.IntegerAttr.get(load_scalar_type, 0)
  if out_aval.shape:  # Vector case.
    load_vector_type = aval_to_ir_type(
        ctx.lowering_context.dynamic_shape_replacement_fn,
        out_aval,
        is_kernel_boundary=True,
    )
    vector_zeros = arith.ConstantOp(
        load_vector_type,
        ir.DenseElementsAttr.get_splat(load_vector_type, const_zero)
    )
    return arith.cmpi(predicate, val, vector_zeros)
  else:  # Scalar case.
    const_zero = arith.ConstantOp(load_scalar_type, const_zero)
    return arith.cmpi(predicate, val, const_zero)


def _maybe_cast_store_to_memref_type(
    ctx: LoweringRuleContext, expected_aval, val: ir.Value
) -> ir.Value:
  """Casts a boolean value back to an integer for storing in a memref."""
  if expected_aval.dtype != jnp.bool_:
    return val
  int_out_type = aval_to_ir_type(
      ctx.lowering_context.dynamic_shape_replacement_fn,
      expected_aval,
      is_kernel_boundary=True,
  )
  return arith.extui(int_out_type, val)


def _masked_swap_lowering_rule(
    ctx: LoweringRuleContext, *args_flat, args_tree, **_
):
  ref, transforms, val, mask = args_tree.unflatten(args_flat)
  ref_aval, transforms_avals, val_aval, mask_aval = args_tree.unflatten(
      ctx.avals_in
  )
  (*prev_transforms, idx) = transforms
  (*_, idx_aval) = transforms_avals

  if mask is not None:
    if  val_aval.dtype.itemsize != 4:
      raise NotImplementedError("masked swap with non-32-bit data")
    if val_aval.shape != mask_aval.shape:
      raise ValueError(
          "Expected value and mask to have the same shape, but got"
          f" value shape {val_aval.shape} vs. mask shape {mask_aval.shape}."
      )

  ref_block_shape, *_ = ctx.block_shapes
  ref, ref_block_shape = _transform_ref(
      ref, ref_aval.dtype, ref_block_shape, prev_transforms
  )

  ref_type = ir.MemRefType(ref.type)
  memory_space = str(ref_type.memory_space)
  is_smem_store = memory_space == "#tpu.memory_space<smem>"
  is_vmem_store = memory_space == "#tpu.memory_space<vmem>"
  (aval_out,) = ctx.avals_out
  if not isinstance(val, ir.Value):
    val = ir_constant(val, mlir_type=_dtype_to_ir_type(val_aval.dtype))
  if any(
      (not isinstance(a, primitives.Slice) and a.shape)
      for a in idx_aval.indices
  ):
    raise ValueError("Cannot do int indexing on TPU")
  if not is_smem_store and not ref_block_shape:
    raise NotImplementedError(
        "Indexing into a ()-shaped Ref not yet supported on TPU.")

  starts, _, strides, _, _ = _indexer_to_start_size_stride(
      idx,
      ref_block_shape,
      cast_to_index=True,
  )
  need_stride = not all((s is None or s == 1) for s in strides)

  if is_smem_store:
    if mask is not None:
      raise ValueError("SMEM store does not support masks")
    if val_aval.shape:
      raise ValueError("Can only store scalars to SMEM")
    result = memref.load(ref, starts)
    result = _maybe_cast_load_to_bool(ctx, val_aval, result)
    val = _maybe_cast_store_to_memref_type(ctx, val_aval, val)
    memref.StoreOp(val, ref, starts)
    return result

  if not is_vmem_store:
    extra = ""
    if memory_space == "#tpu.memory_space<any>":
      extra = " ANY memory space can only be accessed using async_copy."
    raise ValueError(
        "Loads and stores are only allowed on VMEM and SMEM references." + extra
    )

  # handling VMEM store below
  if not val_aval.shape:
    raise ValueError("Cannot store scalars to VMEM")

  mem_slice_shape = list(aval_out.shape)
  for i, a in enumerate(idx_aval.indices):
    if not isinstance(a, primitives.Slice):
      mem_slice_shape.insert(i, 1)
  mem_slice_shape_iter = iter(mem_slice_shape)
  mem_slice_shape = [
      1 if b is pallas_core.mapped else next(mem_slice_shape_iter)
      for b in ref_block_shape
  ]
  mem_aval = aval_out.update(shape=tuple(mem_slice_shape), sharding=None)
  mem_aval_shape = ctx.lowering_context.dynamic_shape_replacement_fn(
      mem_aval.shape
  )
  mem_aval_vec_type = ir.VectorType.get(
      mem_aval_shape, _dtype_to_ir_type(mem_aval.dtype, is_kernel_boundary=True)
  )
  if need_stride:
    result = tpu.strided_load(mem_aval_vec_type, ref, starts, strides)
  else:
    result = vector.load(mem_aval_vec_type, ref, starts)
  val = _maybe_cast_store_to_memref_type(ctx, val_aval, val)
  if mem_aval != aval_out:
    # We are slicing a scalar so provided dummy 1 indices
    result_vec_type = ir.VectorType.get(aval_out.shape,
      _dtype_to_ir_type(aval_out.dtype, is_kernel_boundary=True))
    result = vector.shape_cast(result_vec_type, result)
    val_vec_type = ir.VectorType.get(mem_aval.shape,
      _dtype_to_ir_type(mem_aval.dtype, is_kernel_boundary=True))
    val = vector.shape_cast(val_vec_type, val)
  result = _maybe_cast_load_to_bool(ctx, val_aval, result)

  if need_stride:
    if mask is not None:
      raise NotImplementedError("masked swap with strided store")
    tpu.StridedStoreOp(val, ref, starts, strides)
  elif jaxlib_version <= (0, 4, 35):
    if mask is not None:
      raise NotImplementedError("masked swap with vector store")
    vector.StoreOp(val, ref, starts)
  else:
    tpu.VectorStoreOp(val, ref, starts, [], mask=mask)
  return result


lowering_rules[primitives.swap_p] = _masked_swap_lowering_rule
skip_mlir_conversions.add(primitives.swap_p)


def _multiple_of_lowering_rule(ctx: LoweringRuleContext, val, *, values):
  del ctx
  for multiple in values:
    val = tpu.assume_multiple(val, multiple)
  return val


lowering_rules[primitives.multiple_of_p] = _multiple_of_lowering_rule


def reduce_lowering_rule(reduce_fn, type_to_kind, type_to_identity):
  def _lowering_rule(ctx: LoweringRuleContext, x, *, axes):
    (x_aval,) = ctx.avals_in
    if not ctx.avals_out[0].shape:
      # If reducing to a scalar, we reduce by adding a leading singleton
      # dimension and reducing over all other dimensions. This avoids
      # the materialization of a scalar tensor by the reduction op which
      # is not supported.
      def _proxy_fun(val, *, axes):
        val = val[jnp.newaxis, ...]
        axes = [axis + 1 for axis in axes]
        val = reduce_fn(val, axis=axes, keepdims=True)
        # Squeeze lowers to vector.ExtractOp which will place the final
        # value in a scalar register.
        return jnp.squeeze(val)
      proxy_lowering = lower_fun(
          _proxy_fun, multiple_results=False)
      return proxy_lowering(ctx, x, axes=axes)

    if jnp.issubdtype(x_aval.dtype, jnp.floating):
      kind = type_to_kind[jnp.floating]
      val = type_to_identity[jnp.floating]
      val = ir.FloatAttr.get(
          aval_to_ir_type(
              ctx.lowering_context.dynamic_shape_replacement_fn,
              x_aval,
              shape=(),
          ),
          val,
      )
    elif x_aval.dtype == jnp.int32:
      kind = type_to_kind[jnp.signedinteger]
      val = type_to_identity[jnp.signedinteger]
      val = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), val)
    elif jnp.issubdtype(x_aval.dtype, jnp.unsignedinteger):
      raise NotImplementedError(
          "Reductions over unsigned integers not implemented."
      )
    else:
      raise NotImplementedError(
          f"Reductions over {x_aval.dtype} not implemented.")
    out_type = aval_to_ir_type(
        ctx.lowering_context.dynamic_shape_replacement_fn, ctx.avals_out[0]
    )
    identity = ir.DenseElementsAttr.get_splat(out_type, val)
    acc = arith.ConstantOp(out_type, identity)
    return vector.multi_reduction(kind, x, acc, axes)
  return _lowering_rule


REDUCE_MAX_KINDS = {
    jnp.floating: vector.CombiningKind.MAXIMUMF,
    jnp.signedinteger: vector.CombiningKind.MAXSI,
    jnp.unsignedinteger: vector.CombiningKind.MAXUI,
}
REDUCE_MAX_IDENTITY = {
    jnp.floating: float("-inf"),
    jnp.signedinteger: np.iinfo(np.int32).min,
}
_reduce_max_lowering_rule = reduce_lowering_rule(
    jnp.max, REDUCE_MAX_KINDS, REDUCE_MAX_IDENTITY)
lowering_rules[lax.reduce_max_p] = _reduce_max_lowering_rule


REDUCE_MIN_KINDS = {
    jnp.floating: vector.CombiningKind.MINIMUMF,
    jnp.signedinteger: vector.CombiningKind.MINSI,
    jnp.unsignedinteger: vector.CombiningKind.MINUI,
}
REDUCE_MIN_IDENTITY = {
    jnp.floating: float("inf"),
    jnp.signedinteger: np.iinfo(np.int32).max,
}
_reduce_min_lowering_rule = reduce_lowering_rule(
    jnp.min, REDUCE_MIN_KINDS, REDUCE_MIN_IDENTITY)
lowering_rules[lax.reduce_min_p] = _reduce_min_lowering_rule


REDUCE_SUM_KINDS = {
    jnp.floating: vector.CombiningKind.ADD,
    jnp.signedinteger: vector.CombiningKind.ADD,
    jnp.unsignedinteger: vector.CombiningKind.ADD,
}
REDUCE_SUM_IDENTITY = {
    jnp.floating: 0.0,
    jnp.signedinteger: 0,
}
_reduce_sum_lowering_rule = reduce_lowering_rule(
    jnp.sum, REDUCE_SUM_KINDS, REDUCE_SUM_IDENTITY)
lowering_rules[lax.reduce_sum_p] = _reduce_sum_lowering_rule


def _reduce_and_lowering_rule(ctx: LoweringRuleContext, x, *, axes):
  def _proxy_reduce(arg, *, axes):
    # Mosaic currently only supports float reductions, so we cast the boolean
    # arg to a float and use reduce_min to implement reduce_and.
    # TODO(b/351017807): Implement this logic in Mosaic MultiDimReductionOp
    # instead.
    float_arg = jnp.where(arg, 1.0, 0.0)
    return jnp.min(float_arg, axis=axes) > 0.0
  proxy_lowering = lower_fun(
      _proxy_reduce, multiple_results=False)
  return proxy_lowering(ctx, x, axes=axes)

lowering_rules[lax.reduce_and_p] = _reduce_and_lowering_rule


def _reduce_or_lowering_rule(ctx: LoweringRuleContext, x, *, axes):
  def _proxy_reduce(arg, *, axes):
    # Mosaic currently only supports float reductions, so we cast the boolean
    # arg to a float and use reduce_max to implement reduce_or.
    # TODO(b/351017807): Implement this logic in Mosaic MultiDimReductionOp
    # instead.
    float_arg = jnp.where(arg, 1.0, 0.0)
    return jnp.max(float_arg, axis=axes) > 0.0
  proxy_lowering = lower_fun(
      _proxy_reduce, multiple_results=False)
  return proxy_lowering(ctx, x, axes=axes)

lowering_rules[lax.reduce_or_p] = _reduce_or_lowering_rule


def _broadcast_to_lowering_rule(
    ctx: LoweringRuleContext, x, shape: Sequence[int]
):
  raise RuntimeError(
      "`broadcast_to` is a Triton-specific primitive. Please consider using"
      " `jnp.broadcast_to` instead."
  )


lowering_rules[state_primitives.broadcast_to_p] = _broadcast_to_lowering_rule


def _broadcast_in_dim_lowering_rule(
    ctx: LoweringRuleContext, val, *, shape, broadcast_dimensions, sharding
):
  del sharding
  (aval_in,) = ctx.avals_in
  (aval_out,) = ctx.avals_out

  if jnp.issubdtype(aval_in.dtype, jnp.bool_):
    # Direct broadcasts for bools are not supported in Mosaic due to booleans
    # living in mask registers and broadcast operating on vregs. Broadcast as an
    # integer instead and cast back to a bool.
    # TODO(b/351019164): Implement this logic in Mosaic BroadcastOp instead.
    def _proxy_fun(val, *, shape, broadcast_dimensions):
      int_val = jnp.where(val, 1, 0)
      bcast_val = jax.lax.broadcast_in_dim(int_val, shape, broadcast_dimensions)
      return bcast_val == 1
    proxy_lowering = lower_fun(
        _proxy_fun, multiple_results=False)
    return proxy_lowering(
        ctx, val, shape=shape, broadcast_dimensions=broadcast_dimensions)

  if broadcast_dimensions:
    out_shape_list = [1] * len(shape)
    for i, s in zip(broadcast_dimensions, aval_in.shape):
      out_shape_list[i] = s
    out_shape = tuple(out_shape_list)
    out_type = ir.VectorType.get(
        out_shape, _dtype_to_ir_type(aval_out.dtype)
    )
    val = vector.shape_cast(out_type, val)
    if out_shape == aval_out.shape:
      return val
  out_type = ir.VectorType.get(
      aval_out.shape, _dtype_to_ir_type(aval_out.dtype)
  )
  return vector.broadcast(out_type, val)


lowering_rules[lax.broadcast_in_dim_p] = _broadcast_in_dim_lowering_rule


def jax_dot_dims_to_tpu_dot_dot_dims(dimension_numbers, lhs_shape, rhs_shape):
  """Converts a jax dot dimension numbers to a tpu dot dimension numbers.

  Jax dot dimension numbers are given as a tuple of tuples of sequences of ints
  of the form ((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims,
  rhs_batch_dims)).

  TPU dot dimension numbers are given as an MLIR definition of the form
  #tpu.dot_dimension_numbers - which can be found in the tpu dilect definition
  # file, tpu.td .
  """
  (contracting_dims, batch_dims) = dimension_numbers
  lhs_contracting_dims, rhs_contracting_dims = contracting_dims
  lhs_batch_dims, rhs_batch_dims = batch_dims

  lhs_total_dims = set(range(len(lhs_shape)))
  rhs_total_dims = set(range(len(rhs_shape)))

  lhs_non_contracting_dims = sorted(
      lhs_total_dims - set(lhs_contracting_dims) - set(lhs_batch_dims)
  )
  rhs_non_contracting_dims = sorted(
      rhs_total_dims - set(rhs_contracting_dims) - set(rhs_batch_dims)
  )

  # Create output_dim_order
  # Note: we assume that the output dimensions are ordered as batch dims, lhs_non_contracting_dims,
  # rhs_non_contracting_dims - this assumption is safe to make, as it is
  # the same one made in jax's dot_general.
  output_dim_order = []

  lhs_dim_map = {dim: idx for idx, dim in enumerate(range(len(lhs_shape)))}
  rhs_dim_map = {dim: idx for idx, dim in enumerate(range(len(rhs_shape)))}

  for dim in lhs_batch_dims:
    output_dim_order.append(0)
    output_dim_order.append(lhs_dim_map[dim])

  for dim in lhs_non_contracting_dims:
    output_dim_order.append(0)
    output_dim_order.append(lhs_dim_map[dim])

  for dim in rhs_non_contracting_dims:
    output_dim_order.append(1)
    output_dim_order.append(rhs_dim_map[dim])

  def format_dims(dims):
    return "[" + ", ".join(str(d) for d in dims) + "]"

  all_dims = (
      lhs_contracting_dims,
      rhs_contracting_dims,
      lhs_non_contracting_dims,
      rhs_non_contracting_dims,
      output_dim_order,
      lhs_batch_dims,
      rhs_batch_dims,
  )
  tpu_dim_numbers_str = (
      f"#tpu.dot_dimension_numbers<{','.join(map(format_dims, all_dims))}>"
  )

  return ir.Attribute.parse(tpu_dim_numbers_str)


def _dot_general_lowering_rule(
    ctx: LoweringRuleContext, x, y, dimension_numbers, precision, **_
):
  (lhs_dims, rhs_dims), _ = dimension_numbers
  (aval_out,) = ctx.avals_out
  out_type = aval_to_ir_type(
      ctx.lowering_context.dynamic_shape_replacement_fn, aval_out
  )
  val_type = out_type.element_type
  if any(
      cls.isinstance(val_type)
      for cls in [
          ir.BF16Type,
          ir.F32Type,
          ir.Float8E5M2Type,
          ir.Float8E4M3FNType,
      ]
  ):
    val = ir.FloatAttr.get(val_type, 0.0)
  elif ir.IntegerType.isinstance(val_type):
    val = ir.IntegerAttr.get(val_type, 0)
  else:
    raise NotImplementedError(ctx.avals_out[0].dtype)
  if any(len(a.shape) != 2 for a in ctx.avals_in):
    raise NotImplementedError(
        f"Only 2D tensors supported in dot; received: {ctx.avals_in}"
    )
  lhs_aval, rhs_aval = ctx.avals_in
  # This is really a matrix-vector product. It only looks like matrix-matrix.
  if lhs_dims == (1,) and rhs_dims == (1,) and ctx.avals_in[1].shape[0] == 1:
    if ctx.avals_in[0].shape != ctx.avals_in[1].shape:
      bcast_shape = jnp.broadcast_shapes(
          ctx.avals_in[0].shape, ctx.avals_out[0].shape
      )
      bcast_shape = ir.VectorType.get(
          list(bcast_shape), _dtype_to_ir_type(ctx.avals_out[0].dtype)
      )
      if ctx.avals_in[0].shape != bcast_shape:
        x = vector.broadcast(bcast_shape, x)
      if ctx.avals_in[1].shape != bcast_shape:
        y = vector.broadcast(bcast_shape, y)
    red_type = aval_to_ir_type(
        ctx.lowering_context.dynamic_shape_replacement_fn,
        lhs_aval.update(shape=(lhs_aval.shape[0],)),
    )
    acc = arith.ConstantOp(
        red_type, ir.DenseElementsAttr.get_splat(red_type, val)
    )
    red = vector.MultiDimReductionOp(
        ir.Attribute.parse("#vector.kind<add>"),
        arith.MulFOp(x, y),
        acc,
        [1]
    )
    return vector.shape_cast(out_type, red)

  tpu_dot_dims = jax_dot_dims_to_tpu_dot_dot_dims(
      dimension_numbers, lhs_aval.shape, rhs_aval.shape
  )

  if precision is not None:
    if precision[0] != precision[1]:
      raise NotImplementedError("Per-operand dot precision unsupported")
    precision = precision[0]
  if precision is None or precision == lax.Precision.DEFAULT:
    precision_attr = None  # That's the default in Mosaic.
  elif precision == lax.Precision.HIGHEST:
    precision_attr = ir.Attribute.parse(
        "#tpu.contract_precision<fp32>"
    )
  else:
    raise NotImplementedError(f"Unsupported dot precision: {precision}")
  out_tile = arith.ConstantOp(
      out_type, ir.DenseElementsAttr.get_splat(out_type, val)
  )
  return tpu.matmul(
      out_type,
      x,
      y,
      out_tile,
      dimension_numbers=tpu_dot_dims,
      precision=precision_attr,
  )


lowering_rules[lax.dot_general_p] = _dot_general_lowering_rule

def _convert_helper(x, *, to_dtype):
  # Helper function for dtype conversion
  from_dtype = x.dtype
  if from_dtype == jnp.bool_:
    x = x.astype(jnp.int32)
    return _convert_helper(x, to_dtype=to_dtype)
  if to_dtype == jnp.bool_:
    # Lower float32 or (u)int32 -> bool to cmp neq %in, 0
    # TODO(apaszke,mvoz): Move the upcasts for cmpi to the Mosaic canonicalizer.
    if jnp.issubdtype(from_dtype, jnp.floating):
      if from_dtype.itemsize < 4:
        x = x.astype(jnp.float32)
    elif jnp.issubdtype(from_dtype, jnp.integer):
      if from_dtype.itemsize < 4:
        x = x.astype(jnp.int32)
    return x != jnp.asarray(0, dtype=x.dtype)
  if jnp.issubdtype(from_dtype, jnp.signedinteger):
    if from_dtype.itemsize < 4:
      x = x.astype(jnp.int32)
    if jnp.issubdtype(to_dtype, jnp.floating) and to_dtype.itemsize < 4:
      x = x.astype(jnp.float32)
    return x.astype(to_dtype)
  if jnp.issubdtype(from_dtype, jnp.unsignedinteger):
    if from_dtype.itemsize < 4:
      x = x.astype(jnp.uint32)
    # unsigned -> float is unsupported. We fall through and raise at the bottom.
    if not jnp.issubdtype(to_dtype, jnp.floating):
      return x.astype(to_dtype)
  if jnp.issubdtype(from_dtype, jnp.floating) and jnp.issubdtype(
      to_dtype, jnp.signedinteger
  ):
    if from_dtype.itemsize < 4:
      x = x.astype(jnp.float32)
    if to_dtype.itemsize < 4:
      # Need to clip values to match XLA
      minval, maxval = jnp.iinfo(to_dtype).min, jnp.iinfo(to_dtype).max
      x = jnp.clip(x, minval, maxval)
      return x.astype(jnp.int32).astype(to_dtype)
  raise NotImplementedError(f"Unsupported cast: {from_dtype} -> {to_dtype}")

def _convert_element_type_lowering_rule(
    ctx: LoweringRuleContext, x, *, new_dtype, weak_type, sharding
):
  del weak_type
  del sharding
  out_aval = ctx.avals_out[0]
  in_aval = ctx.avals_in[0]
  old_dtype = in_aval.dtype
  out_type = aval_to_ir_type(
      ctx.lowering_context.dynamic_shape_replacement_fn, out_aval
  )

  if old_dtype == new_dtype:
    return x

  if new_dtype.itemsize == 8:
    raise NotImplementedError("64-bit types are not supported")

  _from = lambda dtype: jnp.issubdtype(old_dtype, dtype)
  _to = lambda dtype: jnp.issubdtype(new_dtype, dtype)
  floating = jnp.floating
  integer = jnp.integer
  signed = jnp.signedinteger
  both_32bit = old_dtype.itemsize == 4 and new_dtype.itemsize == 4
  if _from(floating) and _to(floating):
    if old_dtype.itemsize < new_dtype.itemsize and new_dtype.itemsize == 4:
      return arith.extf(out_type, x)
    elif old_dtype.itemsize > new_dtype.itemsize and old_dtype.itemsize == 4:
      return arith.truncf(out_type, x)
  elif _from(integer) and _to(integer):
    if old_dtype.itemsize < new_dtype.itemsize and new_dtype.itemsize == 4:
      if not (_from(signed) and _to(signed)):
        raise NotImplementedError(f"Unsupported cast: {old_dtype} -> {new_dtype}")
      return arith.extsi(out_type, x)
    elif old_dtype.itemsize > new_dtype.itemsize and old_dtype.itemsize == 4:
      return arith.trunci(out_type, x)
    elif jnp.iinfo(old_dtype).bits == jnp.iinfo(new_dtype).bits:
      # This case triggers when casting signed to unsigned or vice versa.
      return x
  # TODO(apaszke): Remove both_32bit constraints using the Mosaic canonicalizer.
  elif _from(floating) and _to(signed):
    # TODO(apaszke): Remove once a month has passed, along with the
    # _convert_helper float -> signed conversion above.
    if not ctx.forward_compatible or both_32bit:
      return arith.fptosi(out_type, x)
  elif _from(signed) and _to(floating) and both_32bit:
    return arith.sitofp(out_type, x)
  elif old_dtype == jnp.bool_ and _to(integer) and new_dtype.itemsize == 4:
    return arith.extui(out_type, x)
  return lower_fun(functools.partial(_convert_helper, to_dtype=new_dtype),
                   multiple_results=False)(ctx, x)


lowering_rules[lax.convert_element_type_p] = _convert_element_type_lowering_rule


def _reshape_lowering_rule(ctx: LoweringRuleContext, x, new_sizes, dimensions,
                           sharding):
  if dimensions is not None:
    raise NotImplementedError
  if any(d is None for d in new_sizes):
    raise NotImplementedError
  if not ctx.avals_in[0].shape:
    return vector.broadcast(
        aval_to_ir_type(
            ctx.lowering_context.dynamic_shape_replacement_fn, ctx.avals_out[0]
        ),
        x,
    )
  return vector.shape_cast(
      aval_to_ir_type(
          ctx.lowering_context.dynamic_shape_replacement_fn, ctx.avals_out[0]
      ),
      x,
  )


lowering_rules[lax.reshape_p] = _reshape_lowering_rule


def _squeeze_lowering_rule(ctx: LoweringRuleContext, x, dimensions):
  del dimensions  # Unused.
  (aval_in,) = ctx.avals_in
  (aval_out,) = ctx.avals_out
  if not aval_out.shape:
    if aval_out.dtype.itemsize != 4:
      raise ValueError(
          "Only arrays with 32-bit element types can be converted to scalars,"
          f" but got: {aval_out.dtype}. Try casting the input before squeezing"
          " the scalar."
      )
    return vector.extract(x, [], [0] * len(aval_in.shape))
  return vector.shape_cast(
      aval_to_ir_type(
          ctx.lowering_context.dynamic_shape_replacement_fn, ctx.avals_out[0]
      ),
      x,
  )


lowering_rules[lax.squeeze_p] = _squeeze_lowering_rule


def _concatenate_lowering_rule(ctx: LoweringRuleContext, *xs, dimension):
  out_type = aval_to_ir_type(
      ctx.lowering_context.dynamic_shape_replacement_fn, ctx.avals_out[0]
  )
  return tpu.concatenate(out_type, xs, dimension=dimension)


lowering_rules[lax.concatenate_p] = _concatenate_lowering_rule


def _split_lowering_rule(
    ctx: LoweringRuleContext, x, *, sizes, axis
):
  (x_aval,) = ctx.avals_in
  slice_size = np.array(x_aval.shape, dtype=np.int64)
  starts = np.zeros_like(slice_size)
  strides = np.ones_like(slice_size)
  outs = []
  for size, aval_out in zip(sizes, ctx.avals_out):
    slice_size[axis] = size
    outs.append(
        vector.extract_strided_slice(
            aval_to_ir_type(
                ctx.lowering_context.dynamic_shape_replacement_fn, aval_out
            ),
            x,
            starts,
            slice_size,
            strides,
        )
    )
    starts[axis] += size
  return outs

lowering_rules[lax.split_p] = _split_lowering_rule


def _iota_lowering_rule(ctx: LoweringRuleContext, dtype, shape, dimension,
                        sharding):
  out_type = aval_to_ir_type(
      ctx.lowering_context.dynamic_shape_replacement_fn, ctx.avals_out[0]
  )
  return tpu.iota(out_type, dimension=dimension)


lowering_rules[lax.iota_p] = _iota_lowering_rule


def _transpose_lowering_rule(ctx: LoweringRuleContext, x, *, permutation):
  if permutation != (1, 0):
    raise NotImplementedError
  out_type = aval_to_ir_type(
      ctx.lowering_context.dynamic_shape_replacement_fn, ctx.avals_out[0]
  )
  return vector.transpose(out_type, x, permutation)


lowering_rules[lax.transpose_p] = _transpose_lowering_rule


def _bcast(x, y, x_aval, y_aval, out_aval):
  x_dtype = x_aval.dtype
  y_dtype = y_aval.dtype
  if y_aval.weak_type:
    y_dtype = x_aval.dtype
  elif x_aval.weak_type:
    x_dtype = y_aval.dtype
  if isinstance(x, (np.ndarray, np.number, int, float)):
    if getattr(y, "type", None) == ir.IndexType.get():
      mlir_type = y.type
    else:
      mlir_type = _dtype_to_ir_type(x_dtype)
    x = ir_constant(x, mlir_type)
  if isinstance(y, (np.ndarray, np.number, int, float)):
    if getattr(x, "type", None) == ir.IndexType.get():
      mlir_type = x.type
    else:
      mlir_type = _dtype_to_ir_type(y_dtype)
    y = ir_constant(y, mlir_type)
  out_shape = list(out_aval.shape)
  if x_aval.shape != out_aval.shape:
    x_ty = ir.VectorType.get(out_shape, _dtype_to_ir_type(x_dtype))
    x = vector.broadcast(x_ty, x)
  if y_aval.shape != out_aval.shape:
    y_ty = ir.VectorType.get(out_shape, _dtype_to_ir_type(y_dtype))
    y = vector.broadcast(y_ty, y)
  return x, y


def _add_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, ctx.avals_in[0], ctx.avals_in[1], ctx.avals_out[0])
  (aval_out,) = ctx.avals_out
  if jnp.issubdtype(aval_out.dtype, jnp.integer):
    return arith.addi(x, y)
  if jnp.issubdtype(aval_out.dtype, jnp.floating):
    return arith.addf(x, y)
  raise NotImplementedError(aval_out.dtype)


lowering_rules[lax.add_p] = _add_lowering_rule
skip_mlir_conversions.add(lax.add_p)
lowering_rules[ad_util.add_any_p] = _add_lowering_rule
skip_mlir_conversions.add(ad_util.add_any_p)


class FoldingError(Exception):
  pass


def _fold_and_get_constant_value(x):
  def _fold(x, fuel):
    if fuel <= 0:
      raise FoldingError("Folding depth exceeded")
    op_name = getattr(x.owner, "name", None)
    binop_folds = {
        "arith.maxsi": max,
        "arith.minsi": min,
    }
    if op_name == "arith.constant":
      if ir.IntegerType.isinstance(x.type):
        return ir.IntegerAttr(x.owner.attributes["value"]).value
      elif ir.FloatType.isinstance(x.type):
        return ir.FloatAttr(x.owner.attributes["value"]).value
      else:
        raise ValueError(f"Unsupported constant type: {x.type}")
    if op_name in binop_folds:
      return binop_folds[op_name](_fold(v, fuel - 1) for v in x.owner.operands)
    raise FoldingError(f"Folding not supported for {x.owner}")

  try:
    return _fold(x, 10)
  except FoldingError:
    return None


def _max_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, ctx.avals_in[0], ctx.avals_in[1], ctx.avals_out[0])
  (aval_out,) = ctx.avals_out
  if jnp.issubdtype(aval_out.dtype, jnp.signedinteger):
    return arith.maxsi(x, y)
  elif jnp.issubdtype(aval_out.dtype, jnp.unsignedinteger):
    return arith.maxui(x, y)
  elif jnp.issubdtype(aval_out.dtype, jnp.floating):
    return arith.maximumf(x, y)
  raise NotImplementedError(aval_out.dtype)


lowering_rules[lax.max_p] = _max_lowering_rule
skip_mlir_conversions.add(lax.max_p)


def _min_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, ctx.avals_in[0], ctx.avals_in[1], ctx.avals_out[0])
  (aval_out,) = ctx.avals_out
  if jnp.issubdtype(aval_out.dtype, jnp.signedinteger):
    return arith.minsi(x, y)
  elif jnp.issubdtype(aval_out.dtype, jnp.unsignedinteger):
    return arith.minui(x, y)
  elif jnp.issubdtype(aval_out.dtype, jnp.floating):
    return arith.minimumf(x, y)
  raise NotImplementedError(aval_out.dtype)


lowering_rules[lax.min_p] = _min_lowering_rule
skip_mlir_conversions.add(lax.min_p)


def _sub_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, ctx.avals_in[0], ctx.avals_in[1], ctx.avals_out[0])
  (aval_out,) = ctx.avals_out
  if jnp.issubdtype(aval_out.dtype, jnp.integer):
    return arith.subi(x, y)
  if jnp.issubdtype(aval_out.dtype, jnp.floating):
    return arith.subf(x, y)
  raise NotImplementedError(aval_out.dtype)


lowering_rules[lax.sub_p] = _sub_lowering_rule
skip_mlir_conversions.add(lax.sub_p)


def _mul_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, ctx.avals_in[0], ctx.avals_in[1], ctx.avals_out[0])
  (aval_out,) = ctx.avals_out
  if jnp.issubdtype(aval_out.dtype, jnp.integer):
    return arith.muli(x, y)
  if jnp.issubdtype(aval_out.dtype, jnp.floating):
    return arith.mulf(x, y)
  raise NotImplementedError(aval_out.dtype)


lowering_rules[lax.mul_p] = _mul_lowering_rule
skip_mlir_conversions.add(lax.mul_p)


def _div_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, ctx.avals_in[0], ctx.avals_in[1], ctx.avals_out[0])
  (aval_out,) = ctx.avals_out
  if jnp.issubdtype(aval_out.dtype, jnp.signedinteger):
    return arith.divsi(x, y)
  if jnp.issubdtype(aval_out.dtype, jnp.unsignedinteger):
    return arith.divui(x, y)
  elif jnp.issubdtype(aval_out.dtype, jnp.floating):
    return arith.divf(x, y)
  raise NotImplementedError(aval_out.dtype)


lowering_rules[lax.div_p] = _div_lowering_rule
skip_mlir_conversions.add(lax.div_p)


def _rem_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, ctx.avals_in[0], ctx.avals_in[1], ctx.avals_out[0])
  (aval_out,) = ctx.avals_out
  if jnp.issubdtype(aval_out.dtype, jnp.signedinteger):
    return arith.remsi(x, y)
  if jnp.issubdtype(aval_out.dtype, jnp.unsignedinteger):
    return arith.remui(x, y)
  if jnp.issubdtype(aval_out.dtype, jnp.floating):
    return arith.remf(x, y)
  raise NotImplementedError(aval_out.dtype)


lowering_rules[lax.rem_p] = _rem_lowering_rule
skip_mlir_conversions.add(lax.rem_p)


def _abs_lowering_rule(ctx: LoweringRuleContext, x):
  (aval_out,) = ctx.avals_out
  if jnp.issubdtype(aval_out.dtype, jnp.integer):
    return math.absi(x)
  if jnp.issubdtype(aval_out.dtype, jnp.floating):
    return math.absf(x)
  raise NotImplementedError(aval_out.dtype)


lowering_rules[lax.abs_p] = _abs_lowering_rule


def _neg_lowering_rule(ctx: LoweringRuleContext, x):
  (x_aval,) = ctx.avals_in
  new_ctx = ctx.replace(
      avals_in=(jax_core.ShapedArray((), x_aval.dtype), x_aval),
      block_shapes=((), *ctx.block_shapes)
  )
  return _sub_lowering_rule(new_ctx, np.array(0, dtype=x_aval.dtype), x)


lowering_rules[lax.neg_p] = _neg_lowering_rule
skip_mlir_conversions.add(lax.neg_p)


def _sign_lowering_rule(ctx: LoweringRuleContext, x):
  return lower_fun(
      pallas_utils.sign_lowering_helper, multiple_results=False,
  )(ctx, x)


lowering_rules[lax.sign_p] = _sign_lowering_rule


def _nextafter_lowering_rule(ctx: LoweringRuleContext, x, y):
  return lower_fun(
      pallas_utils.nextafter_lowering_helper, multiple_results=False,
  )(ctx, x, y)


lowering_rules[lax.nextafter_p] = _nextafter_lowering_rule


def _rsqrt_lowering_rule(ctx: LoweringRuleContext, x):
  return math.rsqrt(x)


lowering_rules[lax.rsqrt_p] = _rsqrt_lowering_rule


def _sqrt_lowering_rule(ctx: LoweringRuleContext, x):
  return math.sqrt(x)


lowering_rules[lax.sqrt_p] = _sqrt_lowering_rule


def _square_lowering_rule(ctx: LoweringRuleContext, x):
  if jnp.issubdtype(ctx.avals_in[0].dtype, jnp.integer):
    return arith.muli(x, x)
  return arith.mulf(x, x)


lowering_rules[lax.square_p] = _square_lowering_rule


def _exp_lowering_rule(ctx: LoweringRuleContext, x):
  return math.exp(x)


lowering_rules[lax.exp_p] = _exp_lowering_rule


def _pow_lowering_rule(ctx: LoweringRuleContext, x, y):
  # jax accepts float base (x) and integer/float exponent (y), and integer
  # exponent is casted to float.
  out_type = aval_to_ir_type(
      ctx.lowering_context.dynamic_shape_replacement_fn, ctx.avals_out[0]
  )
  if jnp.issubdtype(ctx.avals_in[1].dtype, jnp.integer):
    y = arith.sitofp(out_type, y)
  if not isinstance(x, ir.Value) and x == 2.:
    return math.exp2(y)
  x, y = _bcast(x, y, ctx.avals_in[0], ctx.avals_in[1], ctx.avals_out[0])
  return math.powf(x, y)


lowering_rules[lax.pow_p] = _pow_lowering_rule
skip_mlir_conversions.add(lax.pow_p)


def _integer_pow_lowering_rule(ctx: LoweringRuleContext, x, *, y):
  return lower_fun(lax_internal._integer_pow, multiple_results=False)(
      ctx, x, y=y)


lowering_rules[lax.integer_pow_p] = _integer_pow_lowering_rule


def _exp2_lowering_rule(ctx: LoweringRuleContext, x):
  # exp2 in JAX lowers to exp(ln2 * x), not to pow2. We match that behavior
  # here.
  return lower_fun(
      lambda x: jnp.exp(jnp.astype(np.log(2), x.dtype) * x),
      multiple_results=False,
  )(ctx, x)


lowering_rules[lax.exp2_p] = _exp2_lowering_rule
skip_mlir_conversions.add(lax.exp2_p)


def _logistic_lowering_rule(ctx: LoweringRuleContext, x):
  neg_x = arith.negf(x)
  exp_neg_x = math.exp(neg_x)
  aval_out = ctx.avals_out[0]
  out_type = aval_to_ir_type(
      ctx.lowering_context.dynamic_shape_replacement_fn, aval_out
  )
  if aval_out.shape == ():
    one = ir_constant(1.0, mlir_type=out_type)
  else:
    one = vector.broadcast(out_type, ir_constant(1.0))
  denom = arith.addf(one, exp_neg_x)
  return arith.divf(one, denom)


lowering_rules[lax.logistic_p] = _logistic_lowering_rule


def _sin_lowering_rule(ctx: LoweringRuleContext, x):
  return math.sin(x)


lowering_rules[lax.sin_p] = _sin_lowering_rule


def _cos_lowering_rule(ctx: LoweringRuleContext, x):
  return math.cos(x)


lowering_rules[lax.cos_p] = _cos_lowering_rule


def _tan_lowering_rule(ctx: LoweringRuleContext, x):
  return math.tan(x)


lowering_rules[lax.tan_p] = _tan_lowering_rule


def _tanh_lowering_rule(ctx: LoweringRuleContext, x):
  return math.tanh(x)


lowering_rules[lax.tanh_p] = _tanh_lowering_rule


def _log_lowering_rule(ctx: LoweringRuleContext, x):
  return math.log(x)


lowering_rules[lax.log_p] = _log_lowering_rule


def _log1p_lowering_rule(ctx: LoweringRuleContext, x):
  return math.log1p(x)


lowering_rules[lax.log1p_p] = _log1p_lowering_rule


def _round_lowering_rule(ctx: LoweringRuleContext, x, *, rounding_method):
  if rounding_method == 0:
    return math.round(x)
  elif rounding_method == 1:
    return math.roundeven(x)
  else:
    raise NotImplementedError(f"Unsupported rounding method: {rounding_method}")


lowering_rules[lax.round_p] = _round_lowering_rule


def _ceil_lowering_rule(ctx: LoweringRuleContext, x):
  return math.ceil(x)


lowering_rules[lax.ceil_p] = _ceil_lowering_rule


def _floor_lowering_rule(ctx: LoweringRuleContext, x):
  return math.floor(x)


lowering_rules[lax.floor_p] = _floor_lowering_rule


def _clz_lowering_rule(ctx: LoweringRuleContext, x):
  return math.ctlz(x)

lowering_rules[lax.clz_p] = _clz_lowering_rule


def _population_count_lowering_rule(ctx: LoweringRuleContext, x):
  aval_out = ctx.avals_out[0]
  if aval_out.shape == ():
    raise ValueError("Population count is not supported on scalars")
  return math.ctpop(x)

lowering_rules[lax.population_count_p] = _population_count_lowering_rule


# Mapping for signed integer comparisons.
_cmpsi_lowering_types = {
    lax.eq_p: arith.CmpIPredicate.eq,
    lax.ne_p: arith.CmpIPredicate.ne,
    lax.lt_p: arith.CmpIPredicate.slt,
    lax.le_p: arith.CmpIPredicate.sle,
    lax.gt_p: arith.CmpIPredicate.sgt,
    lax.ge_p: arith.CmpIPredicate.sge,
}

# Mapping for unsigned integer comparisons.
_cmpui_lowering_types = {
    lax.eq_p: arith.CmpIPredicate.eq,
    lax.ne_p: arith.CmpIPredicate.ne,
    lax.lt_p: arith.CmpIPredicate.ult,
    lax.le_p: arith.CmpIPredicate.ule,
    lax.gt_p: arith.CmpIPredicate.ugt,
    lax.ge_p: arith.CmpIPredicate.uge,
}

# Mapping for floating point comparisons.
_cmpf_lowering_types = {
    lax.eq_p: arith.CmpFPredicate.OEQ,
    lax.ne_p: arith.CmpFPredicate.ONE,
    lax.lt_p: arith.CmpFPredicate.OLT,
    lax.le_p: arith.CmpFPredicate.OLE,
    lax.gt_p: arith.CmpFPredicate.OGT,
    lax.ge_p: arith.CmpFPredicate.OGE,
}


# The relationship between comparison operations on booleans and boolean
# algebra is as follows:
# eq(x, y) = !(x ^ y)
# ne(x, y) = x ^ y
# lt(x, y) = !x && y
# le(x, y) = !x || y
# gt(x, y) = x && !y
# ge(x, y) = x || !y
def _cmp_boolean_lowering_helper(primitive, x: Array, y: Array):
  """A helper function for lowering comparison operations for boolean inputs.

  Args:
    primitive: A JAX primitive representing a comparison operation, which is
      one of the following: `lax.eq_p` (equals), `lax.ne_p` (not equals),
      `lax.lt_p` (less than), `lax.le_p` (less than or equal to),
      `lax.gt_p` (greater than), or `lax.ge_p` (greater than or equal to).
    x: A boolean array representing the first operand in the comparison.
    y: A boolean array representing the second operand in the comparison.

  Returns:
    A boolean array that is the result of applying the comparison operation
    between `x` and `y` based on the given primitive.

  Raises:
    ValueError: If an unsupported comparison primitive is provided.
  """
  if primitive == lax.eq_p:
    return jnp.logical_not(jnp.logical_xor(x, y))
  elif primitive == lax.ne_p:
    return jnp.logical_xor(x, y)
  elif primitive == lax.lt_p:
    return jnp.logical_and(jnp.logical_not(x), y)
  elif primitive == lax.le_p:
    return jnp.logical_or(jnp.logical_not(x), y)
  elif primitive == lax.gt_p:
    return jnp.logical_and(x, jnp.logical_not(y))
  elif primitive == lax.ge_p:
    return jnp.logical_or(x, jnp.logical_not(y))
  else:
    raise ValueError(f"Unsupported comparison primitive: {primitive}")


def _cmp_lowering_rule(primitive, ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, ctx.avals_in[0], ctx.avals_in[1], ctx.avals_out[0])
  x_aval, y_aval = ctx.avals_in
  if x_aval.dtype != y_aval.dtype:
    raise ValueError(
        f"Mixed dtype operands in cmp: {x_aval.dtype}, {y_aval.dtype}"
    )
  dtype = x_aval.dtype

  if jnp.issubdtype(dtype, jnp.bool_):
    return lower_fun(
        functools.partial(_cmp_boolean_lowering_helper, primitive),
        multiple_results=False,
    )(ctx, x, y)

  if jnp.issubdtype(dtype, jnp.integer):
    is_uint = jnp.issubdtype(dtype, jnp.unsignedinteger)
    pred = (
        _cmpui_lowering_types if is_uint else _cmpsi_lowering_types
    )[primitive]
    predicate = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), pred)
    return arith.cmpi(predicate, x, y)

  if jnp.issubdtype(dtype, jnp.floating):
    pred = _cmpf_lowering_types[primitive]
    predicate = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), pred)
    return arith.cmpf(predicate, x, y)

  raise NotImplementedError(f"Unsupported dtype in cmp: {dtype}")


lowering_rules[lax.eq_p] = functools.partial(_cmp_lowering_rule, lax.eq_p)
lowering_rules[lax.ne_p] = functools.partial(_cmp_lowering_rule, lax.ne_p)
lowering_rules[lax.lt_p] = functools.partial(_cmp_lowering_rule, lax.lt_p)
lowering_rules[lax.le_p] = functools.partial(_cmp_lowering_rule, lax.le_p)
lowering_rules[lax.gt_p] = functools.partial(_cmp_lowering_rule, lax.gt_p)
lowering_rules[lax.ge_p] = functools.partial(_cmp_lowering_rule, lax.ge_p)


def _and_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, *ctx.avals_in, *ctx.avals_out)
  return arith.andi(x, y)


lowering_rules[lax.and_p] = _and_lowering_rule
skip_mlir_conversions.add(lax.and_p)


def _is_finite_lowering_rule(ctx: LoweringRuleContext, x):
  out_aval, = ctx.avals_out
  out_type = aval_to_ir_type(
      ctx.lowering_context.dynamic_shape_replacement_fn, out_aval
  )
  return _not_lowering_rule(ctx, tpu.weird(out_type, x))


lowering_rules[lax.is_finite_p] = _is_finite_lowering_rule


def _or_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, *ctx.avals_in, *ctx.avals_out)
  return arith.ori(x, y)


lowering_rules[lax.or_p] = _or_lowering_rule
skip_mlir_conversions.add(lax.or_p)


def _not_lowering_rule(ctx: LoweringRuleContext, x):
  # The primitive not_p is lowered to
  # https://github.com/openxla/stablehlo/blob/main/docs/spec.md#not
  # which is arithmetic for integers and logical for booleans.
  # Lowering to:
  # xor x, -1
  # covers both cases.
  out_aval = ctx.avals_out[0]
  out_scalar_type = _dtype_to_ir_type(out_aval.dtype)
  if not out_aval.shape:
    # Create a scalar constant.
    minus_one = ir_constant(-1, out_scalar_type)
  else:
    # Create a vector constant.
    out_type = aval_to_ir_type(
        ctx.lowering_context.dynamic_shape_replacement_fn, out_aval
    )
    scalar_minus_one = ir.IntegerAttr.get(out_scalar_type, -1)
    minus_one = arith.ConstantOp(
        out_type, ir.DenseElementsAttr.get_splat(out_type, scalar_minus_one)
    )
  return arith.xori(x, minus_one)


lowering_rules[lax.not_p] = _not_lowering_rule

def _select_n_lowering_rule(ctx: LoweringRuleContext, pred, x, *args):
  if len(args) > 1:
    raise NotImplementedError("select_n only supported with <= 2 arguments")
  pred_aval, x_aval = ctx.avals_in[:2]
  if pred_aval.dtype != np.dtype(np.bool_):
    lower_ctx = LoweringRuleContext(
        ctx.lowering_context,
        avals_in=[pred_aval],
        avals_out=[pred_aval.update(dtype=np.bool_)],
        block_shapes=[None],
    )
    pred = lower_fun(lambda x: x != 0, multiple_results=False)(lower_ctx, pred)
  if not args:
    return x
  # Assume x and y, which we check above.
  y, = args
  return arith.select(pred, y, x)


lowering_rules[lax.select_n_p] = _select_n_lowering_rule


def _clamp(min, operand, max):
  res = jnp.maximum(operand, min)
  return jnp.minimum(res, max)


def _clamp_lowering_rule(ctx: LoweringRuleContext, min, operand, max):
  """Compute minimum_p(maximum_p(min, operand), max)."""
  return lower_fun(_clamp, multiple_results=False)(ctx, min, operand, max)


lowering_rules[lax.clamp_p] = _clamp_lowering_rule


def _for_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    jaxpr,
    nsteps,
    reverse,
    unroll,
    which_linear,
):
  should_discharge = [
      not isinstance(aval, state.AbstractRef) for aval in ctx.avals_in
  ]
  jaxpr, () = state_discharge.discharge_state(
      jaxpr, (), should_discharge=[False, *should_discharge]
  )
  for i in range(nsteps):
    if reverse:
      i = nsteps - i - 1
    i = ir_constant(i)
    lowering_context = ctx.lowering_context.replace(
        block_shapes=[(), *ctx.block_shapes],
    )
    non_ref_args = jaxpr_subcomp(lowering_context, jaxpr, i, *args)
    non_ref_args_iter = iter(non_ref_args)
    args = [
        next(non_ref_args_iter) if s else a
        for a, s in zip(args, should_discharge)
    ]
  return args


lowering_rules[for_loop.for_p] = _for_lowering_rule


def _lower_jaxpr_to_for_loop(ctx: LoweringRuleContext,
                             jaxpr: jax_core.Jaxpr, start: int | ir.Value,
                             num_steps: int | ir.Value, consts, *args,
                             has_loop_index: bool,
                             unroll: int):
  def _run_body(i, args):
    if has_loop_index:
      lowering_context = ctx.lowering_context.replace(
          block_shapes=ctx.block_shapes)
      args = jaxpr_subcomp(lowering_context, jaxpr, *consts, i, *args)
    else:
      del i
      lowering_context = ctx.lowering_context.replace(
          block_shapes=ctx.block_shapes[:len(consts)]
          + ctx.block_shapes[len(consts) + 1:],
      )
      args = jaxpr_subcomp(lowering_context, jaxpr, *consts, *args)
    return args

  if (
      not isinstance(start, ir.Value)
      and not isinstance(num_steps, ir.Value)
      and num_steps == unroll
  ):
    # No need for an scf.For. We can just unroll completely
    for i in range(start, start + num_steps):
      args = _run_body(
          ir_constant(i, mlir_type=_dtype_to_ir_type(jnp.dtype("int32"))),
          args,
      )
    return args
  if unroll != 1:
    raise NotImplementedError(
        f"Only unroll={num_steps=} and unroll=1 supported. Got {unroll=}.")
  lbd = _ensure_mlir_value(start, pallas_core.index_map_grid_aval)
  ubd = arith.addi(lbd, _ensure_mlir_value(num_steps, pallas_core.index_map_grid_aval))
  step = ir_constant(1, mlir_type=_dtype_to_ir_type(jnp.dtype("int32")))
  for_op = scf.ForOp(lbd, ubd, step, args)
  with ir.InsertionPoint(for_op.body):
    iv = for_op.induction_variable
    inner_args = for_op.inner_iter_args
    inner_out = _run_body(iv, inner_args)
    scf.YieldOp(inner_out)
  return for_op.results


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
  del _split_transpose
  # Can only handle fori_loop-like scans
  num_extensive = len(args) - num_consts - num_carry
  if num_extensive: raise NotImplementedError
  if reverse: raise NotImplementedError
  del linear, num_extensive, reverse

  jaxpr, jaxpr_consts = jaxpr.jaxpr, jaxpr.consts
  if jaxpr_consts: raise NotImplementedError
  del jaxpr_consts

  jaxpr, has_loop_index = pallas_utils.pattern_match_scan_to_fori_loop(
      jaxpr, num_consts, num_carry
  )
  consts, args = split_list(args, [num_consts])
  consts_avals, args_avals = split_list(ctx.avals_in, [num_consts])
  if has_loop_index:
    loop_index_start, *args = args
    args_avals = args_avals[1:]
  else:
    loop_index_start = 0
  consts = map(_ensure_mlir_value, consts, consts_avals)
  args = map(_ensure_mlir_value, args, args_avals)
  out = _lower_jaxpr_to_for_loop(
      ctx, jaxpr, loop_index_start, length,
      consts, *args, has_loop_index=has_loop_index,
      unroll=unroll)
  if has_loop_index:
    out = [ir_constant(length,
                       mlir_type=_dtype_to_ir_type(jnp.dtype('int32'))),
           *out]
  return out
lowering_rules[lax.scan_p] = _scan_lowering_rule
skip_mlir_conversions.add(lax.scan_p)


def _lower_while_via_fori(
    ctx: LoweringRuleContext,
    *args,
    fori_jaxpr,
    cond_nconsts,
    cond_jaxpr,
    body_nconsts,
    body_jaxpr,
):
  _, body_consts, carry = split_list(args, [cond_nconsts, body_nconsts])
  (lb, ub), args = carry[:2], carry[2:]
  for_out = _lower_jaxpr_to_for_loop(
      ctx.replace(
          block_shapes=ctx.block_shapes[: body_nconsts + 1]
          + ctx.block_shapes[body_nconsts + 2 :],
      ),
      fori_jaxpr,
      lb,
      arith.subi(ub, lb),
      body_consts,
      *args,
      has_loop_index=True,
      unroll=1,
  )
  return [ub, ub, *for_out]


def _while_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    cond_nconsts,
    cond_jaxpr,
    body_nconsts,
    body_jaxpr,
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
        cond_jaxpr=cond_jaxpr,
        body_nconsts=body_nconsts,
        body_jaxpr=body_jaxpr,
    )

  # If we fail conversion to fori, fallback to an ordinary while loop.
  cond_consts, body_consts, carry = split_list(
      args, [cond_nconsts, body_nconsts]
  )
  cond_const_block_shapes, body_const_block_shapes, carry_block_shapes = (
      split_list(ctx.block_shapes, [cond_nconsts, body_nconsts])
  )
  carry_types = [a.type for a in carry]
  while_op = scf.WhileOp(carry_types, carry)

  before_block = while_op.before.blocks.append(*carry_types)
  with ir.InsertionPoint.at_block_begin(before_block):
    cond_args = [*cond_consts, *before_block.arguments]
    [cond] = jaxpr_subcomp(
        ctx.lowering_context.replace(
            block_shapes=[*cond_const_block_shapes, *carry_block_shapes]
        ),
        cond_jaxpr.jaxpr,
        *cond_args,
    )
    scf.condition(cond, before_block.arguments)

  after_block = while_op.after.blocks.append(*carry_types)
  with ir.InsertionPoint.at_block_begin(after_block):
    body_args = [*body_consts, *after_block.arguments]
    loop_out = jaxpr_subcomp(
        ctx.lowering_context.replace(
            block_shapes=[*body_const_block_shapes, *carry_block_shapes],
        ),
        body_jaxpr.jaxpr,
        *body_args,
    )
    if loop_out:
      scf.yield_(loop_out)
  return list(while_op.results)


lowering_rules[lax.while_p] = _while_lowering_rule

def _cond_lowering_rule(ctx: LoweringRuleContext, *args, branches):
  index, *args = args
  constant_index = _fold_and_get_constant_value(index)

  if constant_index is not None:
    return jaxpr_subcomp(
        ctx.lowering_context.replace(block_shapes=ctx.block_shapes[1:]), branches[constant_index].jaxpr, *args
    )
  aval_to_ir_type_with_fn = functools.partial(
      aval_to_ir_type, ctx.lowering_context.dynamic_shape_replacement_fn
  )
  out_types = map(aval_to_ir_type_with_fn, ctx.avals_out)
  pred = arith.cmpi(
      arith.CmpIPredicate.ne, index, ir_constant(0, index.type)
  )
  if_op = scf.IfOp(pred, out_types, hasElse=True)
  lowering_context = ctx.lowering_context.replace(
      block_shapes=ctx.block_shapes[1:],
  )
  with ir.InsertionPoint(if_op.then_block):
    # TODO(b/300272065): Use `scf.IndexSwitchOp` instead of a cascade of
    # if/else.
    if len(branches) > 2:
      out = _cond_lowering_rule(
          ctx,
          arith.subi(index, ir_constant(1, index.type)),
          *args,
          branches=branches[1:],
      )
    else:
      out = jaxpr_subcomp(lowering_context, branches[1].jaxpr, *args)
    scf.YieldOp(out)
  with ir.InsertionPoint(if_op.else_block):
    out = jaxpr_subcomp(lowering_context, branches[0].jaxpr, *args)
    scf.YieldOp(out)
  return if_op.results


lowering_rules[lax.cond_p] = _cond_lowering_rule


def _pjit_lowering_rule(ctx: LoweringRuleContext, *args, jaxpr, **_):
  lowering_context = ctx.lowering_context.replace(block_shapes=ctx.block_shapes)
  return jaxpr_subcomp(lowering_context, jaxpr.jaxpr, *args)


lowering_rules[pjit.pjit_p] = _pjit_lowering_rule


def _custom_jvp_call_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    call_jaxpr: jax_core.Jaxpr,
    jvp_jaxpr_thunk: Callable,
    num_consts: int,
    symbolic_zeros: bool,
):
  del jvp_jaxpr_thunk
  if symbolic_zeros: raise NotImplementedError
  if num_consts: raise NotImplementedError
  if call_jaxpr.consts: raise NotImplementedError
  lowering_context = ctx.lowering_context.replace(block_shapes=ctx.block_shapes)
  return jaxpr_subcomp(lowering_context, call_jaxpr.jaxpr, *args)


lowering_rules[custom_derivatives.custom_jvp_call_p] = (
    _custom_jvp_call_lowering_rule)


def _debug_callback_lowering_rule(ctx: LoweringRuleContext, *args, **kwargs):
  del ctx, args, kwargs
  # No-op debug callbacks in Mosaic for now
  return []


lowering_rules[debugging.debug_callback_p] = _debug_callback_lowering_rule


def _program_id_lowering_rule(ctx: LoweringRuleContext, *, axis: int):

  if ctx.lowering_context.user_grid_indices is None:
    raise ValueError(
        f"program id: {axis} was passed, but user did not provide a grid."
    )
  length = len(ctx.lowering_context.user_grid_indices)
  if not (0 <= axis < length):
    raise ValueError(
        f"user passed in program id with axis: {axis}, but grid only has"
        f" length: {length}"
    )
  return ctx.lowering_context.user_grid_indices[axis]
lowering_rules[primitives.program_id_p] = _program_id_lowering_rule

def _num_programs_lowering_rule(ctx: LoweringRuleContext, *, axis: int):
  mapped_axes = set(ctx.lowering_context.mapped_dims)
  seen_user_axes = 0
  for i in range(ctx.lowering_context.grid_rank):
    seen_user_axes += int(i not in mapped_axes)
    if seen_user_axes == axis + 1:
      break
  else:
    raise ValueError(
        f"user passed in program id with axis: {axis}, but grid only has"
        f" length: {len(ctx.lowering_context.grid_rank)}"
    )
  return tpu.iteration_bound(i)
lowering_rules[primitives.num_programs_p] = _num_programs_lowering_rule


def _repeat_lowering_rule(ctx: LoweringRuleContext, x, *, repeats, axis):
  (out_aval,) = ctx.avals_out
  return tpu.repeat(
      aval_to_ir_type(
          ctx.lowering_context.dynamic_shape_replacement_fn, out_aval
      ),
      x,
      axis,
      repeats,
  )


lowering_rules[tpu_primitives.repeat_p] = _repeat_lowering_rule


def _roll_lowering_rule(
    ctx: LoweringRuleContext, x, shift, *, axis, stride, stride_axis
):
  (out_aval,) = ctx.avals_out
  return tpu.dynamic_rotate(
      aval_to_ir_type(
          ctx.lowering_context.dynamic_shape_replacement_fn, out_aval
      ),
      x,
      shift,
      axis,
      stride=stride,
      stride_dimension=stride_axis,
  )


lowering_rules[tpu_primitives.roll_p] = _roll_lowering_rule


def _slice_lowering_rule(
    ctx: LoweringRuleContext, x, limit_indices, start_indices, strides
):
  """Lowers a slice to vector dialect."""
  (aval_out,) = ctx.avals_out
  out_type = aval_to_ir_type(
      ctx.lowering_context.dynamic_shape_replacement_fn, aval_out
  )
  if strides is None:
    strides = [1] * len(start_indices)
  sizes = np.array(limit_indices) - np.array(start_indices)
  return vector.extract_strided_slice(
      out_type, x, start_indices, sizes, strides
  )


lowering_rules[lax.slice_p] = _slice_lowering_rule


def _xor_lowering_rule(ctx: LoweringRuleContext, x, y):
  x, y = _bcast(x, y, *ctx.avals_in, *ctx.avals_out)
  return arith.xori(x, y)


lowering_rules[lax.xor_p] = _xor_lowering_rule
skip_mlir_conversions.add(lax.xor_p)


def _shift_left_lowering_rule(ctx: LoweringRuleContext, x, d):
  x, d = _bcast(x, d, *ctx.avals_in, *ctx.avals_out)
  return arith.shli(x, d)


lowering_rules[lax.shift_left_p] = _shift_left_lowering_rule
skip_mlir_conversions.add(lax.shift_left_p)


def _shift_right_arithmetic_lowering_rule(ctx: LoweringRuleContext, x, d):
  x, d = _bcast(x, d, *ctx.avals_in, *ctx.avals_out)
  return arith.shrsi(x, d)


lowering_rules[lax.shift_right_arithmetic_p] = _shift_right_arithmetic_lowering_rule
skip_mlir_conversions.add(lax.shift_right_arithmetic_p)


def _shift_right_logical_lowering_rules(ctx: LoweringRuleContext, x, d):
  x, d = _bcast(x, d, *ctx.avals_in, *ctx.avals_out)
  return arith.shrui(x, d)


lowering_rules[lax.shift_right_logical_p] = _shift_right_logical_lowering_rules
skip_mlir_conversions.add(lax.shift_right_logical_p)


def _erf_inv_lowering_rule(ctx: LoweringRuleContext, x):
  return lower_fun(
      pallas_utils.erf_inv_lowering_helper, multiple_results=False,
  )(ctx, x)


lowering_rules[lax.erf_inv_p] = _erf_inv_lowering_rule


def _bitcast_lowering_rule(ctx: LoweringRuleContext, x, *, ty):
  del ty
  (out_aval,) = ctx.avals_out
  return tpu.bitcast(
      aval_to_ir_type(
          ctx.lowering_context.dynamic_shape_replacement_fn, out_aval
      ),
      x,
  )

lowering_rules[tpu_primitives.bitcast_p] = _bitcast_lowering_rule

def _bitcast_convert_type_lowering_rule(
    ctx: LoweringRuleContext, x, *, new_dtype):
  (in_aval, ) = ctx.avals_in
  (out_aval,) = ctx.avals_out
  old_bitwidth = pallas_utils.dtype_bitwidth(in_aval.dtype)
  new_bitwidth = pallas_utils.dtype_bitwidth(new_dtype)
  if old_bitwidth != new_bitwidth:
    raise NotImplementedError("Changing bitwidths not supported.")
  return tpu.bitcast(
      aval_to_ir_type(
          ctx.lowering_context.dynamic_shape_replacement_fn, out_aval
      ),
      x,
  )
lowering_rules[lax.bitcast_convert_type_p] = _bitcast_convert_type_lowering_rule


def _alloc_value(
    aval: jax_core.AbstractValue, *, ctx: LoweringRuleContext
) -> ir.Value:
  if isinstance(aval, pallas_core.AbstractMemoryRef):
    memspace = _memory_space_to_mosaic_attribute(aval.memory_space)
    if jnp.issubdtype(aval.dtype, tpu_core.semaphore_dtype):
      assert aval.memory_space == TPUMemorySpace.SEMAPHORE
      memref_type = aval_to_ir_type(
          ctx.lowering_context.dynamic_shape_replacement_fn,
          aval,
          memory_space=TPUMemorySpace.SEMAPHORE,
      )
      return tpu.sem_alloc(memref_type)
    else:
      out_type = ir.MemRefType.get(
          aval.shape,
          _dtype_to_ir_type(aval.dtype, is_kernel_boundary=True),
          memory_space=memspace)
      return memref.alloca(out_type, [], [])
  elif isinstance(aval, tpu_core.AbstractSemaphore):
    memref_type = aval_to_ir_type(
        ctx.lowering_context.dynamic_shape_replacement_fn,
        aval,
        memory_space=TPUMemorySpace.SEMAPHORE,
    )
    return tpu.sem_alloc(memref_type)
  raise NotImplementedError(f"Cannot allocate {type(aval)}.")


def _run_scoped_lowering_rule(ctx: LoweringRuleContext, *consts, jaxpr):
  out_type = [
      aval_to_ir_type(ctx.lowering_context.dynamic_shape_replacement_fn, aval)
      for aval in ctx.avals_out
  ]
  region = tpu.RegionOp(out_type)
  in_avals = [v.aval for v in jaxpr.invars]
  with ctx.lowering_context.grid_name_context():
    jaxpr = pe.convert_constvars_jaxpr(jaxpr)
  with ir.InsertionPoint(region.body):
    alloc_fn = functools.partial(_alloc_value, ctx=ctx)
    args = map(alloc_fn, in_avals)
    block_shapes = tuple(a.shape if isinstance(a, state.AbstractRef) else None
                         for a in in_avals)
    ctx = ctx.lowering_context.replace(
        block_shapes=(*ctx.block_shapes, *block_shapes)
    )
    out = jaxpr_subcomp(ctx, jaxpr, *consts, *args)
    tpu.YieldOp(out)
  return region.results


lowering_rules[primitives.run_scoped_p] = _run_scoped_lowering_rule

def _device_id_to_logical(
    ctx: LoweringRuleContext, device_id,
    device_id_type: tpu_primitives.DeviceIdType):
  if device_id_type is tpu_primitives.DeviceIdType.MESH:
    # Mesh means we are passed the mesh coordinates for the device
    device_ids = tree_util.tree_leaves(device_id)
    mesh_strides = ctx.lowering_context.mesh_context.mesh_strides

    i32 = ir.IntegerType.get_signless(32)
    if len(device_ids) == 0:
      return arith.constant(i32, 0)
    return functools.reduce(
        arith.addi,
        (
            arith.muli(a, arith.constant(i32, b))
            for a, b in zip(device_ids, mesh_strides)
        ),
    )
  elif device_id_type is tpu_primitives.DeviceIdType.LOGICAL:
    return device_id
  raise NotImplementedError(f"Unsupported device id type: {device_id_type}")


def _semaphore_read_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    args_tree,
):
  sem_aval, _ = tree_util.tree_unflatten(args_tree, ctx.avals_in)
  sem, transforms = tree_util.tree_unflatten(args_tree, args)
  sem, _ = _transform_ref(sem, sem_aval.dtype, sem_aval.shape, transforms)
  return tpu.sem_read(sem)


lowering_rules[tpu_primitives.semaphore_read_p] = _semaphore_read_lowering_rule

def _semaphore_signal_lowering_rule(
    ctx: LoweringRuleContext,
    *args,
    args_tree,
    device_id_type: tpu_primitives.DeviceIdType,
):
  sem_aval, _, _, _, _ = tree_util.tree_unflatten(args_tree, ctx.avals_in)
  sem, transforms, value, device_id, core_index = tree_util.tree_unflatten(
      args_tree, args
  )
  sem, _ = _transform_ref(sem, sem_aval.dtype, sem_aval.shape, transforms)
  if device_id is not None:
    device_id = _device_id_to_logical(ctx, device_id, device_id_type)
  tpu.sem_signal(sem, value, device_id=device_id, core_id=core_index)
  return []


lowering_rules[tpu_primitives.semaphore_signal_p] = (
    _semaphore_signal_lowering_rule)


def _semaphore_wait_lowering_rule(ctx: LoweringRuleContext, *args, args_tree):
  sem_aval, _, _ = tree_util.tree_unflatten(args_tree, ctx.avals_in)
  sem, transforms, value = tree_util.tree_unflatten(args_tree, args)
  sem, _ = _transform_ref(sem, sem_aval.dtype, sem_aval.shape, transforms)
  tpu.sem_wait(sem, value)
  return []
lowering_rules[tpu_primitives.semaphore_wait_p] = _semaphore_wait_lowering_rule

def _dma_start_lowering_rule(ctx: LoweringRuleContext, *args, tree,
                             device_id_type: tpu_primitives.DeviceIdType):
  (
      src_ref,
      src_transforms,
      dst_ref,
      dst_transforms,
      sem,
      sem_transforms,
      src_sem,
      src_sem_transforms,
      device_id,
  ) = tree_util.tree_unflatten(tree, args)
  (src_ref_aval, _, dst_ref_aval, _, sem_aval, _, src_sem_aval, _, _) = (
      tree_util.tree_unflatten(tree, ctx.avals_in)
  )
  if src_ref_aval.dtype == jnp.bool_:
    raise NotImplementedError("DMAs with bool dtypes are not supported.")
  block_shapes = tree_util.tree_unflatten(tree, ctx.block_shapes)
  src_ref_block_shape, dst_ref_block_shape = block_shapes[0], block_shapes[2]
  src_ref, _ = _transform_ref(
      src_ref, src_ref_aval.dtype, src_ref_block_shape, src_transforms
  )
  if src_sem is not None:
    src_sem, _ = _transform_ref(
        src_sem, src_sem_aval.dtype, src_sem_aval.shape, src_sem_transforms
    )
  dst_ref, _ = _transform_ref(
      dst_ref, dst_ref_aval.dtype, dst_ref_block_shape, dst_transforms
  )
  sem, _ = _transform_ref(sem, sem_aval.dtype, sem_aval.shape, sem_transforms)
  if device_id is not None:
    device_id = _device_id_to_logical(ctx, device_id, device_id_type)
  tpu.enqueue_dma(src_ref, dst_ref, sem, source_semaphore=src_sem,
                  device_id=device_id)
  return []
lowering_rules[tpu_primitives.dma_start_p] = _dma_start_lowering_rule


def _dma_wait_lowering_rule(ctx: LoweringRuleContext, *args, tree,
                            device_id_type: tpu_primitives.DeviceIdType):
  del device_id_type
  (_, _, ref, transforms, sem, sem_transforms, _, _, _) = tree_util.tree_unflatten(
      tree, args)
  (_, _, ref_aval, _, sem_aval, _, _, _, _) = tree_util.tree_unflatten(
      tree, ctx.avals_in)
  block_shapes = tree_util.tree_unflatten(tree, ctx.block_shapes)
  ref_block_shape = block_shapes[2]
  ref, _ = _transform_ref(ref, ref_aval.dtype, ref_block_shape, transforms)
  sem, _ = _transform_ref(sem, sem_aval.dtype, sem_aval.shape, sem_transforms)
  tpu.wait_dma(sem, ref)
  return []
lowering_rules[tpu_primitives.dma_wait_p] = _dma_wait_lowering_rule

def _device_id_lowering_rule(ctx: LoweringRuleContext):
  return tpu.device_id()
lowering_rules[tpu_primitives.device_id_p] = _device_id_lowering_rule

def _axis_index_rule(ctx: LoweringRuleContext, *, axis_name: Hashable):
  grid_names = ctx.lowering_context.grid_names
  if grid_names and axis_name in grid_names:
    # We are querying a named axis corresponding to a grid dimension.
    return _program_id_lowering_rule(ctx, axis=grid_names.index(axis_name))
  # We are querying a named axis corresponding to a mesh dimension.
  device_id = tpu.device_id()
  mesh_context = ctx.lowering_context.mesh_context
  if mesh_context is None:
    raise ValueError("Mesh context is not set.")
  mesh_shape = mesh_context.mesh_shape
  axis_names = mesh_context.axis_names
  axis_index = axis_names.index(axis_name)
  axis_size = ir_constant(mesh_shape[axis_index])
  minor_divisor = ir_constant(
      np.prod(mesh_shape[axis_index + 1 :], dtype=np.int32)
  )
  return arith.remsi(arith.divsi(device_id, minor_divisor), axis_size)
lowering_rules[lax.axis_index_p] = _axis_index_rule

def _get_barrier_semaphore_rule(ctx: LoweringRuleContext):
  memref_type = aval_to_ir_type(
      ctx.lowering_context.dynamic_shape_replacement_fn, ctx.avals_out[0]
  )
  return tpu.sem_barrier(memref_type)
lowering_rules[tpu_primitives.get_barrier_semaphore_p] = _get_barrier_semaphore_rule


def _delay_rule(ctx: LoweringRuleContext, nanos: int):
  tpu.delay(nanos)
  return []


lowering_rules[tpu_primitives.delay_p] = _delay_rule


def _debug_print_rule(
    ctx: LoweringRuleContext, *args, fmt: str, has_placeholders: bool
):
  is_scalar_inputs = [aval.shape == () for aval in ctx.avals_in]
  is_all_scalars = all(is_scalar_inputs)
  is_single_vector = len(is_scalar_inputs) == 1 and not is_scalar_inputs[0]
  if not (is_all_scalars or is_single_vector):
    raise ValueError(
        "All inputs to debug_print must be all scalars or a single vector, but"
        f" got {ctx.avals_in}"
    )

  # Scalar case.
  if is_all_scalars:
    primitives.check_debug_print_format(fmt, *args)
    if has_placeholders:
      if not all(
          isinstance(arg.type, ir.IntegerType) and arg.type.width == 32
          for arg in args
      ):
        raise TypeError(
            "All arguments must be 32-bit integers when using"
            " placeholders (`{...}`). If you need to print values of other types,"
            " remove placeholders from the format string."
        )

    # TPU expects $0, $1 etc as placeholders.
      fmt = "".join(
          f"{text}${idx}"
          for idx, (text, _, _, _) in enumerate(string.Formatter().parse(fmt))
      )

    tpu.log(args, fmt, formatted=has_placeholders)
    return ()

  # Vector case.
  # Copy the array to vmem for logging.
  # Note that the shape of the array must be explicitly provided here. This is
  # because the underlying implementation aligns shapes to tile boundaries,
  # potentially altering the original shape and making it unrecoverable.
  if len(ctx.avals_in) != 1:
    raise ValueError(
        "Only one vector input to debug_print is supported."
    )
  (aval,) = ctx.avals_in
  (arg,) = args

  if not has_placeholders or not fmt.endswith("{}"):
    raise ValueError("For vector input, the format string must end with {}.")

  fmt = fmt[:-2]

  region = tpu.RegionOp(())
  with ir.InsertionPoint(region.body):
    element_type = _dtype_to_ir_type(aval.dtype)
    ref_type = ir.MemRefType.get(
        aval.shape,
        element_type,
        memory_space=ir.Attribute.parse("#tpu.memory_space<vmem>"),
    )
    ref = memref.alloca(ref_type, [], [])

    index_type = ir.IndexType.get()
    zero = arith.constant(index_type, 0)
    indices = [zero] * len(aval.shape)
    vector.store(arg, ref, indices)
    tpu.log_buffer(ref, aval.shape, fmt)
    tpu.yield_([])
  return ()


lowering_rules[primitives.debug_print_p] = _debug_print_rule


def _prng_seed_lowering_rule(ctx: LoweringRuleContext, *seeds):
  del ctx
  # In the KeyScalarBundle case we unpack the bundle and set the seed with
  # the list of scalars.
  if len(seeds) == 1 and isinstance(seeds[0], KeyScalarBundle):
    tpu.prng_set_seed_32(seeds[0].scalars)
    return []
  # For integer seeds, we can set the seed directly as PRNGSeed32Op natively
  # takes in a list of integers as input.
  all_integers = all(isinstance(seed.type, ir.IntegerType) for seed in seeds)
  if not all_integers:
    seed_types = [seed.type for seed in seeds]
    raise ValueError(f"All seed data must be scalar integers. Got {seed_types}")
  tpu.prng_set_seed_32(seeds)
  return []
lowering_rules[tpu_primitives.prng_seed_p] = _prng_seed_lowering_rule


def _prng_random_bits_lowering_rule(ctx: LoweringRuleContext, *, shape):
  if len(shape) <= 1:
    # TODO(b/342054464): Support implicit dims for PRNGRandomBitsOp.
    raise NotImplementedError("random_bits only supports rank>=2 outputs.")
  out_aval = ctx.avals_out[0]
  out_type = aval_to_ir_type(
      ctx.lowering_context.dynamic_shape_replacement_fn, out_aval
  )
  return tpu.prng_random_bits(out_type)
lowering_rules[tpu_primitives.prng_random_bits_p] = _prng_random_bits_lowering_rule


def random_seed_lowering(ctx, seeds, *, impl):
  seed_lowering = lower_fun(impl.seed, multiple_results=False)
  return seed_lowering(ctx, seeds)
lowering_rules[prng.random_seed_p] = random_seed_lowering


def random_bits_lowering(ctx, keys, *, bit_width, shape):
  assert bit_width == 32, "Only 32-bit PRNG supported."
  aval, = ctx.avals_in
  impl = aval.dtype._impl
  _proxy_fn = impl.random_bits
  if not pl_random.is_pallas_impl(impl):
    def new_lowering(key, bit_width, shape):
      key = jax.random.key_data(key).astype(jnp.uint32)
      return impl.random_bits(key, bit_width, shape)
    _proxy_fn = new_lowering
  bits_lowering = lower_fun(_proxy_fn, multiple_results=False)
  return bits_lowering(ctx, keys, bit_width=bit_width, shape=shape)
lowering_rules[prng.random_bits_p] = random_bits_lowering


def random_fold_in_lowering(ctx, keys, msgs):
  keys_aval, _ = ctx.avals_in
  impl = keys_aval.dtype._impl
  fold_in_lowering = lower_fun(impl.fold_in, multiple_results=False)
  return fold_in_lowering(ctx, keys, msgs)
lowering_rules[prng.random_fold_in_p] = random_fold_in_lowering


def random_unwrap_lowering(ctx, key):
  keys_aval = ctx.avals_in[0]
  impl = keys_aval.dtype._impl
  if not pl_random.is_pallas_impl(impl):
    return key
  assert isinstance(key, KeyScalarBundle)
  # Convert to a vector.
  if tuple(key.key_shape) != (1, 1):
    raise NotImplementedError(
      "Seed key_data of shape != (1, 1) not supported. "
      f"Got: {key.key_shape}")
  scalar = key.scalars[0]
  out_type = ir.VectorType.get(
      key.key_shape, _dtype_to_ir_type(jnp.dtype('int32'))
  )
  val = vector.broadcast(out_type, scalar)
  return val
lowering_rules[prng.random_unwrap_p] = random_unwrap_lowering


def random_wrap_lowering(ctx, key_data, *, impl):
  del ctx
  if not pl_random.is_pallas_impl(impl):
    return key_data
  if isinstance(key_data.type, ir.VectorType):
    # If the key data lives in vregs, need to unpack it to sregs.
    key_data_list = []
    key_data_shape = key_data.type.shape
    if len(key_data_shape) != 2:
      raise NotImplementedError("Seed key_data must be 2D.")
    if tuple(key_data_shape) != (1, 1):
      raise NotImplementedError(
        "Seed key_data of shape != (1, 1) not supported. "
        f"Got: {key_data_shape}")
    for i in range(key_data_shape[1]):
      key_data_list.append(vector.ExtractOp(key_data, [], [0, i]))
    return KeyScalarBundle(
        scalars=key_data_list, key_shape=tuple(key_data_shape))
  if isinstance(key_data, KeyScalarBundle):
    return key_data
  else:
    raise NotImplementedError(f"key_data wrap {type(key_data)}")

lowering_rules[prng.random_wrap_p] = random_wrap_lowering

def _checkify_lowering_rule(
    ctx: LoweringRuleContext, *err_args, err_tree, debug):
  if not tpu_core.runtime_assert_enabled():
    if debug:
      return []
    else:
      raise LoweringException("Non-debug check must be functionalized. "
                              "Enable runtime asserts with "
                              "--jax_pallas_enable_runtime_assert "
                              "or functionalize with checkify.check.")

  assert ctx.lowering_context.ir_context.allow_unregistered_dialects, (
    "allow_unregistered_dialects must be set to True for "
    "runtime assert check.")
  error = jax.tree.unflatten(err_tree, err_args)
  assert len(error._pred) == 1
  assert len(error._metadata) == 1
  assert len(error._payload) == 1
  pred = list(error._pred.items())[0][1]
  metadata = list(error._metadata.items())[0]
  payload = list(error._payload.items())[0][1]
  exception_tree = metadata[1]
  exception = jax.tree.unflatten(exception_tree, payload)
  assert isinstance(exception, checkify.FailedCheckError)

  # check_p has an inverted predicate compared to assert,
  # so we need to compute not(pred) here.
  out_scalar_type = _dtype_to_ir_type(jnp.dtype('bool'))
  minus_one = ir_constant(-1, out_scalar_type)
  not_pred = arith.xori(pred, minus_one)
  attrs = {"msg": ir.StringAttr.get(exception.fmt_string)}
  ir.Operation.create("cf.assert",
                      operands=(not_pred,),
                      attributes=attrs)
  return []
lowering_rules[checkify.check_p] = _checkify_lowering_rule

def _threefry2x32_lowering(ctx, k1, k2, m1, m2):
  def _lower_fun(k1, k2, m1, m2):
    with jax.named_scope("threefry2x32"):
      res = prng._threefry2x32_lowering(k1, k2, m1, m2, use_rolled_loops=False)
    return res

  threefry_lowering = lower_fun(_lower_fun, multiple_results=True)
  return threefry_lowering(ctx, k1, k2, m1, m2)


lowering_rules[prng.threefry2x32_p] = _threefry2x32_lowering


def _iota_2x32_shape_lowering(ctx, *, shape):
  total_elements = np.prod(shape)
  if total_elements > np.iinfo(jnp.int32).max:
    raise NotImplementedError(f"Iota with >{np.iinfo(jnp.int32).max} items.")

  def _lower_fun(shape):
    iota_data = jnp.zeros(shape, dtype=jnp.int32)
    multiplier = 1
    for dim in range(len(shape)-1, -1, -1):
      counts_lo = lax.broadcasted_iota(
          dtype=jnp.int32, shape=shape, dimension=dim
      )
      iota_data += counts_lo * multiplier
      multiplier *= shape[dim]
    counts_hi = jnp.zeros(shape, dtype=jnp.int32)
    return counts_hi, iota_data

  iota_lowering = lower_fun(_lower_fun, multiple_results=True)
  return iota_lowering(ctx, shape=shape)


lowering_rules[prng.iota_2x32_shape_p] = _iota_2x32_shape_lowering


def _pad_lowering_rule(ctx: LoweringRuleContext, *args, **kwargs):
  operand, padding_value = args
  padding_config = kwargs["padding_config"]

  out_type: ir.VectorType = aval_to_ir_type(
      ctx.lowering_context.dynamic_shape_replacement_fn, ctx.avals_in[0]
  )
  if not isinstance(out_type, ir.VectorType):
    raise NotImplementedError("Only vector types are supported.")

  for axis, (low, high, interior) in enumerate(padding_config):
    if low == 0 and high == 0 and interior == 0:
      continue

    def _pad(val):
      shape = list(operand.type.shape)
      shape[axis] = val
      pad_vec_type = ir.VectorType.get(
          shape,
          operand.type.element_type,
      )

      if isinstance(padding_value, ir.OpResult):
        pad = vector.broadcast(pad_vec_type, padding_value)
      else:
        scalar_attr = ir.FloatAttr.get(operand.type.element_type, padding_value)
        pad = arith.ConstantOp(
            pad_vec_type,
            ir.DenseElementsAttr.get_splat(
                pad_vec_type,
                scalar_attr,
            ),
        ).result
      return pad

    if low != 0:
      pad_low = _pad(low)
      new_shape = out_type.shape
      new_shape[axis] += low
      out_type = ir.VectorType.get(
          new_shape,
          out_type.element_type,
      )
      operand = tpu.concatenate(out_type, [pad_low, operand], dimension=axis)

    if high != 0:
      pad_high = _pad(high)
      new_shape = out_type.shape
      new_shape[axis] += high
      out_type = ir.VectorType.get(
          new_shape,
          out_type.element_type,
      )
      operand = tpu.concatenate(out_type, [operand, pad_high], dimension=axis)

    if interior > 0:
      raise NotImplementedError("Not implemented: interior padding")

  return operand


lowering_rules[lax.pad_p] = _pad_lowering_rule


def _platform_index_lowering(
    ctx: mlir.LoweringRuleContext,
    *,
    platforms: Sequence[Sequence[str]],
    has_default: bool,
):
  for i, ps in enumerate(platforms):
    # note - slightly odd structure here, as platforms is a seq[seq[str]]
    if "mosaic" in ps:
      return ir_constant(i)

  if has_default:
    return ir_constant(len(platforms))

  raise NotImplementedError(
      "No mosaic or default platform indexing rule found."
  )


lowering_rules[jax._src.lax.control_flow.platform_index_p] = _platform_index_lowering
