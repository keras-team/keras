# Copyright 2024 The JAX Authors. All Rights Reserved.
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
# ==============================================================================
"""Utilities for code generator."""

from collections.abc import Iterator, Sequence
import contextlib
import dataclasses
import enum
import functools
import math
from typing import Any, Literal

import jax
from jax import numpy as jnp
from jax.interpreters import mlir
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import builtin
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import nvvm
from jaxlib.mlir.dialects import scf
from jaxlib.mlir.dialects import vector
import numpy as np

from jax._src.lib import mosaic_gpu_dialect as dialect  # noqa: F401

# mypy: ignore-errors

WARPGROUP_SIZE: int = 128
DYNAMIC = -9223372036854775808
DYNAMIC32 = -2147483648
MBARRIER_BYTES = 8

# pylint: disable=line-too-long, wildcard-import, missing-function-docstring, bad-continuation, g-bad-todo, protected-access, g-explicit-length-test, missing-class-docstring, g-doc-return-or-yield, g-inconsistent-quotes


def gpu_address_space_to_nvptx(address_space: gpu.AddressSpace) -> int:
  match address_space:
    case gpu.AddressSpace.Global:
      return 1
    case gpu.AddressSpace.Workgroup:
      return 3
    case _:
      raise NotImplementedError(f"address_space not supported: {address_space}")


WORKGROUP_NVPTX_ADDRESS_SPACE = gpu_address_space_to_nvptx(
    gpu.AddressSpace.Workgroup
)


def ptr_as_memref(ptr, memref_ty: ir.MemRefType, ptr_memory_space: int | None = None):
  i64 = ir.IntegerType.get_signless(64)
  rank = len(memref_ty.shape)
  ptr_ty = "ptr" if ptr_memory_space is None else f"ptr<{ptr_memory_space}>"
  if rank > 0:
    desc_ty = ir.Type.parse(
        f"!llvm.struct<({ptr_ty}, {ptr_ty}, i64, array<{rank} x i64>, array<{rank} x i64>)>"
    )
  else:
    desc_ty = ir.Type.parse(f"!llvm.struct<({ptr_ty}, {ptr_ty}, i64)>")
  desc = llvm.UndefOp(desc_ty)
  desc = llvm.InsertValueOp(desc, ptr, [0])  # Allocation
  desc = llvm.InsertValueOp(desc, ptr, [1])  # Aligned Base
  desc = llvm.InsertValueOp(
      desc, llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, 0)), [2]
  )
  if rank > 0:
    for i, s in enumerate(memref_ty.shape):
      desc = llvm.InsertValueOp(
          desc, llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, s)), [3, i]
      )
    for i, s in enumerate(get_contiguous_strides(memref_ty.shape)):
      desc = llvm.InsertValueOp(
          desc, llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, s)), [4, i]
      )
  return builtin.unrealized_conversion_cast([memref_ty], [desc])


def pack_array(values):
  if not values:
    raise ValueError("Empty array")
  elem_ty = values[0].type
  i64 = ir.IntegerType.get_signless(64)
  ptr_ty = ir.Type.parse("!llvm.ptr")
  arr_ptr = llvm.alloca(ptr_ty, c(len(values), i64), elem_ty)
  for i, v in enumerate(values):
    elem_ptr = llvm.getelementptr(ptr_ty, arr_ptr, [], [i], elem_ty)
    llvm.store(v, elem_ptr)
  return arr_ptr


def get_contiguous_strides(xs):
  strides_ret = []
  stride = 1
  for x in xs[::-1]:
    strides_ret.append(stride)
    stride *= x
  return strides_ret[::-1]


def c(val: int | float, ty):
  if ir.IntegerType.isinstance(ty) or ir.IndexType.isinstance(ty):
    if not isinstance(val, (int, np.integer)):
      raise TypeError(type(val))
    attr = ir.IntegerAttr.get(ty, val)
  elif ir.FloatType.isinstance(ty):
    attr = ir.FloatAttr.get(ty, val)
  elif ir.VectorType.isinstance(ty):
    return vector.splat(ty, c(val, ir.VectorType(ty).element_type))
  else:
    raise NotImplementedError(ty)
  return arith.constant(ty, attr)

def _debug_scalar_ty_format(arg):
  if ir.IndexType.isinstance(arg.type):
    return "%llu", arg
  if ir.IntegerType.isinstance(arg.type):
    if ir.IntegerType(arg.type).width < 64:
      arg = arith.extui(ir.IntegerType.get_signless(64), arg)
    return "%llu", arg
  if ir.F32Type.isinstance(arg.type):
    return "%f", arg
  if ir.F16Type.isinstance(arg.type):
    arg = arith.extf(ir.F32Type.get(), arg)
    return "%f", arg
  raise NotImplementedError(f"Can't print the type {arg.type}")

def debug_print(fmt, *args, uniform=True):
  type_formats = []
  new_args = []
  for arg in args:
    if ir.VectorType.isinstance(arg.type):
      index = ir.IndexType.get()
      vec_ty = ir.VectorType(arg.type)
      if len(vec_ty.shape) > 1:
        raise NotImplementedError(vec_ty)
      vec_args = [
          vector.extractelement(arg, position=c(i, index))
          for i in range(vec_ty.shape[0])
      ]
      ty_formats, args = zip(*map(_debug_scalar_ty_format,vec_args))
      ty_format = f"[{','.join(ty_formats)}]"
      new_args += args
    else:
      ty_format, arg = _debug_scalar_ty_format(arg)
      new_args.append(arg)

    if ty_format is None:
      raise NotImplementedError(arg.type)
    type_formats.append(ty_format)
  ctx = (
      functools.partial(single_thread, per_block=False)
      if uniform
      else contextlib.nullcontext
  )
  with ctx():
    gpu.printf(fmt.format(*type_formats) + "\n", new_args)


@dataclasses.dataclass(frozen=True)
class ForResult:
  op: scf.ForOp
  results: tuple[Any, ...]

  @property
  def result(self):
    if len(self.results) != 1:
      raise ValueError
    return self.results[0]


def fori(bound, carrys):
  unwrap = False
  if not isinstance(carrys, (list, tuple)):
    carrys = [carrys]
    unwrap = True
  flat_carrys, carry_treedef = jax.tree.flatten(carrys)

  def wrapper(f):
    c0 = arith.constant(bound.type, 0)
    c1 = arith.constant(bound.type, 1)
    for_op = scf.ForOp(c0, bound, c1, flat_carrys)
    with ir.InsertionPoint(for_op.body):
      i = for_op.induction_variable
      inner_carrys = jax.tree.unflatten(carry_treedef, for_op.inner_iter_args)
      if unwrap:
        [inner_carrys] = inner_carrys
      new_carrys = f(i, inner_carrys)
      if unwrap:
        new_carrys = [new_carrys]
      new_flat_carrys, new_carry_treedef = jax.tree.flatten(new_carrys)
      if new_carry_treedef != carry_treedef:
        raise ValueError(new_carry_treedef, carry_treedef)
      scf.YieldOp(new_flat_carrys)
    final_flat_carrys = for_op.results
    return ForResult(
        for_op, jax.tree.unflatten(carry_treedef, final_flat_carrys)
    )

  return wrapper


@contextlib.contextmanager
def when(cond):
  with ir.InsertionPoint(scf.IfOp(cond).then_block):
    yield
    scf.yield_([])


def thread_idx():
  i32 = ir.IntegerType.get_signless(32)
  as_i32 = lambda x: arith.index_cast(i32, x)
  tidx = as_i32(gpu.thread_id(gpu.Dimension.x))
  stride = as_i32(gpu.block_dim(gpu.Dimension.x))
  for dim in (gpu.Dimension.y, gpu.Dimension.z):
    tidx = arith.addi(tidx, arith.muli(as_i32(gpu.thread_id(dim)), stride))
    stride = arith.muli(stride, as_i32(gpu.block_dim(dim)))
  return tidx


def _warp_bcast(val, lane_idx=0):
  i32 = ir.IntegerType.get_signless(32)
  mask = c(0xFFFFFFFF, i32)
  return nvvm.shfl_sync(
      val.type, mask, val, c(lane_idx, i32), c(0x1F, i32), nvvm.ShflKind.idx
  )


def warp_idx(sync=True):
  i32 = ir.IntegerType.get_signless(32)
  warp_idx = arith.shrui(thread_idx(), c(5, i32))
  # Performing a warp broadcast improves performance as compiler understands
  # that the value is uniform across the warp.
  return _warp_bcast(warp_idx) if sync else warp_idx


def warpgroup_idx(sync=True):
  i32 = ir.IntegerType.get_signless(32)
  wg_idx = arith.shrui(thread_idx(), c(7, i32))
  # Performing a warp broadcast improves performance as compiler understands
  # that the value is uniform across the warp.
  return _warp_bcast(wg_idx) if sync else wg_idx


class ThreadSubset(enum.IntEnum):
  WARPGROUP = enum.auto()
  BLOCK = enum.auto()


# True withon `once()` contexts.
_ONCE_PER: ThreadSubset | None = None


def single_thread_predicate(per_block=True):
  warp = warp_idx()
  if not per_block:
    warp = arith.remui(warp, c(4, warp.type))
  first_warp = arith.cmpi(arith.CmpIPredicate.eq, warp, c(0, warp.type))
  elected = nvvm.elect_sync(ir.IntegerType.get_signless(1))
  return arith.andi(first_warp, elected)


@contextlib.contextmanager
def single_thread(per_block=True):
  """Runs the context only from a single thread.

  Args:
    per_block: If True, only one thread per block will run the context.
      Otherwise, only one thread per warp group will run the context.
  """
  global _ONCE_PER
  scope = ThreadSubset.BLOCK if per_block else ThreadSubset.WARPGROUP
  # If we're already in a single-thread context, we don't have to do anything.
  if _ONCE_PER is not None and _ONCE_PER >= scope:
    yield
    return

  prev_scope = _ONCE_PER
  _ONCE_PER = scope
  try:
    if_op = scf.IfOp(single_thread_predicate(per_block))
    with ir.InsertionPoint(if_op.then_block):
      yield
      scf.YieldOp([])
  finally:
    _ONCE_PER = prev_scope


def clock():
  i32 = ir.IntegerType.get_signless(32)
  return llvm.inline_asm(
      i32, [], "mov.u32  $0,%clock;", "=r", asm_dialect=0, has_side_effects=True
  )


def smid():
  i32 = ir.IntegerType.get_signless(32)
  return llvm.inline_asm(
      i32, [], "mov.u32  $0,%smid;", "=r", asm_dialect=0
  )


def globaltimer(kind: Literal["low", "high"] | None = None):
  if kind is None:
    i64 = ir.IntegerType.get_signless(64)
    return llvm.inline_asm(
        i64, [], "mov.u32  $0,%globaltimer;",
        "=l", asm_dialect=0, has_side_effects=True,
    )
  i32 = ir.IntegerType.get_signless(32)
  return llvm.inline_asm(
      i32, [], f"mov.u32  $0,%globaltimer_{kind[:2]};",
      "=r", asm_dialect=0, has_side_effects=True,
  )


def bytewidth(ty: ir.Type):
  # The actual width of TF32 is 19 bits. However, sinc we need to treat it as
  # 32 bits for compatibility reasons. TF32 used to be 32 bits wide in upstream
  # MLIR, but it changed in
  # https://github.com/llvm/llvm-project/commit/67a1fdb014790a38a205d28e1748634de34471dd.
  if ir.FloatTF32Type.isinstance(ty):
    return 4
  if ir.IntegerType.isinstance(ty):
    return ir.IntegerType(ty).width // 8
  if ir.FloatType.isinstance(ty):
    return ir.FloatType(ty).width // 8
  if dialect is not None and ir.Type.parse("!mosaic_gpu.barrier"):
    return MBARRIER_BYTES
  raise NotImplementedError(ty)


@dataclasses.dataclass(frozen=True)
class DynamicSlice:
  base: ir.Value | int
  length: int

  def __post_init__(self):
    if isinstance(self.base, int) and self.base < 0:
      raise ValueError(f"base must be non-negative, got {self.base}")
    if self.length < 0:
      raise ValueError(f"length must be non-negative, got {self.length}")


ds = DynamicSlice


def memref_slice(ref: ir.Value, index) -> ir.Value:
  ref_ty = ir.MemRefType(ref.type)
  base_indices, slice_shape, is_squeezed = parse_indices(index, ref_ty.shape)
  # TODO(apaszke): Check that slice is within the memref (indices might be
  # dynamic, but we can at least catch some OOB slices).

  memref_strides, offset = ref_ty.get_strides_and_offset()
  dynamic_offset = ir.ShapedType.get_dynamic_stride_or_offset()
  new_offset = offset
  if new_offset != dynamic_offset:
    for idx, stride in zip(base_indices, memref_strides):
      if isinstance(idx, int):
        new_offset += idx * stride
      else:
        new_offset = dynamic_offset
        break
  new_strides = [
      s for s, squeeze in zip(memref_strides, is_squeezed) if not squeeze
  ]
  new_shape = [s for s, squeeze in zip(slice_shape, is_squeezed) if not squeeze]
  new_layout = ir.StridedLayoutAttr.get(new_offset, new_strides)

  ref_slice = memref.subview(
      ref, base_indices, slice_shape, [1] * len(ref_ty.shape),
      result_type=ir.MemRefType.get(
          new_shape, ref_ty.element_type, new_layout, ref_ty.memory_space
      ),
  )
  return ref_slice


def _is_contiguous_shape_slice(
    ref_ty: ir.MemRefType, dim_slice: slice | None = slice(None)
):
  # If it's not a strided layout then we are definitely contiguous.
  if not ir.StridedLayoutAttr.isinstance(ref_ty.layout):
    return True

  strides = ir.StridedLayoutAttr(ref_ty.layout).strides[dim_slice]
  shape = ref_ty.shape[dim_slice]

  # Check that each dimension fits exactly it the immediately larger stride.
  ss = sorted(zip(strides, shape), key=lambda x: x[0], reverse=True)
  for (prev_stride, _), (stride, shape) in zip(ss, ss[1:]):
    if stride * shape != prev_stride:
      return False

  return True


def _reshape(ref: ir.Value, sh0: list[int], sh1: list[int]):
  """Reshapes using only "parallel" folds/unfolds.

  This function uses folds/unfolds that are "parallel" in that they
  only act on original dimensions, i.e. they won't fold into an
  intermediate dimension that they will then unfold.
  """

  i0, i1 = 0, 0
  def fold_until(shape, off , target)  -> tuple[int, int]:
    assert shape[off] < target
    dim = 1
    for to in range(off, len(shape)):
      dim *= shape[to]
      if dim == target:
        return to + 1, dim
      if dim > target:
        # TODO(cperivol): Implement dependent fold-unfolds for subsections
        # of the shape eg (..., 4,5,5, ...) -> (..., 10,10, ...) could be
        # supported without touching any other dimensions.
        raise NotImplementedError(f"Can't reshape {sh0} to {sh1} bu composing independent folds/unfolds.")

    raise AssertionError(f"Unreachable: number of elements don't match in each shape ({sh0} ans {sh1})")

  while i0 < len(sh0) and i1 < len(sh1):
    if sh0[i0] > sh1[i1]:
      # How many dimensions following i1 should we unfold i0 into.
      idx, _ = fold_until(sh1, i1, sh0[i0])
      ref = memref_unfold(ref, i0, sh1[i1:idx])
      sh0[i0:i0+1] = sh1[i1:idx]
      i0 += idx - i1
      i1 = idx
    elif sh0[i0] < sh1[i1]:
      # How many dimensions after i0 should we fold to make dim at i1.
      idx, dim = fold_until(sh0, i0, sh1[i1])
      sh0[i0:idx] = [dim]
      ref = memref_fold(ref, i0, idx - i0)
      i0 += 1
      i1 += 1
    else:
      i0 += 1
      i1 += 1

  # Fold the trailing ones
  if i0 < len(sh0):
    assert i1 == len(sh1)
    ref = memref_fold(ref, i0 - 1, len(sh0) - i0 + 1)

  if i1 < len(sh1):
    assert i0 == len(sh0)
    ref = memref_unfold(ref, i0 - 1, [sh0[i0 - 1]] + [1] * (len(sh1) - i1))

  return ref


def memref_reshape(ref: ir.Value, shape: tuple[int, ...]) -> ir.Value:
  """Reshape by means of folding and unfolding.

  The use of memref fold/unfold may avoid some possible issues with
  strided memrefs.
  """

  ref_ty = ir.MemRefType(ref.type)
  if math.prod(ref_ty.shape) != math.prod(shape):
    raise ValueError("Cannot reshape to a different size")
  if not all(dim > 0 for dim in shape):
    raise ValueError(
        "Shapes must havbe only positive dimensions (no -1 or 0 dimensions"
        f" allowed) {shape}"
    )

  return _reshape(ref, list(ref_ty.shape), list(shape))


def memref_fold(ref: ir.Value, dim, fold_rank) -> ir.Value:
  ref_ty = ir.MemRefType(ref.type)
  new_shape = list(ref_ty.shape)
  new_shape[dim : dim + fold_rank] = [np.prod(new_shape[dim : dim + fold_rank])]
  identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(ref_ty.rank))
  contig_strided_1d = ir.Attribute.parse("strided<[1]>")
  # Not sure why but MLIR expects the strided 1D layout to disappear in this op.
  if ref_ty.layout == identity or ref_ty.layout == contig_strided_1d:
    new_layout = ir.AffineMapAttr.get(
        ir.AffineMap.get_identity(ref_ty.rank - fold_rank + 1)
    )
  elif _is_contiguous_shape_slice(ref_ty, slice(dim, dim + fold_rank)):
    new_strides, offset = ref_ty.get_strides_and_offset()
    new_strides[dim : dim + fold_rank] = [new_strides[dim + fold_rank - 1]]
    new_layout = ir.StridedLayoutAttr.get(offset, new_strides)
  else:
    raise NotImplementedError(
        f"strides={ref_ty.get_strides_and_offset()[0]}, {ref_ty.shape=},"
        f" {dim=}, {fold_rank=}"
    )

  new_ty = ir.MemRefType.get(
      new_shape, ref_ty.element_type, new_layout, ref_ty.memory_space
  )
  assoc = [[d] for d in range(dim)]
  assoc.append([dim + i for i in range(fold_rank)])
  assoc.extend([d] for d in range(dim + fold_rank, ref_ty.rank))
  assert len(assoc) == new_ty.rank
  return memref.collapse_shape(new_ty, ref, assoc)


def memref_unfold(ref: ir.Value, dim, factors) -> ir.Value:
  """Unfolds dim into two dimensions, the size of leading one given be major_factor."""
  ref_ty = ir.MemRefType(ref.type)
  new_shape = list(ref_ty.shape)
  if sum(f is None for f in factors) > 1:
    raise ValueError("Can only infer one dimension")
  known_factor_prod = np.prod([f for f in factors if f is not None])
  if new_shape[dim] % known_factor_prod:
    raise ValueError("Non-divisible unfold:", new_shape[dim], factors)
  factors = tuple(
      new_shape[dim] // known_factor_prod if f is None else f for f in factors
  )
  new_shape[dim : dim + 1] = factors
  identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(ref_ty.rank))
  if ref_ty.layout == identity:
    new_layout = ir.AffineMapAttr.get(
        ir.AffineMap.get_identity(ref_ty.rank + len(factors) - 1)
    )
  else:
    new_strides, offset = ref_ty.get_strides_and_offset()
    prev_stride = new_strides[dim]
    inserted_strides = []
    for f in reversed(factors):
      inserted_strides.append(prev_stride)
      prev_stride *= f
    new_strides[dim : dim + 1] = reversed(inserted_strides)
    new_layout = ir.StridedLayoutAttr.get(offset, new_strides)
  new_ty = ir.MemRefType.get(
      new_shape, ref_ty.element_type, new_layout, ref_ty.memory_space
  )
  if dim == ref_ty.rank:
    assoc = [[d] for d in range(ref_ty.rank)]
    assoc[-1].extend(range(ref_ty.rank, ref_ty.rank + len(factors) - 1))
  else:
    assoc = [[d] for d in range(dim)]
    assoc.append(list(range(dim, dim + len(factors))))
    assoc.extend([d + len(factors) - 1] for d in range(dim + 1, ref_ty.rank))
  assert len(assoc) == ref_ty.rank
  return memref.expand_shape(new_ty, ref, assoc, [], new_ty.shape)


def memref_unsqueeze(ref: ir.Value, dim) -> ir.Value:
  """Inserts a singleton dimension."""
  ref_ty = ir.MemRefType(ref.type)
  if dim == ref_ty.rank:
    new_shape = list(ref_ty.shape)
    new_shape.append(1)
    identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(ref_ty.rank))
    if ref_ty.layout == identity:
      new_layout = ir.AffineMapAttr.get(
          ir.AffineMap.get_identity(ref_ty.rank + 1)
      )
    else:
      new_strides, offset = ref_ty.get_strides_and_offset()
      new_strides.append(1)
      new_layout = ir.StridedLayoutAttr.get(offset, new_strides)
    new_ty = ir.MemRefType.get(
        new_shape, ref_ty.element_type, new_layout, ref_ty.memory_space
    )
    assoc = [[d] for d in range(ref_ty.rank)]
    assoc[-1].append(ref_ty.rank)
    return memref.expand_shape(new_ty, ref, assoc, [], new_ty.shape)
  else:
    return memref_unfold(ref, dim, (1, None))


def memref_transpose(ref: ir.Value, permutation: Sequence[int]) -> ir.Value:
  ref_ty = ir.MemRefType(ref.type)
  strides, offset = ref_ty.get_strides_and_offset()
  new_strides = [strides[p] for p in permutation]
  new_shape = [ref_ty.shape[p] for p in permutation]
  new_layout = ir.StridedLayoutAttr.get(offset, new_strides)
  new_ty = ir.MemRefType.get(
      new_shape, ref_ty.element_type, new_layout, ref_ty.memory_space
  )
  return memref.transpose(
      new_ty, ref, ir.AffineMap.get_permutation(permutation)
  )


def parse_indices(
    index, shape: tuple[int, ...], *, check_oob: bool = True
) -> tuple[list[ir.Value | int], list[int], list[bool]]:
  if not isinstance(index, tuple):
    index = (index,)
  if trailing_dims := len(shape) - len(index):
    index += (slice(None),) * trailing_dims
  base_indices = []
  slice_shape = []
  is_squeezed = []
  for axis, (idx, bound) in enumerate(zip(index, shape)):
    if isinstance(idx, (ir.Operation, ir.OpView)):
      idx = idx.result
    if isinstance(idx, int):
      if check_oob and (idx >= bound or (idx < 0 and -idx > bound)):
        raise IndexError(
            f"Index {idx} along axis {axis} is out of bounds for shape {shape}"
        )
      base_indices.append(idx if idx >= 0 else bound + idx)
      slice_shape.append(1)
      is_squeezed.append(True)
    elif isinstance(idx, slice):
      if idx.step is not None and idx.step != 1:
        raise NotImplementedError("Strided slices not implemented")
      start = idx.start or 0
      if start < 0:
        start = bound + start
      stop = idx.stop or bound
      if stop < 0:
        stop = bound + stop
      if check_oob and (
          start < 0 or start >= bound or stop < 0 or stop > bound
      ):
        raise IndexError(
            f"Slice {idx} along axis {axis} is out of bounds for shape {shape}"
        )
      base_indices.append(start)
      slice_shape.append(stop - start)
      is_squeezed.append(False)
    elif isinstance(idx, DynamicSlice):
      if check_oob and (
          isinstance(idx.base, int) and idx.base + idx.length > bound
      ):
        raise IndexError(
            f"Slice {idx} along axis {axis} is out of bounds for shape {shape}"
        )
      base_indices.append(idx.base)
      slice_shape.append(idx.length)
      is_squeezed.append(False)
    elif isinstance(idx, ir.Value):
      if not ir.IndexType.isinstance(idx.type):
        raise ValueError("Expected an index-typed index")
      base_indices.append(idx)
      slice_shape.append(1)
      is_squeezed.append(True)
    else:
      raise NotImplementedError(type(idx))
  assert len(base_indices) == len(slice_shape) == len(is_squeezed) == len(shape)
  return base_indices, slice_shape, is_squeezed


def commit_shared():
  warpgroup_barrier()
  nvvm.fence_proxy(
      nvvm.ProxyKind.async_shared, space=nvvm.SharedSpace.shared_cta
  )


def warpgroup_barrier():
  # gpu.barrier() uses barrier number 0, and it would be unsafe to reuse it,
  # so we shift the warpgroup index by 1.
  i32 = ir.IntegerType.get_signless(32)
  llvm.inline_asm(
      ir.Type.parse("!llvm.void"),
      [arith.addi(warpgroup_idx(sync=False), c(1, i32))],
      f"bar.sync $0, {WARPGROUP_SIZE};",
      "r",
      has_side_effects=True,
  )


@dataclasses.dataclass(frozen=True)
class BarrierRef:
  base_address: ir.Value
  offset: ir.Value
  phases: ir.Value
  num_barriers: int

  @staticmethod
  def initialize(address: ir.Value, num_barriers: int, arrival_count: int = 1) -> "BarrierRef":
    if num_barriers > 32:
      raise NotImplementedError("Only up to 32 barriers per group supported")
    i32 = ir.IntegerType.get_signless(32)
    i64 = ir.IntegerType.get_signless(64)
    ptr = ir.Type.parse(f"!llvm.ptr<{WORKGROUP_NVPTX_ADDRESS_SPACE}>")
    phases = memref.alloca(ir.MemRefType.get((), i32), [], [])
    memref.store(c(0, i32), phases, [])
    with single_thread(per_block=True):
      for i in range(num_barriers):
        nvvm.mbarrier_init_shared(
            llvm.getelementptr(ptr, address, [], [i], i64),
            c(arrival_count, i32),
        )
    return BarrierRef(address, c(0, i32), phases, num_barriers)

  def __iter__(self) -> Iterator["BarrierRef"]:
    if self.num_barriers == 1:
      yield self
    else:
      for offset in range(self.num_barriers):
        yield self[offset]

  def __getitem__(self, offset: ir.Value | int) -> "BarrierRef":
    i32 = ir.IntegerType.get_signless(32)
    if isinstance(offset, int):
      offset = c(offset, i32)
    elif ir.IndexType.isinstance(offset.type):
      offset = arith.index_castui(i32, offset)
    elif offset.type != i32:
      raise ValueError(f"Expected a dynamic index or an integer, got {offset}")
    return BarrierRef(
        self.base_address,
        arith.addi(self.offset, offset),
        self.phases,
        1,
    )

  def wait_parity(self, parity):
    i32 = ir.IntegerType.get_signless(32)
    ticks = arith.constant(i32, 10000000)
    parity = arith.extui(i32, parity)
    nvvm.mbarrier_try_wait_parity_shared(self.get_ptr(), parity, ticks)

  def wait(self):
    parities = memref.load(self.phases, [])
    parity, new_parities = self.update_parities(parities)
    memref.store(new_parities, self.phases, [])
    self.wait_parity(parity)

  def update_parities(self, parities: ir.Value) -> tuple[ir.Value, ir.Value]:
    i32 = ir.IntegerType.get_signless(32)
    bitmask = arith.shli(c(1, i32), self.offset)
    parity = arith.cmpi(
        arith.CmpIPredicate.ne, arith.andi(parities, bitmask), c(0, i32)
    )
    return parity, arith.xori(parities, bitmask)

  def arrive(self):
    i64 = ir.IntegerType.get_signless(64)
    nvvm.mbarrier_arrive_shared(i64, self.get_ptr())

  def arrive_expect_tx(
      self, bytes: int | ir.Value, predicate: ir.Value | None = None
  ):
    if isinstance(bytes, int):
      bytes = c(bytes, ir.IntegerType.get_signless(32))
    elif ir.IndexType.isinstance(bytes.type):
      i32 = ir.IntegerType.get_signless(32)
      bytes = arith.index_cast(i32, bytes)
    nvvm.mbarrier_arrive_expect_tx_shared(self.get_ptr(), bytes, predicate=predicate)

  def get_ptr(self):
    ptr = ir.Type.parse(f"!llvm.ptr<{WORKGROUP_NVPTX_ADDRESS_SPACE}>")
    i64 = ir.IntegerType.get_signless(64)
    DYNAMIC32 = -2147483648
    return llvm.getelementptr(
        ptr, self.base_address, [self.offset], [DYNAMIC32], i64
    )

  def as_dialect_barrier_memref(self) -> ir.Value:
    shape = () if self.num_barriers == 1 else (self.num_barriers,)
    return ptr_as_memref(
        self.base_address,
        ir.MemRefType.get(shape, ir.Type.parse("!mosaic_gpu.barrier")),
        ptr_memory_space=WORKGROUP_NVPTX_ADDRESS_SPACE,
    )

  @classmethod
  def from_dialect_barrier_memref(cls, barrier: ir.Value):
    """Creates a BarrierRef from a memref of a dialect barrier."""
    memref_type = ir.MemRefType(barrier.type)
    if memref_type.rank > 1 or memref_type.element_type != ir.Type.parse(
        "!mosaic_gpu.barrier"
    ):
      raise ValueError(
          "Expected a memref with rank 0 or 1 and element type "
          f"!mosaic_gpu.barrier, but got {barrier.type}"
      )

    return cls(
        base_address=memref_ptr(
            barrier, memory_space=WORKGROUP_NVPTX_ADDRESS_SPACE
        ),
        offset=c(0, ir.IntegerType.get_signless(64)),
        phases=None,
        num_barriers=(1 if memref_type.rank == 0 else memref_type.shape[0]),
    )


@dataclasses.dataclass(frozen=True)
class CollectiveBarrierRef:
  barrier: BarrierRef
  cluster_mask: ir.Value | None

  @staticmethod
  def initialize(
      address: ir.Value,
      num_barriers: int,
      dims: Sequence[gpu.Dimension | Sequence[gpu.Dimension]],
      cluster_shape: tuple[int, int, int],
  ) -> "CollectiveBarrierRef":
    i32 = ir.IntegerType.get_signless(32)
    # With the exception of the current device, each pair of slices along
    # collective dims is disjoint. Since the current device is overcounted,
    # we must decrease the arrival count a little.
    dims_shape = [
        cluster_shape[d]
        if isinstance(d, gpu.Dimension)
        else math.prod(cluster_shape[dd] for dd in d)
        for d in dims
    ]
    arrival_count = sum(dims_shape) - len(dims) + 1
    if arrival_count == 1:
      assert all(s == 1 for s in dims_shape)
      cluster_mask = None
    else:
      cluster_mask = c(0, i32)
      for d, size in zip(dims, dims_shape):
        if size == 1:
          # Only the current device is in this mask, but it will also be
          # present in one of the non-trivial cluster dims.
          continue
        cluster_mask = arith.ori(
            cluster_mask, cluster_collective_mask(cluster_shape, d)
        )
    barrier = BarrierRef.initialize(address, num_barriers, arrival_count=arrival_count)
    return CollectiveBarrierRef(barrier, cluster_mask)

  def __iter__(self):
    for b in self.barrier:
      yield CollectiveBarrierRef(b, self.cluster_mask)

  def __getitem__(self, offset):
    return CollectiveBarrierRef(self.barrier[offset], self.cluster_mask)

  def arrive(self):
    """Arrives on a barrier in all blocks that share at least one of the coordinates along the collective dimensions.

    Note that unlike in arrive, each warpgroup arrives once.
    """
    if self.barrier.num_barriers != 1:
      raise ValueError("Can only arrive on a single barrier")
    if self.cluster_mask is None:
      with single_thread(per_block=False):
        self.barrier.arrive()
      return
    i32 = ir.IntegerType.get_signless(32)
    thread_in_warpgroup = arith.remui(thread_idx(), c(WARPGROUP_SIZE, i32))
    signaled_block = arith.divui(
        thread_in_warpgroup, c(WARPGROUP_SIZE // 16, i32)
    )
    is_collective_block = arith.cmpi(
        arith.CmpIPredicate.ne,
        arith.andi(self.cluster_mask, arith.shli(c(1, i32), signaled_block)),
        c(0, i32),
    )
    is_signaling_thread = arith.cmpi(
        arith.CmpIPredicate.eq,
        arith.remui(thread_in_warpgroup, c(WARPGROUP_SIZE // 16, i32)),
        c(0, i32),
    )
    should_arrive = arith.andi(is_collective_block, is_signaling_thread)
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [should_arrive, self.barrier.get_ptr(), signaled_block],
        """
    {
        .reg .b32 mapped_addr;
        @$0 mapa.shared::cluster.u32 mapped_addr, $1, $2;
        @$0 mbarrier.arrive.shared::cluster.b64 _, [mapped_addr];
    }""",
        "b,r,r",
        has_side_effects=True,
    )

  def wait(self, *args, **kwargs):
    self.barrier.wait(*args, **kwargs)

  def wait_parity(self, *args, **kwargs):
    self.barrier.wait_parity(*args, **kwargs)


class Partition:
  source_bounds: tuple[int, ...]
  target_bounds: tuple[int, ...]
  partition: tuple[int | None, ...]
  base_offset: tuple[ir.Value, ...] | None

  def __init__(
      self,
      elements: tuple[int, ...],
      *,
      partition: tuple[int | None, ...],
      base_offset: tuple[ir.Value, ...] | None = None,
      num_chunks: tuple[int, ...] | None = None,
      chunk_size: tuple[int, ...] | None = None,
  ):
    self.target_bounds = elements
    self.partition = partition
    self.base_offset = base_offset
    if len(self.target_bounds) != len(self.partition):
      raise ValueError
    if num_chunks is None == chunk_size is None:
      raise ValueError(
          "Exactly one of num_chunks and chunk_size must be specified"
      )
    if num_chunks is not None:
      self.source_bounds = num_chunks
    else:
      if len(chunk_size) != len(self.target_bounds):
        raise ValueError
      source_bounds = []
      for els, chunk in zip(elements, chunk_size):
        if els % chunk:
          raise ValueError("Non-divisible partition", elements, chunk_size)
        source_bounds.append(els // chunk)
      self.source_bounds = tuple(source_bounds)

    seen_dims = set()
    for p in self.partition:
      if p is None:
        continue
      if not (0 <= p < len(self.source_bounds)):
        raise ValueError
      if p in seen_dims:
        raise ValueError
      seen_dims.add(p)
    for tb, p in zip(self.target_bounds, self.partition):
      if p is not None and tb % self.source_bounds[p]:
        raise ValueError("Non-divisible partitioning")

  @property
  def num_chunks(self) -> tuple[int, ...]:
    return self.source_bounds

  @property
  def target_block_shape(self):
    return tuple(tb if p is None else tb // self.source_bounds[p]
                 for tb, p in zip(self.target_bounds, self.partition))

  def get_base(self, *source_coords: ir.Value | int) -> list[ir.Value]:
    coords = []
    index = ir.IndexType.get()
    for i, (tbs, p) in enumerate(zip(self.target_block_shape, self.partition)):
      if p is None:
        dim_base = c(0, index)
      else:
        dim_base = arith.muli(c(tbs, index), source_coords[p])
      if self.base_offset is not None:
        dim_base = arith.addi(self.base_offset[i], dim_base)
      coords.append(dim_base)
    return coords


class Partition1D:
  partition: Partition

  def __init__(
      self,
      elements: int,
      *,
      base_offset: ir.Value | None = None,
      num_chunks: int | None = None,
      chunk_size: int | None = None,
  ):
    self.base_offset = base_offset
    if num_chunks is None == chunk_size is None:
      raise ValueError(
          "Exactly one of num_chunks and chunk_size must be specified"
      )
    common_kwargs = dict(elements=(elements,), partition=(0,))
    if base_offset is not None:
      common_kwargs["base_offset"] = (base_offset,)
    if num_chunks is not None:
      self.partition = Partition(num_chunks=(num_chunks,), **common_kwargs)
    else:
      self.partition = Partition(chunk_size=(chunk_size,), **common_kwargs)

  @property
  def num_chunks(self) -> int:
    return self.partition.source_bounds[0]

  def get_base(self, source_coords: ir.Value) -> ir.Value:
    return self.partition.get_base(source_coords)[0]

  def refine(
      self,
      *,
      chunk: ir.Value | None = None,
      num_chunks: int | None = None,
      chunk_size: int | None = None,
  ):
    return Partition1D(
        self.partition.target_block_shape[0],
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        base_offset=self.get_base(chunk) if chunk is not None else None,
    )


def tile_shape(shape, tiling):
  if len(tiling) > len(shape):
    raise ValueError
  if not tiling:
    return shape
  tiling_rank = len(tiling)
  for s, t in zip(shape[-tiling_rank:], tiling):
    if s % t:
      raise ValueError("Non-divisible tiling:", shape, tiling)
  return (
      *shape[:-tiling_rank],
      *(s // t for s, t in zip(shape[-tiling_rank:], tiling)),
      *tiling,
  )


def warp_tree_reduce(value, op, group_size):
  """Reduce a value across the warpgroup."""
  assert bytewidth(value.type) == 4
  assert 32 % group_size == 0 and group_size <= 32
  i32 = ir.IntegerType.get_signless(32)
  result = value
  iters = np.log2(group_size)
  if not iters.is_integer():
    raise ValueError(f"Warp reduction group size should be a power of 2 (got {group_size})")
  iters = int(iters)
  for i in range(iters):
    other_result = nvvm.shfl_sync(
        result.type,
        c(0xFFFFFFFF, i32),
        result,
        c(1 << i, i32),
        c(0x1F, i32),
        nvvm.ShflKind.bfly,
    )
    result = op(result, other_result)

  return result


def memref_ptr(memref_arg, memory_space=None):
  i64 = ir.IntegerType.get_signless(64)
  memref_ty = ir.MemRefType(memref_arg.type)
  rank = len(memref_ty.shape)
  # TODO: Read out memory space from memref
  space = "" if memory_space is None else "<" + str(memory_space) + ">"
  ptr_ty = ir.Type.parse("!llvm.ptr" + space)
  if rank == 0:
    desc_ty = ir.Type.parse(f"!llvm.struct<({ptr_ty}, {ptr_ty}, i64)>")
  else:
    desc_ty = ir.Type.parse(
        f"!llvm.struct<({ptr_ty}, {ptr_ty}, i64, array<{rank} x i64>,"
        f" array<{rank} x i64>)>"
    )
  desc = builtin.UnrealizedConversionCastOp([desc_ty], [memref_arg])
  aligned_ptr = llvm.extractvalue(ptr_ty, desc, [1])

  elem_bytewidth = bytewidth(memref_ty.element_type)
  offset_elems = llvm.extractvalue(i64, desc, [2])
  offset_bytes = llvm.mul(
      offset_elems,
      c(elem_bytewidth, i64),
      overflow_flags=llvm.IntegerOverflowFlags.none,
  )
  return llvm.inttoptr(
      ptr_ty,
      llvm.add(
          llvm.ptrtoint(i64, aligned_ptr),
          offset_bytes,
          overflow_flags=llvm.IntegerOverflowFlags.none,
      ),
  )


def cluster_collective_mask(
    cluster_shape: tuple[int, int, int],
    collective: Sequence[gpu.Dimension] | gpu.Dimension,
):
  if isinstance(collective, gpu.Dimension):
    collective = (collective,)
  # We first compute the linearized index of the slice along the collective
  # dim that contains the current block. Then, the mask is a sequence of 1s
  # strided by the position of the collective dim, shifted left by the linear
  # slice index.
  # TODO(apaszke): Make sure this gets hoisted outside of any loops.
  # If not, we might need to do it manually.
  i32 = ir.IntegerType.get_signless(32)
  mask_shift = c(0, i32)
  # NOTE: GPU dimensions are minor-to-major.
  cluster_strides = get_contiguous_strides(cluster_shape[::-1])[::-1]
  for stride, cluster_dim in zip(cluster_strides, gpu.Dimension):
    if cluster_dim in collective:
      continue
    if cluster_shape[cluster_dim] != 1:  # Constant-fold multiply by 0.
      dim_idx = arith.index_castui(i32, gpu.cluster_block_id(cluster_dim))
      mask_shift = arith.addi(
          mask_shift, arith.muli(dim_idx, c(stride, i32)),
      )
  mask_unshifted = 0
  collective_strides = [cluster_strides[d] for d in collective]
  collective_shape = tuple(cluster_shape[d] for d in collective)
  for idx in np.ndindex(collective_shape):
    mask_unshifted |= 1 << sum(i * s for i, s in zip(idx, collective_strides))
  return arith.shli(c(mask_unshifted, i32), mask_shift)


def dtype_to_ir_type(dtype: jax.typing.DTypeLike) -> ir.Type:
  dtype = jnp.dtype(dtype)
  if jnp.issubdtype(dtype, jnp.integer):
    # All integer types in Mosaic GPU are signless.
    return ir.IntegerType.get_signless(dtype.itemsize * 8)
  return mlir.dtype_to_ir_type(dtype)


def is_signed(dtype: jax.typing.DTypeLike) -> bool | None:
  if jnp.issubdtype(dtype, jnp.bool_):
    return False
  elif jnp.issubdtype(dtype, jnp.integer):
    return jnp.issubdtype(dtype, jnp.signedinteger)
  return None


def getelementptr(
    ptr: ir.Value, indices: Sequence[ir.Value | int], dtype: ir.Type
) -> ir.Value:
  static_indices = [i if isinstance(i, int) else DYNAMIC32 for i in indices]
  dyn_indices = [i for i in indices if not isinstance(i, int)]
  return llvm.getelementptr(ptr.type, ptr, dyn_indices, static_indices, dtype)


def dyn_dot(x, y):
  return functools.reduce(arith.addi, (arith.muli(a, b) for a, b in zip(x, y)))
