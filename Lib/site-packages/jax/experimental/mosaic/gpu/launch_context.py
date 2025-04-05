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

from collections.abc import Callable, Sequence
import contextlib
import dataclasses
import functools
import math
from typing import Any

from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import func
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import nvvm
import numpy as np

from . import profiler
from . import utils
# mypy: ignore-errors

TMA_DESCRIPTOR_BYTES = 128
TMA_DESCRIPTOR_ALIGNMENT = 64

c = utils.c  # This is too common to fully qualify.

@dataclasses.dataclass(frozen=True)
class MemRefTransform:
  def apply(self, ref: ir.Value) -> ir.Value:
    raise NotImplementedError("Subclasses should override this method")

  def transform_index(self, idx: Sequence[ir.Value]) -> tuple[ir.Value, ...]:
    raise NotImplementedError("Subclasses should override this method")

  def transform_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
    raise NotImplementedError("Subclasses should override this method")

  def batch(self, leading_rank: int) -> 'MemRefTransform':
    """Returns a transform that accepts a ref with the extra `leading_rank` dims.

    The returned transform should leave the leading dimensions unchanged and
    only apply to the suffix of the shape.
    """
    raise NotImplementedError("Subclasses should override this method")


@dataclasses.dataclass(frozen=True)
class TileTransform(MemRefTransform):
  """Tiles a suffix of memref dimensions.

  For example, given a memref of shape (5, 128, 128) and a tiling of (64, 32),
  the shape of the result will be (5, 2, 4, 64, 32). The shape always ends with
  the tile shape, and the size of tiled dimensions is divided by the tile size.
  This is especially useful for swizzled WGMMA, which expect tiled layouts in
  shared memory.
  """
  tiling: tuple[int, ...]

  def apply(self, ref: ir.Value) -> ir.Value:
    untiled_rank = ir.MemRefType(ref.type).rank
    tiling_rank = len(self.tiling)
    tiled_rank = untiled_rank + tiling_rank
    for t, d in zip(self.tiling[::-1], range(untiled_rank)[::-1]):
      s = ir.MemRefType(ref.type).shape[d]
      if s % t and s > t:
        raise ValueError(
            f"Dimension {d} must have size smaller or a multiple of its tiling"
            f" {t}, but got {s}"
        )
      ref = utils.memref_unfold(ref, d, (None, min(t, s)))
    permutation = (
        *range(untiled_rank - tiling_rank),
        *range(untiled_rank - tiling_rank, tiled_rank, 2),
        *range(untiled_rank - tiling_rank + 1, tiled_rank, 2),
    )
    return utils.memref_transpose(ref, permutation)

  def transform_index(self, idx: Sequence[ir.Value]) -> tuple[ir.Value, ...]:
    index = ir.IndexType.get()
    tiling_rank = len(self.tiling)
    return (
        *idx[:-tiling_rank],
        *(
            arith.divui(i, c(t, index))
            for i, t in zip(idx[-tiling_rank:], self.tiling)
        ),
        *(
            arith.remui(i, c(t, index))
            for i, t in zip(idx[-tiling_rank:], self.tiling)
        ),
    )

  def transform_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
    # Note that this also checks that tiled dims are not squeezed. Their slice
    # size would be 1 if so.
    tiling_rank = len(self.tiling)
    for size, tile_size in zip(shape[-tiling_rank:], self.tiling):
      if size % tile_size:
        raise ValueError(
            f"Expected GMEM slice shape {shape} suffix to be a multiple of"
            f" tiling {self.tiling}.\nIf you're using padded async copies, your"
            " slice might need to extend out of bounds of the GMEM buffer (OOB"
            " accesses will be skipped)."
        )
    return (
        *shape[:-tiling_rank],
        *(s // t for s, t in zip(shape[-tiling_rank:], self.tiling)),
        *self.tiling,
    )

  def batch(self, leading_rank: int) -> MemRefTransform:
    return self


@dataclasses.dataclass(frozen=True)
class TransposeTransform(MemRefTransform):
  """Transposes memref dimensions."""
  permutation: tuple[int, ...]

  def __post_init__(self):
    if len(self.permutation) != len(set(self.permutation)):
      raise ValueError("Permutation must be a permutation")

  def apply(self, ref: ir.Value) -> ir.Value:
    return utils.memref_transpose(ref, self.permutation)

  def transform_index(self, idx: Sequence[ir.Value]) -> tuple[ir.Value, ...]:
    return tuple(idx[p] for p in self.permutation)

  def transform_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
    return tuple(shape[p] for p in self.permutation)

  def batch(self, leading_rank: int) -> MemRefTransform:
    return TransposeTransform(
        (*range(leading_rank), *(d + leading_rank for d in self.permutation))
    )


@dataclasses.dataclass(frozen=True)
class CollapseLeadingIndicesTransform(MemRefTransform):
  """Collapses leading indices into one."""
  strides: tuple[int, ...]

  @functools.cached_property
  def common_stride(self) -> int:
    return math.gcd(*self.strides)

  def apply(self, ref: ir.Value) -> ir.Value:
    ref_ty = ir.MemRefType(ref.type)
    strides, offset = ref_ty.get_strides_and_offset()
    if offset == ir.ShapedType.get_dynamic_stride_or_offset():
      raise NotImplementedError("Dynamic offsets are not supported")
    max_bound = sum(
        (d - 1) * s // self.common_stride
        for d, s in zip(
            ref_ty.shape[: len(self.strides)], strides[: len(self.strides)]
        )
    ) + 1
    new_shape = [max_bound, *ref_ty.shape[len(self.strides):]]
    new_strides = [self.common_stride, *strides[len(self.strides):]]
    new_layout = ir.StridedLayoutAttr.get(offset, new_strides)
    new_ref_ty = ir.MemRefType.get(
        new_shape, ref_ty.element_type, new_layout, ref_ty.memory_space
    )
    return memref.reinterpret_cast(
        new_ref_ty, ref, [], [], [],
        static_offsets=[offset],
        static_sizes=new_shape,
        static_strides=new_strides,
    )

  def transform_index(self, idx: Sequence[ir.Value]) -> tuple[ir.Value, ...]:
    index = ir.IndexType.get()
    flat_idx = c(0, index)
    for i, s in zip(idx[:len(self.strides)], self.strides):
      flat_idx = arith.addi(
          flat_idx, arith.muli(i, c(s // self.common_stride, index))
      )
    return (flat_idx, *idx[len(self.strides):])

  def transform_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
    if any(s != 1 for s in shape[:len(self.strides)]):
      raise ValueError("Expected leading indices to be squeezed")
    return (1, *shape[len(self.strides):])

  def batch(self, leading_rank: int) -> MemRefTransform:
    raise NotImplementedError  # Unused


OnDeviceProfiler = profiler.OnDeviceProfiler


@dataclasses.dataclass()
class LaunchContext:
  launch_op: gpu.LaunchOp
  gmem_scratch_ptr: ir.Value
  cluster_size: tuple[int, int, int]
  profiler: OnDeviceProfiler | None = None
  next_scratch_offset: int = 0
  host_scratch_init: list[Callable[[ir.Value], None]] = dataclasses.field(
      default_factory=list, init=False
  )
  tma_descriptors: dict[
      tuple[ir.Value, tuple[int, ...], int | None, tuple[MemRefTransform, ...]],
      ir.Value,
  ] = dataclasses.field(default_factory=dict, init=False)

  @contextlib.contextmanager
  def named_region(self, *args, **kwargs):
    if self.profiler is not None:
      with self.profiler.record(*args, **kwargs):
        yield
    else:
      yield

  def _alloc_scratch(
      self,
      size: int,
      alignment: int | None = None,
      host_init: Callable[[ir.Value], None] = lambda _: None,
      device_init: Callable[[ir.Value], Any] = lambda x: x,
  ) -> ir.Value:
    """Allocates a GMEM scratch buffer.

    The buffer is initialized on the host and then copied to GMEM before the
    kernel launch.
    """
    i8 = ir.IntegerType.get_signless(8)
    ptr_ty = ir.Type.parse("!llvm.ptr")
    if alignment is None:
      alignment = size
    if self.next_scratch_offset % alignment:
      raise NotImplementedError  # TODO(apaszke): Pad to match alignment
    alloc_base = self.next_scratch_offset
    self.next_scratch_offset += size
    def host_init_wrapped(host_ptr):
      host_init(
          llvm.getelementptr(ptr_ty, host_ptr, [], [alloc_base], i8)
      )
    self.host_scratch_init.append(host_init_wrapped)
    # with ir.InsertionPoint(self.gmem_scratch_ptr.owner):
    # There is no way to create an insertion point after an operation...
    gep = llvm.GEPOp(
        ptr_ty, self.gmem_scratch_ptr, [], [alloc_base], i8
    )
    gep.move_after(self.gmem_scratch_ptr.owner)
    return device_init(gep.result)

  def _get_tma_desc(
      self,
      gmem_ref,
      gmem_transform: tuple[MemRefTransform, ...],
      transformed_slice_shape: tuple[int, ...],
      swizzle: int | None,
  ):
    tma_desc_key = (gmem_ref, transformed_slice_shape, swizzle, gmem_transform)
    if (tma_desc := self.tma_descriptors.get(tma_desc_key, None)) is None:
      i64 = ir.IntegerType.get_signless(64)
      ptr_ty = ir.Type.parse("!llvm.ptr")
      def init_tma_desc(host_ptr):
        ref = gmem_ref
        for t in gmem_transform:
          ref = t.apply(ref)
        ref_ty = ir.MemRefType(ref.type)
        # TODO(apaszke): Use utils.memref_ptr to compute base_ptr
        _, offset, *sizes_and_strides = memref.extract_strided_metadata(ref)
        aligned_ptr_idx = memref.extract_aligned_pointer_as_index(ref)
        as_i64 = lambda i: arith.index_cast(i64, i)
        alloc_ptr = llvm.inttoptr(ptr_ty, as_i64(aligned_ptr_idx))
        llvm_dyn = -2147483648  # TODO(apaszke): Improve the MLIR bindings...
        base_ptr = llvm.getelementptr(
            ptr_ty, alloc_ptr, [as_i64(offset)], [llvm_dyn], ref_ty.element_type,
        )
        rank = ref_ty.rank
        assert rank * 2 == len(sizes_and_strides)
        args = [
            host_ptr,
            base_ptr,
            c(utils.bytewidth(ref_ty.element_type), i64),
            c(rank, i64),
            utils.pack_array([as_i64(i) for i in sizes_and_strides[:rank]]),
            utils.pack_array([as_i64(i) for i in sizes_and_strides[rank:]]),
            c(0 if swizzle is None else swizzle, i64),
            utils.pack_array([c(v, i64) for v in transformed_slice_shape]),
        ]
        func.call([], "mosaic_gpu_init_tma_desc", args)
      def cast_tma_desc(device_ptr):
        # TODO(apaszke): Investigate why prefetching can cause launch failures
        # nvvm.prefetch_tensormap(device_ptr)
        return device_ptr
      tma_desc = self._alloc_scratch(
          TMA_DESCRIPTOR_BYTES,
          alignment=TMA_DESCRIPTOR_ALIGNMENT,
          host_init=init_tma_desc,
          device_init=cast_tma_desc,
      )
      self.tma_descriptors[tma_desc_key] = tma_desc
    return tma_desc

  def async_copy(
      self,
      *,
      src_ref,
      dst_ref,
      gmem_slice: Any = (),
      gmem_transform: MemRefTransform | tuple[MemRefTransform, ...] = (),
      barrier: utils.BarrierRef | None = None,
      swizzle: int | None = None,
      arrive: bool | None = None,
      uniform: bool = True,
      collective: Sequence[gpu.Dimension] | gpu.Dimension | None = None,
      predicate: ir.Value | None = None,  # Should select 0 or 1 threads from the WG.
  ):
    index = ir.IndexType.get()
    i16 = ir.IntegerType.get_signless(16)
    i32 = ir.IntegerType.get_signless(32)
    smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
    src_ref_ty = ir.MemRefType(src_ref.type)
    dst_ref_ty = ir.MemRefType(dst_ref.type)
    element_type = src_ref_ty.element_type
    element_bytewidth = utils.bytewidth(element_type)
    if element_type != dst_ref_ty.element_type:
      raise ValueError(
          f"Expected same element type, got {element_type} and"
          f" {dst_ref_ty.element_type}"
      )
    if predicate is not None and not uniform:
      raise ValueError("Predicate can only be defined when uniform is True")
    if not isinstance(gmem_transform, tuple):
      gmem_transform = (gmem_transform,)

    if src_ref_ty.memory_space is None and dst_ref_ty.memory_space == smem:
      gmem_ref, smem_ref = src_ref, dst_ref
      if barrier is None:
        raise ValueError("Barriers are required for GMEM -> SMEM copies")
      if arrive is None:
        arrive = True  # Arrive by default
    elif src_ref_ty.memory_space == smem and dst_ref_ty.memory_space is None:
      gmem_ref, smem_ref = dst_ref, src_ref
      if barrier is not None:
        raise ValueError("Barriers are unsupported for SMEM -> GMEM copies")
      if arrive is not None:
        raise ValueError("arrive is unsupported for SMEM -> GMEM copies")
    else:
      raise ValueError("Only SMEM <-> GMEM copies supported")
    # TODO(apaszke): This is a very approximate check. Improve it!
    expected_name = "builtin.unrealized_conversion_cast"
    if (
        gmem_ref.owner is None
        or gmem_ref.owner.opview.OPERATION_NAME != expected_name
    ):
      raise ValueError("GMEM reference in async_copy must be a kernel argument")
    gmem_ref_ty = ir.MemRefType(gmem_ref.type)
    gmem_strides, _ = gmem_ref_ty.get_strides_and_offset()
    if gmem_strides != utils.get_contiguous_strides(gmem_ref_ty.shape):
      raise NotImplementedError(
          "async_copy assumes the GMEM reference is contiguous"
      )
    if any(s * element_bytewidth % 16 != 0 for s in gmem_strides[:-1]):
      raise ValueError(
          "async_copy requires all GMEM strides except the last one to be a"
          " multiple of 16 bytes"
      )

    # TMA supports OOB indices, so we skip the check.
    base_indices, slice_shape, is_squeezed = utils.parse_indices(
        gmem_slice, ir.MemRefType(gmem_ref.type).shape, check_oob=False
    )
    dyn_base_indices = tuple(
        c(i, index) if not isinstance(i, ir.Value) else i for i in base_indices
    )
    squeezed_dims = [i for i, squeezed in enumerate(is_squeezed) if squeezed]
    sliced_dims = [i for i, squeezed in enumerate(is_squeezed) if not squeezed]
    # Indexing is really slicing + squeezing, and user transforms are meant to
    # apply after that. However, we actually have to apply the indexing last
    # (it's fused into the TMA) and so we need to commute it with all the user
    # transforms. For slicing this is done using transform_index and
    # transform_shape. For squeezing we actually move all the squeezed dims to
    # the front, and then batch each transform, making it ignore the extra dims.
    if squeezed_dims:
      gmem_transform = (TransposeTransform((*squeezed_dims, *sliced_dims)),
                        *(t.batch(len(squeezed_dims)) for t in gmem_transform))

    slice_shape = tuple(slice_shape)
    for t in gmem_transform:
      dyn_base_indices = t.transform_index(dyn_base_indices)
      slice_shape = t.transform_shape(slice_shape)

    num_squeezed_dims = len(squeezed_dims)
    if len(slice_shape) > 5:
      # We can try to collapse all squeezed dims into one.
      if len(slice_shape) - num_squeezed_dims + 1 > 5:
        raise ValueError(
            "Async copies only support striding up to 5 dimensions"
        )
      collapse = CollapseLeadingIndicesTransform(
          tuple(gmem_strides[d] for d in squeezed_dims)
      )
      gmem_transform = (*gmem_transform, collapse)
      dyn_base_indices = collapse.transform_index(dyn_base_indices)
      slice_shape = collapse.transform_shape(slice_shape)
      num_squeezed_dims = 1
    del squeezed_dims, sliced_dims  # Those no longer make sense.

    smem_ref_ty = ir.MemRefType(smem_ref.type)
    # We moved all squeezed dims to the front.
    if slice_shape[num_squeezed_dims:] != tuple(smem_ref_ty.shape):
      raise ValueError(
          "Expected the SMEM reference to have the same shape as the"
          f" transformed slice: {tuple(smem_ref_ty.shape)} != {slice_shape}"
      )
    smem_strides, _ = smem_ref_ty.get_strides_and_offset()
    if smem_strides != utils.get_contiguous_strides(smem_ref_ty.shape):
      raise ValueError(
          "async_copy needs the SMEM reference to be contiguous, but got"
          f" strides {smem_strides} for shape {smem_ref_ty.shape}"
      )

    dyn_base_indices = list(dyn_base_indices)
    slice_shape = list(slice_shape)
    assert all(d == 1 for d in slice_shape[:num_squeezed_dims])
    collective_size = 1
    if collective is not None:
      if isinstance(collective, gpu.Dimension):
        collective = (collective,)
      collective_size = math.prod(self.cluster_size[d] for d in collective)
    if collective_size > 1:
      def partition_dim(dim: int, idx: ir.Value, num_chunks: int):
        # No need to partition squeezed dims. They don't even exist in smem_ref.
        assert dim >= num_squeezed_dims
        nonlocal smem_ref
        slice_shape[dim] //= num_chunks
        block_offset = arith.muli(idx, c(slice_shape[dim], index))
        dyn_base_indices[dim] = arith.addi(dyn_base_indices[dim], block_offset)
        smem_ref = utils.memref_slice(
            smem_ref,
            (slice(None),) * (dim - num_squeezed_dims)
            + (utils.ds(block_offset, slice_shape[dim]),),
        )
      stride = 1
      idx = c(0, index)
      for d in sorted(collective):
        if self.cluster_size[d] == 1:  # Optimize a multiply by 0.
          continue
        idx = arith.addi(idx, arith.muli(gpu.cluster_block_id(d), c(stride, index)))
        stride *= self.cluster_size[d]
      rem_collective_size = collective_size
      for dim, slice_size in enumerate(slice_shape[:-1]):
        if slice_size % rem_collective_size == 0:
          partition_dim(dim, idx, rem_collective_size)
          rem_collective_size = 1
          break
        elif rem_collective_size % slice_size == 0:
          # This is an optimization and it lets us skip squeezed dims.
          if slice_size > 1:
            dim_idx = arith.remui(idx, c(slice_size, index))
            partition_dim(dim, dim_idx, slice_size)
            idx = arith.divui(idx, c(slice_size, index))
            rem_collective_size //= slice_size
        else:
          break  # We failed to partition the leading dimensions.
      del idx  # We overwrote the block index in the loop.
      if rem_collective_size > 1:
        raise ValueError(
            "None of the leading dimensions in the transformed slice shape"
            f" {slice_shape} is divisible by the collective size"
            f" {collective_size}"
        )
      # Make each block load a smaller slice, adjust the GMEM indices and slice
      # the SMEM reference accordingly.
      multicast_mask = arith.trunci(
          i16, utils.cluster_collective_mask(self.cluster_size, collective)
      )
    else:
      multicast_mask = None

    tma_desc = self._get_tma_desc(
        gmem_ref, gmem_transform, tuple(slice_shape), swizzle,
    )

    # We constuct TMA descriptors in column-major order.
    rev_dyn_base_indices = [
        arith.index_cast(i32, idx) for idx in reversed(dyn_base_indices)
    ]

    uniform_ctx = (
        functools.partial(utils.single_thread, per_block=False)
        if uniform and predicate is None
        else contextlib.nullcontext
    )

    if max(slice_shape) > 256:
      raise ValueError(
          "Async copies only support copying <=256 elements along each"
          " dimension"
      )
    if (zeroth_bw := slice_shape[-1] * element_bytewidth) % 16 != 0:
      raise ValueError(
          "Async copies require the number of bytes copied along the last"
          f" dimension to be divisible by 16, but got {zeroth_bw}"
      )
    if swizzle is not None and slice_shape[-1] != swizzle // element_bytewidth:
      raise ValueError(
          f"Async copies with {swizzle=} require last dimension of the slice to"
          f" be exactly {swizzle} bytes"
          f" ({swizzle // element_bytewidth} elements), but got"
          f" {slice_shape[-1]}"
      )
    smem_ptr = utils.memref_ptr(smem_ref, memory_space=3)
    if gmem_ref is src_ref:
      assert barrier is not None  # for pytype
      transfer_bytes = c(
          np.prod(slice_shape) * element_bytewidth * collective_size, i32
      )
      barrier_ptr = barrier.get_ptr()
      with uniform_ctx():
        if arrive:
          nvvm.mbarrier_arrive_expect_tx_shared(
              barrier_ptr, transfer_bytes, predicate=predicate
          )
        nvvm.cp_async_bulk_tensor_shared_cluster_global(
            smem_ptr, tma_desc, rev_dyn_base_indices, barrier_ptr, [],
            multicast_mask=multicast_mask, predicate=predicate
        )
    else:
      with uniform_ctx():
        nvvm.cp_async_bulk_tensor_global_shared_cta(
            tma_desc, smem_ptr, rev_dyn_base_indices, predicate=predicate
        )
        nvvm.cp_async_bulk_commit_group()

  def await_async_copy(
      self, allow_groups: int, await_read_only: bool = False
  ):
    nvvm.cp_async_bulk_wait_group(allow_groups, read=await_read_only)
    utils.warpgroup_barrier()
