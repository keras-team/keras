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

from jax import ShapeDtypeStruct as ShapeDtypeStruct
from jax._src.lib import mosaic_gpu_dialect as dialect  # noqa: F401

from .core import (
    Barrier as Barrier,
    ClusterBarrier as ClusterBarrier,
    TMABarrier as TMABarrier,
    ThreadSemantics as ThreadSemantics,
    Union as Union,
    as_gpu_kernel as as_gpu_kernel,
)

from .launch_context import (
    LaunchContext as LaunchContext,
    MemRefTransform as MemRefTransform,
    TileTransform as TileTransform,
    TransposeTransform as TransposeTransform,
)

if dialect is not None:
  from .dialect_lowering import (
      lower_mgpu_dialect as lower_mgpu_dialect,
  )
  from .layout_inference import (
      infer_layout as infer_layout,
  )
else:
  lower_mgpu_dialect = None
  infer_layout = None


from .fragmented_array import (
    FragmentedArray as FragmentedArray,
    FragmentedLayout as FragmentedLayout,
    WGMMA_LAYOUT as WGMMA_LAYOUT,
    WGMMA_ROW_LAYOUT as WGMMA_ROW_LAYOUT,
    WGMMAFragLayout as WGMMAFragLayout,
    WGMMARowFragLayout as WGMMARowFragLayout,
    WGSplatFragLayout as WGSplatFragLayout,
    WGStridedFragLayout as WGStridedFragLayout,
    optimization_barrier as optimization_barrier,
)
from .layouts import (
    from_strided_fragmented_layout_attr as from_strided_fragmented_layout_attr,
    is_strided_fragmented_layout as is_strided_fragmented_layout,
    to_splat_fragmented_layout_attr as to_splat_fragmented_layout_attr,
    to_strided_fragmented_layout_attr as to_strided_fragmented_layout_attr,
)
from .utils import (
    BarrierRef as BarrierRef,
    CollectiveBarrierRef as CollectiveBarrierRef,
    DynamicSlice as DynamicSlice,
    Partition as Partition,
    Partition1D as Partition1D,
    bytewidth as bytewidth,
    c as c,
    commit_shared as commit_shared,
    debug_print as debug_print,
    ds as ds,
    fori as fori,
    memref_fold as memref_fold,
    memref_slice as memref_slice,
    memref_reshape as memref_reshape,
    memref_transpose as memref_transpose,
    memref_unfold as memref_unfold,
    memref_unsqueeze as memref_unsqueeze,
    single_thread as single_thread,
    single_thread_predicate as single_thread_predicate,
    thread_idx as thread_idx,
    tile_shape as tile_shape,
    warp_idx as warp_idx,
    warpgroup_barrier as warpgroup_barrier,
    warpgroup_idx as warpgroup_idx,
    when as when,
)
from .wgmma import (
    WGMMAAccumulator as WGMMAAccumulator,
    WGMMALayout as WGMMALayout,
    wgmma as wgmma,
)
