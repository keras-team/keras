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

"""Experimental GPU backend for Pallas targeting H100.

These APIs are highly unstable and can change weekly. Use at your own risk.
"""

from jax._src.pallas.mosaic_gpu.core import Barrier as Barrier
from jax._src.pallas.mosaic_gpu.core import GPUBlockSpec as GPUBlockSpec
from jax._src.pallas.mosaic_gpu.core import GPUCompilerParams as GPUCompilerParams
from jax._src.pallas.mosaic_gpu.core import GPUMemorySpace as GPUMemorySpace
from jax._src.pallas.mosaic_gpu.core import GPUMesh as GPUMesh
from jax._src.pallas.mosaic_gpu.core import SwizzleTransform as SwizzleTransform
from jax._src.pallas.mosaic_gpu.core import TilingTransform as TilingTransform
from jax._src.pallas.mosaic_gpu.core import transpose_ref as transpose_ref
from jax._src.pallas.mosaic_gpu.core import TransposeTransform as TransposeTransform
from jax._src.pallas.mosaic_gpu.core import WGMMAAccumulatorRef as ACC  # noqa: F401
from jax._src.pallas.mosaic_gpu.core import WGMMAAccumulatorRef as WGMMAAccumulatorRef
from jax._src.pallas.mosaic_gpu.pipeline import emit_pipeline as emit_pipeline
from jax._src.pallas.mosaic_gpu.primitives import barrier_arrive as barrier_arrive
from jax._src.pallas.mosaic_gpu.primitives import barrier_wait as barrier_wait
from jax._src.pallas.mosaic_gpu.primitives import broadcasted_iota as broadcasted_iota
from jax._src.pallas.mosaic_gpu.primitives import commit_smem as commit_smem
from jax._src.pallas.mosaic_gpu.primitives import copy_gmem_to_smem as copy_gmem_to_smem
from jax._src.pallas.mosaic_gpu.primitives import copy_smem_to_gmem as copy_smem_to_gmem
from jax._src.pallas.mosaic_gpu.primitives import Layout as Layout
from jax._src.pallas.mosaic_gpu.primitives import layout_cast as layout_cast
from jax._src.pallas.mosaic_gpu.primitives import set_max_registers as set_max_registers
from jax._src.pallas.mosaic_gpu.primitives import wait_smem_to_gmem as wait_smem_to_gmem
from jax._src.pallas.mosaic_gpu.primitives import wgmma as wgmma
from jax._src.pallas.mosaic_gpu.primitives import wgmma_wait as wgmma_wait


#: Alias of :data:`jax.experimental.pallas.mosaic_gpu.GPUMemorySpace.GMEM`.
GMEM = GPUMemorySpace.GMEM
#: Alias of :data:`jax.experimental.pallas.mosaic_gpu.GPUMemorySpace.SMEM`.
SMEM = GPUMemorySpace.SMEM
