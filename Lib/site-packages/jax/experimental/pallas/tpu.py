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

"""Mosaic-specific Pallas APIs."""

from jax._src.pallas.mosaic import core as core
from jax._src.pallas.mosaic.core import create_tensorcore_mesh as create_tensorcore_mesh
from jax._src.pallas.mosaic.core import dma_semaphore as dma_semaphore
from jax._src.pallas.mosaic.core import PrefetchScalarGridSpec as PrefetchScalarGridSpec
from jax._src.pallas.mosaic.core import semaphore as semaphore
from jax._src.pallas.mosaic.core import SemaphoreType as SemaphoreType
from jax._src.pallas.mosaic.core import TPUMemorySpace as TPUMemorySpace
from jax._src.pallas.mosaic.core import TPUCompilerParams as TPUCompilerParams
from jax._src.pallas.mosaic.core import runtime_assert_enabled as runtime_assert_enabled
from jax._src.pallas.mosaic.core import _ENABLE_RUNTIME_ASSERT as enable_runtime_assert  # noqa: F401
from jax._src.pallas.mosaic.helpers import sync_copy as sync_copy
from jax._src.pallas.mosaic.lowering import LoweringException as LoweringException
from jax._src.pallas.mosaic.pipeline import ARBITRARY as ARBITRARY
from jax._src.pallas.mosaic.pipeline import BufferedRef as BufferedRef
from jax._src.pallas.mosaic.pipeline import emit_pipeline as emit_pipeline
from jax._src.pallas.mosaic.pipeline import emit_pipeline_with_allocations as emit_pipeline_with_allocations
from jax._src.pallas.mosaic.pipeline import get_pipeline_schedule as get_pipeline_schedule
from jax._src.pallas.mosaic.pipeline import make_pipeline_allocations as make_pipeline_allocations
from jax._src.pallas.mosaic.pipeline import PARALLEL as PARALLEL
from jax._src.pallas.mosaic.primitives import async_copy as async_copy
from jax._src.pallas.mosaic.primitives import async_remote_copy as async_remote_copy
from jax._src.pallas.mosaic.primitives import bitcast as bitcast
from jax._src.pallas.mosaic.primitives import delay as delay
from jax._src.pallas.mosaic.primitives import device_id as device_id
from jax._src.pallas.mosaic.primitives import DeviceIdType as DeviceIdType
from jax._src.pallas.mosaic.primitives import get_barrier_semaphore as get_barrier_semaphore
from jax._src.pallas.mosaic.primitives import make_async_copy as make_async_copy
from jax._src.pallas.mosaic.primitives import make_async_remote_copy as make_async_remote_copy
from jax._src.pallas.mosaic.primitives import prng_random_bits as prng_random_bits
from jax._src.pallas.mosaic.primitives import prng_seed as prng_seed
from jax._src.pallas.mosaic.primitives import repeat as repeat
from jax._src.pallas.mosaic.primitives import roll as roll
from jax._src.pallas.mosaic.primitives import semaphore_read as semaphore_read
from jax._src.pallas.mosaic.primitives import semaphore_signal as semaphore_signal
from jax._src.pallas.mosaic.primitives import semaphore_wait as semaphore_wait
from jax._src.pallas.mosaic.random import to_pallas_key as to_pallas_key

import types
from jax._src.pallas.mosaic.verification import assume
from jax._src.pallas.mosaic.verification import pretend
from jax._src.pallas.mosaic.verification import skip
from jax._src.pallas.mosaic.verification import define_model
verification = types.SimpleNamespace(
    assume=assume, pretend=pretend, skip=skip, define_model=define_model
)
del types, assume, pretend, skip, define_model  # Clean up.

ANY = TPUMemorySpace.ANY
CMEM = TPUMemorySpace.CMEM
SMEM = TPUMemorySpace.SMEM
VMEM = TPUMemorySpace.VMEM
SEMAPHORE = TPUMemorySpace.SEMAPHORE
