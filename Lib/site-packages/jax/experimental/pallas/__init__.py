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

"""Module for Pallas, a JAX extension for custom kernels.

See the Pallas documentation at
https://jax.readthedocs.io/en/latest/pallas.html.
"""

from jax._src.pallas.core import Blocked as Blocked
from jax._src.pallas.core import BlockSpec as BlockSpec
from jax._src.pallas.core import CompilerParams as CompilerParams
from jax._src.pallas.core import core_map as core_map
from jax._src.pallas.core import CostEstimate as CostEstimate
from jax._src.pallas.core import lower_as_mlir as lower_as_mlir
from jax._src.pallas.core import GridSpec as GridSpec
from jax._src.pallas.core import IndexingMode as IndexingMode
from jax._src.pallas.core import MemoryRef as MemoryRef
from jax._src.pallas.core import MemorySpace as MemorySpace
from jax._src.pallas.core import no_block_spec as no_block_spec
from jax._src.pallas.core import Unblocked as Unblocked
from jax._src.pallas.core import unblocked as unblocked
from jax._src.pallas.cost_estimate import estimate_cost as estimate_cost
from jax._src.pallas.helpers import empty as empty
from jax._src.pallas.helpers import empty_like as empty_like
from jax._src.pallas.pallas_call import pallas_call as pallas_call
from jax._src.pallas.pallas_call import pallas_call_p as pallas_call_p
from jax._src.pallas.primitives import atomic_add as atomic_add
from jax._src.pallas.primitives import atomic_and as atomic_and
from jax._src.pallas.primitives import atomic_cas as atomic_cas
from jax._src.pallas.primitives import atomic_max as atomic_max
from jax._src.pallas.primitives import atomic_min as atomic_min
from jax._src.pallas.primitives import atomic_or as atomic_or
from jax._src.pallas.primitives import atomic_xchg as atomic_xchg
from jax._src.pallas.primitives import atomic_xor as atomic_xor
from jax._src.pallas.primitives import debug_print as debug_print
from jax._src.pallas.primitives import dot as dot
from jax._src.pallas.primitives import load as load
from jax._src.pallas.primitives import max_contiguous as max_contiguous
from jax._src.pallas.primitives import multiple_of as multiple_of
from jax._src.pallas.primitives import num_programs as num_programs
from jax._src.pallas.primitives import program_id as program_id
from jax._src.pallas.primitives import run_scoped as run_scoped
from jax._src.pallas.primitives import store as store
from jax._src.pallas.primitives import swap as swap
from jax._src.pallas.utils import cdiv as cdiv
from jax._src.pallas.utils import next_power_of_2 as next_power_of_2
from jax._src.pallas.utils import strides_from_shape as strides_from_shape
from jax._src.pallas.utils import when as when
from jax._src.state.discharge import run_state as run_state
from jax._src.state.indexing import ds as ds
from jax._src.state.indexing import dslice as dslice
from jax._src.state.indexing import Slice as Slice
from jax._src.state.primitives import broadcast_to as broadcast_to


ANY = MemorySpace.ANY
