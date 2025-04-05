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

"""Triton-specific Pallas APIs."""

from jax._src.pallas.triton.core import TritonCompilerParams as TritonCompilerParams
from jax._src.pallas.triton.primitives import approx_tanh as approx_tanh
from jax._src.pallas.triton.primitives import debug_barrier as debug_barrier
from jax._src.pallas.triton.primitives import elementwise_inline_asm as elementwise_inline_asm
