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

"""Contains Triton-specific Pallas abstractions."""
from __future__ import annotations

import dataclasses
from typing import ClassVar

from jax._src.pallas import core as pallas_core

@dataclasses.dataclass(frozen=True)
class TritonCompilerParams(pallas_core.CompilerParams):
  """Compiler parameters for Triton.

  Attributes:
    num_warps: The number of warps to use for the kernel. Each warp consists of
      32 threads.
    num_stages: The number of stages the compiler should use for software
      pipelining loops.
    serialized_metadata: Additional compiler metadata. This field is unstable
      and may be removed in the future.
  """
  PLATFORM: ClassVar[str] = "triton"
  num_warps: int | None = None
  num_stages: int | None = None
  serialized_metadata: bytes | None = None
