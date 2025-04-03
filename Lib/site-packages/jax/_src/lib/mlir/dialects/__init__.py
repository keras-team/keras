# Copyright 2021 The JAX Authors.
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

# ruff: noqa: F401

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
  from jaxlib.mlir.dialects import arith as arith
  from jaxlib.mlir.dialects import builtin as builtin
  from jaxlib.mlir.dialects import chlo as chlo
  from jaxlib.mlir.dialects import func as func
  from jaxlib.mlir.dialects import gpu as gpu
  from jaxlib.mlir.dialects import llvm as llvm
  from jaxlib.mlir.dialects import math as math
  from jaxlib.mlir.dialects import memref as memref
  from jaxlib.mlir.dialects import mhlo as mhlo
  from jaxlib.mlir.dialects import nvgpu as nvgpu
  from jaxlib.mlir.dialects import nvvm as nvvm
  from jaxlib.mlir.dialects import scf as scf
  from jaxlib.mlir.dialects import sparse_tensor as sparse_tensor
  from jaxlib.mlir.dialects import vector as vector
else:
  from jax._src import lazy_loader as _lazy
  __getattr__, __dir__, __all__ = _lazy.attach("jaxlib.mlir.dialects", [
      "arith",
      "builtin",
      "chlo",
      "func",
      "gpu",
      "llvm",
      "math",
      "memref",
      "mhlo",
      "nvgpu",
      "nvvm",
      "scf",
      "sparse_tensor",
      "vector",
  ])
  del _lazy

# TODO(bartchr): Once JAX is released with SDY, remove the try/except.
try:
  from jaxlib.mlir.dialects import sdy as sdy
except ImportError:
  sdy: Any = None  # type: ignore[no-redef]

# Alias that is set up to abstract away the transition from MHLO to StableHLO.
from jaxlib.mlir.dialects import stablehlo as hlo
