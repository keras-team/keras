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

"""Module for GPU-specific JAX primitives."""

from __future__ import annotations

from collections.abc import Sequence

import jax
from jax._src import core as jax_core
from jax._src.lib.mlir.dialects import gpu as gpu_dialect
from jax._src.lib.triton import dialect as tt_dialect
from jax._src.pallas.triton import lowering
from jax.interpreters import mlir
import jax.numpy as jnp


def approx_tanh(x: jax.Array) -> jax.Array:
  r"""Elementwise approximate hyperbolic tangent: :math:`\mathrm{tanh}(x)`.

  See
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-tanh.
  """
  if x.dtype == jnp.float16:
    asm = "tanh.approx.f16 $0, $1;"
    constraint = "h"
  elif x.dtype == jnp.bfloat16:
    asm = "tanh.approx.bf16 $0, $1;"
    constraint = "h"
  elif x.dtype == jnp.float32:
    asm = "tanh.approx.f32 $0, $1;"
    constraint = "f"
  else:
    raise TypeError(f"approx_tanh does not accept {x.dtype} arrays")

  [result] = elementwise_inline_asm(
      asm,
      args=[x],
      constraints=f"={constraint},{constraint}",
      pack=1,
      result_shape_dtypes=[jax.ShapeDtypeStruct(x.shape, x.dtype)],
  )
  return result


def elementwise_inline_asm(
    asm: str,
    *,
    args: Sequence[jax.Array],
    constraints: str,
    pack: int,
    result_shape_dtypes: Sequence[jax.ShapeDtypeStruct],
) -> Sequence[jax.Array]:
  """Inline assembly applying an elementwise operation.

  Args:
    asm: The assembly code to run.
    args: The arguments to pass to the assembly code.
    constraints: LLVM inline assembly `constraints
      <https://llvm.org/docs/LangRef.html#inline-asm-constraint-string>`_.
    pack: The number of elements from each argument expected by a single
      instance of the assembly code.
    result_shape_dtypes: The shapes and dtypes of the results produced by the
      assembly code.

  Returns:
    The results produced by the assembly code.
  """
  return elementwise_inline_asm_p.bind(
      *args,
      asm=asm,
      constraints=constraints,
      pack=pack,
      result_shape_dtypes=result_shape_dtypes,
  )


elementwise_inline_asm_p = jax_core.Primitive("elementwise_inline_asm_p")
elementwise_inline_asm_p.multiple_results = True


@elementwise_inline_asm_p.def_abstract_eval
def _elementwise_inline_asm_abstract_eval(
    *avals: jax_core.ShapedArray, result_shape_dtypes, **kwargs
) -> Sequence[jax_core.ShapedArray]:
  del kwargs  # Unused.
  if not all(x.shape == y.shape for x, y in zip(avals, avals[1:])):
    raise ValueError(
        "All arguments of elementwise_inline_asm must have the same shape"
    )
  return [jax_core.ShapedArray(s.shape, s.dtype) for s in result_shape_dtypes]


@lowering.register_lowering(elementwise_inline_asm_p)
def _elementwise_inline_asm_lowering(
    ctx: lowering.LoweringRuleContext,
    *args,
    asm,
    constraints,
    pack,
    result_shape_dtypes,
):
  del result_shape_dtypes  # Unused.
  return tt_dialect.ElementwiseInlineAsmOp(
      [*map(mlir.aval_to_ir_type, ctx.avals_out)],
      asm,
      constraints=constraints,
      pure=True,
      packed_element=pack,
      args=args,
  ).result


def debug_barrier() -> None:
  """Synchronizes all kernel executions in the grid."""
  return debug_barrier_p.bind()


debug_barrier_p = jax_core.Primitive("debug_barrier_p")
debug_barrier_p.multiple_results = True

@debug_barrier_p.def_abstract_eval
def _debug_barrier_abstract_eval() -> Sequence[jax_core.ShapedArray]:
  return ()

@lowering.register_lowering(debug_barrier_p)
def _debug_barrier_lowering(ctx: lowering.LoweringRuleContext):
  del ctx  # Unused.
  gpu_dialect.barrier()
  return []
