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

"""Module registering a lowering rule for pallas_call on GPU."""


from __future__ import annotations

import os
import time
from typing import Any
import warnings

import jax
from jax._src import core as jax_core
from jax._src.interpreters import mlir
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic_gpu import lowering
import jax.experimental.mosaic.gpu.core as mosaic_core


def pallas_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *args,
    jaxpr: jax_core.Jaxpr,
    name_and_src_info: pallas_core.NameAndSrcInfo,
    interpret: bool,
    debug: bool,
    input_output_aliases: tuple[tuple[int, int], ...],
    grid_mapping: pallas_core.GridMapping,
    compiler_params: dict[str, Any],
    cost_estimate: pallas_core.CostEstimate | None,
    out_avals: tuple[jax_core.AbstractValue, ...],
):
  del interpret, out_avals
  if grid_mapping.num_dynamic_grid_bounds:
    raise NotImplementedError(
        "dynamic grid bounds not supported in the Mosaic GPU backend"
    )

  if debug:
    print(f"\nThe kernel jaxpr for pallas_call {name_and_src_info}:")
    print(jaxpr)
    print(f"The grid mapping for pallas_call {name_and_src_info}:")
    print(grid_mapping)

  lowering_result = lowering.lower_jaxpr_to_module(
      grid_mapping,
      jaxpr,
      name_and_src_info,
      compiler_params,
      cost_estimate,
  )
  if debug:
    print(f"\nThe Mosaic GPU module for pallas_call {name_and_src_info}:")
    print(lowering_result.module.operation)

  module = lowering_result.module
  new_avals_out = [
      jax_core.ShapedArray(t.shape, t.dtype) for t in lowering_result.out_structs
  ]
  outs = mosaic_core._mosaic_gpu_lowering_rule(
      ctx.replace(avals_out=new_avals_out),
      *args,
      module=module,
      out_types=lowering_result.out_structs,
      input_output_aliases=input_output_aliases,
  )
  if (prof_ctx := lowering_result.profiler_context) is not None:
    *outs, prof_buffer = outs
    if (dump_path := prof_ctx.dump_path) == "sponge":
      dump_path = os.getenv("TEST_UNDECLARED_OUTPUTS_DIR")  # type: ignore
    out_file = os.path.join(
        dump_path, f"{name_and_src_info.name}-{time.time_ns()}-trace.json"
    )
    def dump_profile(prof_buffer):
      try:
        with open(out_file, "x") as f:
          prof_ctx.spec.dump(
              prof_buffer,
              f,
              grid=lowering_result.grid,
              block=lowering_result.block,
          )
      except FileExistsError:
        warnings.warn(
            f"Failed to dump profile for pallas_call {name_and_src_info}, "
            f"profile already exists at {out_file}"
        )
    def do_callback(prof_buffer):
      jax.debug.callback(dump_profile, prof_buffer)
      return ()
    mlir.lower_fun(do_callback, multiple_results=True)(
        ctx.replace(avals_in=(new_avals_out[-1],)), prof_buffer
    )
  return outs
