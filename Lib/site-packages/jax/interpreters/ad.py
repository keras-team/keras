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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from __future__ import annotations

from jax._src.interpreters.ad import (
  JVPTrace as JVPTrace,
  JVPTracer as JVPTracer,
  UndefinedPrimal as UndefinedPrimal,
  Zero as Zero,
  add_jaxvals as add_jaxvals,
  add_jaxvals_p as add_jaxvals_p,
  add_tangents as add_tangents,
  backward_pass as backward_pass_internal,
  bilinear_transpose as bilinear_transpose,
  call_param_updaters as call_param_updaters,
  call_transpose as call_transpose,
  call_transpose_param_updaters as call_transpose_param_updaters,
  closed_backward_pass as closed_backward_pass,
  custom_lin_p as custom_lin_p,
  defbilinear as defbilinear,
  defjvp as defjvp,
  defjvp2 as defjvp2,
  defjvp_zero as defjvp_zero,
  deflinear as deflinear,
  deflinear2 as deflinear2,
  f_jvp_traceable as f_jvp_traceable,
  get_primitive_transpose as get_primitive_transpose,
  instantiate_zeros as instantiate_zeros,
  is_undefined_primal as is_undefined_primal,
  jvp as jvp,
  jvp_jaxpr as jvp_jaxpr,
  jvp_subtrace as jvp_subtrace,
  jvp_subtrace_aux as jvp_subtrace_aux,
  jvpfun as jvpfun,
  linear_jvp as linear_jvp,
  linear_transpose as linear_transpose,
  linear_transpose2 as linear_transpose2,
  linearize as linearize,
  map_transpose as map_transpose,
  nonzero_outputs as nonzero_outputs,
  nonzero_tangent_outputs as nonzero_tangent_outputs,
  primitive_jvps as primitive_jvps,
  primitive_transposes as primitive_transposes,
  rearrange_binders as rearrange_binders,
  reducing_transposes as reducing_transposes,
  standard_jvp as standard_jvp,
  standard_jvp2 as standard_jvp2,
  traceable as traceable,
  unpair_pval as unpair_pval,
  vjp as vjp,
  zero_jvp as zero_jvp,
  zeros_like_aval as zeros_like_aval,
  zeros_like_p as zeros_like_p,
)


def backward_pass(jaxpr, reduce_axes, transform_stack,
                  consts, primals_in, cotangents_in):
  if reduce_axes:
    raise NotImplementedError("reduce_axes on ad.backward_pass is deprecated")
  del reduce_axes
  return backward_pass_internal(
      jaxpr, transform_stack, consts, primals_in, cotangents_in)
