# Copyright 2018 The JAX Authors.
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

from jax._src.interpreters.partial_eval import (
  AbstractedAxesSpec as AbstractedAxesSpec,
  AbstractedAxisName as AbstractedAxisName,
  BoundedAxisSize as BoundedAxisSize,
  Const as Const,
  ConstFoldRule as ConstFoldRule,
  ConstVar as ConstVar,
  DCERule as DCERule,
  DynamicJaxprTrace as DynamicJaxprTrace,
  DynamicJaxprTracer as DynamicJaxprTracer,
  ForwardingRule as ForwardingRule,
  FreeVar as FreeVar,
  JaxprEqnRecipe as JaxprEqnRecipe,
  JaxprStackFrame as JaxprStackFrame,
  JaxprTrace as JaxprTrace,
  JaxprTracer as JaxprTracer,
  JaxprTracerRecipe as JaxprTracerRecipe,
  LambdaBinding as LambdaBinding,
  ParamsUpdater as ParamsUpdater,
  PartialEvalCustomResult as PartialEvalCustomResult,
  PartialEvalCustomRule as PartialEvalCustomRule,
  PartialVal as PartialVal,
  ResAvalUpdater as ResAvalUpdater,
  TracerAsName as TracerAsName,
  TracerId as TracerId,
  Val as Val,
  abstract_eval_fun as abstract_eval_fun,
  call_padding_rule as call_padding_rule,
  call_param_updaters as call_param_updaters,
  call_partial_eval_custom_rule as call_partial_eval_custom_rule,
  call_partial_eval_rules as call_partial_eval_rules,
  close_jaxpr as close_jaxpr,
  closed_call_partial_eval_custom_rule as closed_call_partial_eval_custom_rule,
  config as config,
  const_fold_rules as const_fold_rules,
  convert_constvars_jaxpr as convert_constvars_jaxpr,
  convert_envvars_to_constvars as convert_envvars_to_constvars,
  convert_invars_to_constvars as convert_invars_to_constvars,
  custom_partial_eval_rules as custom_partial_eval_rules,
  custom_staging_rules as custom_staging_rules,
  dce_jaxpr as dce_jaxpr,
  dce_jaxpr_call_rule as dce_jaxpr_call_rule,
  dce_jaxpr_closed_call_rule as dce_jaxpr_closed_call_rule,
  dce_jaxpr_consts as dce_jaxpr_consts,
  dce_rules as dce_rules,
  tracing_debug_info as tracing_debug_info,
  tracing_debug_info_final as tracing_debug_info_final,
  def_trivial_padding as def_trivial_padding,
  forwarding_rules as forwarding_rules,
  has_effects as has_effects,
  infer_lambda_input_type as infer_lambda_input_type,
  instantiate_const_at as instantiate_const_at,
  make_jaxpr_effects as make_jaxpr_effects,
  move_binders_to_back as move_binders_to_back,
  move_binders_to_front as move_binders_to_front,
  new_eqn_recipe as new_eqn_recipe,
  pad_jaxpr as pad_jaxpr,
  padding_rules as padding_rules,
  partial_eval_jaxpr_custom as partial_eval_jaxpr_custom,
  partial_eval_jaxpr_custom_rule_not_implemented as partial_eval_jaxpr_custom_rule_not_implemented,
  partial_eval_jaxpr_custom_rules as partial_eval_jaxpr_custom_rules,
  partial_eval_jaxpr_nounits as partial_eval_jaxpr_nounits,
  partial_eval_wrapper_nounits as partial_eval_wrapper_nounits,
  partition_pvals as partition_pvals,
  recipe_to_eqn as recipe_to_eqn,
  trace_to_jaxpr_dynamic as _trace_to_jaxpr_dynamic,
  trace_to_jaxpr_dynamic2 as trace_to_jaxpr_dynamic2,
  trace_to_jaxpr_nounits as trace_to_jaxpr_nounits,
  trace_to_subjaxpr_nounits as trace_to_subjaxpr_nounits,
  trace_to_subjaxpr_nounits_fwd as trace_to_subjaxpr_nounits_fwd,
  tracers_to_jaxpr as tracers_to_jaxpr,
  trivial_ctx as trivial_ctx,
)


# TODO(mattjj): remove temporary shim when trace_to_jaxpr_dynamic sig stabilizes
def trace_to_jaxpr_dynamic(fun, in_avals, debug_info=None, *, keep_inputs=None):  # noqa
  jaxpr, out_avals, consts, () = _trace_to_jaxpr_dynamic(
      fun, in_avals, debug_info, keep_inputs=keep_inputs)
  return jaxpr, out_avals, consts


from jax._src.core import Jaxpr as Jaxpr
