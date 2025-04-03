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

from jax._src.interpreters.mlir import (
  AxisContext as AxisContext,
  ConstantHandler as ConstantHandler,
  DEVICE_TO_DEVICE_TYPE as DEVICE_TO_DEVICE_TYPE,
  LoweringParameters as LoweringParameters,
  LoweringResult as LoweringResult,
  LoweringRule as LoweringRule,
  LoweringRuleContext as LoweringRuleContext,
  ModuleContext as ModuleContext,
  RECV_FROM_HOST_TYPE as RECV_FROM_HOST_TYPE,
  SEND_TO_HOST_TYPE as SEND_TO_HOST_TYPE,
  ShapePolyLoweringState as ShapePolyLoweringState,
  Token as Token,
  TokenSet as TokenSet,
  Value as Value,
  call_lowering as _call_lowering,  # noqa: F401
  _lowerings as _lowerings,
  _platform_specific_lowerings as _platform_specific_lowerings,
  aval_to_ir_type as aval_to_ir_type,
  aval_to_ir_types as aval_to_ir_types,
  core_call_lowering as core_call_lowering,
  custom_call as custom_call,
  dense_bool_elements as dense_bool_elements,
  dense_bool_array as dense_bool_array,
  dense_int_array as dense_int_array,
  dense_int_elements as dense_int_elements,
  dtype_to_ir_type as dtype_to_ir_type,
  emit_python_callback as emit_python_callback,
  flatten_ir_types as flatten_ir_types,
  flatten_ir_values as flatten_lowering_ir_args,  # TODO(phawkins): remove me  # noqa: F401
  flatten_ir_values as flatten_ir_values,
  unflatten_ir_values_like_types as unflatten_ir_values_like_types,
  func_dialect as func_dialect,
  hlo as hlo,
  i32_attr as i32_attr,
  i64_attr as i64_attr,
  ir as ir,
  ir_constant as ir_constant,
  ir_type_handlers as ir_type_handlers,
  jaxpr_subcomp as jaxpr_subcomp,
  lower_fun as lower_fun,
  lower_jaxpr_to_fun as lower_jaxpr_to_fun,
  lower_jaxpr_to_module as lower_jaxpr_to_module,
  lowerable_effects as lowerable_effects,
  make_ir_context as make_ir_context,
  merge_mlir_modules as merge_mlir_modules,
  module_to_bytecode as module_to_bytecode,
  module_to_string as module_to_string,
  register_constant_handler as register_constant_handler,
  register_lowering as register_lowering,
  shape_tensor as shape_tensor,
  token_type as token_type,
  xla_computation_to_mlir_module as xla_computation_to_mlir_module,
)

from jax._src.mesh import Mesh as Mesh
from jax._src.sharding_impls import (
  MeshAxisName as MeshAxisName,
  ReplicaAxisContext as ReplicaAxisContext,
  SPMDAxisContext as SPMDAxisContext,
  ShardingContext as ShardingContext,
)
