# Copyright 2020 The JAX Authors.
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

from jax._src.custom_derivatives import (
  _initial_style_jaxpr,  # noqa: F401
  _sum_tangents,  # noqa: F401
  _zeros_like_pytree,  # noqa: F401
  closure_convert as closure_convert,
  custom_gradient as custom_gradient,
  custom_jvp as custom_jvp,
  custom_jvp_call_p as custom_jvp_call_p,
  custom_jvp_call_jaxpr_p as custom_jvp_call_jaxpr_p,
  custom_vjp as custom_vjp,
  custom_vjp_call_p as custom_vjp_call_p,
  custom_vjp_call_jaxpr_p as custom_vjp_call_jaxpr_p,
  custom_vjp_primal_tree_values as custom_vjp_primal_tree_values,
  CustomVJPPrimal as CustomVJPPrimal,
  linear_call as linear_call,
  remat_opt_p as remat_opt_p,
)

from jax._src.ad_util import (
  SymbolicZero as SymbolicZero,
  zero_from_primal as zero_from_primal
)
