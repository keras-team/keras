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
"""Helper tool for automatic cost estimation."""
import dataclasses
import functools
import math
from typing import Any, Sequence

import jax
from jax._src import api_util
from jax._src import core as jax_core
from jax._src import custom_derivatives
from jax._src import linear_util as lu
from jax._src import pjit
from jax._src.state import discharge
from jax._src.pallas import core as pallas_core
from jax._src.interpreters import partial_eval as pe
from jax._src.util import safe_map
from jax._src.util import safe_zip
from jax._src.lax import lax

map, unsafe_map = safe_map, map  # pylint: disable=redefined-builtin
zip, unsafe_zip = safe_zip, zip  # pylint: disable=redefined-builtin

_cost_rules = {}

@dataclasses.dataclass(frozen=True)
class CostEstimate:
  flops: int
  transcendentals: int
  bytes_accessed: int

  def __add__(self, other: 'CostEstimate') -> 'CostEstimate':
    return CostEstimate(
        flops=self.flops + other.flops,
        transcendentals=self.transcendentals + other.transcendentals,
        bytes_accessed=self.bytes_accessed + other.bytes_accessed,
    )

def register_cost_rule(primitive: jax_core.Primitive, rule):
  _cost_rules[primitive] = rule

@dataclasses.dataclass(frozen=True)
class Context:
  avals_in: Sequence[Any]
  avals_out: Sequence[Any]

def cost_estimate_jaxpr(
    jaxpr: jax_core.ClosedJaxpr,
) -> pallas_core.CostEstimate:
  """Returns the cost estimate for the given Jaxpr."""
  jaxpr, _ = jaxpr.jaxpr, jaxpr.consts
  total_cost = CostEstimate(flops=0, transcendentals=0, bytes_accessed=0)

  for eqn in jaxpr.eqns:
    _, bind_params = eqn.primitive.get_bind_params(eqn.params)
    rule = _cost_rules.get(eqn.primitive, None)
    if rule is not None:
      context = Context(avals_in=[v.aval for v in eqn.invars],
                        avals_out=[v.aval for v in eqn.outvars])
      op_cost = rule(context, **bind_params)
      total_cost = total_cost + op_cost
  return pallas_core.CostEstimate(
      flops=total_cost.flops,
      transcendentals=total_cost.transcendentals,
      bytes_accessed=total_cost.bytes_accessed,
  )

def estimate_cost(fun, *args, **kwargs) -> pallas_core.CostEstimate:
  """Computes a cost estimate for the given function.

  Args:
    fun: The function to compute the cost estimate for.
    *args: The arguments to the function. Can be jax.ShapeDtypeStruct or
      jax.Array.
    **kwargs: The keyword arguments to the function.

  Returns:
    A pallas_core.CostEstimate object containing the cost estimate.
  """
  flattened_args, treedef = jax.tree.flatten(args)
  partial_fun = functools.partial(fun, **kwargs)
  wrapped_fun, _ = api_util.flatten_fun_nokwargs(lu.wrap_init(partial_fun),
                                                 treedef)
  avals = [jax_core.ShapedArray(a.shape, a.dtype) for a in flattened_args]
  jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(wrapped_fun, avals)
  estimate = cost_estimate_jaxpr(jax_core.ClosedJaxpr(jaxpr, consts))
  input_bytes = sum(
      math.prod(a.shape) * a.dtype.itemsize for a in flattened_args)
  output_bytes = sum(
      math.prod(a.aval.shape) * a.aval.dtype.itemsize for a in jaxpr.outvars)
  return pallas_core.CostEstimate(
      flops=estimate.flops,
      transcendentals=estimate.transcendentals,
      bytes_accessed=estimate.bytes_accessed + input_bytes + output_bytes,
  )

def binary_cost_rule(ctx: Context, **_) -> CostEstimate:
  aval_out, = ctx.avals_out
  out_flops = math.prod(aval_out.shape)
  return CostEstimate(
      flops=out_flops,
      transcendentals=0,
      bytes_accessed=0,
  )
BINARY_OPS = [
    lax.add_p,
    lax.mul_p,
    lax.sub_p,
    lax.div_p,
    lax.min_p,
    lax.max_p,
    lax.or_p,
    lax.and_p,
    lax.xor_p,
]
for op in BINARY_OPS:
  register_cost_rule(op, binary_cost_rule)


def unary_cost_rule(transcendental: bool):
  def cost_rule(ctx: Context, **_) -> CostEstimate:
    x_aval, = ctx.avals_in
    new_flops = 0
    new_transcendentals = 0
    if transcendental:
      new_transcendentals += math.prod(x_aval.shape)
    else:
      new_flops += math.prod(x_aval.shape)
    return CostEstimate(
        flops=new_flops,
        transcendentals=new_transcendentals,
        bytes_accessed=0,
    )
  return cost_rule

UN_OPS = [
    lax.neg_p,
    lax.floor_p,
    lax.ceil_p,
    lax.round_p,
    lax.not_p,
]
for op in UN_OPS:
  register_cost_rule(op, unary_cost_rule(transcendental=False))

TRANSCENDENTAL_OPS = [
    lax.cos_p,
    lax.sin_p,
    lax.tan_p,
    lax.sinh_p,
    lax.cosh_p,
    lax.tanh_p,
    lax.acos_p,
    lax.asin_p,
    lax.atan_p,
    lax.exp_p,
    lax.log_p,
    lax.logistic_p,
    lax.sqrt_p,
]
for op in TRANSCENDENTAL_OPS:
  register_cost_rule(op, unary_cost_rule(transcendental=True))

def _integer_pow_cost_rule(ctx: Context, *, y: int) -> CostEstimate:
  x_aval, = ctx.avals_in
  num_elements = math.prod(x_aval.shape)
  if y == 0 or y == 1:
    # No flops, the result is 0 or a copy of the input.
    cost_per_element = 0
  else:
    # We assume integer pow is implemented using repeated squaring.
    # The cost is log(y) squarings, plus one multiply per non-zero bit.
    highest_bit = math.floor(math.log(y, 2))
    cost_per_element = highest_bit + y.bit_count()
  return CostEstimate(
      flops=num_elements * cost_per_element,
      transcendentals=0,
      bytes_accessed=0,
  )
register_cost_rule(lax.integer_pow_p, _integer_pow_cost_rule)

def dot_general_cost_rule(ctx: Context,
                          dimension_numbers: lax.DotDimensionNumbers,
                          **_) -> CostEstimate:
  x_aval, y_aval = ctx.avals_in
  x_shape, y_shape = x_aval.shape, y_aval.shape
  (lhs_contracting_dims, rhs_contracting_dims), (
      lhs_batch_dims, rhs_batch_dims) = dimension_numbers
  assert len(lhs_contracting_dims) == len(rhs_contracting_dims)
  assert len(lhs_batch_dims) == len(rhs_batch_dims)
  flops = 1
  # Flops along a contracting dim is 2*dim (addition and multiplication)
  for i in range(len(lhs_contracting_dims)):
    lhs_dim, rhs_dim = lhs_contracting_dims[i], rhs_contracting_dims[i]
    assert x_shape[lhs_dim] == y_shape[rhs_dim]
    flops *= 2 * x_shape[lhs_dim]
  # Now we handle all other dimensions.
  for i, lhs_dim in enumerate(x_shape):
    if i in lhs_contracting_dims:
      continue
    flops *= lhs_dim
  for i, rhs_dim in enumerate(y_shape):
    if i in rhs_contracting_dims:
      continue
    # Don't double-count batch dims (we already counted for LHS)
    if i in rhs_batch_dims:
      continue
    flops *= rhs_dim
  return CostEstimate(
      flops=flops,
      transcendentals=0,
      bytes_accessed=0,
  )
register_cost_rule(lax.dot_general_p, dot_general_cost_rule)

# Higher-order primitives
def _pjit_cost_rule(ctx, *, jaxpr: jax_core.ClosedJaxpr, **_):
  del ctx
  inner_cost = cost_estimate_jaxpr(jaxpr)
  return CostEstimate(
      flops=inner_cost.flops,
      transcendentals=inner_cost.transcendentals,
      bytes_accessed=inner_cost.bytes_accessed,
  )
register_cost_rule(pjit.pjit_p, _pjit_cost_rule)

def _custom_vjp_rule(ctx, *, fun_jaxpr: jax_core.ClosedJaxpr, **_):
  del ctx
  inner_cost = cost_estimate_jaxpr(fun_jaxpr)
  return CostEstimate(
      flops=inner_cost.flops,
      transcendentals=inner_cost.transcendentals,
      bytes_accessed=inner_cost.bytes_accessed,
  )
register_cost_rule(custom_derivatives.custom_vjp_call_jaxpr_p, _custom_vjp_rule)

def _run_state_rule(*_, jaxpr: jax_core.Jaxpr, **_2):
  inner_cost = cost_estimate_jaxpr(pe.close_jaxpr(jaxpr))
  return CostEstimate(
      flops=inner_cost.flops,
      transcendentals=inner_cost.transcendentals,
      bytes_accessed=inner_cost.bytes_accessed,
  )
register_cost_rule(discharge.run_state_p, _run_state_rule)
