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

"""Pallas utility functions."""

from __future__ import annotations
from typing import overload

import jax
from jax import lax
from jax._src import core as jax_core
from jax._src.util import split_list
import jax.numpy as jnp
import numpy as np


def when(condition):
  def _wrapped(f):
    if isinstance(condition, bool):
      if condition:
        f()
    else:
      lax.cond(condition, f, lambda: None)
  return _wrapped

@overload
def cdiv(a: int, b: int) -> int:
  ...

@overload
def cdiv(a: int, b: jax.Array) -> jax.Array:
  ...

@overload
def cdiv(a: jax.Array, b: int) -> jax.Array:
  ...

@overload
def cdiv(a: jax.Array, b: jax.Array) -> jax.Array:
  ...

def cdiv(a: int | jax.Array, b: int | jax.Array) -> int | jax.Array:
  if isinstance(a, int) and isinstance(b, int):
    return (a + b - 1) // b
  return lax.div(a + b - 1, b)


def strides_from_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
  size = np.prod(shape)
  strides = []
  for s in shape:
    size = size // s
    strides.append(int(size))
  return tuple(strides)


def next_power_of_2(x: int) -> int:
  """Returns the next power of two greater than or equal to `x`."""
  if x < 0:
    raise ValueError("`next_power_of_2` requires a non-negative integer.")
  return 1 if x == 0 else 2 ** (x - 1).bit_length()

def dtype_bitwidth(dtype: np.dtype | jnp.dtype) -> int:
  if jnp.issubdtype(dtype, jnp.integer):
    return jnp.iinfo(dtype).bits
  return np.dtype(dtype).itemsize * 8

def pattern_match_scan_to_fori_loop(
    jaxpr: jax_core.Jaxpr, num_consts: int, num_carry: int
) -> tuple[jax_core.Jaxpr, bool]:
  if num_carry > 0:
    # Pattern match onto fori_loop:
    # We expect the first carry argument to the jaxpr to be the loop index and
    # for the loop index + 1 to be returned as the first value out of the loop.
    in_index_var = jaxpr.invars[num_consts]
    out_index_var = jaxpr.outvars[0]
    # Check that the loop index argument is an int32 scalar
    if (in_index_var.aval.shape or
        in_index_var.aval.dtype not in (jnp.int32, jnp.int64)):
      raise NotImplementedError(
          f"not a fori_loop index in: {in_index_var.aval} {jaxpr=}")
    if (out_index_var.aval.shape or
        out_index_var.aval.dtype not in (jnp.int32, jnp.int64)):
      raise NotImplementedError(
          f"not a fori_loop index out: {out_index_var.aval} {jaxpr=}")
    # Look for the equation that increments the loop index
    for i, eqn in enumerate(jaxpr.eqns):
      if eqn.primitive == lax.add_p:
        if eqn.invars[0] == in_index_var:
          if isinstance(eqn.invars[1], jax_core.Literal):
            if eqn.invars[1].val == 1:
              if eqn.outvars[0] == out_index_var:
                eqn_index = i
                break
    else:
      raise NotImplementedError("Unable to match fori_loop pattern")
    # Delete the equation that increments and remove the loop index from the
    # output. Incrementing the loop index will be done implicitly.
    jaxpr = jaxpr.replace(
        eqns=jaxpr.eqns[:eqn_index] + jaxpr.eqns[eqn_index + 1:],
        outvars=jaxpr.outvars[1:])
    has_loop_index = True
  else:
    # If there's no carry, the loop index has been DCEd and the body does *not*
    # expect a loop index as an argument.
    has_loop_index = False
  return jaxpr, has_loop_index


def pattern_match_while_to_fori_loop(
    cond_jaxpr: jax_core.Jaxpr,
    cond_nconsts: int,
    body_jaxpr: jax_core.Jaxpr,
    body_nconsts: int,
) -> tuple[jax_core.Jaxpr | None, str | None]:
  # Try to pattern match to fori loop.
  # Successful matches produce (jaxpr, None), while failures use the str
  # component of the return tuple to capture information about the failure.
  if cond_nconsts:
    return (None, "Conditional jaxpr can't contain consts.")
  _, cond_invars = split_list(cond_jaxpr.jaxpr.invars, [cond_nconsts])
  cond_in_avals = [v.aval for v in cond_invars]
  if len(cond_in_avals) < 2:
    return (None, "Conditional jaxpr have only two carry args.")
  # Check that the first two carry values are scalar ints
  a1, a2 = cond_in_avals[:2]
  if a1.shape or a1.dtype not in (jnp.int32, jnp.int64):
    return (None, "First conditional jaxpr carry arg is not a scalar int.")
  if a2.shape or a2.dtype not in (jnp.int32, jnp.int64):
    return (None, "Second conditional jaxpr carry arg is not a scalar int.")
  # Check that the only eqn in the cond checks the loop index condition
  v1, v2 = cond_invars[:2]
  outvar = cond_jaxpr.jaxpr.outvars[0]
  assert outvar.aval.dtype == jnp.bool_
  if len(cond_jaxpr.jaxpr.eqns) != 1:
    return (None, "Non-trivial conditional jaxprs not supported.")
  eqn = cond_jaxpr.jaxpr.eqns[0]
  if eqn.primitive != lax.lt_p:
    return (None, "Non-trivial conditional jaxprs not supported.")
  if eqn.outvars != [outvar]:
    return (None, "Non-trivial conditional jaxprs not supported.")
  if eqn.invars != [v1, v2]:
    return (None, "Non-trivial conditional jaxprs not supported.")
  # Check that the carry is updated in the body appropriately
  _, body_invars = split_list(body_jaxpr.jaxpr.invars, [body_nconsts])
  v1, v2 = body_invars[:2]
  vo1, vo2 = body_jaxpr.jaxpr.outvars[:2]
  # Upper bound should be constant
  if v2 is not vo2:
    return (None, "Loop upper bound is not constant.")
  # Check that we increment the loop index in the body
  for i, eqn in enumerate(body_jaxpr.jaxpr.eqns):
    if eqn.primitive is lax.add_p:
      if eqn.invars[0] is v1:
        if isinstance(eqn.invars[1], jax_core.Literal):
          if eqn.invars[1].val == 1:
            if eqn.outvars[0] == vo1:
              eqn_index = i
              break
  else:
    return (None, "Loop index not incremented in body.")
  jaxpr = body_jaxpr.jaxpr
  new_invars = (
      *jaxpr.invars[:body_nconsts],
      jaxpr.invars[body_nconsts],
      *jaxpr.invars[body_nconsts + 2 :],
  )
  new_outvars = tuple(jaxpr.outvars[2:])
  jaxpr = jaxpr.replace(
      eqns=jaxpr.eqns[:eqn_index] + jaxpr.eqns[eqn_index + 1 :],
      invars=new_invars,
      outvars=new_outvars,
  )
  return jaxpr, None


# based on https://github.com/openxla/xla/blob/a7a09d56c3599123f8148bbf3e44c9ebc04624b9/xla/mlir_hlo/mhlo/transforms/chlo_legalize_to_hlo/chlo_legalize_to_hlo.cc#L644-L802
def _erf_inv_32_lowering_helper(x):
  k_degree = 9
  w_lt_5_constants = [
    2.81022636e-08,  3.43273939e-07, -3.5233877e-06,
    -4.39150654e-06, 0.00021858087,  -0.00125372503,
    -0.00417768164,  0.246640727,    1.50140941,
  ]
  w_gt_5_constants = [
    -0.000200214257, 0.000100950558, 0.00134934322,
    -0.00367342844,  0.00573950773,  -0.0076224613,
    0.00943887047,   1.00167406,     2.83297682,
  ]

  w = -jnp.log1p(x * -x)
  w_lt_5 = w < 5.0

  w = jnp.where(w_lt_5, w - 2.5, jnp.sqrt(w) - 3.0)

  p = jnp.where(w_lt_5, w_lt_5_constants[0], w_gt_5_constants[0])
  for i in range(1, k_degree):
    c = jnp.where(w_lt_5, w_lt_5_constants[i], w_gt_5_constants[i])
    p = c + p * w

  return jnp.where(jnp.abs(x) == 1.0, jnp.inf * x, p * x)


# based on https://github.com/openxla/xla/blob/a7a09d56c3599123f8148bbf3e44c9ebc04624b9/xla/mlir_hlo/mhlo/transforms/chlo_legalize_to_hlo/chlo_legalize_to_hlo.cc#L696-L802
def _erf_inv_64_lowering_helper(x):
  w_lt_625_constants = [
    -3.6444120640178196996e-21, -1.685059138182016589e-19,
    1.2858480715256400167e-18,  1.115787767802518096e-17,
    -1.333171662854620906e-16,  2.0972767875968561637e-17,
    6.6376381343583238325e-15,  -4.0545662729752068639e-14,
    -8.1519341976054721522e-14, 2.6335093153082322977e-12,
    -1.2975133253453532498e-11, -5.4154120542946279317e-11,
    1.051212273321532285e-09,   -4.1126339803469836976e-09,
    -2.9070369957882005086e-08, 4.2347877827932403518e-07,
    -1.3654692000834678645e-06, -1.3882523362786468719e-05,
    0.0001867342080340571352,   -0.00074070253416626697512,
    -0.0060336708714301490533,  0.24015818242558961693,
    1.6536545626831027356
  ]

  w_lt_16_constants = [
    2.2137376921775787049e-09,  9.0756561938885390979e-08,
    -2.7517406297064545428e-07, 1.8239629214389227755e-08,
    1.5027403968909827627e-06,  -4.013867526981545969e-06,
    2.9234449089955446044e-06,  1.2475304481671778723e-05,
    -4.7318229009055733981e-05, 6.8284851459573175448e-05,
    2.4031110387097893999e-05,  -0.0003550375203628474796,
    0.00095328937973738049703,  -0.0016882755560235047313,
    0.0024914420961078508066,   -0.0037512085075692412107,
    0.005370914553590063617,    1.0052589676941592334,
    3.0838856104922207635,
  ]

  w_gt_16_constants = [
    -2.7109920616438573243e-11, -2.5556418169965252055e-10,
    1.5076572693500548083e-09,  -3.7894654401267369937e-09,
    7.6157012080783393804e-09,  -1.4960026627149240478e-08,
    2.9147953450901080826e-08,  -6.7711997758452339498e-08,
    2.2900482228026654717e-07,  -9.9298272942317002539e-07,
    4.5260625972231537039e-06,  -1.9681778105531670567e-05,
    7.5995277030017761139e-05,  -0.00021503011930044477347,
    -0.00013871931833623122026, 1.0103004648645343977,
    4.8499064014085844221,
  ]  # should add "as jnp.float64 array"?

  w = -jnp.log1p(x * -x)
  w_lt_625 = w < 6.25
  w_lt_16 = w < 16.0

  def get_coefficient(i):
    c = w_lt_625_constants[i]
    if i < 19:
      c = jnp.where(w_lt_625, c, w_lt_16_constants[i])
    if i < 17:
      c = jnp.where(w_lt_16, c, w_gt_16_constants[i])
    return c

  select2 = jnp.where(w_lt_16, 3.25, 5.0)
  select2_result = jnp.sqrt(w) - select2
  w = jnp.where(w_lt_625, w - 3.125, select2_result)

  p = get_coefficient(0)
  for i in range(1, 17):
    p = get_coefficient(i) + p * w
  for i in range(17, 19):
    p = jnp.where(w_lt_16, get_coefficient(i) + p * w, p)
  for i in range(19, 23):
    p = jnp.where(w_lt_625, get_coefficient(i) + p * w, p)

  return jnp.where(jnp.abs(x) == 1.0, np.inf * x, p * x)


def erf_inv_lowering_helper(x):
  if x.dtype == jnp.float32:
    return _erf_inv_32_lowering_helper(x)
  if x.dtype == jnp.float64:
    return _erf_inv_64_lowering_helper(x)
  raise NotImplementedError(f"erf_inv_lowering_helper not implemented for {x.dtype}")


def sign_lowering_helper(x):
  if jnp.issubdtype(x.dtype, jnp.unsignedinteger):
    return (x != 0).astype(x.dtype)

  if jnp.issubdtype(x.dtype, jnp.integer):
    return (x > 0).astype(x.dtype) - (x < 0).astype(x.dtype)

  if jnp.issubdtype(x.dtype, jnp.floating):
    out = (x > 0.).astype(x.dtype) - (x < 0.).astype(x.dtype)
    return jnp.where(jnp.isnan(x), jnp.nan, out)

  raise NotImplementedError(f"sign_lowering_helper not implemented for {x.dtype}")


# based on https://github.com/openxla/xla/blob/a7a09d56c3599123f8148bbf3e44c9ebc04624b9/xla/mlir_hlo/mhlo/transforms/chlo_legalize_to_hlo/chlo_legalize_to_hlo.cc#L1339-L1422
def nextafter_lowering_helper(x, y):
  if x.dtype != y.dtype:
    raise ValueError(
        "The two inputs to `nextafter` must have the same dtype, but got"
        f" {x.dtype} and {y.dtype}"
    )

  if x.dtype not in (jnp.float32, jnp.float64):
    raise ValueError(
        f"`nextafter` only supports float32 and float64, but got {x.dtype}"
    )

  jnp_float, jnp_uint, np_float, np_uint, np_int = (
      jnp.float32, jnp.uint32, np.float32, np.uint32, np.int32,
  ) if x.dtype == jnp.float32 else (
      jnp.float64, jnp.uint64, np.float64, np.uint64, np.int64,
  )

  bitwidth = dtype_bitwidth(x.dtype)

  x_as_int = x.view(jnp_uint)
  y_as_int = y.view(jnp_uint)

  # The result is NaN if either "x" or "y" are NaN.
  nan_input = jnp.isnan(x) | jnp.isnan(y)
  result_for_nan = jnp.full_like(x_as_int, np_float(np.nan).view(np_uint))

  # The sign bit is the MSB.
  sign_bit = jnp_uint(1 << (bitwidth - 1))
  # Discard the sign bit to make the result non-negative.
  sign_mask = sign_bit
  negated_sign_mask = ~sign_bit
  x_abs = x_as_int & negated_sign_mask
  y_abs = y_as_int & negated_sign_mask

  # When both "x" and "y" are equal, the result is "y".
  x_and_y_are_equal = x == y
  result_for_equal = y_as_int

  # When both "x" and "y" are 0, the result is "y". This is a separate case
  # from above because "x" and "y" might have a different sign.
  zero = jnp.zeros_like(x_as_int)
  x_is_zero = x_abs == zero
  y_is_zero = y_abs == zero
  result_for_both_zero = y_as_int

  x_sign = x_as_int & sign_mask
  y_sign = y_as_int & sign_mask

  # If x == 0 && y != 0, we need to return the smallest subnormal number
  # signed like "y".
  one = jnp.ones_like(x_as_int)
  result_for_x_zero_y_non_zero = y_sign | one

  # If the sign of "x" and "y" disagree:
  # - we need to make the magnitude of "from" smaller so that it is closer to
  #   zero.
  #
  # Otherwise the signs agree:
  # - "x" with a magnitude larger than "y" means we need to make the magnitude
  #   smaller.
  # - "x" with a magnitude smaller than "y" means we need to make the magnitude
  #   larger.
  signs_disagree = x_sign != y_sign
  x_magnitude_larger_than_y = x_abs > y_abs
  result_has_smaller_magnitude = x_magnitude_larger_than_y | signs_disagree
  minus_one = jnp.full_like(x_as_int, np_int(-1).view(np_uint))
  magnitude_adjustment = jnp.where(result_has_smaller_magnitude, minus_one, one)
  result = x_as_int + magnitude_adjustment

  # Handle x == +-0.
  result = jnp.where(
      x_is_zero,
      jnp.where(y_is_zero, result_for_both_zero, result_for_x_zero_y_non_zero),
      result,
  )

  # Handle x == y.
  result = jnp.where(x_and_y_are_equal, result_for_equal, result)

  # Handle isnan(x) || isnan(y).
  result = jnp.where(nan_input, result_for_nan, result)

  # Cast back to the original type.
  return result.view(jnp_float)
