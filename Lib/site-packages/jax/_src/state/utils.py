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
"""Utilities for tracing stateful functions."""

from functools import partial
from typing import Callable

import jax
from jax._src import core
from jax._src import dtypes
from jax._src import linear_util as lu
from jax._src.interpreters import partial_eval as pe
from jax._src.state import AbstractRef
from jax._src.state.primitives import ref_get
from jax._src.typing import DTypeLike
from jax._src.util import safe_map, safe_zip, split_list

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip


def hoist_consts_to_refs(
    jaxpr: core.Jaxpr,
    *,
    index: int = 0,
    make_abstract_ref: Callable[[core.AbstractValue], AbstractRef] = lambda aval: AbstractRef(aval)
) -> core.Jaxpr:
  """Hoists the constants in the given jaxpr into invars.

  Args:
    jaxpr: The jaxpr.
    index: The index where the invars for the constants should be inserted.
      By default, the new invars are inserted *before* any existing invars.
    make_abstract_ref: a callable to construct an AbstractRef, or subtype
      thereof, from a constant AbstractValue.

  Returns:
    A new jaxpr where the constants were hoisted into invars as ``Ref``s.
  """
  if not jaxpr.constvars:
    return jaxpr  # Nothing to hoist.

  is_const_ref = [
      isinstance(var.aval, AbstractRef) for var in jaxpr.constvars
  ]
  const_avals = [
      var.aval if is_ref else make_abstract_ref(var.aval)
      for is_ref, var in zip(is_const_ref, jaxpr.constvars)
  ]
  in_avals = [var.aval for var in jaxpr.invars]
  in_avals[index:index] = const_avals

  def _hoist(*consts_args):
    args0, all_consts, args1 = split_list(
        consts_args, [index, len(const_avals)]
    )
    # We immediately read the const values out of the `Ref`s.
    all_consts = [
        c if is_ref else ref_get(c, ())
        for is_ref, c in zip(is_const_ref, all_consts)
    ]
    return core.eval_jaxpr(jaxpr, all_consts, *args0, *args1)

  hoisted_jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(_hoist), in_avals)
  assert not consts, "All consts should have been converted to refs"
  return hoisted_jaxpr


def val_to_ref_aval(x) -> AbstractRef:
  aval = core.get_aval(x)
  if type(aval) is not core.ShapedArray:
    raise TypeError(f"can't make ref from {x}")
  return AbstractRef(aval)


def dtype_bitwidth(dtype: DTypeLike) -> int:
  if dtypes.isdtype(dtype, "integral"):
    return dtypes.iinfo(dtype).bits
  return dtypes.dtype(dtype).itemsize * 8


def bitcast(x, dtype: DTypeLike):
  x_bitwidth = dtype_bitwidth(x.dtype)
  y_bitwidth = dtype_bitwidth(dtype)
  shape = list(x.shape)
  if x_bitwidth != y_bitwidth:
    if len(shape) < 2:
      raise NotImplementedError(
          "Bitcast 1D ref with bitwidth change is not supported."
      )
    # Note: this is only valid on TPU.
    if shape[-2] * x_bitwidth % y_bitwidth != 0:
      raise ValueError(
          "Expected input and output shapes are the same after multiplying"
          " the second-minor dimension by the bitwidths."
      )
  shape[-2] = shape[-2] * x_bitwidth // y_bitwidth
  if x_bitwidth < y_bitwidth:
    ratio = y_bitwidth // x_bitwidth
    x = x.reshape(*x.shape[:-2], x.shape[-2] // ratio, ratio, -1).swapaxes(
        -1, -2
    )
  y = jax.lax.bitcast_convert_type(x, dtype)
  if x_bitwidth > y_bitwidth:
    y = y.swapaxes(-1, -2).reshape(shape)
  return y


def eval_bitcast_shape(x, dtype: DTypeLike):
  f = partial(bitcast, dtype=dtype)
  return jax.eval_shape(f, jax.ShapeDtypeStruct(x.shape, x.dtype)).shape
