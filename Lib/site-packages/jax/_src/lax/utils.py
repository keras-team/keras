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

# This module contains utility functions split out of jax._src.lax.lax to
# avoid cyclic dependencies. Definitions that are used at import time by
# multiple modules can go here.

from functools import partial

from jax._src import core
from jax._src import dispatch
from jax._src import config
from jax._src import dtypes
from jax._src import mesh as mesh_lib
from jax._src.util import safe_zip

zip, unsafe_zip = safe_zip, zip

import numpy as np

def _input_dtype(x, *_, **__):
  return dtypes.canonicalize_dtype(x.dtype, allow_extended_dtype=True)

def _argnum_weak_type(*argnums):
  return lambda *args, **_: all(args[i].weak_type for i in argnums)

def standard_primitive(shape_rule, dtype_rule, name,
                       weak_type_rule=None, sharding_rule=None):
  weak_type_rule = weak_type_rule or _standard_weak_type_rule
  prim = core.Primitive(name)
  prim.def_impl(partial(dispatch.apply_primitive, prim))
  prim.def_abstract_eval(
      partial(standard_abstract_eval, prim, shape_rule, dtype_rule,
              weak_type_rule, sharding_rule))
  return prim

def _get_array_abstraction_level(a): return a.array_abstraction_level

def call_sharding_rule(rule, num_out, *avals, **kwargs):
  if config.sharding_in_types.value:
    if rule is None and mesh_lib.get_abstract_mesh()._are_all_axes_hidden:  # type: ignore
      return None if num_out is None else [None] * num_out
    return rule(*avals, **kwargs)
  return None if num_out is None else [None] * num_out

def standard_abstract_eval(prim, shape_rule, dtype_rule, weak_type_rule,
                           sharding_rule, *avals, **kwargs):
  assert all(isinstance(aval, core.UnshapedArray) for aval in avals), avals
  assert not prim.multiple_results
  weak_type = weak_type_rule(*avals, **kwargs)
  least_specialized = type(max(avals, key=_get_array_abstraction_level))
  if least_specialized is core.ShapedArray:
    core.check_avals_context_mesh(avals, prim.name)
    out_aval = core.ShapedArray(
        shape_rule(*avals, **kwargs), dtype_rule(*avals, **kwargs),
        weak_type=weak_type,
        sharding=call_sharding_rule(sharding_rule, None, *avals, **kwargs))
    core.check_avals_context_mesh([out_aval], prim.name)
    return out_aval
  elif least_specialized is core.DShapedArray:
    shape = shape_rule(*avals, **kwargs)
    ty = (core.ShapedArray if all(type(d) is int for d in shape)
          else core.DShapedArray)
    return ty(shape, dtype_rule(*avals, **kwargs), weak_type)
  elif least_specialized is core.UnshapedArray:
    return core.UnshapedArray(dtype_rule(*avals, **kwargs), weak_type=weak_type)
  else:
    raise TypeError(avals, least_specialized)

def standard_multi_result_abstract_eval(
    prim, shape_rule, dtype_rule, weak_type_rule, sharding_rule,
    *avals, **kwargs):
  assert prim.multiple_results
  assert all(isinstance(aval, core.UnshapedArray) for aval in avals), avals
  least_specialized = max(map(type, avals), key=_get_array_abstraction_level)
  weak_types = weak_type_rule(*avals, **kwargs)
  if least_specialized is core.ShapedArray:
    out_shapes = shape_rule(*avals, **kwargs)
    out_dtypes = dtype_rule(*avals, **kwargs)
    core.check_avals_context_mesh(avals, prim.name)
    out_shardings = call_sharding_rule(
        sharding_rule, len(out_shapes), *avals, **kwargs)
    out_avals = [core.ShapedArray(s, d, weak_type=weak_type, sharding=sh)
                 for s, d, weak_type, sh in zip(out_shapes, out_dtypes,
                                                weak_types, out_shardings)]
    core.check_avals_context_mesh(out_avals, prim.name)
    return out_avals
  elif least_specialized is core.UnshapedArray:
    out_dtypes = dtype_rule(*avals, **kwargs)
    return [core.UnshapedArray(dtype, weak_type=weak_type)
            for dtype, weak_type in zip(out_dtypes, weak_types)]
  else:
    raise TypeError(avals, least_specialized)


def _standard_weak_type_rule(*avals, **kwargs):
  return all(aval.weak_type for aval in avals)

def dtype_to_string(dtype):
  try:
    return str(np.dtype(dtype).name)
  except TypeError:
    pass
  try:
    return dtype.name
  except AttributeError:
    pass
  return str(dtype)
