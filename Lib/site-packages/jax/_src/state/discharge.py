# Copyright 2022 The JAX Authors.
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
"""Module for discharging state primitives."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
from functools import partial
import math
import operator
from typing import Any, Protocol, TypeVar

from jax._src import ad_util
from jax._src import api_util
from jax._src import core
from jax._src import linear_util as lu
from jax._src import source_info_util
from jax._src import tree_util
from jax._src.interpreters import ad
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import lax
from jax._src.lax import slicing as lax_slicing
from jax._src.state import indexing
from jax._src.state.primitives import addupdate_p, get_p, swap_p
from jax._src.state.types import (
    AbstractRef,
    RefBitcaster,
    RefEffect,
    RefReshaper,
    get_ref_aval_from_value,
    uninitialized,
)
from jax._src.state.utils import bitcast, hoist_consts_to_refs
from jax._src.typing import Array
from jax._src.util import (
    merge_lists,
    partition_list,
    safe_map,
    safe_zip,
    split_dict,
    split_list,
    unzip2,
    weakref_lru_cache,
)
import numpy as np

## JAX utilities

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip
PyTreeDef = tree_util.PyTreeDef

## Discharging state

# Let's say we have a jaxpr that takes in `Ref`s and outputs regular JAX values
# (`Ref`s should never be outputs from jaxprs). We'd like to convert that jaxpr
# into a "pure" jaxpr that takes in and outputs values and no longer has the
# `Read/Write/Accum` effects.

def discharge_state(jaxpr: core.Jaxpr, consts: Sequence[Any], * ,
                    should_discharge: bool | Sequence[bool] = True
                    ) -> tuple[core.Jaxpr, list[Any]]:
  """Converts a jaxpr that takes in `Ref`s into one that doesn't."""
  if isinstance(should_discharge, bool):
    should_discharge = [should_discharge] * len(jaxpr.invars)
  in_avals = [v.aval.inner_aval
              if isinstance(v.aval, AbstractRef) and d
              else v.aval for v, d in zip(jaxpr.invars, should_discharge)]
  eval_jaxpr = lu.wrap_init(partial(_eval_jaxpr_discharge_state, jaxpr,
                                    should_discharge, consts))
  new_jaxpr, _ , new_consts, () = pe.trace_to_jaxpr_dynamic(eval_jaxpr, in_avals)
  return new_jaxpr, new_consts

@dataclasses.dataclass
class Environment:
  env: dict[core.Var, Any]

  def read(self, v: core.Atom) -> Any:
    if type(v) is core.Literal:
      return v.val
    assert isinstance(v, core.Var)
    return self.env[v]

  def write(self, v: core.Var, val: Any) -> None:
    self.env[v] = val

class DischargeRule(Protocol):

  def __call__(self, in_avals: Sequence[core.AbstractValue],
      out_avals: Sequence[core.AbstractValue], *args: Any,
      **params: Any) -> tuple[Sequence[Any | None], Sequence[Any]]:
    ...

_discharge_rules: dict[core.Primitive, DischargeRule] = {}

class PartialDischargeRule(Protocol):
  """A partial discharge rule.

  Exactly like a discharge rule only it accepts a `should_discharge`
  argument that indicates which inputs should be discharged and the
  return value returns a tuple of which the first element is the new
  inputs or none but only the ones that correspond to `True` entries
  in `should_charge`.
  """

  def __call__(self, should_discharge: Sequence[bool],
      in_avals: Sequence[core.AbstractValue],
      out_avals: Sequence[core.AbstractValue], *args: Any,
      **params: Any) -> tuple[Sequence[Any | None], Sequence[Any]]:
    ...

_partial_discharge_rules: dict[core.Primitive, PartialDischargeRule] = {}

def register_discharge_rule(prim: core.Primitive):
  def register(f: DischargeRule):
    _discharge_rules[prim] = f
  return register

def register_partial_discharge_rule(prim: core.Primitive):
  def register(f: PartialDischargeRule):
    _partial_discharge_rules[prim] = f
  return register


def _eval_jaxpr_discharge_state(
    jaxpr: core.Jaxpr, should_discharge: Sequence[bool], consts: Sequence[Any],
    *args: Any):
  env = Environment({})

  map(env.write, jaxpr.constvars, consts)
  # Here some args may correspond to `Ref` avals but they'll be treated like
  # regular values in this interpreter.
  map(env.write, jaxpr.invars, args)

  refs_to_discharge = {id(v.aval) for v, d in zip(jaxpr.invars, should_discharge)
                       if d and isinstance(v.aval, AbstractRef)}

  for eqn in jaxpr.eqns:
    name_stack = (
        source_info_util.current_name_stack() + eqn.source_info.name_stack
    )
    traceback = eqn.source_info.traceback
    with source_info_util.user_context(
        traceback, name_stack=name_stack), eqn.ctx.manager:
      should_discharge = [id(v.aval) in refs_to_discharge for v in eqn.invars]
      if eqn.primitive is core.mutable_array_p:
        [invar], [outvar] = eqn.invars, eqn.outvars
        ans = env.read(invar)
        refs_to_discharge.add(id(outvar.aval))
      elif eqn.primitive is core.freeze_p:
        [invar], [outvar] = eqn.invars, eqn.outvars
        ans = env.read(invar)
        refs_to_discharge.remove(id(invar.aval))
      elif (any(should_discharge)
            or core.internal_mutable_array_effect in eqn.effects
        ):
        if eqn.primitive in _partial_discharge_rules:
          rule: DischargeRule = partial(_partial_discharge_rules[eqn.primitive], should_discharge)
        elif eqn.primitive in _discharge_rules:
          rule = _discharge_rules[eqn.primitive]
        else:
          raise NotImplementedError("No state discharge rule implemented for "
              f"primitive: {eqn.primitive}")
        invals = map(env.read, eqn.invars)
        in_avals = [v.aval for v in eqn.invars]
        out_avals = [v.aval for v in eqn.outvars]
        new_invals, ans = rule(
            in_avals, out_avals, *invals, **eqn.params)
        for invar, should, new_inval in zip(eqn.invars, should_discharge, new_invals):
          if new_inval is not None:
            if not should:
              raise ValueError(
                  f"Did not ask for inval to be discharged but it was. ({invar=},"
                  f" {new_inval=})"
              )
            env.write(invar, new_inval)  # type: ignore[arg-type]
      else:
        # Default primitive rule, similar to `core.eval_jaxpr`. Note that here
        # we assume any higher-order primitives inside of the jaxpr are *not*
        # stateful.
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        ans = eqn.primitive.bind(*subfuns, *map(env.read, eqn.invars),
                                **bind_params)
    if eqn.primitive.multiple_results:
      map(env.write, eqn.outvars, ans)
    else:
      env.write(eqn.outvars[0], ans)
  # By convention, we return the outputs of the jaxpr first and then the final
  # values of the `Ref`s. Callers to this function should be able to split
  # them up by looking at `len(jaxpr.outvars)`.
  out_vals = map(env.read, jaxpr.outvars)
  ref_vals = map(
      env.read, [v for v in jaxpr.invars if id(v.aval) in refs_to_discharge])
  return out_vals + ref_vals

def _is_trivial_indexer(indexer: indexing.NDIndexer):
  for s, idx in zip(indexer.shape, indexer.indices):
    if not isinstance(idx, indexing.Slice):
      return False
    if not isinstance(idx.start, int):
      return False
    if idx.start:
      return False
    if idx.size != s:
      return False
  return True


def _maybe_convert_to_slice(
    indexer: indexing.NDIndexer
) -> list[tuple[int, int, int]] | None:
  args = []

  for i in indexer.indices:
    if not isinstance(i, indexing.Slice):
      return None

    start = i.start
    end = i.start + (i.size - 1) * i.stride + 1
    stride = i.stride

    # cannot convert to static `slice` if `start` or `end` is dynamic
    if not isinstance(start, int) or not isinstance(end, int):
      return None

    args.append((start, end, stride))

  return args


def _maybe_convert_to_dynamic_slice(
    indexer: indexing.NDIndexer,
) -> (
    tuple[tuple[Array | int, ...], tuple[Array | int, ...], tuple[int, ...]]
    | None
):
  # An NDIndexer only corresponds to a `dynamic_slice` or `dynamic_update_slice`
  # if each of the indexers is a `Slice` or a ()-shaped value.
  if not all(isinstance(i, indexing.Slice) or not np.shape(i)
             for i in indexer.indices):
    return None

  # `lax.dynamic_slice` does not handle striding
  for i in indexer.indices:
    if isinstance(i, indexing.Slice) and i.stride > 1:
      return None

  _convert_i32 = lambda x: lax.convert_element_type(x, np.dtype("int32"))
  starts = tuple(
      _convert_i32(i.start) if isinstance(i, indexing.Slice)
      else _convert_i32(i) for i in indexer.indices
  )
  sizes = tuple(
      i.size if isinstance(i, indexing.Slice) else 1 for i in indexer.indices
  )
  squeeze_dims = tuple(
      i
      for i, idx in enumerate(indexer.indices)
      if not isinstance(idx, indexing.Slice)
  )
  return starts, sizes, squeeze_dims


def _convert_to_array_indexer(indexer: indexing.NDIndexer
                              ) -> tuple[int | Array, ...]:
  # This is the general gather case. We need to create the gather arrays.
  is_integer_indexer, _, integer_indexer = (
      indexing.unpack_ndindexer(indexer)
  )
  total_shape = indexer.get_indexer_shape()
  int_indexer_shape = indexer.int_indexer_shape
  slice_shape = total_shape[len(int_indexer_shape):]
  slice_dims = tuple(
      i + len(int_indexer_shape) for i in range(len(slice_shape))
  )
  slice_dim_iter = iter(slice_dims)
  slice_indexer: list[Array] = []
  for idx, is_int_index in zip(indexer.indices, is_integer_indexer):
    if not is_int_index:
      assert isinstance(idx, indexing.Slice)
      slice_indices = lax.broadcasted_iota(
          np.dtype("int32"), total_shape, next(slice_dim_iter)
      ) * idx.stride + idx.start
      slice_indexer.append(slice_indices)
      integer_indexer = tuple(
          lax.expand_dims(idx, (-1,)) for idx in integer_indexer
      )
      continue
  assert next(slice_dim_iter, None) is None
  return tuple(merge_lists(is_integer_indexer, slice_indexer, integer_indexer))


@register_discharge_rule(get_p)
def _get_discharge_rule(
    in_avals: Sequence[core.AbstractValue],
    out_avals: Sequence[core.AbstractValue], x, *idx,
    tree):
  del in_avals, out_avals
  y = _get_discharge(x, idx, tree)
  return (None,) * (len(idx) + 1), y

def _prepend_gather(x, indexer):
  # NumPy advanced int indexing won't prepend w/ only one dim, so add dummy.
  return x[None][(np.array(0, 'int32'), *indexer)]

def _prepend_scatter(x, indexer, val, *, add=False):
  # NumPy advanced int indexing won't prepend w/ only one dim, so add dummy.
  # However, since this is scatter, we need to remove the 1-sized dimension
  # we added at the front.
  if add:
    return x[None].at[(0, *indexer)].add(val)[0]
  return x[None].at[(0, *indexer)].set(val)[0]


def _index_array(x, indexer):
  if _is_trivial_indexer(indexer):
    return x
  # Try the three APIs in the following order: `lax.slice`,
  # `lax.dynamic_slice` and gather
  if maybe_slice := _maybe_convert_to_slice(indexer):
    x = lax_slicing.slice(x, *zip(*maybe_slice))
  # If everything in the indexer is a slice or ()-shaped, we can also
  # use `lax.dynamic_slice` with 1-sized slices for ()-shaped indices.
  # We need to squeeze out the 1-sized slices at the end.
  elif maybe_slice := _maybe_convert_to_dynamic_slice(indexer):
    starts, sizes, squeeze_dims = maybe_slice
    y = lax_slicing.dynamic_slice(x, starts, sizes)
    x = lax.squeeze(y, squeeze_dims)
  else:
    indexer = _convert_to_array_indexer(indexer)
    x = x[None][(np.array(0, "int32"), *indexer)]
  return x


def transform_array(x, transforms):
  if transforms is None:
    transforms = []
  result = x
  for transform in transforms:
    if transform is None:
      continue
    match transform:
      case indexing.NDIndexer():
        result = _index_array(result, transform)
      case RefBitcaster():
        result = bitcast(result, transform.dtype)
      case RefReshaper():
        result = result.reshape(transform.shape)
      case _:
        raise NotImplementedError(f"Unsupported transform: {transform}")
  return result

def transform_swap_array(x, transforms, val):
  if transforms is None:
    transforms = []
  result = x
  result_val = val
  # Compute updated "val" (result).
  _results = [x]
  for transform in transforms:
    match transform:
      case indexing.NDIndexer():
        indexer = transform
        if _is_trivial_indexer(indexer):
          _results.append(_results[-1])
          continue
        # If everything in the indexer is a slice or ()-shaped, we can also
        # use `lax.dynamic_slice` with 1-sized slices for ()-shaped indices.
        # We need to squeeze out the 1-sized slices at the end.
        if maybe_slice := _maybe_convert_to_dynamic_slice(indexer):
          starts, sizes, squeeze_dims = maybe_slice
          result_old = lax_slicing.dynamic_slice(result, starts, sizes)
          result = lax.squeeze(result_old, squeeze_dims)
        else:
          indexer = _convert_to_array_indexer(indexer)
          result = _prepend_gather(result, indexer)
        _results.append(result)
      case RefBitcaster():
        _results.append(bitcast(result, transform.dtype))
      case RefReshaper():
        _results.append(result.reshape(transform.shape))
      case _:
        raise NotImplementedError(f"Unsupported transform: {transform}")

  # Compute updated "x" (result_val)
  for i, transform in reversed(list(enumerate(transforms))):
    if isinstance(transform, indexing.NDIndexer):
      indexer = transform
      if _is_trivial_indexer(indexer):
        continue
      if maybe_slice := _maybe_convert_to_dynamic_slice(indexer):
        starts, _, squeeze_dims = maybe_slice
        result_val = lax.expand_dims(result_val, squeeze_dims)
        result_val = lax_slicing.dynamic_update_slice(
            _results[i], result_val, starts
        )
      else:
        indexer = _convert_to_array_indexer(indexer)
        result_val = _prepend_scatter(_results[i], indexer, result_val)
    else:
      raise NotImplementedError(f"Unsupported transform: {transform}")
  return result, result_val

def _get_discharge(x, idx, tree):
  transforms = tree_util.tree_unflatten(tree, idx)
  return transform_array(x, transforms)

@register_discharge_rule(swap_p)
def _swap_discharge_rule(
    in_avals: Sequence[core.AbstractValue],
    out_avals: Sequence[core.AbstractValue], x, val, *idx,
    tree):
  del in_avals, out_avals
  z, x_new = _swap_discharge(x, val, idx, tree)
  return (x_new, None) + (None,) * len(idx), z

def _swap_discharge(x, val, idx, tree):
  transforms = tree_util.tree_unflatten(tree, idx)
  return transform_swap_array(x, transforms, val)

@register_discharge_rule(addupdate_p)
def _addupdate_discharge_rule(
    in_avals: Sequence[core.AbstractValue],
    out_avals: Sequence[core.AbstractValue], x, val, *idx,
    tree):
  del in_avals, out_avals
  ans = _addupdate_discharge(x, val, idx, tree)
  return (ans, None) + (None,) * len(idx), []

def _addupdate_discharge(x, val, idx, tree):
  transforms = tree_util.tree_unflatten(tree, idx)
  if len(transforms) > 1:
    raise NotImplementedError("Only single indexer is supported.")
  indexer = transforms[0]
  if _is_trivial_indexer(indexer):
    return x + val
  # If everything in the indexer is a slice or ()-shaped, we can also
  # use `lax.dynamic_slice` with 1-sized slices for ()-shaped indices.
  # We need to squeeze out the 1-sized slices at the end.
  if maybe_slice := _maybe_convert_to_dynamic_slice(indexer):
    starts, sizes, squeeze_dims = maybe_slice
    x_old = lax_slicing.dynamic_slice(x, starts, sizes)
    val = lax.expand_dims(val, squeeze_dims)
    y = lax_slicing.dynamic_update_slice(x, x_old + val, starts)
    return y
  indexer = _convert_to_array_indexer(indexer)
  return _prepend_scatter(x, indexer, val, add=True)

@weakref_lru_cache
def _cached_closed_jaxpr_discharge(closed_jaxpr):
  jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.consts
  num_outs = len(jaxpr.outvars)
  discharged_jaxpr, discharged_consts = discharge_state(jaxpr, consts)
  discharged_closed_jaxpr = core.ClosedJaxpr(discharged_jaxpr, discharged_consts)
  fun = lu.wrap_init(core.jaxpr_as_fun(discharged_closed_jaxpr))
  return discharged_closed_jaxpr, num_outs, fun

@register_discharge_rule(core.closed_call_p)
def _closed_call_discharge_rule(
    in_avals: Sequence[core.AbstractValue], _,*args,
    call_jaxpr: core.ClosedJaxpr):
  discharged_closed_jaxpr, num_outs, fun = _cached_closed_jaxpr_discharge(call_jaxpr)
  out_and_ref_vals = core.closed_call_p.bind(fun, *args,
                                             call_jaxpr=discharged_closed_jaxpr)
  out_vals, ref_vals = split_list(out_and_ref_vals, [num_outs])
  ref_vals_iter = iter(ref_vals)
  new_invals = tuple(next(ref_vals_iter) if isinstance(aval, AbstractRef)
                     else None for aval in in_avals)
  sentinel = object()
  assert next(ref_vals_iter, sentinel) is sentinel
  return new_invals, out_vals

# # `run_state`

run_state_p = core.Primitive("run_state")
run_state_p.multiple_results = True

def _default_initialization(x):
  assert hasattr(x, 'shape')
  assert hasattr(x, 'dtype')
  dtype = np.dtype(x)
  if np.issubdtype(dtype, np.integer):
    value = np.iinfo(dtype).min
  else:
    value = math.nan
  return lax.full(x.shape, value, dtype)

def _run_state_impl(*args: Any, jaxpr: core.Jaxpr,
                    which_linear: tuple[bool, ...],
                    is_initialized: tuple[bool, ...]):
  del which_linear
  discharged_jaxpr, consts = discharge_state(jaxpr, ())
  # Initialize the args that are not initialized.
  args_it = iter(args)
  args = tuple(
      next(args_it) if is_init else _default_initialization(var.aval)
      for is_init, var in zip(is_initialized, discharged_jaxpr.invars)
  )
  return core.eval_jaxpr(discharged_jaxpr, consts, *args)
run_state_p.def_impl(_run_state_impl)
mlir.register_lowering(run_state_p, mlir.lower_fun(_run_state_impl))

def _run_state_abstract_eval(*avals: core.AbstractValue, jaxpr: core.Jaxpr,
                             which_linear: tuple[bool, ...],
                             is_initialized: tuple[bool, ...]):
  del which_linear
  assert sum(is_initialized) == len(avals)
  # When we abstractly evaluate `run_state`, we want to keep track of which
  # input avals are `Ref`s and which are not. If an aval is a `Ref`, we want to
  # "propagate" out its inner effects. Otherwise, the effects are local to this
  # `run_state`.
  inner_to_outer_aval_mapping = {}
  outer_ref_index = 0
  for i, is_init in enumerate(is_initialized):
    if not is_init:
      pass
    inner_to_outer_aval_mapping[i] = outer_ref_index
    outer_ref_index += 1
  nonlocal_effects = set()
  is_ref = {i for i, aval in enumerate(avals) if isinstance(aval, AbstractRef)}
  for eff in jaxpr.effects:
    if not isinstance(eff, RefEffect):
      nonlocal_effects.add(eff)
      continue
    if eff.input_index not in inner_to_outer_aval_mapping:
      # This means that this effect corresponds to an uninitialized Ref and
      # should not propagate out of the primitive.
      continue
    # If we do propagate the effect, we need to update the input index to
    # correspond to the outer index.
    outer_index = inner_to_outer_aval_mapping[eff.input_index]
    if outer_index in is_ref:
      # This means that the effect corresponds to a Ref from an outside scope.
      nonlocal_effects.add(
          eff.replace(input_index=inner_to_outer_aval_mapping[eff.input_index])
      )
  return avals, nonlocal_effects
run_state_p.def_effectful_abstract_eval(_run_state_abstract_eval)

def _run_state_jvp(primals: Sequence[Any], tangents: Sequence[Any], *,
                   jaxpr: core.Jaxpr, which_linear: tuple[bool, ...],
                   is_initialized: tuple[bool, ...]):
  if not all(is_initialized):
    raise NotImplementedError("Uninitialized Refs are not supported in jvp.")
  nonzero_tangents = [not isinstance(t, ad_util.Zero) for t in tangents]
  discharged_jaxpr, body_consts = discharge_state(jaxpr, ())
  for _ in range(len(nonzero_tangents)):
    _, out_nonzero_tangents = ad.jvp_jaxpr(
        core.ClosedJaxpr(discharged_jaxpr, body_consts),
        nonzero_tangents, instantiate=nonzero_tangents)
    if out_nonzero_tangents == nonzero_tangents:
      break
    nonzero_tangents = map(operator.or_, nonzero_tangents, out_nonzero_tangents)
  else:
    raise Exception("Invalid fixpoint")
  del discharged_jaxpr, body_consts, out_nonzero_tangents
  tangents = [ad.instantiate_zeros(t) if inst else t
              for t, inst in zip(tangents, nonzero_tangents)]
  tangents = [t for t in tangents if type(t) is not ad_util.Zero]
  closed_jvp_jaxpr, _ = ad.jvp_jaxpr(pe.close_jaxpr(jaxpr),
                                     nonzero_tangents, [])
  jvp_jaxpr_, jvp_consts = closed_jvp_jaxpr.jaxpr, closed_jvp_jaxpr.consts
  jvp_jaxpr = hoist_consts_to_refs(jvp_jaxpr_)
  jvp_which_linear = (*(False,) * len(jvp_consts), *which_linear, *(True,) * len(tangents))
  out = run_state_p.bind(*jvp_consts, *primals, *tangents, jaxpr=jvp_jaxpr,
                         which_linear=jvp_which_linear,
                         # TODO(sharadmv): compute this in the general case
                         is_initialized=(True,) * len(jvp_jaxpr.invars))
  out_consts, out_primals, out_tangents = split_list(out, [len(jvp_consts),
                                                           len(primals)])
  del out_consts
  out_tangents_iter = iter(out_tangents)
  out_tangents = [next(out_tangents_iter) if nz else ad_util.Zero.from_primal_value(p)
                  for p, nz in zip(out_primals, nonzero_tangents)]
  return out_primals, out_tangents
ad.primitive_jvps[run_state_p] = _run_state_jvp

_save_everything = lambda *_, **__: True

def _convert_outputs_to_writes(
    jaxpr: core.Jaxpr) -> tuple[core.Jaxpr, list[core.ShapedArray]]:
  assert not jaxpr.constvars, "Jaxpr shouldn't have constvars."

  in_avals = [v.aval for v in jaxpr.invars]
  @lu.wrap_init
  def eval_jaxpr(*refs):
    # We split the refs into the original input refs and the dummy residual
    # refs.
    orig_refs, residual_refs = split_list(refs, [len(in_avals)])
    residual_vals = core.eval_jaxpr(jaxpr, (), *orig_refs)
    for res_ref, res_val in zip(residual_refs, residual_vals):
      res_ref[...] = res_val
    return []
  res_ref_avals = [AbstractRef(v.aval) if not isinstance(v.aval, AbstractRef)
                   else v.aval for v in jaxpr.outvars]
  jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(
      eval_jaxpr, [*in_avals, *res_ref_avals])
  assert not consts
  return jaxpr, [core.ShapedArray(a.shape, a.dtype) for a in res_ref_avals]

def _convert_inputs_to_reads(num_res: int, jaxpr: core.Jaxpr) -> core.Jaxpr:
  assert not jaxpr.constvars, "Jaxpr should not have constvars"

  @lu.wrap_init
  def eval_jaxpr(*refs):
    residual_refs, orig_refs = split_list(refs, [num_res])
    residual_vals = [r[...] for r in residual_refs]
    () = core.eval_jaxpr(jaxpr, (), *residual_vals, *orig_refs)
    return []

  res_val_avals, orig_ref_avals = \
      split_list([v.aval for v in jaxpr.invars], [num_res])
  res_ref_avals = [AbstractRef(aval) if not isinstance(aval, AbstractRef) else
                   aval for aval in res_val_avals]
  jaxpr, _, (), () = pe.trace_to_jaxpr_dynamic(
      eval_jaxpr, [*res_ref_avals, *orig_ref_avals])
  return jaxpr

def _run_state_partial_eval(trace: pe.JaxprTrace, *tracers: pe.JaxprTracer,
                            jaxpr: core.Jaxpr, which_linear: tuple[bool, ...],
                            is_initialized: tuple[bool, ...]):
  if not all(is_initialized):
    raise NotImplementedError(
        "Uninitialized Refs are not supported in partial_eval."
    )
  num_inputs = len(tracers)
  assert num_inputs == len(jaxpr.invars)
  in_unknowns = [not t.pval.is_known() for t in tracers]
  # We first need to run a fixpoint to determine which of the `Ref`s are unknown
  # after running the for loop. We want to use the jaxpr to determine which
  # `Ref`s are unknown after executing the for loop body given which `Ref`s are
  # unknown before. However, the jaxpr has no outputs. Instead, we discharge
  # the body and run the fixpoint with the discharged jaxpr. We can do this
  # because the outputs of the jaxpr are one-to-one with the inputs.
  discharged_jaxpr_, discharged_consts = discharge_state(jaxpr, ())
  discharged_jaxpr = pe.convert_constvars_jaxpr(discharged_jaxpr_)
  for _ in range(num_inputs):
    jaxpr_in_unknowns = [False] * len(discharged_consts) + in_unknowns
    _, _, out_unknowns, out_inst, _, _ = pe.partial_eval_jaxpr_stateful(
        discharged_jaxpr, jaxpr_in_unknowns, jaxpr_in_unknowns,
          in_unknowns, False, _save_everything)
    # assert out_inst == out_unknowns
    out_unknowns = list(out_unknowns)
    if out_unknowns == in_unknowns:
      break
    in_unknowns = map(operator.or_, in_unknowns, out_unknowns)
  else:
    raise Exception("Invalid fixpoint")
  del out_unknowns  # redundant since it's the same as `in_unknowns`
  tracers = tuple(trace.instantiate_const(t) if uk else t
                  for t, uk in zip(tracers, in_unknowns))

  # We use `partial_eval_jaxpr_stateful` here because it won't remove effectful
  # primitives like `get`/`set`.
  jaxpr_known_resout, jaxpr_unknown_resin_, _, _, num_res_out, num_res_ref = \
        pe.partial_eval_jaxpr_stateful(jaxpr, in_unknowns, in_inst=in_unknowns,
                                     ensure_out_unknowns=[], ensure_out_inst=[],
                                     saveable=_save_everything)
  # # `partial_eval_jaxpr_stateful` will give us jaxprs that have hybrid `Ref`
  # and regular valued input/outputs. However, we'd like to bind these jaxprs to
  # a `for`, which expects only `Ref` inputs and no output. We need to convert
  # both of these jaxprs into ones that are compatible with `for`.

  # `jaxpr_known_resout` is a jaxpr that maps from all the input `Refs`
  # to output residual values (none of them should be `Ref`s). We'll need to
  # convert the output residual values into `Ref`s that are initially empty
  # `Ref`s that are written to at the end of the jaxpr.
  num_res = num_res_out + num_res_ref

  num_invars = len(jaxpr_known_resout.invars) - num_res_ref
  _, res_ref_avals = split_list(
      [v.aval for v in jaxpr_known_resout.invars], [num_invars])
  res_avals = [a.inner_aval for a in res_ref_avals]  # pytype: disable=attribute-error
  jaxpr_known, new_res_avals = _convert_outputs_to_writes(jaxpr_known_resout)
  # We now run the known jaxpr to obtain our residual values.
  known_tracers, _ = partition_list(in_unknowns, tracers)
  known_which_linear, _ = partition_list(in_unknowns, which_linear)
  known_vals = [t.pval.get_known() for t in known_tracers]
  all_res_avals = [*res_avals, *new_res_avals]
  empty_res = map(ad_util.zeros_like_aval, all_res_avals)
  jaxpr_known_args = [*known_vals, *empty_res]

  jaxpr_known_which_linear = (*known_which_linear, *(False,) * num_res)
  out_flat = run_state_p.bind(*jaxpr_known_args, jaxpr=jaxpr_known,
                              which_linear=jaxpr_known_which_linear,
                              # TODO(sharadmv): compute this in the general case
                              is_initialized=(True,) * len(jaxpr_known.invars))
  known_outputs, residuals = split_list(out_flat, [len(known_tracers)])
  residuals = map(trace.new_instantiated_const, residuals)
  ref_res, nonref_res = split_list(residuals, [num_res_ref])

  # Now we handle the `jaxpr_unknown` that expects residual values as inputs.
  # This jaxpr is the output of `partial_eval_jaxpr_stateful` that marks which
  # inputs are actually used.
  # `partial_eval_jaxpr_stateful` doesn't remove extra inputs/outputs for you
  # so we use `dce_jaxpr` here to do that.
  # To make it compatible with `for`, we need to convert those residual values
  # into `Ref`s.
  jaxpr_unknown = _convert_inputs_to_reads(len(new_res_avals),
                                           jaxpr_unknown_resin_)
  _, unknown_tracers = partition_list(in_unknowns, tracers)
  _, uk_which_linear = partition_list(in_unknowns, which_linear)
  unknown_which_linear = (False,) * num_res + tuple(uk_which_linear)
  unknown_inputs = [*nonref_res, *ref_res, *unknown_tracers]
  # Outputs match inputs so we construct output tracers that look like the input
  # tracers.
  res_ref_unknown_outputs = [
      pe.JaxprTracer(trace, pe.PartialVal.unknown(t.aval), None)
      for t in unknown_inputs]
  name_stack = source_info_util.current_name_stack()[len(trace.name_stack):]
  source = source_info_util.current().replace(name_stack=name_stack)

  assert len(unknown_inputs) == len(res_ref_unknown_outputs)
  assert len(unknown_inputs) == len(jaxpr_unknown.invars)
  uk_params = dict(jaxpr=jaxpr_unknown, which_linear=unknown_which_linear,
                   # TODO(sharadmv); compute this in the general case
                   is_initialized=(True,) * len(jaxpr_unknown.invars))
  _, eqn_effects = run_state_p.abstract_eval(*[v.aval for v in unknown_inputs],
                                             **uk_params)
  eqn = pe.new_eqn_recipe(unknown_inputs, res_ref_unknown_outputs,
                          run_state_p, uk_params,
                          eqn_effects, source)
  for t in res_ref_unknown_outputs: t.recipe = eqn
  _, unknown_outputs = split_list(res_ref_unknown_outputs, [num_res])
  return merge_lists(in_unknowns, known_outputs, unknown_outputs)
pe.custom_partial_eval_rules[run_state_p] = _run_state_partial_eval

def _run_state_partial_eval_custom(
    saveable: Callable[..., pe.RematCases_],
    in_unknowns: Sequence[bool],
    in_inst: Sequence[bool],
    eqn: core.JaxprEqn):
  if not any(in_unknowns):
    return eqn, None, in_unknowns, [False] * len(in_unknowns), []
  jaxpr, which_linear, is_initialized = split_dict(
      eqn.params, ["jaxpr", "which_linear", "is_initialized"]
  )
  if not all(is_initialized):
    raise NotImplementedError(
        "Uninitialized Refs are not supported in partial_eval_custom."
    )
  num_inputs = len(eqn.invars)
  # We first need to run a fixpoint to determine which of the `Ref`s are unknown
  # after running the for loop. However, the jaxpr has no outputs. Instead, we
  # discharge the body and run the fixpoint with the discharged jaxpr. We can do
  # this because the outputs of the discharged jaxpr are one-to-one with the
  # inputs.
  discharged_jaxpr, discharged_consts = discharge_state(jaxpr, ())
  discharged_jaxpr = discharged_jaxpr.replace(
      invars=discharged_jaxpr.constvars + discharged_jaxpr.invars,
      constvars=[])
  in_unknowns, in_inst = list(in_unknowns), list(in_inst)
  out_unknowns, out_inst =  in_unknowns, in_unknowns
  for _ in range(num_inputs):
    jaxpr_in_unknowns = [False] * len(discharged_consts) + in_unknowns
    _, _, out_unknowns, out_inst, _, _ = pe.partial_eval_jaxpr_stateful(
        discharged_jaxpr,
        in_unknowns=jaxpr_in_unknowns,
        in_inst=jaxpr_in_unknowns,
        ensure_out_unknowns=in_unknowns,
        ensure_out_inst=in_unknowns,
        saveable=saveable)
    out_unknowns = list(out_unknowns)
    if out_unknowns == in_unknowns:
      break
    in_unknowns = map(operator.or_, in_unknowns, out_unknowns)
  else:
    if num_inputs > 0:
      raise Exception("Invalid fixpoint")
  del out_unknowns # Redundant since it's the same as `in_unknowns`
  new_inst = [x for x, already, inst in zip(eqn.invars, in_inst, out_inst)
              if type(x) is core.Var and inst and not already]

  # We use `partial_eval_jaxpr_stateful` here because it won't remove effectful
  # primitives like `get`/`set`.
  jaxpr_known_resout, jaxpr_staged_resin_, _, _, num_res_out, num_res_ref = \
        pe.partial_eval_jaxpr_stateful(jaxpr, in_unknowns,
            in_unknowns, [], [], saveable)
  num_res = num_res_ref + num_res_out
  # `partial_eval_jaxpr_stateful` will give us jaxprs that have hybrid `Ref` and
  # non-Ref input/outputs. However, we'd like to bind these jaxprs to a
  # `for`, which expects only `Ref` inputs and no output. We need to convert
  # both of these jaxprs into ones that are compatible with `for`.
  # TODO(sharadmv,mattjj): implement "passthrough" optimization.

  # `jaxpr_known_resout` is a jaxpr that maps from all the input `Refs`
  # to output residual values (none of them should be `Ref`s). We'll need to
  # convert the output residual values into `Ref`s that are initially empty
  # `Ref`s that are written to at the end of the jaxpr.
  jaxpr_known, res_avals = _convert_outputs_to_writes(jaxpr_known_resout)

  # In a stateful partial_eval, the residuals should be `Ref`s.
  res_avals = map(AbstractRef, res_avals)  # type: ignore

  known_invars, staged_invars = partition_list(in_unknowns, eqn.invars)
  known_outvars, staged_outvars = partition_list(in_unknowns, eqn.outvars)
  newvar = core.gensym()
  _, res_ref_avals = split_list([v.aval for v in jaxpr_known_resout.invars],
                                [len(known_invars)])
  nonref_resvars = map(newvar, res_avals)
  ref_resvars = map(newvar, res_ref_avals)
  known_out_resvars = map(newvar, [*res_ref_avals, *res_avals])

  known_which_linear, _ = partition_list(in_unknowns, which_linear)
  jaxpr_known_which_linear = (*known_which_linear, *(False,) * num_res)
  known_and_res_invars = [*known_invars, *ref_resvars, *nonref_resvars]

  known_params = dict(jaxpr=jaxpr_known, which_linear=jaxpr_known_which_linear,
                      # TODO(sharadmv): compute this in the general case
                      is_initialized=(True,) * len(jaxpr_known.invars))
  _, known_effects = run_state_p.abstract_eval(
      *[v.aval for v in known_and_res_invars], **known_params)
  eqn_known = pe.new_jaxpr_eqn(known_and_res_invars,
                               [*known_outvars, *known_out_resvars],
                               run_state_p, known_params,
                               known_effects, eqn.source_info)

  jaxpr_staged = _convert_inputs_to_reads(len(res_avals), jaxpr_staged_resin_)

  _, staged_which_linear = partition_list(in_unknowns, which_linear)
  which_linear_unknown = (*[False] * num_res, *staged_which_linear)
  staged_params = dict(jaxpr=jaxpr_staged, which_linear=which_linear_unknown,
                       # TODO(sharadmv): compute this in the general case
                       is_initialized=(True,) * len(jaxpr_staged.invars))
  rejiggered_resvars = [*nonref_resvars, *ref_resvars]
  _, staged_invars = partition_list(in_unknowns, eqn.invars)
  res_staged_invars = [*rejiggered_resvars, *staged_invars]
  _, staged_effects = run_state_p.abstract_eval(
      *[v.aval for v in res_staged_invars], **staged_params)
  _, staged_outvars = partition_list(in_unknowns, eqn.outvars)
  if num_res:
    @lu.wrap_init
    def staged(*args):
      out = run_state_p.bind(*args, **staged_params)
      return out[num_res:]
    staged_call_jaxpr, _, (), () = pe.trace_to_jaxpr_dynamic(staged,
                                                             [v.aval for v in res_staged_invars])
    eqn_staged = pe.new_jaxpr_eqn(res_staged_invars,
                                  staged_outvars,
                                  core.closed_call_p,
                                  dict(call_jaxpr=pe.close_jaxpr(staged_call_jaxpr)),
                                  staged_effects, eqn.source_info)
    assert len(res_staged_invars) == len(staged_call_jaxpr.invars)
    assert len(staged_outvars) == len(staged_call_jaxpr.outvars)
  else:
    eqn_staged = pe.new_jaxpr_eqn(staged_invars,
                                  staged_outvars,
                                  run_state_p,
                                  staged_params,
                                  staged_effects, eqn.source_info)
  new_vars = [*new_inst, *nonref_resvars, *ref_resvars]
  return eqn_known, eqn_staged, in_unknowns, in_unknowns, new_vars
pe.partial_eval_jaxpr_custom_rules[run_state_p] = _run_state_partial_eval_custom

def _transpose_jaxpr(jaxpr: core.Jaxpr, which_linear: Sequence[bool],
                     is_initialized: tuple[bool, ...]) -> tuple[core.Jaxpr, Any]:
  if not all(is_initialized):
    raise NotImplementedError(
        "Uninitialized Refs are not supported in transpose."
    )
  def trans(*args):
    # First we want to run the computation to read all the residual refs. We can
    # do that by using partial evaluation with all linear inputs unknown.
    res_jaxpr_, tangent_jaxpr_, *_, num_res_out, num_res_ref = \
        pe.partial_eval_jaxpr_stateful(jaxpr, which_linear, in_inst=which_linear,
                                       ensure_out_inst=[],
                                       ensure_out_unknowns=[],
                                       saveable=_save_everything)

    num_unknown = sum(which_linear)
    num_known = len(jaxpr.invars) - num_unknown
    res_args, _ = partition_list(which_linear, args)
    res_jaxpr_avals = [v.aval for v in res_jaxpr_.invars]
    _, res_avals = split_list(res_jaxpr_avals, [num_known])
    res_avals = [a.inner_aval for a in res_avals]  # pytype: disable=attribute-error
    all_avals = [*res_avals, *[v.aval for v in res_jaxpr_.outvars]]
    empty_res = map(ad.zeros_like_aval, all_avals)
    res_jaxpr, _ = _convert_outputs_to_writes(res_jaxpr_)
    res = run_state_p.bind(
        *res_args,
        *empty_res,
        jaxpr=res_jaxpr,
        which_linear=(False,) * (len(res_args) + len(empty_res)),
        # TODO(sharadmv): compute this in the general case
        is_initialized=(True,) * len(res_jaxpr.invars),
    )
    res = res[len(res_args):]
    ref_res_, nonref_res_ = split_list(res, [num_res_ref])

    # Now that we have residual values, we run the tangent jaxpr. It takes as
    # input the residuals, the loop index, and all the refs (at least, the ones
    # that are used in the body). Luckily, `tangent_jaxpr_` has all known and
    # unknown inputs!
    tangent_jaxpr, used_inputs = pe.dce_jaxpr(tangent_jaxpr_, [])
    used_res, used_cts = split_list(used_inputs, [len(res)])
    used_nonref_res, used_ref_res = split_list(used_res, [num_res_out])
    _, nonref_res = partition_list(used_nonref_res, nonref_res_)
    _, ref_res = partition_list(used_ref_res, ref_res_)
    primals_args = [*nonref_res, *ref_res]
    _, tangent_args = partition_list(which_linear, args)
    _, ct_args = partition_list(used_cts, tangent_args)
    ad.backward_pass(tangent_jaxpr, False, (), (*primals_args, *ct_args), ())
    return []
  jaxpr_trans, _, consts, () = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(trans), [v.aval for v in jaxpr.invars])
  return jaxpr_trans, consts

def _run_state_transpose(in_cts, *args, jaxpr: core.Jaxpr,
                         which_linear: tuple[bool, ...],
                         is_initialized: tuple[bool, ...]):
  if not all(is_initialized):
    raise NotImplementedError(
        "Uninitialized Refs are not supported in transpose."
    )
  # if any in_ct is nonzero, we definitely want it in args_ (and the
  # corresponding x in args could be an undefined primal, but doesn't have to be)
  # for non-res stuff:
  #   getting and setting => (nonzero ct, UndefinedPrimal arg)
  #   just setting =>        (nonzero ct, not UndefinedPrimal, dummy value)
  #   just getting =>        (zero ct   , UndefinedPrimal arg)
  # for res stuff:
  #                          (zero ct   , not UndefinedPrimal)
  assert any(which_linear)
  transpose_args = []
  for x, ct in zip(args, in_cts):
    if   type(ct) is     ad_util.Zero and not ad.is_undefined_primal(x):
      # this is a residual, take x!
      transpose_args.append(x)
    elif type(ct) is     ad_util.Zero and     ad.is_undefined_primal(x):
      # the loop was 'just getting', plug in a zero
      transpose_args.append(ad_util.zeros_like_aval(x.aval))
    elif type(ct) is not ad_util.Zero and not ad.is_undefined_primal(x):
      # the loop was 'just setting', grab that cotangent! x is dummy
      transpose_args.append(ct)
    elif type(ct) is not ad_util.Zero and     ad.is_undefined_primal(x):
      # the loop was 'getting and setting', grab that cotangent!
      transpose_args.append(ct)
  jaxpr_transpose_, consts = _transpose_jaxpr(
      jaxpr, which_linear, is_initialized
  )
  jaxpr_transpose = hoist_consts_to_refs(jaxpr_transpose_)
  which_linear = (*[False] * len(consts), *which_linear)
  const_all_outs = run_state_p.bind(
      *consts,
      *transpose_args,
      jaxpr=jaxpr_transpose,
      which_linear=which_linear,
      # TODO(sharadmv): compute this in the general case
      is_initialized=(True,) * len(jaxpr_transpose.invars),
  )
  _, all_outs = split_list(const_all_outs, [len(consts)])
  ct_outs = [ct if ad.is_undefined_primal(x) else None
             for x, ct in zip(args, all_outs)]
  return ct_outs
ad.primitive_transposes[run_state_p] = _run_state_transpose

@register_discharge_rule(run_state_p)
def _run_state_discharge_rule(in_avals: Sequence[core.AbstractValue],
                              out_avals: Sequence[core.AbstractValue],
                              *args: Any, jaxpr: core.Jaxpr,
                              which_linear: Sequence[bool],
                              is_initialized: tuple[bool, ...]):
  if not all(is_initialized):
    raise NotImplementedError(
        "Uninitialized Refs are not supported in discharge."
    )
  del out_avals
  out_vals = run_state_p.bind(*args, jaxpr=jaxpr, which_linear=which_linear,
                              is_initialized=is_initialized)
  new_invals = []
  for aval, out_val in zip(in_avals, out_vals):
    new_invals.append(out_val if isinstance(aval, AbstractRef) else None)
  return new_invals, out_vals

def initial_style_jaxpr(
    fun: Callable, in_tree: PyTreeDef, in_avals: Sequence[core.AbstractValue]
  ) -> tuple[core.Jaxpr, list[Any], PyTreeDef]:
  return _initial_style_jaxpr(fun, in_tree, tuple(in_avals))

@weakref_lru_cache
def _initial_style_jaxpr(fun, in_tree, in_avals):
  fun_, out_tree_thunk = api_util.flatten_fun_nokwargs(lu.wrap_init(fun),
      tree_util.treedef_tuple((in_tree,)))
  debug = pe.tracing_debug_info(fun_, in_tree, out_tree_thunk, False, 'run_state')
  jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(fun_, in_avals, debug)
  return jaxpr, consts, out_tree_thunk()


T = TypeVar('T')
def run_state(f: Callable[..., None]) -> Callable[[T], T]:
  def wrapped(args):
    flat_args, in_tree = tree_util.tree_flatten(args)
    ref_avals, ref_args = unzip2(map(get_ref_aval_from_value, flat_args))
    # There may be some uninitialized values here in ref_args.
    jaxpr_, consts, _ = initial_style_jaxpr(f, in_tree, ref_avals)
    jaxpr = hoist_consts_to_refs(jaxpr_)
    which_linear = (False,) * (len(consts) + len(ref_args))
    refs_is_initialized = tuple(r is not uninitialized for r in ref_args)
    init_args = tuple(r for r in ref_args if r is not uninitialized)
    # Consts are always initialized.
    is_initialized = (True,) * len(consts) + refs_is_initialized
    out_const_flat = run_state_p.bind(*consts, *init_args, jaxpr=jaxpr,
                                      which_linear=which_linear,
                                      is_initialized=is_initialized)
    _, out_flat = split_list(out_const_flat, [len(consts)])
    return in_tree.unflatten(out_flat)
  return wrapped

def run_state_reference(f: Callable[..., None]):
  def wrapped(args):
    flat_args, in_tree = tree_util.tree_flatten(args)
    ref_avals, ref_args = unzip2(map(get_ref_aval_from_value, flat_args))
    jaxpr_, consts, _ = initial_style_jaxpr(f, in_tree, ref_avals)
    jaxpr = hoist_consts_to_refs(jaxpr_)
    discharged_jaxpr, discharged_consts = discharge_state(jaxpr, ())

    # Initialize any uninitialized values here in ref_args in the reference.
    ref_args = [
        _default_initialization(aval) if r is uninitialized else r
        for r, aval in zip(ref_args, ref_avals)
    ]

    out_const_flat = core.eval_jaxpr(discharged_jaxpr, discharged_consts,
                                     *consts, *ref_args)
    _, out_flat = split_list(out_const_flat, [len(consts)])
    return in_tree.unflatten(out_flat)
  return wrapped
