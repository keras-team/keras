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

from __future__ import annotations

from collections.abc import Callable, Sequence
import contextlib
import functools
import itertools as it
from functools import partial
from typing import Any

from jax._src import config
from jax._src import linear_util as lu
from jax._src.interpreters import partial_eval as pe
from jax.tree_util import (tree_flatten, tree_unflatten,
                           register_pytree_node, Partial)
from jax._src import core
from jax._src import source_info_util
from jax._src.ad_util import (
    add_jaxvals, replace_internal_symbolic_zeros,
    replace_rule_output_symbolic_zeros, Zero, zeros_like_aval, SymbolicZero)
from jax._src.ad_util import zeros_like_p, add_jaxvals_p  # noqa: F401
from jax._src.api_util import flatten_fun, flatten_fun_nokwargs
from jax._src.core import (Trace, Tracer, get_aval, call_p, Primitive, Literal)
from jax._src.dtypes import dtype, float0
from jax._src.util import (unzip2, safe_map, safe_zip, split_list, wrap_name,
                           as_hashable_function, weakref_lru_cache,
                           partition_list)

zip = safe_zip
map = safe_map
def identity(x): return x

def _update_annotation(
    f: lu.WrappedFun,
    orig_type: tuple[tuple[core.AbstractValue, bool], ...] | None,
    explicit_nonzeros: list[bool]
  ) -> lu.WrappedFun:
  if orig_type is None:
    return f
  # By convention, `explicit_nonzeros` only accounts for explicit arguments.
  assert len(explicit_nonzeros) == sum(explicit for _, explicit in orig_type)
  # Implicit arguments never have tangents, so generate the tangent part of the
  # type annotation from explicit arguments only.
  explicit_avals = [aval for aval, explicit in orig_type if explicit]
  tan_types = [(aval.to_tangent_aval(), True)
               for nz, aval in zip(explicit_nonzeros, explicit_avals) if nz]
  return lu.annotate(f, (*orig_type, *tan_types))

def jvp(fun: lu.WrappedFun, has_aux=False, instantiate=True,
    transform_stack=True) -> Any:
  if not has_aux:
    return jvpfun(jvp_subtrace(fun), instantiate, transform_stack)
  else:
    fun, aux = jvp_subtrace_aux(fun)
    return jvpfun(fun, instantiate, transform_stack), aux

@lu.transformation2
def jvpfun(f, instantiate, transform_stack, primals, tangents):
  tag = core.TraceTag()
  tangents = [Zero.from_primal_value(t) if not isinstance(t, Zero)
              and dtype(t) == float0 else t for t in tangents]
  ctx = (source_info_util.transform_name_stack('jvp') if transform_stack
         else contextlib.nullcontext())
  with ctx:
    out_primals, out_tangents = f(tag, primals, tangents)
  if type(instantiate) is bool:
    instantiate = [instantiate] * len(out_tangents)
  out_tangents = [instantiate_zeros(t) if inst else t for t, inst
                  in zip(out_tangents, instantiate)]
  return out_primals, out_tangents

@lu.transformation_with_aux2
def linearize_subtrace(_f: Callable, _store, _tag, nzs_in, *primals, **params):
  with core.take_current_trace() as parent_trace:
    tangent_trace = pe.DynamicJaxprTrace(None)  # TODO(necula): fill-in the debug info (use JAX_USE_DIRECT_LINEARIZATION=1)
    linearize_trace = LinearizeTrace(parent_trace, tangent_trace, tag=_tag)
    tracers = [LinearizeTracer(linearize_trace, p,
                               tangent_trace.new_arg(get_aval(p).to_tangent_aval()))
               if nz else p
               for p, nz in zip(primals, nzs_in)]
    with core.set_current_trace(linearize_trace, check_leaks=True):
      ans = _f(*tracers)
      out_primals, out_tangents = unzip2(map(linearize_trace.to_primal_tangent_pair, ans))
      del linearize_trace, ans, tracers
  nzs_out = tuple(type(t) is not Zero for t in out_tangents)
  out_tangents = tuple(t for t, nz in zip(out_tangents, nzs_out) if nz)
  out_tangents = map(tangent_trace.to_jaxpr_tracer, out_tangents)  # type: ignore[assignment]
  jaxpr, consts, attrs_tracked = tangent_trace.to_jaxpr(out_tangents)
  residual_avals = map(get_aval, consts)
  if attrs_tracked:
    raise NotImplementedError("TODO: attrs")
  _store.store((residual_avals, nzs_out, jaxpr))
  return tuple(consts) + tuple(out_primals)

@lu.transformation2
def jvp_subtrace(f, tag, primals, tangents):
  with core.take_current_trace() as parent_trace:
    trace = JVPTrace(parent_trace, tag)
    in_tracers = [maybe_jvp_tracer(trace, x, t)
                  for x, t in zip(primals, tangents)]
    with core.set_current_trace(trace):
      ans = f(*in_tracers)
    out = unzip2(map(trace.to_primal_tangent_pair, ans))
  return out

@lu.transformation_with_aux2
def jvp_subtrace_aux(f, store, tag, primals, tangents):
  with core.take_current_trace() as parent_trace:
    trace = JVPTrace(parent_trace, tag)
    with core.set_current_trace(trace):
      ans, aux = f(*(map(partial(maybe_jvp_tracer, trace), primals, tangents)))
    out_primals, out_tangents = unzip2(map(trace.to_primal_tangent_pair, ans))
    aux_primals = [x.primal if isinstance(x, JVPTracer) and x._trace.tag is tag
                   else x for x in aux]
  store.store(aux_primals)
  return out_primals, out_tangents

def convert_constvars_jaxpr_constvars_at_end(jaxpr: core.Jaxpr) -> core.Jaxpr:
  dbg = jaxpr.debug_info and jaxpr.debug_info._replace(
      arg_names=jaxpr.debug_info.arg_names + (None,) * len(jaxpr.constvars))
  return core.Jaxpr(constvars=(),
                    invars=jaxpr.invars + jaxpr.constvars,
                    outvars=jaxpr.outvars, eqns=jaxpr.eqns,
                    effects=jaxpr.effects, debug_info=dbg)

def linearize_jaxpr(
    jaxpr: core.ClosedJaxpr,
    nonzeros: Sequence[bool]
  ) -> tuple[core.ClosedJaxpr, int, Sequence[bool], core.ClosedJaxpr]:
  return _linearize_jaxpr(jaxpr, tuple(nonzeros))

@weakref_lru_cache
def _linearize_jaxpr(
    jaxpr: core.ClosedJaxpr,
    nonzeros: tuple[bool, ...]
  ) -> tuple[core.ClosedJaxpr, int, Sequence[bool], core.ClosedJaxpr]:
  dbg = lu.TracingDebugInfo.from_jaxpr(jaxpr)
  primal_trace = pe.DynamicJaxprTrace(dbg)
  tangent_trace = pe.DynamicJaxprTrace(dbg)
  lin_trace = LinearizeTrace(primal_trace, tangent_trace)

  def new_arg(trace, primal_aval, nz):
    primal = primal_trace.new_arg(primal_aval)
    tangent_aval = primal_aval.to_tangent_aval()
    tangent = tangent_trace.new_arg(tangent_aval) if nz else Zero(tangent_aval)
    return LinearizeTracer(trace, primal, tangent)

  tracers = [new_arg(lin_trace, v.aval, nz)
             for (v, nz) in zip(jaxpr.jaxpr.invars, nonzeros)]

  with core.set_current_trace(lin_trace, check_leaks=True):
    ans = core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *tracers)
    out_primals, out_tangents = unzip2(map(lin_trace.to_primal_tangent_pair, ans))
    del lin_trace, ans, tracers, new_arg

  nzs_out = [type(t) is not Zero for t in out_tangents]
  out_tangents = tuple(tangent_trace.to_jaxpr_tracer(t)
                       for (nz, t) in zip(nzs_out, out_tangents) if nz)
  tangent_jaxpr, tangent_consts, attrs_tracked = tangent_trace.to_jaxpr(out_tangents)
  tangent_trace.invalidate()
  if attrs_tracked:
    raise NotImplementedError("TODO: attrs")
  residuals_and_primals = (*tangent_consts, *out_primals)
  residuals_and_primals = map(primal_trace.to_jaxpr_tracer, residuals_and_primals)  # type: ignore[assignment]
  primal_jaxpr, primal_consts, attrs_tracked = primal_trace.to_jaxpr(residuals_and_primals)
  primal_trace.invalidate()
  num_residuals = len(tangent_consts)
  tangent_jaxpr = pe.close_jaxpr(convert_constvars_jaxpr_constvars_at_end(tangent_jaxpr))
  if attrs_tracked:
    raise NotImplementedError("TODO: attrs")
  return core.ClosedJaxpr(primal_jaxpr, primal_consts), num_residuals, nzs_out, tangent_jaxpr

def direct_linearize(traceable: lu.WrappedFun,
                     primals, kwargs, *, has_aux=False, tag=None):
  with core.take_current_trace() as parent_trace:
    tangent_trace = pe.DynamicJaxprTrace(traceable.debug_info)
    tangents = [tangent_trace.new_arg(get_aval(p).to_tangent_aval()) for p in primals]
    tangents = [Zero.from_primal_value(t) if dtype(t) == float0 else t for t in tangents]
    linearize_trace = LinearizeTrace(parent_trace, tangent_trace, tag=tag)
    tracers = [LinearizeTracer(linearize_trace, p, t) for p, t in zip(primals, tangents)]
    tracers = [t.full_lower() for t in tracers]
    with core.set_current_trace(linearize_trace, check_leaks=True):
      if has_aux:
        ans, aux = traceable.call_wrapped(*tracers)
        aux_primals = [x.primal
                       if isinstance(x, LinearizeTracer)
                       and x._trace.tag is linearize_trace.tag
                       else x for x in aux]
      else:
        ans = traceable.call_wrapped(*tracers)
        aux = None
      out_primals, out_tangents = unzip2(map(linearize_trace.to_primal_tangent_pair, ans))
      del linearize_trace, ans, tracers
  out_nzs = [type(t) is not Zero for t in out_tangents]
  out_nz_tangents = [t for t, nz in zip(out_tangents, out_nzs) if nz]
  out_nz_tangents = map(tangent_trace.to_jaxpr_tracer, out_nz_tangents)  # type: ignore
  jaxpr, consts, attrs_tracked = tangent_trace.to_jaxpr(out_nz_tangents)
  tangent_trace.invalidate()
  out_tangents_pvals = [pe.PartialVal.unknown(core.get_aval(t)) if nz else
                        pe.PartialVal.known(zeros_like_aval(t.aval))
                        for t, nz in zip(out_tangents, out_nzs)]
  if attrs_tracked:
    raise NotImplementedError("TODO: attrs")
  if has_aux:
    return out_primals, out_tangents_pvals, jaxpr, consts, aux_primals
  else:
    return out_primals, out_tangents_pvals, jaxpr, consts

def linearize(traceable: lu.WrappedFun,
              *primals, **kwargs):
  has_aux = kwargs.pop('has_aux', False)
  if config.use_direct_linearize.value:
    return direct_linearize(traceable, primals, kwargs, has_aux=has_aux)
  if not has_aux:
    jvpfun = jvp(traceable)
  else:
    jvpfun, aux = jvp(traceable, has_aux=True)

  in_pvals = (tuple(pe.PartialVal.known(p) for p in primals)
              + tuple(pe.PartialVal.unknown(get_aval(p).to_tangent_aval())
                      for p in primals))
  _, in_tree = tree_flatten(((primals, primals), {}))
  jvpfun_flat, out_tree = flatten_fun(jvpfun, in_tree)
  jaxpr, out_pvals, consts = pe.trace_to_jaxpr_nounits(jvpfun_flat, in_pvals)
  out_primals_pvals, out_tangents_pvals = tree_unflatten(out_tree(), out_pvals)
  if any(not out_primal_pval.is_known() for out_primal_pval in out_primals_pvals):
    raise ValueError(
        "Linearization failed to produce known values for all output primals. "
        "This is typically caused by attempting to differentiate a function "
        "uses an operation that does not support reverse-mode autodiff.")
  out_primals_consts = [pval.get_known() for pval in out_primals_pvals]
  if not has_aux:
    return out_primals_consts, out_tangents_pvals, jaxpr, consts
  else:
    return out_primals_consts, out_tangents_pvals, jaxpr, consts, aux()

def vjp(traceable: lu.WrappedFun, primals, has_aux=False):
  if not has_aux:
    out_primals, pvals, jaxpr, consts = linearize(traceable, *primals)
  else:
    out_primals, pvals, jaxpr, consts, aux = linearize(traceable, *primals, has_aux=True)

  def unbound_vjp(pvals, jaxpr, consts, *cts):
    cts = tuple(ct for ct, pval in zip(cts, pvals) if not pval.is_known())
    dummy_args = [UndefinedPrimal(v.aval) for v in jaxpr.invars]
    arg_cts = backward_pass(jaxpr, True, consts, dummy_args, cts)
    return map(instantiate_zeros, arg_cts)

  # Ensure that vjp_ is a PyTree so that we can pass it from the forward to the backward
  # pass in a custom VJP.
  vjp_ =  Partial(partial(unbound_vjp, pvals, jaxpr), consts)
  if not has_aux:
    return out_primals, vjp_
  else:
    return out_primals, vjp_, aux

def unpair_pval(pval):
  aval, const = pval
  const_1, const_2 = const
  if aval is None:
    return (None, const_1), (None, const_2)
  else:
    aval_1, aval_2 = aval
    return (aval_1, const_1), (aval_2, const_2)

# NOTE: The FIXMEs below are caused by primal/tangent mixups (type
# errors if you will)
def backward_pass(jaxpr: core.Jaxpr, transform_stack,
                  consts, primals_in, cotangents_in):
  if all(type(ct) is Zero for ct in cotangents_in) and not jaxpr.effects:
    return map(lambda v: Zero(v.aval), jaxpr.invars)

  def write_cotangent(prim, v, ct):
    # assert v not in primal_env
    assert ct is not Zero, (prim, v.aval)  # check for an old harmless type error
    if ct is None or type(v) is Literal:
      return
    if type(ct) is Zero:
      # FIXME: This triggers a lot of failures!
      # assert v.aval == ct.aval, (prim, v.aval, ct.aval)
      return
    ct_env[v] = add_tangents(ct_env[v], ct) if v in ct_env else ct
    # TODO(mattjj): add back these checks for dynamic shapes
    # if config.enable_checks.value:
    #   ct_aval = core.get_aval(ct_env[v])
    #   joined_aval = core.lattice_join(v.aval, ct_aval).strip_weak_type()
    #   assert v.aval.strip_weak_type() == joined_aval, (prim, v.aval, ct_aval)

  def read_cotangent(v):
    return ct_env.pop(v, Zero(v.aval.to_tangent_aval()))

  def read_primal(v):
    if type(v) is Literal:
      return v.val
    else:
      a = v.aval
      if type(a) is core.DShapedArray:
        shape = [primal_env[d] if type(d) is core.Var else d for d in a.shape]
        a = a.update(shape=tuple(shape))
      return primal_env.get(v, UndefinedPrimal(a))

  def write_primal(v, val):
    if not is_undefined_primal(val):
      primal_env[v] = val

  primal_env: dict[Any, Any] = {}
  map(write_primal, jaxpr.constvars, consts)
  # FIXME: invars can contain both primal and tangent values, and this line
  #        forces primal_in to contain UndefinedPrimals for tangent values!
  map(write_primal, jaxpr.invars, primals_in)

  ct_env: dict[Any, Any] = {}
  ctx = (source_info_util.transform_name_stack('transpose') if transform_stack
         else contextlib.nullcontext())
  with ctx:
    map(partial(write_cotangent, 'outvars'), jaxpr.outvars, cotangents_in)
    for eqn in jaxpr.eqns[::-1]:
      if eqn.primitive.ref_primitive:
        if eqn.primitive is core.mutable_array_p:
          val_var, = eqn.invars
          ref_var, = eqn.outvars
          ref = read_primal(ref_var)
          ct_out = core.freeze(ref)
          write_cotangent(eqn.primitive, val_var, ct_out)
        elif eqn.primitive is core.freeze_p:
          val_var, = eqn.outvars  # type: ignore
          ref_var, = eqn.invars   # type: ignore
          ct_in = instantiate_zeros(read_cotangent(val_var))
          write_primal(ref_var, core.mutable_array(ct_in))
        continue

      invals = map(read_primal, eqn.invars)
      if eqn.primitive.multiple_results:
        cts_in = map(read_cotangent, eqn.outvars)
      else:
        cts_in, = map(read_cotangent, eqn.outvars)
      name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
      with source_info_util.user_context(
          eqn.source_info.traceback, name_stack=name_stack), eqn.ctx.manager:
        if eqn.primitive.call_primitive or eqn.primitive.map_primitive:
          cts_in_avals = [v.aval for v in eqn.outvars]
          params = dict(eqn.params)
          call_jaxpr = params.pop('call_jaxpr')
          cts_out = get_primitive_transpose(eqn.primitive)(
              params, call_jaxpr, invals, cts_in, cts_in_avals)
        else:
          cts_out = get_primitive_transpose(eqn.primitive)(
              cts_in, *invals, **eqn.params)
        cts_out = [Zero(v.aval) for v in eqn.invars] if cts_out is Zero else cts_out
        # FIXME: Some invars correspond to primals!
        map(partial(write_cotangent, eqn.primitive), eqn.invars, cts_out)

  cotangents_out = map(read_cotangent, jaxpr.invars)
  return cotangents_out

def closed_backward_pass(jaxpr: core.ClosedJaxpr, transform_stack,
                         primals_in, cotangents_in):
  return backward_pass(jaxpr.jaxpr, transform_stack, jaxpr.consts,
                       primals_in, cotangents_in)


class UndefinedPrimal:
  __slots__ = ['aval']
  def __init__(self, aval):
    self.aval = aval
  def __repr__(self):
    return f'UndefinedPrimal({self.aval})'

def is_undefined_primal(x):
  return type(x) is UndefinedPrimal

register_pytree_node(UndefinedPrimal,
                     lambda z: ((), z.aval),
                     lambda aval, _: UndefinedPrimal(aval))

def get_primitive_transpose(p):
  try:
    return primitive_transposes[p]
  except KeyError as err:
    raise NotImplementedError(
        "Transpose rule (for reverse-mode differentiation) for '{}' "
        "not implemented".format(p)) from err

@lu.transformation_with_aux2
def nonzero_tangent_outputs(f, store, *args, **kwargs):
  results = (_, tangents_out) = f(*args, **kwargs)
  store.store([type(r) is not Zero for r in tangents_out])
  return results


class JVPTrace(Trace):
  def __init__(self, parent_trace, tag):
    self.tag = tag
    self.parent_trace = parent_trace

  def to_primal_tangent_pair(self, val):
    if isinstance(val, JVPTracer) and val._trace.tag is self.tag:
      return (val.primal, val.tangent)
    else:
      tangent_zero = Zero.from_primal_value(val)
      return (val, tangent_zero)

  def process_primitive(self, primitive, tracers, params):
    primals_in, tangents_in = unzip2(map(self.to_primal_tangent_pair, tracers))
    if all(type(t) is Zero for t in tangents_in):
      return primitive.bind_with_trace(self.parent_trace, primals_in, params)
    jvp = primitive_jvps.get(primitive)
    if not jvp:
      msg = f"Differentiation rule for '{primitive}' not implemented"
      raise NotImplementedError(msg)
    with core.set_current_trace(self.parent_trace):
      primal_out, tangent_out = jvp(primals_in, tangents_in, **params)

    if primitive.multiple_results:
      return [maybe_jvp_tracer(self, x, t) for x, t in zip(primal_out, tangent_out)]
    else:
      return maybe_jvp_tracer(self, primal_out, tangent_out)

  def process_call(self, call_primitive, f, tracers, params):
    assert call_primitive.multiple_results
    primals, tangents = unzip2(map(self.to_primal_tangent_pair, tracers))
    which_nz = [     type(t) is not Zero           for t in tangents]
    tangents = [t if type(t) is not Zero else None for t in tangents]
    args, in_tree = tree_flatten((primals, tangents))
    f_jvp = jvp_subtrace(f, self.tag)
    f_jvp, which_nz_out = nonzero_tangent_outputs(f_jvp)
    if isinstance(call_primitive, core.MapPrimitive):
      in_axes = params['in_axes']
      tangent_in_axes = [ax for ax, nz in zip(in_axes, which_nz) if nz]
      out_axes_thunk = params['out_axes_thunk']
      # NOTE: This assumes that the output tangents being zero is a
      # deterministic function of which input tangents were zero.
      @as_hashable_function(closure=out_axes_thunk)
      def new_out_axes_thunk():
        out_ax = out_axes_thunk()
        return (*out_ax, *(ax for ax, nz in zip(out_ax, which_nz_out()) if nz))
      params = dict(params, in_axes=(*in_axes, *tangent_in_axes),
                    out_axes_thunk=new_out_axes_thunk)
    f_jvp, out_tree = traceable(f_jvp, in_tree)
    update_params = call_param_updaters.get(call_primitive)
    new_params = update_params(params, which_nz) if update_params else params
    fun_and_args = (_update_annotation(f_jvp, f.in_type, which_nz),) + tuple(args)
    result = call_primitive.bind_with_trace(self.parent_trace, fun_and_args, new_params)
    primal_out, tangent_out = tree_unflatten(out_tree(), result)
    tangent_out = [Zero.from_primal_value(p) if t is None else t
                   for p, t in zip(primal_out, tangent_out)]
    return [maybe_jvp_tracer(self, p, t) for p, t in zip(primal_out, tangent_out)]

  # The only difference between process_map and process_call is that
  # the `in_axes` and `out_axes_thunk` params must be updated;
  # that's handled in process_call.
  process_map = process_call

  def process_custom_jvp_call(self, prim, fun, f_jvp, tracers, *, symbolic_zeros):
    primals_in, tangents_in = unzip2(map(self.to_primal_tangent_pair, tracers))
    if all(type(t) is Zero for t in tangents_in):
      return prim.bind_with_trace(self.parent_trace, (fun, f_jvp, *primals_in),
                                  dict(symbolic_zeros=symbolic_zeros))
    with core.set_current_trace(self.parent_trace):
      if not symbolic_zeros:
        tangents_in = map(instantiate_zeros, tangents_in)
      else:
        tangents_in = map(replace_internal_symbolic_zeros, tangents_in)
      outs = f_jvp.call_wrapped(*(tuple(primals_in) + tuple(tangents_in)))

    primals_out, tangents_out = split_list(outs, [len(outs) // 2])
    tangents_out = map(replace_rule_output_symbolic_zeros, tangents_out)
    return map(partial(maybe_jvp_tracer, self), primals_out, tangents_out)

  def process_custom_vjp_call(self, prim, fun, fwd, bwd, tracers, out_trees,
                              symbolic_zeros):
    primals_in, tangents_in = unzip2(map(self.to_primal_tangent_pair, tracers))
    if all(type(t) is Zero for t in tangents_in):
      return prim.bind_with_trace(self.parent_trace,
                                  (fun, fwd, bwd, *primals_in),
                                  dict(out_trees=out_trees, symbolic_zeros=symbolic_zeros))
    fwd_in = [(p, type(t) is not Zero) for p, t in zip(primals_in, tangents_in)]
    fwd_in = [x for pair in fwd_in for x in pair]   # flatten
    with core.set_current_trace(self.parent_trace):
      res_and_primals_out = fwd.call_wrapped(*fwd_in)

    _, res_tree = out_trees()
    res, primals_out = split_list(res_and_primals_out, [res_tree.num_leaves])
    avals_out = [core.get_aval(x).to_tangent_aval() for x in primals_out]
    # TODO(frostig,mattjj): avoid instantiating zeros when we don't have to!
    with core.set_current_trace(self.parent_trace):
      tangents_in = map(instantiate_zeros, tangents_in)
      tangents_out = custom_lin_p.bind(
        *res, *tangents_in, num_res=res_tree.num_leaves, bwd=bwd,
        out_avals=avals_out, symbolic_zeros=symbolic_zeros)
    return map(partial(maybe_jvp_tracer, self), primals_out, tangents_out)

  def process_custom_transpose(self, prim, call, tracers, **params):
    ps_in, ts_in = unzip2(map(self.to_primal_tangent_pair, tracers))
    res_ps_in, lin_ps_in = split_list(ps_in, [params['res_tree'].num_leaves])
    res_ts_in, lin_ts_in = split_list(ts_in, [params['res_tree'].num_leaves])

    # TODO(frostig): Handle differentiation with respect to residual
    # operands. Calling `call` twice on all operands invalid, since it
    # isn't linear in the residuals. However, we know that if we
    # write:
    #
    #   jvp_call_res = lambda x: partial(jvp, lambda r: call(r, x))
    #
    # then:
    #
    #   jvp(call, (r, x), (dr, dx)) == jvp_call_res(x)(r, dr) + call(r, dx)
    #
    # In words: a possible strategy is to take the jvp of `call` with
    # respect to residuals, and with linear arguments fixed, then add
    # that to a custom-transpose call to `call` (i.e. what we already
    # do below in the all-linear argument case).

    if any(type(t) is not Zero for t in res_ts_in):
      raise NotImplementedError(
        'JVP of custom transpose with respect to non-symbolic-zero residuals')

    with core.set_current_trace(self.parent_trace):
      ps_out = prim.bind(call, *ps_in, **params)
      lin_ts_in = map(instantiate_zeros, lin_ts_in)
      ts_out = prim.bind(call, *res_ps_in, *lin_ts_in, **params)

    return map(partial(maybe_jvp_tracer, self), ps_out, ts_out)

def maybe_jvp_tracer(trace, primal, tangent):
  if type(tangent) is Zero or dtype(tangent) == float0:
    return primal
  else:
    return JVPTracer(trace, primal, tangent)

class JVPTracer(Tracer):
  __slots__ = ['primal', 'tangent']

  def __init__(self, trace, primal, tangent):
    if config.enable_checks.value:
      _primal_tangent_shapes_match(primal, tangent)
    self._trace = trace
    self.primal = primal
    self.tangent = tangent

  @property
  def aval(self):
    return get_aval(self.primal)

  def full_lower(self):
    if type(self.tangent) is Zero:
      return core.full_lower(self.primal)
    else:
      return self

  def to_concrete_value(self):
    return core.to_concrete_value(self.primal)

  def get_referent(self):
    return core.get_referent(self.primal)

def _primal_tangent_shapes_match(primal, tangent):
  if type(tangent) is not Zero:
    primal_aval = get_aval(primal).strip_weak_type()
    tangent_aval = get_aval(tangent).strip_weak_type()
    assert core.definitely_equal_shape(primal_aval.shape, tangent_aval.shape)
    expected_tangent_dtype = core.primal_dtype_to_tangent_dtype(primal_aval.dtype)
    assert expected_tangent_dtype == tangent_aval.dtype, (expected_tangent_dtype, tangent_aval.dtype)

call_param_updaters: dict[core.Primitive, Callable] = {}
call_linearize_param_updaters: dict[core.Primitive, Callable] = {}
call_transpose_param_updaters: dict[core.Primitive, Callable] = {}

# -------------------- Linearize trace --------------------

class LinearizeTrace(Trace):

  def __init__(self, parent_trace, tangent_trace, tag=None):
    self.tag = core.TraceTag() if tag is None else tag
    self.parent_trace = parent_trace
    self.tangent_trace = tangent_trace

  def to_primal_tangent_pair(self, val):
    if isinstance(val, LinearizeTracer) and val._trace.tag is self.tag:
      return (val.primal, val.tangent)
    else:
      tangent_zero = Zero.from_primal_value(val)
      return (val, tangent_zero)

  def process_primitive(self, primitive, args, params):
    primals_in, tangents_in = unzip2(map(self.to_primal_tangent_pair, args))
    tangent_nzs = [type(t) is not Zero for t in tangents_in]
    if all(type(t) is Zero for t in tangents_in):
      return primitive.bind_with_trace(self.parent_trace, primals_in, params)
    fallback = partial(fallback_linearize_rule, primitive)
    lin = primitive_linearizations.get(primitive, fallback)
    with core.set_current_trace(self.parent_trace):
      primal_out, tangent_nzs_out, residuals, linearized = lin(
          tangent_nzs, *primals_in, **params)
    with core.set_current_trace(self.tangent_trace):
      tangent_out = linearized(residuals, *tangents_in)
    if primitive.multiple_results:
      return [maybe_linearize_tracer(self, x, nz, t)
              for x, nz, t in zip(primal_out, tangent_nzs_out, tangent_out)]
    else:
      return maybe_linearize_tracer(self, primal_out, tangent_nzs_out, tangent_out)

  def process_custom_jvp_call(self, prim, fun, f_jvp, tracers, *, symbolic_zeros):
    primals_in, tangents_in = unzip2(map(self.to_primal_tangent_pair, tracers))
    if all(type(t) is Zero for t in tangents_in):
      return prim.bind_with_trace(self.parent_trace, (fun, f_jvp, *primals_in),
                                  dict(symbolic_zeros=symbolic_zeros))

    def _f_jvp(primals, tangents):
      outs = f_jvp.call_wrapped(*primals, *tangents)
      primals_out, tangents_out = split_list(outs, [len(outs) // 2])
      return primals_out, tangents_out

    with core.set_current_trace(self.parent_trace):
      instantiate_zeros = not symbolic_zeros
      nonzeros_in = [type(t) is not Zero for t in tangents_in]
      primals_out, tangent_nzs_out, residuals, linearized = linearize_from_jvp(
          _f_jvp, True, nonzeros_in, symbolic_zeros, instantiate_zeros, primals_in, {})

    with core.set_current_trace(self.tangent_trace):
      tangents_out = linearized(residuals, *tangents_in)
    tangents_out = map(replace_rule_output_symbolic_zeros, tangents_out)
    return [maybe_linearize_tracer(self, x, nz, t)
            for x, nz, t in zip(primals_out, tangent_nzs_out, tangents_out)]

  def process_custom_vjp_call(self, prim, fun, fwd, bwd, tracers, out_trees,
                              symbolic_zeros):
    primals_in, tangents_in = unzip2(map(self.to_primal_tangent_pair, tracers))
    if all(type(t) is Zero for t in tangents_in):
      return prim.bind_with_trace(self.parent_trace,
                                  (fun, fwd, bwd, *primals_in),
                                  dict(out_trees=out_trees, symbolic_zeros=symbolic_zeros))
    fwd_in = [(p, type(t) is not Zero) for p, t in zip(primals_in, tangents_in)]
    fwd_in = [x for pair in fwd_in for x in pair]   # flatten
    with core.set_current_trace(self.parent_trace):
      res_and_primals_out = fwd.call_wrapped(*fwd_in)

    _, res_tree = out_trees()
    res, primals_out = split_list(res_and_primals_out, [res_tree.num_leaves])
    avals_out = [core.get_aval(x).to_tangent_aval() for x in primals_out]

    tangents_in = map(instantiate_zeros, tangents_in)
    with core.set_current_trace(self.tangent_trace):
      tangents_out = custom_lin_p.bind(
        *res, *tangents_in, num_res=res_tree.num_leaves, bwd=bwd,
        out_avals=avals_out, symbolic_zeros=symbolic_zeros)
    tangent_nzs_out = [type(t) is not Zero for t in tangents_out]
    return map(partial(maybe_linearize_tracer, self), primals_out, tangent_nzs_out, tangents_out)

  def process_call(self, call_primitive, f, tracers, params):
    assert call_primitive.multiple_results
    primals, tangents = unzip2(map(self.to_primal_tangent_pair, tracers))
    nzs_in = tuple(type(t) is not Zero for t in tangents)
    f_primal, linearize_outs_thunk = linearize_subtrace(f, self.tag, nzs_in)
    if isinstance(call_primitive, core.MapPrimitive):
      @as_hashable_function(closure=(linearize_outs_thunk))
      def new_out_axes_thunk():
        residual_avals, _, _ = linearize_outs_thunk()
        out_axes = params['out_axes_thunk']()
        return (*(0 for _ in residual_avals), *out_axes)
      primal_params = dict(params, out_axes_thunk=new_out_axes_thunk)
    else:
      primal_params = params

    all_primal_results = call_primitive.bind_with_trace(self.parent_trace, (f_primal, *primals), primal_params)
    residual_avals, nzs_out, lin_jaxpr = linearize_outs_thunk()
    num_residuals = len(residual_avals)
    residuals = all_primal_results[:num_residuals]
    primals_out = all_primal_results[num_residuals:]

    if isinstance(call_primitive, core.MapPrimitive):
      in_axes = params['in_axes']
      out_axes = params['out_axes_thunk']()
      residual_avals = map(get_aval, residuals)
      new_in_axes = (*(0 for _ in residual_avals),
                     *(ax for ax, nz in zip(in_axes, nzs_in) if nz))
      new_out_axes = (*(ax for ax, nz in zip(out_axes, nzs_out) if nz),)
      # NOTE: This assumes that the output tangents being zero is a
      # deterministic function of which input tangents were zero.
      @as_hashable_function(closure=(new_out_axes))
      def new_out_axes_thunk():
        return new_out_axes
      params = dict(params,
                    in_axes=new_in_axes,
                    out_axes_thunk=new_out_axes_thunk)

    update_params = call_linearize_param_updaters.get(call_primitive)
    new_params = update_params(params, residual_avals, nzs_in) if update_params else params

    def f_tangent(*args):
      residuals = args[:num_residuals]
      nz_tangents = args[num_residuals:]
      return core.eval_jaxpr(lin_jaxpr, residuals, *nz_tangents)

    nz_tangents_in = [t for (t, nz) in zip(tangents, nzs_in) if nz]
    nz_tangents_out = call_primitive.bind_with_trace(
        self.tangent_trace, (lu.wrap_init(f_tangent), *residuals, *nz_tangents_in), new_params)
    nz_tangents_out_iter = iter(nz_tangents_out)
    tangents_out = [next(nz_tangents_out_iter) if nz else Zero.from_primal_value(primal)
                    for nz, primal in zip(nzs_out, primals_out)]
    return map(partial(maybe_linearize_tracer, self), primals_out, nzs_out, tangents_out)

  # The only difference between process_map and process_call is that
  # the `in_axes` and `out_axes_thunk` params must be updated;
  # that's handled in process_call.
  process_map = process_call

def maybe_linearize_tracer(trace, primal, is_nonzero, tangent):
  if is_nonzero:
    assert not type(tangent) is Zero
    return LinearizeTracer(trace, primal, tangent)
  else:
    assert type(tangent) is Zero
    return primal

def fallback_linearize_rule(_prim, _nonzeros, *primals, **params):
  jvp = primitive_jvps.get(_prim)
  if not jvp:
    msg = f"Differentiation rule for '{_prim}' not implemented"
    raise NotImplementedError(msg)
  return linearize_from_jvp(jvp, _prim.multiple_results, _nonzeros, False, False, primals, params)

def linearize_from_jvp(jvp, multiple_results, nonzeros,
                       user_facing_symbolic_zeros, instantiate_input_zeros, primals, params):
  current_name_stack = source_info_util.current_name_stack()
  with core.take_current_trace() as parent_trace:
    trace = pe.JaxprTrace(parent_trace, current_name_stack, core.TraceTag())
    tangent_avals = [get_aval(p).to_tangent_aval() for p in primals]

    def make_zero(aval):
      if instantiate_input_zeros:
        return zeros_like_aval(aval)
      elif user_facing_symbolic_zeros:
        return SymbolicZero(aval)
      else:
        return Zero(aval)

    if user_facing_symbolic_zeros:
      zero_type = SymbolicZero
    else:
      zero_type = Zero

    tangent_args = tuple(trace.new_arg(pe.PartialVal.unknown(aval)) if nz else make_zero(aval)
                         for aval, nz in zip(tangent_avals, nonzeros))
    with core.set_current_trace(trace):
      out_primals, out_tangents = jvp(primals, tangent_args, **params)

    if not multiple_results:
      out_primals = [out_primals]
      out_tangents = [out_tangents]

    out_primals = [trace.to_jaxpr_tracer(p).pval.get_known() for p in out_primals]

    out_nzs = [type(t) is not zero_type and not trace.to_jaxpr_tracer(t).is_known()
               for t in out_tangents]
    out_tangent_avals = [get_aval(p).to_tangent_aval() for p in out_primals]
    out_nz_tracers = [trace.to_jaxpr_tracer(r)
                      for (r, nz) in zip(out_tangents, out_nzs) if nz]
    in_tracers = [t for t, nz in zip(tangent_args, nonzeros) if nz]
    jaxpr, out_consts, _ = pe.tracers_to_jaxpr(in_tracers, out_nz_tracers)

    def linearized(residuals, *tangents):
      nz_tangents_in = [t for (t, nz) in zip(tangents, nonzeros) if nz]
      nz_tangents_out = core.eval_jaxpr(jaxpr, residuals, *nz_tangents_in)
      nz_tangents_out_iter = iter(nz_tangents_out)
      all_out_tangents = [next(nz_tangents_out_iter) if nz else Zero(aval)
                          for (aval, nz) in zip(out_tangent_avals, out_nzs)]
      if multiple_results:
        return all_out_tangents
      else:
        out_tangent, = all_out_tangents
        return out_tangent

  if multiple_results:
    return out_primals, out_nzs, out_consts, linearized
  else:
    out_primal, = out_primals
    out_nz, = out_nzs
    return out_primal, out_nz, out_consts, linearized

class LinearizeTracer(Tracer):
  __slots__ = ['primal', 'tangent']

  def __init__(self, trace, primal, tangent):
    if config.enable_checks.value:
      _primal_tangent_shapes_match(primal, tangent)
    self._trace = trace
    self.primal = primal
    self.tangent = tangent

  @property
  def aval(self):
    return get_aval(self.primal)

  def full_lower(self):
    if type(self.tangent) is Zero:
      return core.full_lower(self.primal)
    else:
      return self

  def to_concrete_value(self):
    return core.to_concrete_value(self.primal)


# -------------------- Primitives --------------------

primitive_jvps : dict[core.Primitive, Callable] = {}
primitive_transposes: dict[core.Primitive, Callable] = {}
primitive_linearizations : dict[core.Primitive, Callable]  = {}

def deflinear(primitive, transpose_rule):
  primitive_jvps[primitive] = partial(linear_jvp, primitive)
  primitive_transposes[primitive] = partial(linear_transpose, transpose_rule)

def linear_jvp(primitive, primals, tangents, **params):
  val_out = primitive.bind(*primals, **params)
  if all(type(tangent) is Zero for tangent in tangents):
    if primitive.multiple_results:
      return val_out, map(Zero.from_primal_value, val_out)
    return val_out, Zero.from_primal_value(val_out)
  else:
    tangents = map(instantiate_zeros, tangents)
    return val_out, primitive.bind(*tangents, **params)

def linear_transpose(transpose_rule, cotangent, *args, **kwargs):
  return Zero if type(cotangent) is Zero else transpose_rule(cotangent, **kwargs)


def deflinear2(primitive, transpose_rule):
  primitive_jvps[primitive] = partial(linear_jvp, primitive)
  primitive_transposes[primitive] = partial(linear_transpose2, transpose_rule)

def linear_transpose2(transpose_rule, cotangent, *args, **kwargs):
  return Zero if type(cotangent) is Zero else transpose_rule(cotangent, *args, **kwargs)


def defjvp(primitive, *jvprules):
  assert isinstance(primitive, Primitive)
  assert not primitive.multiple_results
  primitive_jvps[primitive] = partial(standard_jvp, jvprules, primitive)


def standard_jvp(jvprules, primitive, primals, tangents, **params):
  val_out = primitive.bind(*primals, **params)
  tangents_out = [rule(t, *primals, **params) for rule, t in zip(jvprules, tangents)
                  if rule is not None and type(t) is not Zero]
  return val_out, functools.reduce(add_tangents, tangents_out, Zero.from_primal_value(val_out))

def defjvp2(primitive, *jvprules):
  assert isinstance(primitive, Primitive)
  assert not primitive.multiple_results
  primitive_jvps[primitive] = partial(standard_jvp2, jvprules, primitive)

def standard_jvp2(jvprules, primitive, primals, tangents, **params):
  val_out = primitive.bind(*primals, **params)
  tangents_out = (rule(t, val_out, *primals, **params) for rule, t in zip(jvprules, tangents)
                  if rule is not None and type(t) is not Zero)
  tangents_out = list(tangents_out)
  return val_out, functools.reduce(add_tangents, tangents_out, Zero.from_primal_value(val_out))

def add_tangents(x, y):
  if type(x) is Zero:
    return y
  elif type(y) is Zero:
    return x
  else:
    return add_jaxvals(x, y)

def defbilinear(prim, lhs_rule, rhs_rule):
  assert isinstance(prim, Primitive)
  lhs_jvp = lambda g, x, y, **kwargs: prim.bind(g, y, **kwargs)
  rhs_jvp = lambda g, x, y, **kwargs: prim.bind(x, g, **kwargs)
  defjvp(prim, lhs_jvp, rhs_jvp)
  primitive_transposes[prim] = partial(bilinear_transpose, lhs_rule, rhs_rule)

def bilinear_transpose(lhs_rule, rhs_rule, cotangent, x, y, **kwargs):
  assert is_undefined_primal(x) ^ is_undefined_primal(y)
  if type(cotangent) is Zero:
    return Zero
  if is_undefined_primal(x):
    out = lhs_rule(cotangent, x, y, **kwargs)
    return Zero if out is Zero else (out, None)
  else:
    out = rhs_rule(cotangent, x, y, **kwargs)
    return Zero if out is Zero else (None, out)


def defjvp_zero(primitive):
  assert isinstance(primitive, Primitive)
  primitive_jvps[primitive] = partial(zero_jvp, primitive)

def zero_jvp(primitive, primals, tangents, **params):
  r = primitive.bind(*primals, **params)
  return r, Zero.from_primal_value(r)

deflinear2(add_jaxvals_p, lambda t, *args: (t, t))


def instantiate_zeros(tangent):
  return zeros_like_aval(tangent.aval) if type(tangent) is Zero else tangent

@lu.transformation_with_aux2
def traceable(f, store, in_tree, *primals_and_tangents):
  primals, tangents = tree_unflatten(in_tree, primals_and_tangents)
  tangents = [Zero.from_primal_value(p) if t is None else t
              for p, t in zip(primals, tangents)]
  primals_out, tangents_out = f(primals, tangents)
  tangents_out = [None if type(t) is Zero else t for t in tangents_out]
  out_flat, out_tree = tree_flatten((primals_out, tangents_out))
  store.store(out_tree)
  return out_flat


def call_transpose(primitive, params, call_jaxpr, args, ct, _):
  if isinstance(call_jaxpr, core.ClosedJaxpr):
    call_jaxpr, consts = call_jaxpr.jaxpr, call_jaxpr.consts
  else:
    consts = ()
  all_args, in_tree_def = tree_flatten((consts, args, ct))
  fun = lu.hashable_partial(lu.wrap_init(backward_pass), call_jaxpr, False)
  fun, out_tree = flatten_fun_nokwargs(fun, in_tree_def)
  update_params = call_transpose_param_updaters.get(primitive)
  if update_params:
    params = update_params(params, map(is_undefined_primal, args),
                           [type(x) is not Zero for x in ct])
  if config.dynamic_shapes.value:
    # TODO(mattjj,dougalm): handle consts, for now assume just args
    which_lin = [is_undefined_primal(x) for x in args]
    res_invars, _ = partition_list(which_lin, call_jaxpr.invars)
    new_invars = [*res_invars, *call_jaxpr.outvars]
    dbidx_map = {v: core.DBIdx(i) for i, v in enumerate(new_invars)}
    in_type = [(v.aval.update(shape=tuple(dbidx_map.get(d, d) for d in v.aval.shape))
                if type(v.aval) is core.DShapedArray else v.aval, True) for v in new_invars]
    fun = lu.annotate(fun, tuple(in_type))
  out_flat = primitive.bind(fun, *all_args, **params)
  return tree_unflatten(out_tree(), out_flat)
primitive_transposes[core.call_p] = partial(call_transpose, call_p)


def _closed_call_transpose(params, jaxpr, args, ct, cts_in_avals):
  jaxpr_, consts = jaxpr.jaxpr, jaxpr.consts
  jaxpr_ = pe.convert_constvars_jaxpr(jaxpr_)
  return call_transpose(core.closed_call_p, params, jaxpr_, (*consts, *args),
                        ct, cts_in_avals)
primitive_transposes[core.closed_call_p] = _closed_call_transpose


@lu.transformation_with_aux2
def nonzero_outputs(f, store, *args, **kwargs):
  results = f(*args, **kwargs)
  store.store([type(r) is not Zero for r in results])
  return results

def map_transpose(primitive, params, call_jaxpr, args, ct, _):
  all_args, in_tree_def = tree_flatten(((), args, ct))  # empty consts
  fun = lu.hashable_partial(lu.wrap_init(backward_pass), call_jaxpr, False)
  fun, nz_arg_cts = nonzero_outputs(fun)
  fun, out_tree = flatten_fun_nokwargs(fun, in_tree_def)
  # Preserve axis for primal arguments, skip tangents (represented as undefined primals).
  in_axes, out_axes = params['in_axes'], params['out_axes']
  new_in_axes = (*[axis for axis, x in zip(in_axes, args)
                   if not is_undefined_primal(x)],
                 *[axis for axis, x in zip(out_axes, ct)
                   if type(x) is not Zero])
  if any(out_axis is None for out_axis in out_axes):
    raise NotImplementedError(
        "autodiff of pmap functions with out_axes=None is not supported. "
        "Consider using shard_map instead.")
  assert all(out_axis is not None for out_axis in out_axes), out_axes
  # NOTE: This assumes that the output cotangents being zero is a deterministic
  #       function of which input cotangents were zero.
  @as_hashable_function(closure=(in_axes, tuple(type(c) is Zero for c in ct)))
  def out_axes_thunk():
    return tuple(axis or 0 for axis, nz in zip(in_axes, nz_arg_cts()) if nz)
  new_params = dict(params, name=wrap_name(params['name'], 'transpose'),
                    in_axes=new_in_axes, out_axes_thunk=out_axes_thunk)
  del new_params['out_axes']
  update_params = call_transpose_param_updaters.get(primitive)
  if update_params:
    new_params = update_params(new_params, map(is_undefined_primal, args),
                               [type(x) is not Zero for x in ct])
  out_flat = primitive.bind(fun, *all_args, **new_params)
  arg_cts = tree_unflatten(out_tree(), out_flat)

  # The freevars are being fanned out (not mapped). During transpose the
  # dual of fan-out is fan-in-sum. We apply it to the unmapped invars.
  assert len(in_axes) == len(arg_cts)
  def unmap_zero(zero, in_axis):
    return (zero if in_axis is None else
            Zero(core.unmapped_aval(params['axis_size'], params['axis_name'], in_axis, zero.aval)))
  arg_cts = (unmap_zero(arg_ct, in_axis) if type(arg_ct) is Zero else
             arg_ct if in_axis is not None else
             arg_ct.sum(0)
             for arg_ct, in_axis in zip(arg_cts, in_axes))
  return tuple(arg_cts)


def jvp_jaxpr(jaxpr: core.ClosedJaxpr, nonzeros: Sequence[bool],
              instantiate: bool | Sequence[bool]
              ) -> tuple[core.ClosedJaxpr, list[bool]]:
  if type(instantiate) is bool:
    instantiate = (instantiate,) * len(jaxpr.out_avals)
  return _jvp_jaxpr(jaxpr, tuple(nonzeros), tuple(instantiate))

@weakref_lru_cache
def _jvp_jaxpr(jaxpr: core.ClosedJaxpr,
               nonzeros: Sequence[bool], instantiate: Sequence[bool]):
  assert len(jaxpr.in_avals) == len(nonzeros)
  f = lu.wrap_init(core.jaxpr_as_fun(jaxpr))
  f_jvp, out_nonzeros = f_jvp_traceable(jvp(f, instantiate=instantiate, transform_stack=False),
                                        nonzeros)
  tangent_avals = [aval.to_tangent_aval() for aval, nz in zip(jaxpr.in_avals, nonzeros) if nz]
  avals_in = list(it.chain(jaxpr.in_avals, tangent_avals))
  jaxpr_out, avals_out, literals_out, () = pe.trace_to_jaxpr_dynamic(f_jvp, avals_in)
  return core.ClosedJaxpr(jaxpr_out, literals_out), out_nonzeros()

@lu.transformation_with_aux2
def f_jvp_traceable(f, store, nonzeros, *primals_and_nztangents):
  num_primals = len(nonzeros)
  primals = list(primals_and_nztangents[:num_primals])
  nonzero_tangents = iter(primals_and_nztangents[num_primals:])
  tangents = [next(nonzero_tangents) if nz else Zero.from_primal_value(p)
              for p, nz in zip(primals, nonzeros)]
  primals_out, tangents_out = f(primals, tangents)
  out_nonzeros = [type(t) is not Zero for t in tangents_out]
  nonzero_tangents_out = [t for t in tangents_out if type(t) is not Zero]
  store.store(out_nonzeros)
  return list(primals_out) + nonzero_tangents_out

def rearrange_binders(jaxpr: core.ClosedJaxpr, primals_in, tangents_in, primals_out, tangents_out):
  new_invars = _perm(primals_in, tangents_in, jaxpr.jaxpr.invars)
  new_outvars = _perm(primals_out, tangents_out, jaxpr.jaxpr.outvars)
  new_jaxpr = core.Jaxpr(jaxpr.jaxpr.constvars,
                         new_invars, new_outvars, jaxpr.jaxpr.eqns,
                         jaxpr.jaxpr.effects)
  return core.ClosedJaxpr(new_jaxpr, jaxpr.consts)

def _perm(primal_counts, tangent_counts, lst):
  n = sum(primal_counts)
  primals, tangents = lst[:n], lst[n:]
  primal_groups = split_list(primals, primal_counts[:-1])
  tangent_groups = split_list(tangents, tangent_counts[:-1])
  return _interleave(primal_groups, tangent_groups)

def _interleave(xs, ys):
  assert len(xs) == len(ys)
  return [e for pair in zip(xs, ys) for l in pair for e in l]


custom_lin_p: core.Primitive = core.Primitive('custom_lin')
custom_lin_p.def_abstract_eval(lambda *_, out_avals, **__: out_avals)
custom_lin_p.multiple_results = True

def raise_custom_vjp_error_on_jvp(*_, **__):
  raise TypeError("can't apply forward-mode autodiff (jvp) to a custom_vjp "
                  "function.")
custom_lin_p.def_impl(raise_custom_vjp_error_on_jvp)

def _custom_lin_transpose(cts_out, *invals, num_res, bwd, out_avals,
                          symbolic_zeros):
  res, _ = split_list(invals, [num_res])
  if symbolic_zeros:
    cts_out = map(replace_internal_symbolic_zeros, cts_out)
  else:
    cts_out = map(instantiate_zeros, cts_out)
  cts_in = bwd(*res, *cts_out)
  cts_in = map(replace_rule_output_symbolic_zeros, cts_in)
  return [None] * num_res + list(cts_in)
primitive_transposes[custom_lin_p] = _custom_lin_transpose


class CustomJVPException(Exception):
  def __init__(self):
    # TODO(mattjj): track source provenance on AD tracers, improve error
    msg = ("Detected differentiation of a custom_jvp function with respect to "
           "a closed-over value. That isn't supported because the custom JVP "
           "rule only specifies how to differentiate the custom_jvp function "
           "with respect to explicit input parameters. Try passing the "
           "closed-over value into the custom_jvp function as an argument, and "
           "adapting the custom_jvp rule.")
    super().__init__(msg)

class CustomVJPException(Exception):
  def __init__(self):
    # TODO(mattjj): track source provenance on AD tracers, improve error
    msg = ("Detected differentiation of a custom_vjp function with respect to "
           "a closed-over value. That isn't supported because the custom VJP "
           "rule only specifies how to differentiate the custom_vjp function "
           "with respect to explicit input parameters. Try passing the "
           "closed-over value into the custom_vjp function as an argument, and "
           "adapting the custom_vjp fwd and bwd rules.")
    super().__init__(msg)

# TODO(mattjj): remove this vestigial dict
reducing_transposes: dict[core.Primitive, Callable] = {}
