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
"""Module for conditional control flow primitives."""
from __future__ import annotations

import collections
from collections.abc import Callable, Sequence
import functools
from functools import partial
import inspect
import itertools
import operator
from typing import Any, TypeVar

from jax.tree_util import tree_flatten, tree_unflatten
from jax._src import ad_util
from jax._src.api_util import (
    _check_no_aliased_ref_args, _check_no_aliased_closed_over_refs)
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import effects
from jax._src import linear_util as lu
from jax._src import source_info_util
from jax._src import util
from jax._src.state.discharge import register_partial_discharge_rule, discharge_state
from jax._src.state.types import AbstractRef, RefEffect
from jax._src.core import replace_jaxpr_effects
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import xla
from jax._src.lax import lax
from jax._src.traceback_util import api_boundary
from jax._src.util import (safe_map, split_list, partition_list)
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
import numpy as np

from jax._src.lax.control_flow.common import (
    _avals_short,
    _check_tree_and_avals,
    _initial_style_jaxprs_with_common_consts,
    _make_closed_jaxpr,
    _prune_zeros,
    _typecheck_param,
    )

map, unsafe_map = safe_map, map


# For backward compatibility with a previous switch/cond calling convention,
# we allow a single (pytree) `operand` argument to be passed by keyword. We use
# a sentinel object as its default value to indicate when it is _not_ passed.
_no_operand_sentinel = object()

@api_boundary
def switch(index, branches: Sequence[Callable], *operands,
           operand=_no_operand_sentinel):
  """Apply exactly one of the ``branches`` given by ``index``.

  If ``index`` is out of bounds, it is clamped to within bounds.

  Has the semantics of the following Python::

    def switch(index, branches, *operands):
      index = clamp(0, index, len(branches) - 1)
      return branches[index](*operands)

  Internally this wraps XLA's `Conditional
  <https://www.tensorflow.org/xla/operation_semantics#conditional>`_
  operator. However, when transformed with :func:`~jax.vmap` to operate over a
  batch of predicates, ``cond`` is converted to :func:`~jax.lax.select`.

  Args:
    index: Integer scalar type, indicating which branch function to apply.
    branches: Sequence of functions (A -> B) to be applied based on ``index``.
      All branches must return the same output structure.
    operands: Operands (A) input to whichever branch is applied.

  Returns:
    Value (B) of ``branch(*operands)`` for the branch that was selected based
    on ``index``.
  """
  if not all(callable(branch) for branch in branches):
    raise TypeError("lax.switch: branches argument should be a sequence of callables.")
  if operand is not _no_operand_sentinel:
    if operands:
      raise TypeError("if 'operand' keyword is passed then no positional "
                      f"operands can be passed, got {operand=} "
                      f"and positional operands {operands}")
    operands = (operand,)
  del operand

  if len(np.shape(index)) != 0:
    raise TypeError(
        f"Branch index must be scalar, "
        f"got {index} of shape {np.shape(index)}.")

  try:
    index_dtype = dtypes.result_type(index)
  except TypeError as err:
    msg = f"Index type must be an integer, got {index}."
    raise TypeError(msg) from err

  if index_dtype.kind not in 'iu':
    raise TypeError(
        f"Index type must be an integer, got {index} as {index_dtype}")

  branches = tuple(branches)

  if len(branches) == 0:
    raise ValueError("Empty branch sequence")
  elif len(branches) == 1:
    return branches[0](*operands)

  index = lax.convert_element_type(index, np.int32)
  lo = np.array(0, np.int32)
  hi = np.array(len(branches) - 1, np.int32)
  index = lax.clamp(lo, index, hi)

  if (config.disable_jit.value and core.is_concrete(index)):
    return branches[int(index)](*operands)

  ops, ops_tree = tree_flatten(operands)
  ops_avals = tuple(map(core.get_aval, ops))

  if config.mutable_array_checks.value:
    dbg = pe.tracing_debug_info(branches[0], ops_tree, None, False, 'switch')
    _check_no_aliased_ref_args(dbg, ops_avals, ops)

  jaxprs, consts, out_trees = _initial_style_jaxprs_with_common_consts(
      branches, ops_tree, ops_avals, primitive_name='switch')
  if config.mutable_array_checks.value:
    _check_no_aliased_closed_over_refs(dbg, (*jaxprs[0].consts, *consts), ops)
  for i, (out_tree, jaxpr) in enumerate(zip(out_trees[1:], jaxprs[1:])):
    _check_tree_and_avals("branch 0 output",
                          out_trees[0], jaxprs[0].out_avals,
                          f"branch {i + 1} output",
                          out_tree, jaxpr.out_avals)
  joined_effects = core.join_effects(*(jaxpr.effects for jaxpr in jaxprs))
  disallowed_effects = effects.control_flow_allowed_effects.filter_not_in(joined_effects)
  if disallowed_effects:
    raise NotImplementedError(
        f'Effects not supported in `switch`: {disallowed_effects}')
  out = cond_p.bind(index, *consts, *ops, branches=tuple(jaxprs))
  return tree_unflatten(out_trees[0], out)


def _cond(pred, true_fun: Callable, false_fun: Callable, *operands,
          operand=_no_operand_sentinel):
  """Conditionally apply ``true_fun`` or ``false_fun``.

  Wraps XLA's `Conditional
  <https://www.tensorflow.org/xla/operation_semantics#conditional>`_
  operator.

  Provided arguments are correctly typed, ``cond()`` has equivalent
  semantics to this Python implementation, where ``pred`` must be a
  scalar type::

    def cond(pred, true_fun, false_fun, *operands):
      if pred:
        return true_fun(*operands)
      else:
        return false_fun(*operands)


  In contrast with :func:`jax.lax.select`, using ``cond`` indicates that only one of
  the two branches is executed (up to compiler rewrites and optimizations).
  However, when transformed with :func:`~jax.vmap` to operate over a batch of
  predicates, ``cond`` is converted to :func:`~jax.lax.select`.

  Args:
    pred: Boolean scalar type, indicating which branch function to apply.
    true_fun: Function (A -> B), to be applied if ``pred`` is True.
    false_fun: Function (A -> B), to be applied if ``pred`` is False.
    operands: Operands (A) input to either branch depending on ``pred``. The
      type can be a scalar, array, or any pytree (nested Python tuple/list/dict)
      thereof.

  Returns:
    Value (B) of either ``true_fun(*operands)`` or ``false_fun(*operands)``,
    depending on the value of ``pred``. The type can be a scalar, array, or any
    pytree (nested Python tuple/list/dict) thereof.
  """
  if not (callable(true_fun) and callable(false_fun)):
    raise TypeError("lax.cond: true_fun and false_fun arguments should be callable.")
  if operand is not _no_operand_sentinel:
    if operands:
      raise TypeError("if 'operand' keyword is passed then no positional "
                      f"operands can be passed, got {operand=} "
                      f"and positional operands {operands}")
    operands = (operand,)
  del operand

  if pred is None:
    raise TypeError("cond predicate is None")
  if isinstance(pred, Sequence) or np.ndim(pred) != 0:
    raise TypeError(
        f"Pred must be a scalar, got {pred} of " +
        (f"type {type(pred)}" if isinstance(pred, Sequence)
         else f"shape {np.shape(pred)}."))

  try:
    pred_dtype = dtypes.result_type(pred)
  except TypeError as err:
    msg = ("Pred type must be either boolean or number, got {}.")
    raise TypeError(msg.format(pred)) from err

  if pred_dtype.kind != 'b':
    if pred_dtype.kind in 'iuf':
      pred = pred != 0
    else:
      msg = ("Pred type must be either boolean or number, got {}.")
      raise TypeError(msg.format(pred_dtype))

  if config.disable_jit.value and core.is_concrete(pred):
    if pred:
      return true_fun(*operands)
    else:
      return false_fun(*operands)

  ops, ops_tree = tree_flatten(operands)
  ops_avals = tuple(map(core.get_aval, ops))

  if config.mutable_array_checks.value:
    dbg = pe.tracing_debug_info(true_fun, ops_tree, None, False, 'cond')
    _check_no_aliased_ref_args(dbg, ops_avals, ops)
  jaxprs, consts, out_trees = _initial_style_jaxprs_with_common_consts(
      (true_fun, false_fun), ops_tree, ops_avals, 'cond')
  true_jaxpr, false_jaxpr = jaxprs
  if config.mutable_array_checks.value:
    _check_no_aliased_closed_over_refs(dbg, (*true_jaxpr.consts, *consts), ops)

  out_tree, false_out_tree = out_trees
  if any(isinstance(out_aval, AbstractRef) for out_aval in
         true_jaxpr.out_avals + false_jaxpr.out_avals):
    raise ValueError("Cannot return `Ref`s from `cond`.")

  _check_tree_and_avals("true_fun output",
                        out_tree, true_jaxpr.out_avals,
                        "false_fun output",
                        false_out_tree, false_jaxpr.out_avals)
  # prune passhtrough outputs
  true_fwds = pe._jaxpr_forwarding(true_jaxpr.jaxpr)
  false_fwds = pe._jaxpr_forwarding(false_jaxpr.jaxpr)
  in_fwd = [i if i == j else None for i, j in zip(true_fwds, false_fwds)]
  keep = [f is None for f in in_fwd]
  true_jaxpr = pe.prune_closed_jaxpr_outputs(true_jaxpr, keep)
  false_jaxpr = pe.prune_closed_jaxpr_outputs(false_jaxpr, keep)

  joined_effects = core.join_effects(true_jaxpr.effects, false_jaxpr.effects)
  disallowed_effects = effects.control_flow_allowed_effects.filter_not_in(joined_effects)
  if disallowed_effects:
    raise NotImplementedError(
        f'Effects not supported in `cond`: {disallowed_effects}')

  index = lax.convert_element_type(pred, np.int32)
  false_jaxpr = replace_jaxpr_effects(false_jaxpr, joined_effects)
  true_jaxpr = replace_jaxpr_effects(true_jaxpr, joined_effects)

  out = cond_p.bind(index, *consts, *ops, branches=(false_jaxpr, true_jaxpr))
  num_consts = len(consts)
  out_ = iter(out)

  out = [
    next(out_) if fwd is None else lax.asarray(ops[fwd - num_consts])
    for fwd in in_fwd
  ]
  assert next(out_, None) is None
  return tree_unflatten(out_tree, out)

@api_boundary
@functools.wraps(_cond)
def cond(*args, **kwargs):
  # detect an attempt to call the former, deprecated cond
  try:
    ba = inspect.signature(_cond_with_per_branch_args).bind(*args, **kwargs)
  except TypeError:
    pass
  else:
    assert not ba.kwargs  # no catch-all **kwargs in _cond_with_per_branch
    _, true_operand, true_fun, false_operand, false_fun = ba.args
    if callable(true_operand) and callable(true_fun):
      # treat this as modern cond (with two operands)
      return _cond(*args, **kwargs)
    if callable(true_fun) and callable(false_fun):
      return _cond_with_per_branch_args(*ba.args)

  return _cond(*args, **kwargs)

def _cond_with_per_branch_args(pred,
                               true_operand, true_fun: Callable,
                               false_operand, false_fun: Callable):
  """Conditionally apply ``true_fun`` or ``false_fun``.

  Has equivalent semantics to this Python implementation::

    def cond(pred, true_operand, true_fun, false_operand, false_fun):
      if pred:
        return true_fun(true_operand)
      else:
        return false_fun(false_operand)

  Pred has to be a scalar type, collection types (list, tuple) are not supported
  """
  if not (callable(true_fun) and callable(false_fun)):
    raise TypeError("lax.cond: true_fun and false_fun arguments should be callable.")
  return _cond(pred,
               lambda op: true_fun(op[0]),
               lambda op: false_fun(op[1]),
               (true_operand, false_operand))

def _join_cond_effects(branches: Sequence[core.Jaxpr]) -> effects.Effects:
  joined_effects = set()
  for b in branches:
    for eff in b.effects:
      if isinstance(eff, effects.JaxprInputEffect):
        # Offset index to handle predicate
        eff = eff.replace(input_index=eff.input_index + 1)
      joined_effects.add(eff)
  return joined_effects

def _cond_abstract_eval(*avals, branches, **_):
  joined_effects = _join_cond_effects(branches)
  disallowed_effects = effects.control_flow_allowed_effects.filter_not_in(joined_effects)
  if disallowed_effects:
    raise NotImplementedError(
        f'Effects not supported in `cond`: {disallowed_effects}')
  return branches[0].out_avals, joined_effects

def _bcast_select(pred, on_true, on_false):
  if np.ndim(pred) != np.ndim(on_true):
    idx = list(range(np.ndim(pred)))
    pred = lax.broadcast_in_dim(pred, np.shape(on_true), idx)
  return lax.select(pred, on_true, on_false)

def _bcast_select_n(pred, *cases):
  if np.ndim(pred) != np.ndim(cases[0]):
    idx = list(range(np.ndim(pred)))
    pred = lax.broadcast_in_dim(pred, np.shape(cases[0]), idx)
  return lax.select_n(pred, *cases)

def _cond_batching_rule(axis_data, args, dims, branches):
  index, *ops = args
  index_dim, *op_dims = dims
  # TODO(sharadmv): clean this up by adding a specific blocklist
  if any(isinstance(eff, RefEffect) for branch in branches for eff in
      branch.jaxpr.effects):
    raise NotImplementedError(
        "State effect not supported in vmap-of-cond.")
  from jax._src.callback import _IOEffect, _OrderedIOEffect
  if any(eff in branch.effects for eff in [_IOEffect, _OrderedIOEffect]
      for branch in branches):
    raise NotImplementedError(
        "IO effect not supported in vmap-of-cond.")


  if index_dim is not batching.not_mapped:
    # Convert to a lax.select. While we could get away with not broadcasting
    # some operands yet, because all outputs must be broadcast together anyway
    # for the select we broadcast the input operands for simplicity and leave
    # optimizations to XLA.
    # TODO(mattjj,frostig): assumes branches are side-effect-free, revise!
    index, *ops = (
        batching.bdim_at_front(x, d, axis_data.size) for x, d in zip(args, dims))

    in_batched  = [True] * len(branches[0].in_avals)
    out_batched = [True] * len(branches[0].out_avals)

    branches_batched = [
        batching.batch_jaxpr(jaxpr, axis_data, in_batched, out_batched)[0]
        for jaxpr in branches]

    branch_outs = []
    for i, jaxpr in enumerate(branches_batched):
      # Perform a select on the inputs for safety of reverse-mode autodiff; see
      # https://github.com/jax-ml/jax/issues/1052
      predicate = lax.eq(index, lax._const(index, i))
      ops_ = [_bcast_select(predicate, x, lax.stop_gradient(x)) for x in ops]
      branch_outs.append(core.jaxpr_as_fun(jaxpr)(*ops_))
    out = [_bcast_select_n(index, *outs) for outs in zip(*branch_outs)]
    return out, [0 if b else None for b in out_batched]
  else:
    ops_bat = [d is not batching.not_mapped for d in op_dims]
    ops = [batching.moveaxis(x, d, 0) if b else x
           for b, x, d in zip(ops_bat, ops, op_dims)]

    branches_out_bat = [
        batching.batch_jaxpr(jaxpr, axis_data, ops_bat, False)[1]
        for jaxpr in branches]
    out_bat = [any(bat) for bat in zip(*branches_out_bat)]
    branches_batched = tuple(
        batching.batch_jaxpr(jaxpr, axis_data, ops_bat, out_bat)[0]
        for jaxpr in branches)

    out_dims = [0 if b else batching.not_mapped for b in out_bat]
    out = cond_p.bind(index, *ops, branches=branches_batched)
    return out, out_dims

def _cond_jvp(primals, tangents, branches):
  nonzeros = [type(t) is not ad_util.Zero for t in tangents]

  index_nz, *ops_nz = nonzeros
  assert index_nz is False

  branches_out_nz = [ad.jvp_jaxpr(jaxpr, ops_nz, instantiate=False)[1]
                     for jaxpr in branches]
  out_nz = [any(nz) for nz in zip(*branches_out_nz)]

  branches_jvp = tuple(ad.jvp_jaxpr(jaxpr, ops_nz, instantiate=out_nz)[0]
                       for jaxpr in branches)

  index, *ops = primals
  _, *ops_dot = tangents
  ops_dot = _prune_zeros(ops_dot)

  out = cond_p.bind(index, *ops, *ops_dot, branches=branches_jvp)
  out_primals, out_tangents = split_list(out, [len(out_nz)])
  out_tangents_iter = iter(out_tangents)
  out_tangents = [next(out_tangents_iter) if nz else ad_util.Zero.from_primal_value(p)
                  for p, nz in zip(out_primals, out_nz)]
  return out_primals, out_tangents

def _cond_partial_eval(trace, *tracers, branches):
  in_unknowns = [t.pval[0] is not None for t in tracers]
  index_uk, *ops_uk = in_unknowns
  if any(isinstance(eff, RefEffect) for branch in branches for eff in
      branch.jaxpr.effects):
    raise NotImplementedError(
        "State effect not supported in cond partial-eval.")

  if index_uk:
    # When the branch index is unknown, we stage out the whole cond.
    # TODO(mattjj): remove this path when old remat is removed
    params = dict(branches=branches)
    return trace.default_process_primitive(cond_p, tracers, params)

  branches_out_uks = []
  for branch_jaxpr in branches:
    _, _, out_uks, _ = pe.partial_eval_jaxpr_nounits(
        branch_jaxpr, ops_uk, instantiate=False)
    branches_out_uks.append(out_uks)
  out_uks = [any(uks) for uks in zip(*branches_out_uks)]

  branches_known, branches_unknown, branch_res_avals = [], [], []
  for branch_jaxpr in branches:
    branch_jaxpr_known, branch_jaxpr_unknown, _, res_avals = \
        pe.partial_eval_jaxpr_nounits(branch_jaxpr, ops_uk, instantiate=out_uks)
    branches_known.append(branch_jaxpr_known)
    branches_unknown.append(branch_jaxpr_unknown)
    branch_res_avals.append(res_avals)

  all_res_avals, res_avals_per_branch = _merge_branch_residuals(branch_res_avals)
  num_res = len(all_res_avals)

  num_known_outs = len(out_uks) - sum(out_uks)
  branches_known = _join_cond_outputs(
      branches_known, all_res_avals, res_avals_per_branch, num_known_outs)
  branches_unknown = _join_cond_pe_staged_jaxpr_inputs(
      branches_unknown, all_res_avals, res_avals_per_branch)
  assert all(all(map(core.typematch, j.out_avals, branches_known[0].out_avals))
             for j in branches_known[1:])

  in_consts = [t.pval.get_known() for t in tracers if t.pval.is_known()]
  out_consts_res = cond_p.bind(*in_consts, branches=branches_known)
  out_consts, res = split_list(out_consts_res, [len(out_consts_res) - num_res])

  index_tracer = trace.instantiate_const(tracers[0])
  ops_tracers = [trace.instantiate_const(t)
                 for uk, t in zip(in_unknowns[1:], tracers[1:]) if uk]
  res_tracers = map(trace.new_instantiated_const, res)
  out_tracers = [pe.JaxprTracer(trace, pe.PartialVal.unknown(aval), None)
                 for aval in branches_unknown[0].out_avals]
  params = dict(branches=branches_unknown)
  name_stack = source_info_util.current_name_stack()[len(trace.name_stack):]
  source = source_info_util.current().replace(name_stack=name_stack)
  eqn = pe.new_eqn_recipe(
      [index_tracer] + res_tracers + ops_tracers, out_tracers, cond_p, params,
      core.join_effects(*(j.effects for j in branches_unknown)), source)
  for t in out_tracers: t.recipe = eqn
  return util.merge_lists(out_uks, out_consts, out_tracers)

# TODO(mattjj): de-duplicate with _cond_partial_eval
def _cond_partial_eval_custom(saveable, unks_in, inst_in, eqn):
  index_uk, *ops_uk = unks_in
  branches = eqn.params['branches']

  # Instantiate all inputs (b/c jaxpr_staged will take all inputs).
  new_inst = [x for x, inst in zip(eqn.invars, inst_in)
              if type(x) is core.Var and not inst]
  del inst_in

  # NOTE(mattjj): I think it should be impossible for the index to be unknown,
  # but asserting that caused a test failure in diffrax. So we handle it: if it
  # is unknown, stage out the whole cond.
  if index_uk:
    all_true = [True] * len(branches[0].out_avals)
    return None, eqn, all_true, all_true, new_inst

  # First, compute output unknowns (unks_out), where an output of the cond is
  # unknown if it would be unknown on any of the branches.
  unks_out: list[bool] = [False] * len(eqn.outvars)
  for jaxpr in branches:
    _, _, unks_out_, _, _ = pe.partial_eval_jaxpr_custom(
        jaxpr.jaxpr, in_unknowns=ops_uk, in_inst=True,
        ensure_out_unknowns=False, ensure_out_inst=True, saveable=saveable)
    unks_out = map(operator.or_, unks_out, unks_out_)

  # Next, use the computed output unknowns to build a known jaxpr and a staged
  # jaxpr for each branch.
  branches_known_ : list[core.ClosedJaxpr] = []
  branches_staged_: list[core.ClosedJaxpr] = []
  branch_res_avals: list[list[core.AbstractValue]] = []
  for jaxpr in branches:
    jaxpr_known, jaxpr_staged, _, inst_out, num_res = \
        pe.partial_eval_jaxpr_custom(
            jaxpr.jaxpr, in_unknowns=ops_uk, in_inst=True,
            ensure_out_unknowns=unks_out, ensure_out_inst=True,
            saveable=saveable)
    branches_known_.append( core.ClosedJaxpr(jaxpr_known,  jaxpr.consts))
    branches_staged_.append(core.ClosedJaxpr(jaxpr_staged, jaxpr.consts))
    branch_res_avals.append(branches_staged_[-1].in_avals[:num_res])

  # Residuals may differ across branches, so we merge them, then use the merged
  # residuals to join the outputs of all branches to the same type.
  all_res_avals, res_avals_per_branch = _merge_branch_residuals(branch_res_avals)
  num_res = len(all_res_avals)
  num_known_outs = len(unks_out) - sum(unks_out)
  branches_known = _join_cond_outputs(
      branches_known_, all_res_avals, res_avals_per_branch, num_known_outs)
  branches_staged = _join_cond_pe_staged_jaxpr_inputs(
      branches_staged_, all_res_avals, res_avals_per_branch)
  assert all(all(map(core.typematch, j.out_avals, branches_known[0].out_avals))
             for j in branches_known[1:])

  # Create residual variables.
  newvar = core.gensym()
  res_binders = map(newvar, all_res_avals)

  # Build the known eqn.
  ins_known, _ = partition_list(unks_in, eqn.invars)  # includes index invar
  out_binders_known, _ = partition_list(unks_out, eqn.outvars)
  params_known = dict(branches=branches_known)
  effects_known = _join_cond_effects(branches_known)
  eqn_known = pe.new_jaxpr_eqn(
      ins_known, [*out_binders_known, *res_binders], cond_p, params_known,
      effects_known, eqn.source_info)

  # Build the staged eqn.
  _, out_binders_staged = partition_list(inst_out, eqn.outvars)
  params_staged = dict(branches=branches_staged)
  effects_staged = _join_cond_effects(branches_staged)
  eqn_staged = pe.new_jaxpr_eqn(
      [eqn.invars[0], *res_binders, *eqn.invars[1:]], out_binders_staged,
      cond_p, params_staged, effects_staged, eqn.source_info)

  new_vars = [*new_inst, *res_binders]
  return eqn_known, eqn_staged, unks_out, inst_out, new_vars

# When partially evaluating conditionals, each branch produces residuals
# depending on the computation carried out by the branch, and a corresponding
# staged jaxpr that accepts those residuals as its first few inputs. The
# residual-producing branches are staged as jaxprs and bound right away in a
# conditional. The residual-consuming jaxprs are assembled together in a jaxpr
# conditional. The following helper functions ensure that both collections of
# jaxprs (those evaluated and those staged) are valid for joint use under their
# respective conditionals.
#
# In particular, the residuals derived from each original branch may have
# distinct types. Because the branches of conditionals must have identical type
# signatures, we join residuals together across branches into a common format.

# In order to set up a type signature that all branches can conform to, it would
# suffice to concatenate all branches' residuals. But concatenation can result
# in redundant inputs and outputs, and might lead to memory allocation that
# scales unnecessarily with the branch count. This function finds common
# residual types across branches for reuse, so as to avoid redundant
# allocation. It returns a list L of types (avals) representing the collection
# of residuals merged according to type, and, for each branch, a lookup table to
# match its residuals to their positions/types in L. Example input/output:
#
# [x], [y], [x, x]             -> [x, y, x],    [[0], [1], [0, 2]]
# [x], [x], [x, x]             -> [x, x],       [[0], [0], [0, 1]]
# [y, x, x], [x, z, y], [z, x] -> [y, x, x, z], [[0, 1, 2], [1, 3, 0], [3, 1]]
def _merge_branch_residuals(branch_res_avals):
  def enumerate_equal(xs):
    counts = {v: itertools.count() for v in set(xs)}
    return [(x, next(counts[x])) for x in xs]
  branch_res_tagged_avals = map(enumerate_equal, branch_res_avals)
  all_tagged_avals = _ordered_unique(util.concatenate(branch_res_tagged_avals))
  indices = {v: i for i, v in enumerate(all_tagged_avals)}
  branch_indices = [
      [indices[aval] for aval in avals] for avals in branch_res_tagged_avals]
  all_avals = [x for x, _ in all_tagged_avals]
  return all_avals, branch_indices

# This function augments branch outputs to agree with the merged residual
# format: each branch is made to return zero-filled values in the places of
# residual outputs that it does not populate.
def _join_cond_outputs(jaxprs, all_res_avals, res_aval_indices_per_jaxpr,
                       num_non_res_outputs):
  def augment_jaxpr(jaxpr, res_indices):
    @lu.wrap_init
    def f_aug(*args):
      outs_and_residuals = core.jaxpr_as_fun(jaxpr)(*args)
      outs, residuals = split_list(outs_and_residuals, [num_non_res_outputs])
      aug_residuals = map(ad_util.zeros_like_aval, all_res_avals)
      aug_residuals = util.subvals(aug_residuals, zip(res_indices, residuals))
      return outs + list(aug_residuals)

    return _make_closed_jaxpr(f_aug, jaxpr.in_avals)

  return tuple(map(augment_jaxpr, jaxprs, res_aval_indices_per_jaxpr))

# This function augments branch inputs to agree with the merged residual format:
# each branch is made to accept all residuals, even though it will ignore those
# that it does not read.
def _join_cond_pe_staged_jaxpr_inputs(jaxprs, all_res_avals,
                                      res_aval_indices_per_jaxpr):
  newvar = core.gensym(suffix='_')
  all_res_vars = map(newvar, all_res_avals)

  def augment_jaxpr(jaxpr, res_indices):
    num_res = len(res_indices)
    res_vars = jaxpr.jaxpr.invars[:num_res]
    non_res_vars = jaxpr.jaxpr.invars[num_res:]

    aug_res_vars = list(util.subvals(all_res_vars, zip(res_indices, res_vars)))
    aug_invars = aug_res_vars + non_res_vars
    jaxpr_aug = core.Jaxpr(jaxpr.jaxpr.constvars, aug_invars,
                           jaxpr.jaxpr.outvars, jaxpr.jaxpr.eqns,
                           jaxpr.jaxpr.effects)
    jaxpr_aug = core.ClosedJaxpr(jaxpr_aug, jaxpr.consts)
    return jaxpr_aug

  return tuple(map(augment_jaxpr, jaxprs, res_aval_indices_per_jaxpr))

def _ordered_unique(xs):
  d = collections.OrderedDict((x, None) for x in xs)
  return list(d.keys())

def _cond_dce_rule(used_outputs: list[bool], eqn: core.JaxprEqn,
                   ) -> tuple[list[bool], core.JaxprEqn | None]:

  if not any(used_outputs) and not pe.has_effects(eqn):
    return [False] * len(eqn.invars), None

  closed_branches = eqn.params['branches']
  branches = [closed_jaxpr.jaxpr for closed_jaxpr in closed_branches]

  # First, compute which inputs are used in any branch (not including `pred`).
  used_inputs: list[bool] = [False] * (len(eqn.invars) - 1)  # -1 for pred
  for jaxpr in branches:
    _, used_inputs_ = pe.dce_jaxpr(jaxpr, used_outputs, instantiate=False)
    used_inputs = map(operator.or_, used_inputs, used_inputs_)

  # Next, compute DCEd branches, instantiating according to used_inputs.
  dce_branches_ = [pe.dce_jaxpr(jaxpr, used_outputs, instantiate=used_inputs)[0]
                   for jaxpr in branches]
  dce_branches = [core.ClosedJaxpr(jaxpr, closed_jaxpr.consts)
                  for closed_jaxpr, jaxpr in zip(closed_branches, dce_branches_)]

  # Finally, update parameters and form the new eqn.
  new_params = dict(eqn.params, branches=tuple(dce_branches))
  new_effects = core.join_effects(*(b.effects for b in dce_branches))
  new_effects = _join_cond_effects(dce_branches_)
  new_eqn = pe.new_jaxpr_eqn(
      [v for v, used in zip(eqn.invars, [True, *used_inputs]) if used],
      [v for v, used in zip(eqn.outvars, used_outputs) if used],
      eqn.primitive, new_params, new_effects, eqn.source_info)

  assert all(len(new_eqn.invars ) == 1 + len(jaxpr.in_avals )
             for jaxpr in new_params['branches'])
  assert all(len(new_eqn.outvars) == len(jaxpr.out_avals)
             for jaxpr in new_params['branches'])
  return [True, *used_inputs], new_eqn


def _transpose_cond_jaxpr(jaxpr, num_res):
  res_avals, primal_avals = split_list(jaxpr.in_avals, [num_res])

  @lu.wrap_init
  def transposed(*args):
    res, cts_out = split_list(args, [num_res])
    primals = res + [ad.UndefinedPrimal(aval) for aval in primal_avals]
    cts_in = ad.backward_pass(
        jaxpr.jaxpr, False, jaxpr.consts, primals, cts_out)
    _, cts_in = split_list(cts_in, [num_res])
    return map(ad.instantiate_zeros, cts_in)

  return _make_closed_jaxpr(transposed, res_avals + jaxpr.out_avals)

def _cond_transpose(cts, *args, branches):
  index, *ops = args
  assert type(index) is not ad.UndefinedPrimal
  linear = [type(x) is ad.UndefinedPrimal for x in ops]
  in_avals = branches[0].in_avals
  num_res = len(ops) - sum(linear)
  if any(isinstance(eff, RefEffect) for branch in branches for eff in
      branch.jaxpr.effects):
    raise NotImplementedError("State effect not supported in cond transpose.")

  branches_trans = tuple(
      _transpose_cond_jaxpr(jaxpr, num_res) for jaxpr in branches)
  lin_in_avals = [a.strip_weak_type() for a, l in zip(in_avals, linear) if l]
  assert all(core.typematch(out_aval, lin_in_aval)
             for jaxpr in branches_trans
             for out_aval, lin_in_aval in zip(jaxpr.out_avals, lin_in_avals))

  res = ops[:num_res]
  cts = map(ad.instantiate_zeros, cts)

  out = cond_p.bind(index, *res, *cts, branches=branches_trans)
  assert all(map(core.typecheck, lin_in_avals, out))

  out_iter = iter(out)
  out = [next(out_iter) if l else None for l in linear]
  assert next(out_iter, None) is None
  return [None] + out

def _cond_typecheck(bind_time, *in_atoms, branches):
  if not bind_time:
    _, *in_atoms = in_atoms
  avals = [x.aval for x in in_atoms]
  tc = partial(_typecheck_param, 'cond')
  tc(branches, 'branches', 'tuple of ClosedJaxpr',
     type(branches) is tuple and
     all(type(x) is core.ClosedJaxpr for x in branches))

  if len(branches) == 0:
    raise core.JaxprTypeError('cond requires at least one branch function')

  jaxpr0 = branches[0]
  jaxpr0_in_avals_str = _avals_short(jaxpr0.in_avals)
  jaxpr0_out_avals_str = _avals_short(jaxpr0.out_avals)
  joined_effects = _join_cond_effects(branches)
  disallowed_effects = effects.control_flow_allowed_effects.filter_not_in(joined_effects)
  if disallowed_effects:
    raise NotImplementedError(
        f'Effects not supported in `cond`: {disallowed_effects}')

  for i, jaxpr in enumerate(branches[1:]):
    if len(jaxpr0.in_avals) != len(jaxpr.in_avals):
      raise core.JaxprTypeError(
        f'cond branch 0 takes {len(jaxpr0.in_avals)} inputs, '
        f'branch {i+1} takes {len(jaxpr.in_avals)}')
    if len(jaxpr0.out_avals) != len(jaxpr.out_avals):
      raise core.JaxprTypeError(
        f'cond branch 0 outputs {len(jaxpr0.out_avals)} values, '
        f'branch {i+1} outputs {len(jaxpr.out_avals)}')
    if not all(map(core.typematch, jaxpr0.in_avals, jaxpr.in_avals)):
      raise core.JaxprTypeError(
        f'cond branches 0 and {i+1} have mismatching input types: '
        f'{jaxpr0_in_avals_str} vs {_avals_short(jaxpr.in_avals)}')
    if not all(map(core.typematch, jaxpr0.out_avals, jaxpr.out_avals)):
      raise core.JaxprTypeError(
        f'cond branches 0 and {i+1} have mismatching output types: '
        f'{jaxpr0_out_avals_str} vs {_avals_short(jaxpr.out_avals)}')

  if len(avals) != 1 + len(jaxpr0.in_avals):
    raise core.JaxprTypeError(
      f'cond called with {len(avals) - 1} non-predicate operands, '
      f'but branches take {len(jaxpr0.in_avals)} inputs')

  index_aval, *op_avals = avals
  if index_aval.dtype != np.int32:
    raise core.JaxprTypeError(
      f'cond called with index of type {index_aval.dtype} instead of int32')
  if not all(map(core.typecompat, jaxpr0.in_avals, op_avals)):
    raise core.JaxprTypeError(
      f'cond branches take input types {jaxpr0_in_avals_str}, '
      f'called with operands of type {_avals_short(op_avals)}')
  return jaxpr0.out_avals, joined_effects

cond_p = core.Primitive('cond')
cond_p.multiple_results = True
cond_p.def_impl(partial(dispatch.apply_primitive, cond_p))
cond_p.def_effectful_abstract_eval(_cond_abstract_eval)
ad.primitive_jvps[cond_p] = _cond_jvp
ad.primitive_transposes[cond_p] = _cond_transpose
pe.custom_partial_eval_rules[cond_p] = _cond_partial_eval
batching.fancy_primitive_batchers[cond_p] = _cond_batching_rule
xla.register_initial_style_primitive(cond_p)
core.custom_typechecks[cond_p] = partial(_cond_typecheck, False)
pe.partial_eval_jaxpr_custom_rules[cond_p] = _cond_partial_eval_custom
pe.dce_rules[cond_p] = _cond_dce_rule
batching.ragged_prop_rules[cond_p] = batching.ragged_mask_assert_no_op_rule

def _cond_lowering(ctx, index, *args, branches):
  joined_effects = core.join_effects(*(branch.effects for branch in branches))
  ordered_effects = list(effects.ordered_effects.filter_in(joined_effects))
  num_tokens = len(ordered_effects)
  tokens_in = ctx.tokens_in.subset(ordered_effects)
  output_token_types = [mlir.token_type() for _ in ordered_effects]
  output_types = [
      *output_token_types, *map(mlir.aval_to_ir_type, ctx.avals_out)]
  flat_output_types = mlir.flatten_ir_types(output_types)

  # CaseOp takes a single argument 'index' and the corresponding blocks
  # have no arguments; the computation within the block uses implicit
  # captures.
  case_op = hlo.CaseOp(flat_output_types, index=index,
                       num_branches=len(branches))
  name_stack = ctx.name_stack.extend('cond')
  for i, jaxpr in enumerate(branches):
    branch = case_op.regions[i].blocks.append()
    with ir.InsertionPoint(branch):
      consts = [mlir.ir_constant(xla.canonicalize_dtype(x)) for x in jaxpr.consts]
      out_vals, tokens_out = mlir.jaxpr_subcomp(
          ctx.module_context, jaxpr.jaxpr, name_stack.extend(f'branch_{i}_fun'),
          tokens_in, consts, *args,
          dim_var_values=ctx.dim_var_values)
      out_tokens = [tokens_out.get(eff) for eff in ordered_effects]
      out_vals = [*out_tokens, *out_vals]
      hlo.return_(mlir.flatten_ir_values(out_vals))

  tokens_and_outputs = mlir.unflatten_ir_values_like_types(
    case_op.results, output_types)
  tokens, outputs = util.split_list(tokens_and_outputs, [num_tokens])
  ctx.set_tokens_out(mlir.TokenSet(zip(ordered_effects, tokens)))
  return outputs

mlir.register_lowering(cond_p, _cond_lowering)

@register_partial_discharge_rule(cond_p)
def _cond_state_discharge_rule(should_discharge, in_avals, out_avals, index, *args, branches):
  assert not should_discharge[0], "Can't discharge the index."
  discharged_branches = tuple(
      discharge_state(branch.jaxpr, (), should_discharge=should_discharge[1:])[0]
      for branch in branches
  )
  # Don't thread the ref values through the cond if they never change.
  forwarded_outvars = None
  for branch in discharged_branches:
    invar_pos = {v: i for i, v in enumerate(branch.invars)}
    branch_forwarding = [
        invar_pos.get(v, None) if isinstance(v, core.Var) else None
        for v in branch.outvars[len(out_avals) :]
    ]
    if forwarded_outvars is None:
      forwarded_outvars = branch_forwarding
    else:
      forwarded_outvars = [
          i if i == j else None
          for i, j in zip(forwarded_outvars, branch_forwarding)
      ]
  assert forwarded_outvars is not None
  all_outvars_fwd = [None] * len(out_avals) + forwarded_outvars
  new_branches = tuple(
      core.ClosedJaxpr(
          branch.replace(outvars=[v for v, fwd in zip(branch.outvars, all_outvars_fwd)
                                  if fwd is None]), ())
      for branch in discharged_branches
  )
  out_vals_no_fwd = cond_p.bind(index, *args, branches=new_branches)
  out_vals, out_ref_vals_no_fwd = util.split_list(out_vals_no_fwd, [len(out_avals)])
  # Insert forwarded values into reference outputs
  ref_val_no_fwd_iter = iter(out_ref_vals_no_fwd)
  out_ref_vals = [next(ref_val_no_fwd_iter) if fwd is None else args[fwd]
                  for fwd in forwarded_outvars]
  # Map reference outputs back to their invars
  ref_val_iter = iter(out_ref_vals)
  new_invals = []
  for should, aval in zip(should_discharge, in_avals):
    discharged_inval = isinstance(aval, AbstractRef) and should
    new_invals.append(next(ref_val_iter) if discharged_inval else None)
  return new_invals, out_vals


_T = TypeVar("_T")
def platform_dependent(*args: Any,
                       default: Callable[..., _T] | None = None,
                       **per_platform: Callable[..., _T]):
  """Stages out platform-specific code.

  In JAX the actual platform on which a computation is run is determined
  very late, e.g., based on where the data is located. When using AOT
  lowering or serialization, the computation may be compiled and executed
  on a different machine, or even on a platform that is not available at
  lowering time. This means that it is not safe to write platform-dependent
  code using Python conditionals, e.g., based on the current default
  JAX platform. Instead, one can use ``platform_dependent``:

  Usage::

      def cpu_code(*args): ...
      def tpu_code(*args): ...
      def other_platforms_code(*args): ...
      res = platform_dependent(*args, cpu=cpu_code, tpu=tpu_code,
                               default=other_platforms_code)

  When the staged out code is executed on a CPU, this is equivalent to
  ``cpu_code(*args)``, on a TPU is equivalent to ``tpu_code(*args)`` and on
  any other platform to ``other_platforms_code(*args)``.
  Unlike a Python conditional, all alternatives are traced
  and staged out to Jaxpr. This is similar to, and is implemented in terms of,
  :func:`~switch`, from which it inherits the behavior
  under transformations.

  Unlike a :func:`~switch` the choice of what gets executed is made earlier:
  in most cases during lowering when the lowering platform is known; in the
  rare case of multi-platform lowering and serialization, the StableHLO code
  will contain a conditional on the actual platform. This conditional is
  resolved just in time prior to compilation when the compilation platform is
  known. This means that the compiler actually never sees a conditional.

  Args:
    *args: JAX arrays passed to each of the branches. May be PyTrees.
    **per_platform: branches to use for different platforms. The branches are
      JAX callables invoked with ``*args``. The keywords are platform names,
      e.g., 'cpu', 'tpu', 'cuda', 'rocm'.
    default: optional default branch to use for a platform not mentioned in
      ``per_platform``. If there is no ``default`` there will be an error when
      the code is lowered for a platform not mentioned in ``per_platform``.

  Returns:
    The value ``per_platform[execution_platform](*args)``.
  """
  # Join identical branches
  platform_branches: list[tuple[list[str], Callable]] = []
  for pname, pbranch in per_platform.items():
    if pname == "gpu":
      raise ValueError("Use 'cuda' or 'rocm' for lax.platform_dependent.")
    for ps, b in platform_branches:
      if b == pbranch:
        ps.append(pname)
        break
    else:
      platform_branches.append(([pname], pbranch))

  platforms_lists, branches = util.unzip2(platform_branches)
  platform_index = platform_index_p.bind(
    platforms=tuple(tuple(ps) for ps in platforms_lists),
    has_default=(default is not None))

  if default is not None:
    branches = branches + (default,)
  # Use a switch, to get the proper transformation rules for free. Since
  # platform index has no dependence on the input data, it won't be vectorized
  # under vmap.
  # If the switch and the platform_index_p above are in the same compilation
  # unit then constant-folding will remove the unnecessary branches. However,
  # if we run in eager mode the switch below cannot be constant-folded and
  # the compilation may fail if some of the branches contain custom calls not
  # recognized on the compilation platform. Detect eager mode and keep only the
  # needed branch.
  try:
    # Note/TODO(mvoz): This actually rarely seems to concretize - we could look into
    # core.ensure_compile_time_eval to get better single-branch selection.
    platform_index_concrete = core.concrete_or_error(operator.index, platform_index)
  except core.ConcretizationTypeError:
    return switch(platform_index, branches, *args)
  else:
    assert 0 <= platform_index_concrete < len(branches)
    return branches[platform_index_concrete](*args)

# A primitive to compute the index of a platform into a list of platforms.
# Args:
#   platforms: Sequence[Sequence[str]]: a sequence of sequences of platform
#     names. If the current lowering platform is in one of the inner sequences
#     returns the index of that inner sequence in the outer sequence.
#   has_default: if True, and if the lowering platform is not found in
#     `platforms` then return `len(platforms)`. Otherwise, raise an error.
platform_index_p = core.Primitive("platform_index")
platform_index_p.multiple_results = False
platform_index_p.def_impl(functools.partial(dispatch.apply_primitive,
                                            platform_index_p))

@platform_index_p.def_abstract_eval
def _platform_index_aval(*_, **__):
  return core.ShapedArray((), np.int32)

def _platform_index_lowering(ctx: mlir.LoweringRuleContext,
                             *,
                             platforms: Sequence[Sequence[str]],
                             has_default: bool):
  def lower_constant(
      ctx: mlir.LoweringRuleContext, *, i: int
  ) -> Sequence[ir.Value]:
    v = mlir.ir_constant(np.int32(i))
    assert isinstance(v, ir.Value), v
    return [v]
  platform_rules: dict[str, mlir.LoweringRule] = {}
  for i, ps in enumerate(platforms):
    rule = partial(lower_constant, i=i)
    for p in ps:
      platform_rules[p] = rule

  default_rule = (
    partial(lower_constant, i=len(platforms)) if has_default else None)
  return mlir.lower_per_platform(
    ctx,
    f"platform_index(platforms={platforms}, has_default={has_default})",
    platform_rules, default_rule, effects.no_effects)

mlir.register_lowering(platform_index_p, _platform_index_lowering)
