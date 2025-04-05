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
"""Module for the common control flow utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import os
from functools import partial
from typing import Any

from jax._src import core
from jax._src import linear_util as lu
from jax._src.lax import lax
from jax._src import effects
from jax._src import ad_util
from jax._src import state
from jax._src import util
from jax._src.util import weakref_lru_cache, safe_map, partition_list
from jax.api_util import flatten_fun_nokwargs
from jax._src.interpreters import partial_eval as pe
from jax.tree_util import tree_map, tree_unflatten, keystr
from jax._src.tree_util import equality_errors_pytreedef

map, unsafe_map = safe_map, map

effects.control_flow_allowed_effects.add_type(lax.InOutFeedEffect)


def _typecheck_param(prim, param, name, msg_required, pred):
  if not pred:
    msg = (f'invalid {prim} param {name} of type {type(param).__name__}, '
           f'{msg_required} required:')
    param_str = str(param)
    # Avoid using os.linesep here to have the same multi-line error message
    # format on different platforms.
    sep = os.linesep if '\n' in param_str or '\r' in param_str else ' '
    msg = sep.join([msg, param_str])
    raise core.JaxprTypeError(msg)

@weakref_lru_cache
def _initial_style_open_jaxpr(fun: Callable, in_tree, in_avals,
                              primitive_name: str | None = None):
  wrapped_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
  debug = pe.tracing_debug_info(fun, in_tree, out_tree, False,
                                primitive_name or "<unknown>")
  jaxpr, _, consts, attrs_tracked = pe.trace_to_jaxpr_dynamic(
      wrapped_fun, in_avals, debug)
  return jaxpr, consts, out_tree(), attrs_tracked

@weakref_lru_cache
def _initial_style_jaxpr(fun: Callable, in_tree, in_avals,
                         primitive_name: str | None = None):
  jaxpr, consts, out_tree, () = _initial_style_open_jaxpr(
      fun, in_tree, in_avals, primitive_name)
  closed_jaxpr = pe.close_jaxpr(pe.convert_constvars_jaxpr(jaxpr))
  return closed_jaxpr, consts, out_tree

def _initial_style_jaxpr_attrs(fun: Callable, in_tree, in_avals,
                               primitive_name: str | None = None):
  jaxpr, consts, out_tree, attrs_tracked = _initial_style_open_jaxpr(
      fun, in_tree, in_avals, primitive_name)
  closed_jaxpr = pe.close_jaxpr(pe.convert_constvars_jaxpr(jaxpr))
  return closed_jaxpr, consts, out_tree, attrs_tracked

def _initial_style_jaxprs_with_common_consts(
    funs: Sequence[Callable], in_tree, in_avals, primitive_name: str):
  # When staging the branches of a conditional into jaxprs, constants are
  # extracted from each branch and converted to jaxpr arguments. To use the
  # staged jaxprs as the branches to a conditional *primitive*, we need for
  # their (input) signatures to match. This function "joins" the staged jaxprs:
  # for each one, it makes another that accepts *all* constants, but only uses
  # those that it needs (dropping the rest).

  jaxpr_data = [_initial_style_open_jaxpr(fn, in_tree, in_avals, primitive_name)
                for fn in funs]
  if not jaxpr_data:
    return [], [], []

  jaxprs, all_consts, all_out_trees, all_attrs_tracked = zip(*jaxpr_data)
  all_const_avals = [map(core.get_aval, consts) for consts in all_consts]

  # TODO(sharadmv,mattjj): we could dedup *all consts* instead of just the Refs.

  # We don't want two different Refs in a jaxpr's input to refer to the same
  # Ref in the caller. We call this the "Ref aliasing problem" and it introduces
  # difficulties when discharging Refs and when reasoning about programs with
  # state effects. When unifying the arguments to each branch in a cond,
  # however, we might naively pass the same Ref in multiple times.
  #
  # Here we dedup any `Ref`s that were closed over across the branches and
  # pad out constants used across different branches.
  # Let's consider an example case. For the following branch jaxprs, we will
  # produce the following const lists, where `t_` indicates a tracer (a Ref).
  # { lambda x:i32[] a:Ref{float64[]} c:Ref[float64[]}; . let
  #    a[] <- 1.0
  #    c[] <- 3.14
  #   in () }

  # { lambda  d:Ref[float64[]} b:Ref{float64[]} y:i32[]; . let
  #     d[] <- 6.28
  #     b[] <- 2.0
  #   in () }
  # consts = [[0, t_e, t_f], [t_g, t_e, 1]]
  #
  # Notice how `t_e` is duplicated. To deduplicate the `Ref`s we first
  # 1) Detecting duplicate `Ref` tracers. We keep track of duplicates in
  #    `tracer_id_to_canonical_id.` We store the deduped `Ref` tracers in a
  #    list called `canonical_refs`. We remove the `Ref`s from the consts.
  #    We should have the following lists:
  #    canonical_refs = [t_e, t_f, t_g]
  #    consts = [[0], [1]]
  # 2) We need to munge the branch jaxprs to take in *all* the canonical Refs
  #    and ignore the ones it doesn't actually use. We do this by keeping track
  #    for each jaxpr for each of its input Refs which canonical_ref it
  #    corresponds to, producing the following list:
  #    canonical_ref_indices = [[0, 1], [2, 0]]
  #
  # Afterwards, we proceed by rewriting the jaxprs to be the following:
  # { lambda a:Ref{float64[]} c:Ref[float64[]} b_:Ref{float64[]} x:i32[]; . let
  #    a[] <- 1.0
  #    c[] <- 3.14
  #   in () }
  # { lambda b:Ref{float64[]} _:Ref{float64[]} d:Ref{float64[]} y:i32[]; . let
  #     d[] <- 6.28
  #     b[] <- 2.0
  #   in () }
  canonical_ref_indices = []
  canonical_refs: list[Any] = []
  tracer_id_to_canonical_id = {}
  all_nonref_consts = []
  canonical_ref_avals = []
  all_nonref_const_avals = []
  for consts, consts_avals in zip(all_consts, all_const_avals):
    ref_indices = []
    nonref_consts = []
    nonref_const_avals = []
    for c, aval in zip(consts, consts_avals):
      if isinstance(aval, state.AbstractRef):
        tracer_id = id(c)
        if tracer_id not in tracer_id_to_canonical_id:
          canonical_id = len(canonical_refs)
          canonical_refs.append(c)
          tracer_id_to_canonical_id[tracer_id] = canonical_id
          canonical_ref_avals.append(aval)
        canonical_id = tracer_id_to_canonical_id[tracer_id]
        ref_indices.append(canonical_id)
      else:
        nonref_consts.append(c)
        nonref_const_avals.append(aval)
    all_nonref_consts.append(nonref_consts)
    all_nonref_const_avals.append(tuple(nonref_const_avals))
    canonical_ref_indices.append(tuple(ref_indices))

  consts = [*canonical_refs, *util.concatenate(all_nonref_consts)]
  jaxprs = tuple(_pad_jaxpr_constvars(jaxpr, i, (*canonical_ref_avals,), (*canonical_ref_indices,), (*all_nonref_const_avals,))
                 for i, jaxpr in enumerate(jaxprs))
  return jaxprs, consts, all_out_trees

@weakref_lru_cache
def _pad_jaxpr_constvars(jaxpr, i, canonical_ref_avals, canonical_ref_indices,
                         all_nonref_const_avals):
  is_ref = [isinstance(v.aval, state.AbstractRef) for v in jaxpr.constvars]
  nonref_constvars, ref_constvars = partition_list(is_ref, jaxpr.constvars)
  newvar = core.gensym(suffix='_')
  unused_const_vars = [tuple(map(newvar, const_avals))
                       for const_avals in all_nonref_const_avals]
  padded_ref_constvars  = map(newvar, canonical_ref_avals)
  for canonical_id, ref_var in zip(canonical_ref_indices[i], ref_constvars):
    padded_ref_constvars[canonical_id] = ref_var
  const_prefix = util.concatenate(unused_const_vars[:i])
  const_suffix = util.concatenate(unused_const_vars[i + 1:])
  constvars = [*padded_ref_constvars, *const_prefix, *nonref_constvars,
                *const_suffix]
  jaxpr = jaxpr.replace(constvars=constvars)
  effects = pe.make_jaxpr_effects(jaxpr.constvars, jaxpr.invars,
                                  jaxpr.outvars, jaxpr.eqns)
  jaxpr = jaxpr.replace(effects=effects)
  return core.ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr), ())

def _check_tree_and_avals(what1, tree1, avals1, what2, tree2, avals2):
  """Raises TypeError if (tree1, avals1) does not match (tree2, avals2).

  Corresponding `tree` and `avals` must match in the sense that the number of
  leaves in `tree` must be equal to the length of `avals`. `what1` and
  `what2` describe what the `tree1` and `tree2` represent.
  """
  if tree1 != tree2:
    errs = list(equality_errors_pytreedef(tree1, tree2))
    msg = []
    msg.append(
        f"{what1} must have same type structure as {what2}, but there are differences: ")
    for path, thing1, thing2, explanation in errs:
      msg.append(
          f"    * at output{keystr(tuple(path))}, {what1} has {thing1} and "
          f"{what2} has {thing2}, so {explanation}")
    raise TypeError('\n'.join(msg))

  if not all(map(core.typematch, avals1, avals2)):
    diff = tree_map(_show_diff, tree_unflatten(tree1, avals1),
                    tree_unflatten(tree2, avals2))
    raise TypeError(f"{what1} and {what2} must have identical types, got\n{diff}.")

def _check_tree(func_name, expected_name, actual_tree, expected_tree, has_aux=False):
  if has_aux:
    actual_tree_children = actual_tree.children()

    if len(actual_tree_children) == 2:
      # select first child as result tree
      actual_tree = actual_tree_children[0]
    else:
      raise ValueError(
        f"{func_name}() produced a pytree with structure "
        f"{actual_tree}, but a pytree tuple with auxiliary "
        f"output was expected because has_aux was set to True.")

  if actual_tree != expected_tree:
    raise TypeError(
        f"{func_name}() output pytree structure must match {expected_name}, "
        f"got {actual_tree} and {expected_tree}.")

def _prune_zeros(ts):
  return [t for t in ts if type(t) is not ad_util.Zero]

def _make_closed_jaxpr(traceable: lu.WrappedFun, in_avals: Sequence[core.AbstractValue]):
  jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(traceable, in_avals)
  return core.ClosedJaxpr(jaxpr, consts)

def _make_closed_jaxpr_attrs(traceable: lu.WrappedFun, in_avals: Sequence[core.AbstractValue]):
  jaxpr, _, consts, attrs_tracked = pe.trace_to_jaxpr_dynamic(traceable, in_avals)
  return core.ClosedJaxpr(jaxpr, consts), attrs_tracked


def _show_diff(array1, array2):
  if core.typematch(array1, array2):
    return f"{array1}"
  return f"DIFFERENT {array1} vs. {array2}"

def _avals_short(avals):
  to_str = lambda aval: getattr(aval, 'str_short', partial(str, aval))()
  return ' '.join(map(to_str, avals))
