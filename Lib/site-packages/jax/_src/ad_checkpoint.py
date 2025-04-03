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

from __future__ import annotations

from collections.abc import Callable, Sequence
import functools
from functools import partial
import logging
from typing import Any
import types

import numpy as np

from jax._src import ad_util
from jax._src import api
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import linear_util as lu
from jax._src import effects
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src.api_util import (
    flatten_fun, debug_info, fun_sourceinfo, fun_signature)
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import lax as lax_internal
from jax._src.lax import convolution as lax_convolution
from jax._src.lib.mlir.dialects import hlo
from jax._src.traceback_util import api_boundary
from jax._src.tree_util import tree_flatten, tree_unflatten, tree_structure
from jax._src.util import (unzip2, wraps, split_list, partition_list, safe_map,
                           safe_zip, merge_lists, weakref_lru_cache)

source_info_util.register_exclusion(__file__)
traceback_util.register_exclusion(__file__)

map = safe_map
zip = safe_zip

logger = logging.getLogger(__name__)

### Policies

def everything_saveable(*_, **__) -> bool:
  # This is the effective policy without any use of jax.remat.
  return True

def nothing_saveable(*_, **__) -> bool:
  # This is the effective policy when using jax.remat without explicit policy.
  return False

def dots_saveable(prim, *_, **__) -> bool:
  # Matrix multiplies are expensive, so let's save them (and nothing else).
  return prim in {lax_internal.dot_general_p,
                  lax_convolution.conv_general_dilated_p}
checkpoint_dots = dots_saveable

def dot_with_no_batch_dims_saveable(prim, *_, **params) -> bool:
  # This is a useful heuristic for transformers.
  if prim is lax_internal.dot_general_p:
    (_, _), (lhs_b, rhs_b) = params['dimension_numbers']
    if not lhs_b and not rhs_b:
      return True
  return False

def offload_dot_with_no_batch_dims(offload_src, offload_dst):
  def policy(prim, *_, **params):
    # This is a useful heuristic for transformers.
    if prim is lax_internal.dot_general_p:
      (_, _), (lhs_b, rhs_b) = params['dimension_numbers']
      if not lhs_b and not rhs_b:
        return pe.Offloadable(src=offload_src, dst=offload_dst)
    return pe.Recompute
  return policy


name_p = core.Primitive('name')

def save_anything_except_these_names(*names_not_to_save):
  """Save any values (not just named ones) excluding the names given."""
  names_not_to_save = frozenset(names_not_to_save)
  def policy(prim, *_, **params):
    if prim is name_p:
      return params['name'] not in names_not_to_save
    return True  # allow saving anything which is not named
  return policy

def save_any_names_but_these(*names_not_to_save):
  """Save only named values, excluding the names given."""
  names_not_to_save = frozenset(names_not_to_save)
  def policy(prim, *_, **params):
    if prim is name_p:
      return params['name'] not in names_not_to_save
    return False  # only allow saving named values
  return policy

def save_only_these_names(*names_which_can_be_saved):
  """Save only named values, and only among the names given."""
  names_which_can_be_saved = set(names_which_can_be_saved)
  def policy(prim, *_, **params):
    if prim is name_p:
      return params['name'] in names_which_can_be_saved
    return False  # not saveable unless it's in the allow-list
  return policy

def save_and_offload_only_these_names(
    *, names_which_can_be_saved, names_which_can_be_offloaded,
    offload_src, offload_dst):
  names_which_can_be_saved = set(names_which_can_be_saved)
  names_which_can_be_offloaded = set(names_which_can_be_offloaded)
  intersection = names_which_can_be_saved.intersection(names_which_can_be_offloaded)
  if intersection:
    raise ValueError(
        "The names should be exclusive and should not intersect in"
        " `names_which_can_be_saved` and `names_which_can_be_offloaded`. Got"
        f" names_which_can_be_saved={names_which_can_be_saved},"
        f" names_which_can_be_offloaded={names_which_can_be_offloaded} and the"
        f" intersection={intersection}")
  def policy(prim, *_, **params):
    if prim is name_p and params['name'] in names_which_can_be_saved:
      return pe.Saveable
    if prim is name_p and params['name'] in names_which_can_be_offloaded:
      return pe.Offloadable(src=offload_src, dst=offload_dst)
    return pe.Recompute  # not saveable unless it's in the allow-list
  return policy


def save_from_both_policies(policy_1, policy_2):

  def policy(prim, *args, **params):
    out1 = policy_1(prim, *args, **params)
    out2 = policy_2(prim, *args, **params)
    if not (isinstance(out1, bool) and isinstance(out2, bool)):
      raise ValueError(
          "The return value of the policies should be a boolean. Got:"
          f" {out1} and {out2}. Please write a custom policy function directly,"
          " rather than using this helper function.")
    return out1 or out2
  return policy


# Please update the file docs/gradient-checkpointing.md with any new
# policies to keep the doc in sync.
checkpoint_policies = types.SimpleNamespace(
    everything_saveable=everything_saveable,
    nothing_saveable=nothing_saveable,
    dots_saveable=dots_saveable,
    checkpoint_dots=dots_saveable,
    dots_with_no_batch_dims_saveable=dot_with_no_batch_dims_saveable,
    checkpoint_dots_with_no_batch_dims=dot_with_no_batch_dims_saveable,
    offload_dot_with_no_batch_dims=offload_dot_with_no_batch_dims,
    save_anything_except_these_names=save_anything_except_these_names,
    save_any_names_but_these=save_any_names_but_these,
    save_only_these_names=save_only_these_names,
    save_from_both_policies=save_from_both_policies,
    save_and_offload_only_these_names=save_and_offload_only_these_names)


### Main API

@api_boundary
def checkpoint(fun: Callable, *, prevent_cse: bool = True,
               policy: Callable[..., bool] | None = None,
               static_argnums: int | tuple[int, ...] = (),
               ) -> Callable:
  """Make ``fun`` recompute internal linearization points when differentiated.

  The :func:`jax.checkpoint` decorator, aliased to :func:`jax.remat`, provides a
  way to trade off computation time and memory cost in the context of automatic
  differentiation, especially with reverse-mode autodiff like :func:`jax.grad`
  and :func:`jax.vjp` but also with :func:`jax.linearize`.

  When differentiating a function in reverse-mode, by default all the
  linearization points (e.g. inputs to elementwise nonlinear primitive
  operations) are stored when evaluating the forward pass so that they can be
  reused on the backward pass. This evaluation strategy can lead to a high
  memory cost, or even to poor performance on hardware accelerators where memory
  access is much more expensive than FLOPs.

  An alternative evaluation strategy is for some of the linearization points to
  be recomputed (i.e. rematerialized) rather than stored. This approach can
  reduce memory usage at the cost of increased computation.

  This function decorator produces a new version of ``fun`` which follows
  the rematerialization strategy rather than the default store-everything
  strategy. That is, it returns a new version of ``fun`` which, when
  differentiated, doesn't store any of its intermediate linearization points.
  Instead, these linearization points are recomputed from the function's saved
  inputs.

  See the examples below.

  Args:
    fun: Function for which the autodiff evaluation strategy is to be changed
      from the default of storing all intermediate linearization points to
      recomputing them. Its arguments and return value should be arrays,
      scalars, or (nested) standard Python containers (tuple/list/dict) thereof.
    prevent_cse: Optional, boolean keyword-only argument indicating whether to
      prevent common subexpression elimination (CSE) optimizations in the HLO
      generated from differentiation. This CSE prevention has costs because it
      can foil other optimizations, and because it can incur high overheads on
      some backends, especially GPU. The default is True because otherwise,
      under a :func:`~jax.jit` or :func:`~jax.pmap`, CSE can defeat the purpose
      of this decorator.
      But in some settings, like when used inside a :func:`~jax.lax.scan`, this
      CSE prevention mechanism is unnecessary, in which case ``prevent_cse`` can
      be set to False.
    static_argnums: Optional, int or sequence of ints, a keyword-only argument
      indicating which argument values on which to specialize for tracing and
      caching purposes. Specifying arguments as static can avoid
      ConcretizationTypeErrors when tracing, but at the cost of more retracing
      overheads. See the example below.
    policy: Optional, callable keyword-only argument. It should be one of the
      attributes of ``jax.checkpoint_policies``. The callable takes as input a
      type-level specification of a first-order primitive application and
      returns a boolean indicating whether the corresponding output value(s) can
      be saved as residuals (or instead must be recomputed in the (co)tangent
      computation if needed).

  Returns:
    A function (callable) with the same input/output behavior as ``fun`` but
    which, when differentiated using e.g. :func:`jax.grad`, :func:`jax.vjp`, or
    :func:`jax.linearize`, recomputes rather than stores intermediate
    linearization points, thus potentially saving memory at the cost of extra
    computation.

  Here is a simple example:

  >>> import jax
  >>> import jax.numpy as jnp

  >>> @jax.checkpoint
  ... def g(x):
  ...   y = jnp.sin(x)
  ...   z = jnp.sin(y)
  ...   return z
  ...
  >>> jax.value_and_grad(g)(2.0)
  (Array(0.78907233, dtype=float32, weak_type=True), Array(-0.2556391, dtype=float32, weak_type=True))

  Here, the same value is produced whether or not the :func:`jax.checkpoint`
  decorator is present. When the decorator is not present, the values
  ``jnp.cos(2.0)`` and ``jnp.cos(jnp.sin(2.0))`` are computed on the forward
  pass and are stored for use in the backward pass, because they are needed
  on the backward pass and depend only on the primal inputs. When using
  :func:`jax.checkpoint`, the forward pass will compute only the primal outputs
  and only the primal inputs (``2.0``) will be stored for the backward pass.
  At that time, the value ``jnp.sin(2.0)`` is recomputed, along with the values
  ``jnp.cos(2.0)`` and ``jnp.cos(jnp.sin(2.0))``.

  While :func:`jax.checkpoint` controls what values are stored from the
  forward-pass to be used on the backward pass, the total amount of memory
  required to evaluate a function or its VJP depends on many additional internal
  details of that function. Those details include which numerical primitives are
  used, how they're composed, where jit and control flow primitives like scan
  are used, and other factors.

  The :func:`jax.checkpoint` decorator can be applied recursively to express
  sophisticated autodiff rematerialization strategies. For example:

  >>> def recursive_checkpoint(funs):
  ...   if len(funs) == 1:
  ...     return funs[0]
  ...   elif len(funs) == 2:
  ...     f1, f2 = funs
  ...     return lambda x: f1(f2(x))
  ...   else:
  ...     f1 = recursive_checkpoint(funs[:len(funs)//2])
  ...     f2 = recursive_checkpoint(funs[len(funs)//2:])
  ...     return lambda x: f1(jax.checkpoint(f2)(x))
  ...

  If ``fun`` involves Python control flow that depends on argument values,
  it may be necessary to use the ``static_argnums`` parameter. For example,
  consider a boolean flag argument::

    from functools import partial

    @partial(jax.checkpoint, static_argnums=(1,))
    def foo(x, is_training):
      if is_training:
        ...
      else:
        ...

  Here, the use of ``static_argnums`` allows the ``if`` statement's condition
  to depends on the value of ``is_training``. The cost to using
  ``static_argnums`` is that it introduces re-tracing overheads across calls:
  in the example, ``foo`` is re-traced every time it is called with a new value
  of ``is_training``. In some situations, ``jax.ensure_compile_time_eval``
  is needed as well::

    @partial(jax.checkpoint, static_argnums=(1,))
    def foo(x, y):
      with jax.ensure_compile_time_eval():
        y_pos = y > 0
      if y_pos:
        ...
      else:
        ...

  As an alternative to using ``static_argnums`` (and
  ``jax.ensure_compile_time_eval``), it may be easier to compute some values
  outside the :func:`jax.checkpoint`-decorated function and then close over them.
  """
  if isinstance(static_argnums, int):
    static_argnums = static_argnums,

  @wraps(fun)
  @api_boundary
  def fun_remat(*args, **kwargs):
    debug = debug_info("checkpoint / remat", fun_sourceinfo(fun),
                       fun_signature(fun), args, kwargs, static_argnums, ())
    fun_, args = _remat_static_argnums(fun, static_argnums, args)
    args_flat, in_tree = tree_flatten((args, kwargs))
    in_avals = [core.shaped_abstractify(x) for x in args_flat]
    jaxpr, consts, out_tree = _trace_to_jaxpr(fun_, in_tree, tuple(in_avals), debug)
    out_flat = remat_p.bind(
        *consts, *args_flat, jaxpr=jaxpr, prevent_cse=prevent_cse,
        differentiated=False, policy=policy)
    return tree_unflatten(out_tree, out_flat)
  return fun_remat

remat = checkpoint  # alias

# This function is similar to api_util.argnums_partial, except the error
# messages are specific to jax.remat (and thus more actionable), the
# hashing/caching behavior is slightly different, and this function accepts a
# boolean for static_argnums. Perhaps the two could be de-duplicated.
def _remat_static_argnums(fun, static_argnums, args):
  if type(static_argnums) is int:
    static_argnums = (static_argnums,)
  elif not (type(static_argnums) is tuple and
            all(type(d) is int for d in static_argnums)):
    raise TypeError("the `static_argnums` argument to `jax.checkpoint` / "
                    "`jax.remat` must be an int, tuple of ints or, bool, but "
                    f"got value {static_argnums}")

  if not all(-len(args) <= d < len(args) for d in static_argnums):
    raise ValueError("the `static_argnums` argument to `jax.checkpoint` / "
                     "`jax.remat` can only take integer values greater than or "
                     "equal to `-len(args)` and less than `len(args)`, but got "
                     f"{static_argnums}")

  if not static_argnums:
    return fun, args
  nargs = len(args)
  static_argnums_ = frozenset(d % len(args) for d in static_argnums)
  dyn_args, static_args = [], []
  for i, x in enumerate(args):
    if i in static_argnums_: static_args.append(WrapHashably(x))
    else: dyn_args.append(x)
  new_fun = _dyn_args_fun(fun, static_argnums_, tuple(static_args), nargs)
  return new_fun, dyn_args

class WrapHashably:
  val: Any
  hash: int
  hashable: bool

  def __init__(self, val):
    self.val = val
    try:
      self.hash = hash(val)
      self.hashable = True
    except:
      self.hash = id(val)
      self.hashable = False
  def __hash__(self):
    return self.hash
  def __eq__(self, other):
    if isinstance(other, WrapHashably):
      if self.hashable and other.hashable:
        return self.val == other.val
      else:
        return self.val is other.val
    return False

# This caching is useful to avoid retracing even when static_argnums is used.
# See api_benchmark.py:bench_remat_eager_retracing_overheads_static_argnums.
# On that benchmark, including this caching makes a ~10x difference (which can
# be made arbitrary large by involving larger functions to be traced).
def _dyn_args_fun(fun: Callable, static_argnums: frozenset[int],
                  static_args: tuple[WrapHashably, ...], nargs: int):
  if any(isinstance(x.val, core.Tracer) for x in static_args):
    return _dyn_args_fun_uncached(fun, static_argnums, static_args, nargs)
  return _dyn_args_fun_cached(fun, static_argnums, static_args, nargs)

def _dyn_args_fun_uncached(fun: Callable, static_argnums: frozenset[int],
                           static_args: tuple[WrapHashably, ...], nargs: int):
  def new_fun(*dyn_args, **kwargs):
    static_args_, dyn_args_ = iter(static_args), iter(dyn_args)
    full_args = [next(static_args_).val if i in static_argnums
                 else next(dyn_args_) for i in range(nargs)]
    return fun(*full_args, **kwargs)
  return new_fun

_dyn_args_fun_cached = weakref_lru_cache(_dyn_args_fun_uncached)

# This helper is similar to those in control_flow/common.py, but with
# remat-specific errors.
@weakref_lru_cache
def _trace_to_jaxpr(fun, in_tree, in_avals, debug):
  flat_fun, out_tree = flatten_fun(lu.wrap_init(fun), in_tree)
  try:
    jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals, debug)
  except core.ConcretizationTypeError as e:
    msg, = e.args
    if 'for checkpoint' in msg:
      msg += "\n\n" + (
          "Consider using the `static_argnums` parameter for `jax.remat` or "
          "`jax.checkpoint`. See the `jax.checkpoint` docstring and its example "
          "involving `static_argnums`:\n"
          "https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html"
          "\n")
      e.args = msg,
    raise
  return pe.convert_constvars_jaxpr(jaxpr), consts, out_tree()


### Utilities

def saved_residuals(f, *args, **kwargs) -> list[tuple[core.AbstractValue, str]]:
  in_leaves, in_tree = tree_flatten((args, kwargs))

  def f_(*args):
    args, kwargs = tree_unflatten(in_tree, args)
    return f(*args, **kwargs)

  out = api.make_jaxpr(lambda *args: api.linearize(f_, *args)[1],
                       return_shape=True)(*in_leaves)
  assert isinstance(out, tuple)
  jaxpr_, out_shape = out
  jaxpr = jaxpr_.jaxpr
  out_tree = lambda: tree_structure(out_shape)
  assert len(jaxpr.invars) == len(in_leaves)
  dbg = pe.tracing_debug_info(f, in_tree, out_tree, True, "saved_residuals")
  return _saved_residuals(jaxpr, dbg.arg_names)  # type: ignore

def _saved_residuals(jaxpr, arg_names) -> list[tuple[core.AbstractValue, str]]:
  res_lits = [x for x in jaxpr.outvars if     isinstance(x, core.Literal)]
  res_vars = {x for x in jaxpr.outvars if not isinstance(x, core.Literal)}

  results = []

  for x in res_lits:
    results.append((x.aval, 'from a literal'))

  for v in jaxpr.constvars:
    if v in res_vars:
      results.append((v.aval, 'from a constant'))

  for i, v in enumerate(jaxpr.invars):
    if v in res_vars:
      if arg_names is not None:
        src = f'from the argument {arg_names[i]}'
      else:
        src = 'from the argument at flattened index {i}'
      results.append((v.aval, src))

  named_vars = {v: e for e in jaxpr.eqns if e.primitive is name_p
                for v in e.invars}

  for eqn in jaxpr.eqns:
    src = source_info_util.summarize(eqn.source_info)
    for v in eqn.outvars:
      if v in res_vars:
        if eqn.primitive is name_p or v in named_vars and (eqn := named_vars[v]):
          results.append((v.aval, f"named '{eqn.params['name']}' from {src}"))
        elif str(eqn.primitive) == 'pjit':
          results.append((v.aval,
                          f"output of jitted function '{eqn.params['name']}' "
                          f"from {src}"))
        else:
          results.append((v.aval, f'output of {eqn.primitive.name} from {src}'))

  assert len(results) == len(jaxpr.outvars)
  return results

def print_saved_residuals(f, *args, **kwargs):
  for aval, src in saved_residuals(f, *args, **kwargs):
    print(f'{aval.str_short(short_dtypes=True)} {src}')


### Implementation

remat_p = core.Primitive('remat2')
remat_p.multiple_results = True

@remat_p.def_impl
def remat_impl(*args, jaxpr, prevent_cse, differentiated, policy):
  del prevent_cse, differentiated, policy  # Unused.
  return core.eval_jaxpr(jaxpr, (), *args)

@remat_p.def_effectful_abstract_eval
def remat_abstract_eval(*args, jaxpr, prevent_cse, differentiated, policy):
  del args, prevent_cse, differentiated, policy  # Unused.
  return [v.aval for v in jaxpr.outvars], jaxpr.effects

def remat_jvp(primals, tangents, jaxpr, prevent_cse, differentiated, policy):
  assert not jaxpr.constvars
  in_nonzeros = [type(t) is not ad_util.Zero for t in tangents]
  jaxpr_jvp_, out_nz = ad.jvp_jaxpr(pe.close_jaxpr(jaxpr), in_nonzeros, False)
  nonzero_tangents = [t for t in tangents if type(t) is not ad_util.Zero]
  jaxpr_jvp = pe.convert_constvars_jaxpr(jaxpr_jvp_.jaxpr)
  outs = remat_p.bind(
      *jaxpr_jvp_.consts, *primals, *nonzero_tangents, jaxpr=jaxpr_jvp,
      prevent_cse=prevent_cse, differentiated=differentiated, policy=policy)
  out_primals, out_tangents_ = split_list(outs, [len(jaxpr.outvars)])
  out_tangents_ = iter(out_tangents_)
  out_tangents = [next(out_tangents_) if nz else ad_util.Zero.from_primal_value(p)
                  for p, nz in zip(out_primals, out_nz)]
  return out_primals, out_tangents
ad.primitive_jvps[remat_p] = remat_jvp

effects.remat_allowed_effects.add_type(lax_internal.InOutFeedEffect)

def remat_partial_eval(trace, *tracers, jaxpr, **params):
  assert not jaxpr.constvars
  disallowed_effects = effects.remat_allowed_effects.filter_not_in(jaxpr.effects)
  if disallowed_effects:
    raise NotImplementedError(
        'Effects not supported in partial-eval of `checkpoint`/`remat`: '
        f'{disallowed_effects}')
  policy = params['policy'] or nothing_saveable
  in_unknowns = [not t.is_known() for t in tracers]
  jaxpr_known, jaxpr_staged, out_unknowns, out_inst, num_res = \
      pe.partial_eval_jaxpr_custom(
          jaxpr, in_unknowns, [True] * len(in_unknowns), False, False, policy)

  # DCE jaxpr_staged, keeping only instantiated outputs which are unknown
  _, out_inst_unknown = partition_list(out_inst, out_unknowns)
  jaxpr_unknown, in_used_staged = pe.dce_jaxpr(jaxpr_staged, out_inst_unknown)
  used_res, in_used_staged = split_list(in_used_staged, [num_res])

  # DCE jaxpr_known, keeping all known outputs but discarding dce'd res
  out_used_known = [True] * (len(out_unknowns) - sum(out_unknowns)) + used_res
  jaxpr_known, in_used_known = pe.dce_jaxpr(jaxpr_known, out_used_known)
  num_res = sum(used_res)

  # To avoid precision mismatches in fwd and bwd passes due to XLA excess
  # precision, insert explicit x = reduce_precision(x, **finfo(x.dtype)) calls
  # on producers of any residuals. See https://github.com/jax-ml/jax/pull/22244.
  jaxpr_known_ = _insert_reduce_precision(jaxpr_known, num_res)

  # compute known outputs and residuals (hoisted out of remat primitive)
  _, in_consts_ = unzip2(t.pval for t in tracers if t.pval.is_known())
  _, in_consts = partition_list(in_used_known, in_consts_)
  out_consts = core.eval_jaxpr(jaxpr_known_, (), *in_consts)
  out_knowns, residuals = split_list(out_consts, [len(out_consts)-num_res])

  # set up unknown outputs with a recipe to call remat
  res_tracers = map(trace.new_instantiated_const, residuals)
  _, tracers_staged = partition_list(in_used_staged, tracers)
  in_jaxpr_tracers = res_tracers + map(trace.instantiate_const, tracers_staged)
  out_jaxpr_tracers = [pe.JaxprTracer(trace, pe.PartialVal.unknown(x.aval), None)
                       for x in jaxpr_unknown.outvars]
  new_params = dict(params, jaxpr=jaxpr_unknown, differentiated=True)
  recipe = pe.new_eqn_recipe(in_jaxpr_tracers, out_jaxpr_tracers, remat_p,
                             new_params, jaxpr_unknown.effects,
                             source_info_util.current())

  # log info about saved residuals
  log_level = logging.WARNING if config.log_checkpoint_residuals.value else logging.DEBUG
  if logger.isEnabledFor(log_level):
    try:
      _, staged_unk = partition_list(in_used_staged, in_unknowns)
      res_invars, _ = partition_list(staged_unk, jaxpr_unknown.invars[num_res:])
      res_outvars = jaxpr_known.outvars[len(jaxpr_known.outvars) - num_res:]
      body_res = _saved_residuals(jaxpr_known.replace(outvars=res_outvars), None)
      logger.log(log_level,
                'remat-decorated function ' +
                'saving inputs with shapes:\n' * bool(res_invars) +
                '  %s\n' * len(res_invars) +
                'and ' * bool(res_invars) * bool(body_res) +
                'saving these intermediates:\n' * bool(body_res) +
                '  %s from %s\n' * len(body_res),
                *[v.aval.str_short() for v in res_invars],
                *[elt for (a, s) in body_res for elt in [a.str_short(), s]])
    except:
      pass  # just don't log anything on failure

  for t in out_jaxpr_tracers: t.recipe = recipe

  # zip together known and unknown outputs
  return merge_lists(out_unknowns, out_knowns, out_jaxpr_tracers)
pe.custom_partial_eval_rules[remat_p] = remat_partial_eval

@weakref_lru_cache
def _insert_reduce_precision(jaxpr: core.Jaxpr, num_res: int) -> core.Jaxpr:
  res_vars = jaxpr.outvars[len(jaxpr.outvars) - num_res:]
  used_vars = {x for e in jaxpr.eqns for x in e.invars if isinstance(x, core.Var)}
  invars, constvars, eqns = jaxpr.invars[:], jaxpr.constvars[:], jaxpr.eqns[:]
  for v in res_vars:
    if (not isinstance(v.aval, core.UnshapedArray) or
        not dtypes.issubdtype(v.aval.dtype, np.inexact)):
      continue
    if v not in used_vars:
      continue
    assert isinstance(v, core.Var)
    newvar = core.Var(v.suffix, v.aval)
    finfo = dtypes.finfo(v.aval.dtype)
    params = dict(exponent_bits=finfo.nexp, mantissa_bits=finfo.nmant)
    if v in constvars or v in invars:
      lst = constvars if v in constvars else invars
      new_eqn = core.new_jaxpr_eqn(
          [newvar], [v], lax_internal.reduce_precision_p, params, set())
      lst[lst.index(v)] = newvar
      eqns.insert(0, new_eqn)
    else:
      (eqn_idx, eqn), = ((i, e) for i, e in enumerate(eqns) if v in e.outvars)
      if (eqn.primitive == lax_internal.reduce_precision_p and
          eqn.params == params):
        continue
      replace_eqn = eqn.replace(outvars=[v_ if v_ != v else newvar
                                         for v_ in eqn.outvars])
      new_eqn = core.new_jaxpr_eqn(
          [newvar], [v], lax_internal.reduce_precision_p, params, set(),
          eqn.source_info, eqn.ctx)
      eqns[eqn_idx] = replace_eqn
      eqns.insert(eqn_idx+1, new_eqn)
  new_jaxpr = jaxpr.replace(invars=invars, constvars=constvars, eqns=eqns)
  config.enable_checks.value and core.check_jaxpr(new_jaxpr)
  return new_jaxpr

def remat_partial_eval_custom_params_updater(*args):
  *_, params_known, params_staged = args
  return params_known, dict(params_staged, differentiated=True)
pe.partial_eval_jaxpr_custom_rules[remat_p] = \
    partial(pe.call_partial_eval_custom_rule, 'jaxpr',
            remat_partial_eval_custom_params_updater)

def remat_transpose(out_cts, *in_primals, jaxpr, **params):
  assert not jaxpr.constvars
  in_linear = [ad.is_undefined_primal(x) for x in in_primals]
  out_zeros = [type(ct) is ad_util.Zero for ct in out_cts]
  transposed_jaxpr_, in_zeros = transpose_jaxpr(
      pe.close_jaxpr(jaxpr), in_linear, out_zeros)
  transposed_jaxpr, consts = transposed_jaxpr_.jaxpr, transposed_jaxpr_.consts
  transposed_jaxpr = pe.convert_constvars_jaxpr(transposed_jaxpr)
  args, _ = tree_flatten((in_primals, out_cts))
  in_cts_nz = remat_p.bind(*consts, *args, jaxpr=transposed_jaxpr, **params)
  in_cts_nz_, in_zeros_ = iter(in_cts_nz), iter(in_zeros)
  in_cts = [None if not ad.is_undefined_primal(x) else
            ad_util.Zero(x.aval) if next(in_zeros_) else next(in_cts_nz_)
            for x in in_primals]
  assert next(in_cts_nz_, None) is next(in_zeros_, None) is None
  return in_cts
ad.primitive_transposes[remat_p] = remat_transpose

# TODO(mattjj): move this to ad.py
def transpose_jaxpr(jaxpr: core.ClosedJaxpr, in_linear: bool | Sequence[bool],
                    out_zeros: bool | Sequence[bool],
                    ) -> tuple[core.ClosedJaxpr, list[bool]]:
  if type(in_linear) is bool:
    in_linear = (in_linear,) * len(jaxpr.in_avals)
  if type(out_zeros) is bool:
    out_zeros = (out_zeros,) * len(jaxpr.out_avals)
  return _transpose_jaxpr(jaxpr, tuple(in_linear), tuple(out_zeros))

@weakref_lru_cache
def _transpose_jaxpr(jaxpr, in_lin, out_zeros):
  in_avals = ([a for a,  lin in zip(jaxpr.in_avals,  in_lin   ) if not lin] +
              [a for a, zero in zip(jaxpr.out_avals, out_zeros) if not zero])
  cell = lambda: None

  @lu.wrap_init
  def transposed(*args_flat):
    ins_flat, out_cts_flat = split_list(args_flat, [len(in_lin) - sum(in_lin)])

    # Evaluate nonlinear parts using partial evaluation to get a linear jaxpr.
    ins_iter = iter(ins_flat)
    in_pvals = [pe.PartialVal.unknown(aval) if lin else
                pe.PartialVal.known(next(ins_iter))
                for aval, lin in zip(jaxpr.in_avals, in_lin)]
    assert next(ins_iter, None) is None
    with source_info_util.extend_name_stack('rematted_computation'):
      lin_jaxpr, _, consts = pe.trace_to_jaxpr_nounits(
          lu.wrap_init(core.jaxpr_as_fun(jaxpr)), in_pvals, False)

    # Transpose the linear jaxpr (which only has linear inputs).
    out_cts_iter = iter(out_cts_flat)
    out_cts = [ad_util.Zero(aval) if zero else next(out_cts_iter)
               for aval, zero in zip(jaxpr.out_avals, out_zeros)]
    assert next(out_cts_iter, None) is None
    dummy_args = [ad.UndefinedPrimal(v.aval) for v in lin_jaxpr.invars]
    in_cts = ad.backward_pass(lin_jaxpr, False, consts, dummy_args, out_cts)

    # Identify symbolic zeros in the resulting cotangents, and return nonzeros.
    in_zeros = cell.in_cts_zero = [type(ct) is ad_util.Zero for ct in in_cts]
    in_cts_nz, _ = partition_list(in_zeros, in_cts)
    return in_cts_nz

  transposed_jaxpr_, _, consts, () = pe.trace_to_jaxpr_dynamic(transposed, in_avals)
  transposed_jaxpr = core.ClosedJaxpr(transposed_jaxpr_, consts)
  return transposed_jaxpr, cell.in_cts_zero  # pytype: disable=attribute-error

def remat_vmap(axis_data, args, dims, *, jaxpr, **params):
  assert not jaxpr.constvars
  jaxpr_batched_, out_batched = batching.batch_jaxpr_axes(
      pe.close_jaxpr(jaxpr), axis_data, dims,
      [batching.zero_if_mapped] * len(jaxpr.outvars))
  jaxpr_batched, consts = jaxpr_batched_.jaxpr, jaxpr_batched_.consts
  if consts:
    jaxpr_batched = pe.convert_constvars_jaxpr(jaxpr_batched)
  out_dims = [0 if b else None for b in out_batched]
  return remat_p.bind(*consts, *args, jaxpr=jaxpr_batched, **params), out_dims
batching.fancy_primitive_batchers[remat_p] = remat_vmap

# TODO(mattjj,sharadmv): de-duplicate with pe.dce_jaxpr_call_rule
def remat_dce(used_outputs: list[bool], eqn: core.JaxprEqn
              ) -> tuple[list[bool], core.JaxprEqn | None]:
  if not any(used_outputs) and not pe.has_effects(eqn):
    return [False] * len(eqn.invars), None
  new_jaxpr, used_inputs = pe.dce_jaxpr(eqn.params['jaxpr'], used_outputs)
  new_params = dict(eqn.params, jaxpr=new_jaxpr)
  if (not any(used_inputs) and not any(used_outputs) and
      _has_effects(new_jaxpr.effects)):
    return used_inputs, None
  else:
    new_eqn = pe.new_jaxpr_eqn(
        [v for v, used in zip(eqn.invars, used_inputs) if used],
        [v for v, used in zip(eqn.outvars, used_outputs) if used],
        eqn.primitive, new_params, new_jaxpr.effects, eqn.source_info, eqn.ctx)
    return used_inputs, new_eqn
pe.dce_rules[remat_p] = remat_dce

def _has_effects(effects) -> bool:
  return bool({e for e in effects if not isinstance(e, core.NamedAxisEffect)})


def remat_expansion(*args, jaxpr: core.Jaxpr, prevent_cse: bool,
                    differentiated: bool, is_gpu_platform: bool = False,
                    **_):
  assert not jaxpr.constvars

  if differentiated and prevent_cse:
    if config.remat_opt_barrier.value:
      translation_rule = _remat_translation_using_opt_barrier
    elif is_gpu_platform:
      translation_rule = _remat_translation_using_while
    else:
      translation_rule = _remat_translation_using_cond
  else:
    translation_rule = lambda *args, jaxpr: core.eval_jaxpr(jaxpr, (), *args)

  return api.named_call(translation_rule, name="checkpoint")(*args, jaxpr=jaxpr)

def _remat_translation_using_opt_barrier(*args, jaxpr: core.Jaxpr):
  args = lax_internal.optimization_barrier(args)
  return core.eval_jaxpr(jaxpr, (), *args)

# TODO(mattjj): add core utility for 'create dummy value for this type'?
def _dummy_like(aval: core.AbstractValue) -> Any:
  if aval is core.abstract_token:
    return lax_internal.create_token()
  elif isinstance(aval, (core.ShapedArray, core.DShapedArray)):
    return lax_internal.broadcast(lax_internal.empty(aval.dtype), aval.shape)  # type: ignore
  else:
    raise ValueError(aval)

def _remat_translation_using_while(*args, jaxpr: core.Jaxpr):
  # Implements:
  #  for(counter=0, result=0; counter < rng(1, 2); counter ++) {
  #     result = eval_jaxpr(*args)
  #  }
  # The loop carry is a tuple: (counter, result, args)
  from jax._src.lax import control_flow as lax_control_flow

  avals_out = tuple(v.aval for v in jaxpr.outvars)
  carry_init = (np.int32(0), tuple(map(_dummy_like, avals_out)), args)
  def cond(carry):
    counter, _, _ = carry
    unif = lax_internal.rng_uniform(np.int32(1), np.int32(2), shape=())
    return counter < unif

  def body(carry):
    counter, _, args = carry
    results = core.eval_jaxpr(jaxpr, (), *args)
    return (counter + 1, tuple(results), args)

  carry_res = lax_control_flow.while_loop(cond, body, carry_init)
  return carry_res[1]

def _remat_translation_using_cond(*args, jaxpr: core.Jaxpr):
  # Implements:
  #  if(rng(0, 1) < 2)
  #    return eval_jaxpr(*args)
  #  else:
  #    return 0
  from jax._src.lax import control_flow as lax_control_flow

  avals_out = tuple(v.aval for v in jaxpr.outvars)

  def remat_comp(*args):
    return tuple(core.eval_jaxpr(jaxpr, (), *args))
  def dummy_comp(*args):
    return tuple(map(_dummy_like, avals_out))

  unif = lax_internal.rng_uniform(np.float32(0), np.float32(1), shape=())
  return lax_control_flow.cond(unif < np.float32(2), remat_comp, dummy_comp, *args)

def _remat_lowering(ctx, *args, jaxpr: core.Jaxpr, prevent_cse: bool,
                   differentiated: bool, policy, is_gpu_platform=False):
  jaxpr_args: Sequence[mlir.IrValues]
  if differentiated and prevent_cse:
    # If we're using the loop or cond lowerings, use the slower lower_fun
    # based path.
    if not config.remat_opt_barrier.value:
      return mlir.lower_fun(remat_expansion, multiple_results=True)(
          ctx, *args, jaxpr=jaxpr, prevent_cse=prevent_cse,
          differentiated=differentiated, policy=policy,
          is_gpu_platform=is_gpu_platform)

    arg_types = map(mlir.aval_to_ir_type, ctx.avals_in)
    flat_args = mlir.flatten_ir_values(args)
    barrier_op = hlo.OptimizationBarrierOp(flat_args)
    jaxpr_args = mlir.unflatten_ir_values_like_types(
      barrier_op.results, arg_types)
  else:
    jaxpr_args = args
  outs, tokens_out = mlir.jaxpr_subcomp(
      ctx.module_context, jaxpr, ctx.name_stack.extend('checkpoint'),
      ctx.tokens_in, (), *jaxpr_args, dim_var_values=ctx.dim_var_values)
  ctx.set_tokens_out(tokens_out)
  return outs

mlir.register_lowering(remat_p, _remat_lowering)
mlir.register_lowering(remat_p, partial(_remat_lowering, is_gpu_platform=True),
                       platform="gpu")


def checkpoint_name(x, name):
  return name_p.bind(x, name=name)

name_p.def_impl(lambda x, *, name: x)
name_p.def_abstract_eval(lambda x, *, name: x)

def name_jvp(primals, tangents, *, name):
  (x,), (xdot,) = primals, tangents
  return name_p.bind(x, name=name), xdot  # don't name the tangent value
ad.primitive_jvps[name_p] = name_jvp

mlir.register_lowering(name_p, lambda ctx, x, *, name: [x])

def name_batcher(args, dims, *, name):
  (x,), (d,) = args, dims
  return name_p.bind(x, name=name), d
batching.primitive_batchers[name_p] = name_batcher


@functools.wraps(checkpoint)
def checkpoint_wrapper(
    fun: Callable,
    *,
    concrete: bool = False,
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy: Callable[..., bool] | None = None,
) -> Callable:
  if concrete:
    msg = ("The 'concrete' option to jax.checkpoint / jax.remat is deprecated; "
           "in its place, you can use its `static_argnums` option, and if "
           "necessary the `jax.ensure_compile_time_eval()` context manager.\n"
           "\n"
           "For example, if using `concrete=True` for an `is_training` flag:\n"
           "\n"
           "  from functools import partial\n"
           "\n"
           "  @partial(jax.checkpoint, concrete=True)\n"
           "  def foo(x, is_training):\n"
           "    if is_training:\n"
           "      return f(x)\n"
           "    else:\n"
           "      return g(x)\n"
           "\n"
           "replace it with a use of `static_argnums`:\n"
           "\n"
           "  @partial(jax.checkpoint, static_argnums=(1,))\n"
           "  def foo(x, is_training):\n"
           "    ...\n"
           "\n"
           "If jax.numpy operations need to be performed on static arguments, "
           "we can use the `jax.ensure_compile_time_eval()` context manager. "
           "For example, we can replace this use of `concrete=True`\n:"
           "\n"
           "  @partial(jax.checkpoint, concrete=True)\n"
           "  def foo(x, y):\n"
           "    if y > 0:\n"
           "      return f(x)\n"
           "    else:\n"
           "      return g(x)\n"
           "\n"
           "with this combination of `static_argnums` and "
           "`jax.ensure_compile_time_eval()`:\n"
           "\n"
           "  @partial(jax.checkpoint, static_argnums=(1,))\n"
           "  def foo(x, y):\n"
           "    with jax.ensure_compile_time_eval():\n"
           "      y_pos = y > 0\n"
           "    if y_pos:\n"
           "      return f(x)\n"
           "    else:\n"
           "      return g(x)\n"
           "\n"
           "See https://jax.readthedocs.io/en/latest/jep/11830-new-remat-checkpoint.html\n")
    raise NotImplementedError(msg)
  return checkpoint(fun, prevent_cse=prevent_cse, policy=policy,
                    static_argnums=static_argnums)

# TODO(phawkins): update users to refer to the public name.
_optimization_barrier = lax_internal.optimization_barrier
