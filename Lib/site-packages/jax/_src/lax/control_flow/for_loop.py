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
"""Module for the `for_loop` primitive."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import functools
import operator
from typing import Any, Generic, TypeVar

from jax import lax
from jax.api_util import flatten_fun_nokwargs
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax.tree_util import (tree_flatten, tree_structure, tree_unflatten,
                           treedef_tuple, tree_map, tree_leaves, PyTreeDef)

from jax._src import ad_util
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import linear_util as lu
from jax._src import source_info_util
from jax._src.state.types import (ReadEffect, AbstractRef, StateEffect)
from jax._src.state import discharge as state_discharge
from jax._src.state import primitives as state_primitives
from jax._src.state import utils as state_utils
from jax._src.state import types as state_types
from jax._src.typing import Array
from jax._src.util import (partition_list, merge_lists, safe_map, safe_zip,
                           split_list, split_dict, weakref_lru_cache)
from jax._src.lax.control_flow import loops
from jax._src.lax.control_flow.common import _initial_style_jaxpr
import numpy as np

## JAX utilities

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

## Helpful type aliases
S = TypeVar('S')
T = TypeVar('T')
class Ref(Generic[T]): pass

ref_set = state_primitives.ref_set
ref_get = state_primitives.ref_get
ref_addupdate = state_primitives.ref_addupdate
discharge_state = state_discharge.discharge_state


## `for_loop` implementation

for_p = core.Primitive('for')
for_p.multiple_results = True

### Tracing utilities

def _trace_to_jaxpr_with_refs(f, state_tree: PyTreeDef,
                              state_avals: Sequence[core.AbstractValue]
                              ) -> tuple[core.Jaxpr, list[Any], PyTreeDef]:
  f, out_tree_thunk = flatten_fun_nokwargs(
      lu.wrap_init(f), treedef_tuple((tree_structure(0), state_tree)))
  jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(
      f, state_avals)
  return jaxpr, consts, out_tree_thunk()

def for_loop(nsteps: int | Sequence[int],
             body: Callable[[Array, Ref[S]], None], init_state: S,
             *, reverse: bool = False, unroll: int = 1) -> S:
  """A for-loop combinator that allows read/write semantics in the loop body.

  `for_loop` is a higher-order function that enables writing loops that can be
  staged out in JIT-ted JAX computations. Unlike `jax.lax.fori_loop`, it allows
  mutation in its body using `Ref`s.

  `for_loop` will initialize `Ref`s with the values in `init_state`. Each
  iteration, `body` will be called with the current `Ref`s, which can be read
  from and written to using `ref_get` and `ref_set`.

  `for_loop` is semantically equivalent to the following Python code:

  ```python
  def for_loop(nsteps, body, init_state):
    refs = tree_map(make_ref, init_state)
    for i in range(nsteps):
      body(i, refs)
    return tree_map(ref_get, refs)
  ```

  Args:
    nsteps: Number of iterations
    body: A callable that takes in the iteration number as its first argument
      and `Ref`s corresponding to `init_state` as its second argument.
      `body` is free to read from and write to its `Ref`s. `body` should
       not return anything.
    init_state: A Pytree of JAX-compatible values used to initialize the `Ref`s
      that will be passed into the for loop body.
    unroll: A positive int specifying, in the underlying operation of the
      `for` primitive, how many iterations to unroll within a single iteration
      of a loop. Higher values may speed up execution time at the cost of longer
      compilation time.
  Returns:
    A Pytree of values representing the output of the for loop.
  """
  if unroll < 1:
    raise ValueError("`unroll` must be a positive integer.")
  if isinstance(nsteps, int):
    nsteps = [nsteps]
  if len(nsteps) > 1:
    outer_step, *rest_steps = nsteps
    def wrapped_body(i, refs):
      vals = tree_map(lambda ref: ref_get(ref, ()), refs)
      vals = for_loop(
          rest_steps, functools.partial(body, i), vals, unroll=unroll)
      tree_map(lambda ref, val: ref_set(ref, (), val), refs, vals)
    return for_loop(outer_step, wrapped_body, init_state, unroll=unroll)
  nsteps, = nsteps
  flat_state, state_tree = tree_flatten(init_state)
  state_avals = map(state_utils.val_to_ref_aval, flat_state)
  idx_aval = core.ShapedArray((), dtypes.canonicalize_dtype(np.int64))
  jaxpr, consts, out_tree = _trace_to_jaxpr_with_refs(
      body, state_tree, [idx_aval, *state_avals])
  if out_tree != tree_structure(None):
    raise Exception("`body` should not return anything.")
  jaxpr = state_utils.hoist_consts_to_refs(jaxpr, index=1)
  which_linear = (False,) * (len(consts) + len(flat_state))
  out_flat = for_p.bind(*consts, *flat_state, jaxpr=jaxpr, nsteps=int(nsteps),
                        reverse=reverse, which_linear=which_linear,
                        unroll=unroll)
  # Consts are `Ref`s so they are both inputs and outputs. We remove them from
  # the outputs.
  out_flat = out_flat[len(consts):]
  return tree_unflatten(state_tree, out_flat)

Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')

def scan(f: Callable[[Carry, X], tuple[Carry, Y]],
         init: Carry,
         xs: X | None = None,
         length: int | None = None,
         reverse: bool = False,
         unroll: int = 1) -> tuple[Carry, Y]:
  if not callable(f):
    raise TypeError("scan: f argument should be a callable.")
  if unroll < 1:
    raise ValueError("`unroll` must be a positive integer.")
  xs_flat, xs_tree = tree_flatten(xs)

  try:
    lengths = [x.shape[0] for x in xs_flat]
  except AttributeError as err:
    msg = "scan got value with no leading axis to scan over: {}."
    raise ValueError(
      msg.format(', '.join(str(x) for x in xs_flat
                           if not hasattr(x, 'shape')))) from err

  if length is not None:
    length = int(length)
    if not all(length == l for l in lengths):
      msg = ("scan got `length` argument of {} which disagrees with "
             "leading axis sizes {}.")
      raise ValueError(msg.format(length, [x.shape[0] for x in xs_flat]))
  else:
    unique_lengths = set(lengths)
    if len(unique_lengths) > 1:
      msg = "scan got values with different leading axis sizes: {}."
      raise ValueError(msg.format(', '.join(str(x.shape[0]) for x in xs_flat)))
    elif len(unique_lengths) == 0:
      msg = "scan got no values to scan over and `length` not provided."
      raise ValueError(msg)
    else:
      length, = unique_lengths

  x_shapes = [x.shape[1:] for x in xs_flat]
  x_dtypes = [dtypes.canonicalize_dtype(x.dtype) for x in xs_flat]
  x_avals = tuple(map(core.ShapedArray, x_shapes, x_dtypes))

  def _create_jaxpr(init):
    init_flat = tree_leaves(init)
    _, in_tree = tree_flatten((init, xs))

    carry_avals = tuple(map(core.get_aval, init_flat))
    jaxpr, _, out_tree = _initial_style_jaxpr(
        f, in_tree, carry_avals + x_avals, "scan")
    return jaxpr, out_tree
  jaxpr, out_tree = _create_jaxpr(init)
  _, ys_avals = tree_unflatten(out_tree, jaxpr.out_avals)
  ys = tree_map(lambda aval: lax.full([length, *aval.shape], 0, aval.dtype),
                ys_avals)
  def for_body(i, refs):
    carry_refs, xs_refs, ys_refs = refs
    carry = tree_map(lambda x: x[()], carry_refs)
    x = tree_map(lambda x: x[i], xs_refs)
    carry, y = f(carry, x)
    tree_map(lambda c_ref, c: ref_set(c_ref, (), c), carry_refs, carry)
    tree_map(lambda y_ref, y: ref_set(y_ref, (i,), y), ys_refs, y)
  assert isinstance(length, int)
  init, _, ys = for_loop(length, for_body, (init, xs, ys), reverse=reverse,
                         unroll=unroll)
  return init, ys

@for_p.def_effectful_abstract_eval
def _for_abstract_eval(*avals, jaxpr, **__):
  # Find out for each of the `Ref`s in our jaxpr what effects they have.
  jaxpr_aval_effects = state_types.get_ref_state_effects(
      [v.aval for v in jaxpr.invars], jaxpr.effects)[1:]
  aval_effects = [{eff.replace(input_index=eff.input_index - 1)
                      for eff in effs} for aval, effs
                  in zip(avals, jaxpr_aval_effects)
                  if isinstance(aval, AbstractRef)]
  nonlocal_state_effects = core.join_effects(*aval_effects)
  return list(avals), nonlocal_state_effects

@state_discharge.register_discharge_rule(for_p)
def _for_discharge_rule(in_avals, _, *args: Any, jaxpr: core.Jaxpr,
                        reverse: bool, which_linear: Sequence[bool],
                        nsteps: int, unroll: int
                        ) -> tuple[Sequence[Any | None], Sequence[Any]]:
  out_vals = for_p.bind(*args, jaxpr=jaxpr, reverse=reverse,
                        which_linear=which_linear, nsteps=nsteps,
                        unroll=unroll)
  new_invals = []
  for aval, out_val in zip(in_avals, out_vals):
    new_invals.append(out_val if isinstance(aval, AbstractRef) else None)
  return new_invals, out_vals

def _for_impl(*args, jaxpr, nsteps, reverse, which_linear, unroll):
  del which_linear
  discharged_jaxpr, consts = discharge_state(jaxpr, ())
  def body(i, state):
    i_ = nsteps - i - 1 if reverse else i
    return core.eval_jaxpr(discharged_jaxpr, consts, i_, *state)
  return _for_impl_unrolled(body, nsteps, unroll, *args)

def _for_impl_unrolled(body, nsteps, unroll, *args):
  remainder = nsteps % unroll
  i = lax.full((), 0, dtypes.canonicalize_dtype(np.int64))
  state = list(args)

  for _ in range(remainder):
    state = body(i, state)
    i = i + 1

  def cond(carry):
    i, _ = carry
    return i < nsteps
  def while_body(carry):
    i, state = carry
    for _ in range(unroll):
      state = body(i, state)
      i = i + 1
    return i, state
  _, state = lax.while_loop(cond, while_body, (i, state))
  return state

mlir.register_lowering(for_p, mlir.lower_fun(_for_impl, multiple_results=True))
for_p.def_impl(functools.partial(dispatch.apply_primitive, for_p))

@weakref_lru_cache
def _cached_for_jaxpr(jaxpr):
  discharged_jaxpr, body_consts = discharge_state(jaxpr, ())
  return core.ClosedJaxpr(discharged_jaxpr, body_consts)

def _for_vmap(axis_data, args, dims, *,
              jaxpr, nsteps, reverse, which_linear, unroll):
  init_batched = [d is not batching.not_mapped for d in dims]
  closed_jaxpr = _cached_for_jaxpr(jaxpr)
  batched = init_batched
  for _ in range(len(batched)):
    _, out_batched = batching.batch_jaxpr(
        closed_jaxpr, axis_data, [False] + batched, instantiate=batched)
    if out_batched == batched:
      break
    batched = map(operator.or_, batched, out_batched)
  else:
    raise Exception("Invalid fixpoint")
  args = [batching.broadcast(x, axis_data.size, 0) if now_bat and not was_bat
          else batching.moveaxis(x, d, 0) if now_bat else x
          for x, d, was_bat, now_bat in zip(args, dims, init_batched, batched)]
  batched_jaxpr_, _ = batching.batch_jaxpr(
      pe.close_jaxpr(jaxpr), axis_data, [False] + batched, [])
  batched_jaxpr, () = batched_jaxpr_.jaxpr, batched_jaxpr_.consts  # TODO consts
  out_flat = for_p.bind(*args, jaxpr=batched_jaxpr, nsteps=nsteps,
                        reverse=reverse, which_linear=which_linear,
                        unroll=unroll)
  return out_flat, [0 if b else batching.not_mapped for b in batched]
batching.fancy_primitive_batchers[for_p] = _for_vmap

def _for_jvp(primals, tangents, *, jaxpr, nsteps, reverse, which_linear,
             unroll):
  nonzero_tangents = [not isinstance(t, ad_util.Zero) for t in tangents]
  # We need to find out which `Ref`s have nonzero tangents after running the
  # for loop. Ordinarily we do this with a fixed point on the body jaxpr but
  # a `for` body jaxpr is stateful and has no outputs. We therefore discharge
  # the state effect from the jaxpr and we will now have a "symmetric" jaxpr
  # where the inputs line up with the outputs. We use this discharged jaxpr
  # for the fixed point.
  closed_jaxpr = _cached_for_jaxpr(jaxpr)
  for _ in range(len(nonzero_tangents)):
    _, out_nonzero_tangents = ad.jvp_jaxpr(
        closed_jaxpr,
        [False] + nonzero_tangents, instantiate=nonzero_tangents)
    if out_nonzero_tangents == nonzero_tangents:
      break
    nonzero_tangents = map(operator.or_, nonzero_tangents, out_nonzero_tangents)
  else:
    raise Exception("Invalid fixpoint")
  tangents = [ad.instantiate_zeros(t) if inst else t
              for t, inst in zip(tangents, nonzero_tangents)]
  tangents = [t for t in tangents if type(t) is not ad_util.Zero]
  closed_jaxpr = pe.close_jaxpr(jaxpr)
  jvp_jaxpr_, _ = ad.jvp_jaxpr(closed_jaxpr, [False] + nonzero_tangents, [])
  jvp_jaxpr, () = jvp_jaxpr_.jaxpr, jvp_jaxpr_.consts  # TODO consts
  jvp_which_linear = which_linear + (True,) * len(tangents)
  out_flat = for_p.bind(*primals, *tangents, jaxpr=jvp_jaxpr,
                        nsteps=nsteps, reverse=reverse,
                        which_linear=jvp_which_linear, unroll=unroll)
  # `out_flat` includes constant inputs into the `for_loop` which are converted
  # into outputs as well. We don't care about these in AD so we throw them out.
  out_primals, out_tangents = split_list(out_flat, [len(primals)])
  out_tangents_iter = iter(out_tangents)
  out_tangents = [next(out_tangents_iter) if nz else ad_util.Zero.from_primal_value(p)
                  for p, nz in zip(out_primals, nonzero_tangents)]
  return out_primals, out_tangents
ad.primitive_jvps[for_p] = _for_jvp


def _partial_eval_jaxpr_custom(jaxpr, in_unknowns, policy):
  # A simple wrapper around `pe.partial_eval_jaxpr_custom` that assumes all
  # inputs are instantiated and doesn't ensure any outputs are unknown or
  # instantiated.
  return pe.partial_eval_jaxpr_custom(
      jaxpr, in_unknowns, [True] * len(in_unknowns), False, False, policy)

_save_everything = lambda *_, **__: True

def _is_read_only(ref_effects: set[StateEffect]) -> bool:
  assert len(ref_effects) > 0
  if len(ref_effects) > 1:
    # Means we must have a write or accum effect so not read-only
    return False
  eff, = ref_effects
  return isinstance(eff, ReadEffect)

def _loop_invariant_outputs(jaxpr: core.Jaxpr) -> list[bool]:
  # Get effects for each of the jaxpr inputs and remove the loop index.
  ref_effects = state_types.get_ref_state_effects(
      [v.aval for v in jaxpr.invars], jaxpr.effects)[1:]
  # We first assume that *read-only `Ref`s* are loop-invariant. We can safely do
  # this because the only way something can be loop-varying is if we write to it
  # at some point. It's *possible* that read-write `Ref`s are loop-invariant but
  # we conservatively assume they aren't.
  loop_invar_refs = [_is_read_only(effs) if effs else True
                     for effs in ref_effects]
  loop_var_refs = map(operator.not_, loop_invar_refs)

  # We'd like to detect if the outputs of the jaxpr are loop-invariant. An
  # output is loop-invariant if it is downstream of only loop-invariant values
  # (seeded by the read-only `Ref`s). If at any point, a loop-varying value
  # interacts with a loop-invariant value, we produce a loop-varying value. We
  # can use `partial_eval` to perform this analysis by treating loop-varying
  # values as "unknown" and loop-invariant values as "known", since when a known
  # and unknown value interact, they produce an unknown value.
  loop_var_inputs = [True, *loop_var_refs]
  _, _, loop_var_outputs, _, _, = _partial_eval_jaxpr_custom(
      jaxpr, loop_var_inputs, _save_everything)
  return map(operator.not_, loop_var_outputs)


def _for_partial_eval(trace: pe.JaxprTrace, *tracers: pe.JaxprTracer,
                      jaxpr: core.Jaxpr, nsteps: int, reverse: bool,
                      which_linear: tuple[bool, ...],
                      unroll: int) -> list[pe.JaxprTracer]:
  num_inputs = len(tracers)
  assert num_inputs == len(jaxpr.invars) - 1
  in_unknowns = [not t.pval.is_known() for t in tracers]
  # We first need to run a fixpoint to determine which of the `Ref`s are unknown
  # after running the for loop. We want to use the jaxpr to determine which
  # `Ref`s are unknown after executing the for loop body given which `Ref`s are
  # unknown before. However, the jaxpr has no outputs. Instead, we discharge
  # the body and run the fixpoint with the discharged jaxpr. We can do this
  # because the outputs of the jaxpr are one-to-one with the inputs.
  discharged_jaxpr, discharged_consts = discharge_state(jaxpr, ())
  discharged_jaxpr = discharged_jaxpr.replace(
      invars=discharged_jaxpr.constvars + discharged_jaxpr.invars,
      constvars=[])
  for _ in range(num_inputs):
    jaxpr_in_unknowns = [False] * len(discharged_consts) + [False, *in_unknowns]
    _, _, out_unknowns, _, _, = pe.partial_eval_jaxpr_custom(
        discharged_jaxpr, jaxpr_in_unknowns, [True] * len(jaxpr_in_unknowns),
          in_unknowns, False, _save_everything)
    out_unknowns = list(out_unknowns)
    if out_unknowns == in_unknowns:
      break
    in_unknowns = map(operator.or_, in_unknowns, out_unknowns)
  else:
    raise Exception("Invalid fixpoint")
  del out_unknowns  # redundant since it's the same as `in_unknowns`
  tracers = tuple(trace.instantiate_const(t) if uk else t
                  for t, uk in zip(tracers, in_unknowns))

  # We use `partial_eval_jaxpr_custom` here because it won't remove effectful
  # primitives like `get`/`set`.
  jaxpr_known_resout, jaxpr_unknown_resin_, uk_out, inst_out, num_res = \
        _partial_eval_jaxpr_custom(jaxpr, [False, *in_unknowns],
                                   _save_everything)
  # # `partial_eval_jaxpr_custom` will give us jaxprs that have hybrid `Ref` and
  # regular valued input/outputs. However, we'd like to bind these jaxprs to a
  # `for`, which expects only `Ref` inputs and no output. We need to convert
  # both of these jaxprs into ones that are compatible with `for`.
  # TODO(sharadmv,mattjj): implement "passthrough" optimization.
  # TODO(sharadmv,mattjj): rematerialize loop-dependent values instead of
  # passing the loop index as a residual

  # `jaxpr_known_resout` is a jaxpr that maps from all the input `Refs`
  # to output residual values (none of them should be `Ref`s). We'll need to
  # convert the output residual values into `Ref`s that are initially empty
  # `Ref`s that are written to at the end of the jaxpr.

  # # Loop-invariant residual optimization
  # Here we are interested in finding out which of the residuals are *not*
  # dependent on the loop index. If a residual is not dependent on the loop
  # index, we don't need add an extra loop dimension we're reading from when we
  # convert it from an output into a write.
  loop_invar_res = _loop_invariant_outputs(jaxpr_known_resout)

  jaxpr_known, res_avals = _convert_outputs_to_writes(nsteps,
                                                      jaxpr_known_resout,
                                                      loop_invar_res)
  # We now run the known jaxpr to obtain our residual values.
  known_tracers, _ = partition_list(in_unknowns, tracers)
  known_vals = [t.pval.get_known() for t in known_tracers]
  empty_res = map(ad_util.zeros_like_aval, res_avals)
  jaxpr_known_args = [*known_vals, *empty_res]
  # We assume the known inputs are nonlinear which is okay to do for AD but not
  # necessarily okay for general partial eval.
  jaxpr_known_which_linear = (False,) * len(jaxpr_known_args)
  out_flat = for_p.bind(*jaxpr_known_args, jaxpr=jaxpr_known, nsteps=nsteps,
                        reverse=reverse, which_linear=jaxpr_known_which_linear,
                        unroll=unroll)
  known_outputs, residuals = split_list(out_flat, [len(known_tracers)])
  residuals = map(trace.new_instantiated_const, residuals)

  # Now we handle the `jaxpr_unknown` that expects residual values as inputs.
  # This jaxpr is the output of `partial_eval_jaxpr_custom` that marks which
  # inputs are actually used.
  # `partial_eval_jaxpr_custom` doesn't remove extra inputs/outputs for you
  # so we use `dce_jaxpr` here to do that.
  jaxpr_unknown_resin, used_inputs = pe.dce_jaxpr(
        jaxpr_unknown_resin_, [], [True] * num_res + [True, *in_unknowns])
  used_res, (used_i,), used_refs = split_list(used_inputs, [num_res, 1])
  assert all(used_res), "All residuals should be used"
  # To make it compatible with `for`, we need to convert those residual values
  # into `Ref`s.
  jaxpr_unknown = _convert_inputs_to_reads(nsteps, len(res_avals),
                                           jaxpr_unknown_resin,
                                           loop_invar_res)
  # Since not all inputs are used in jaxpr_unknown, we filter the input tracers
  # down using the output of `dce_jaxpr`.
  used_and_known = map(operator.and_, used_refs, map(operator.not_, in_unknowns))
  tracers = [trace.instantiate_const(t) if u_and_k else t for t, u_and_k  # type: ignore
             in zip(tracers, used_and_known)]
  _, known_used = partition_list(used_refs, used_and_known)
  _, used_tracers = partition_list(used_refs, tracers)
  _, used_which_linear = partition_list(used_refs, which_linear)
  which_linear_unknown = (False,) * num_res + tuple(used_which_linear)
  unknown_inputs = [*residuals, *used_tracers]
  # Outputs match inputs so we construct output tracers that look like the input
  # tracers.
  res_ref_unknown_outputs = [
      pe.JaxprTracer(trace, pe.PartialVal.unknown(t.aval), None)
      for t in unknown_inputs]
  name_stack = source_info_util.current_name_stack()[len(trace.name_stack):]
  source = source_info_util.current().replace(name_stack=name_stack)

  assert len(unknown_inputs) == len(res_ref_unknown_outputs)
  assert len(unknown_inputs) == len(jaxpr_unknown.invars) - 1
  eqn = pe.new_eqn_recipe(unknown_inputs, res_ref_unknown_outputs,
                          for_p, dict(jaxpr=jaxpr_unknown, nsteps=nsteps,
                                      reverse=reverse,
                                      which_linear=which_linear_unknown,
                                      unroll=unroll),
                          core.no_effects, source)
  for t in res_ref_unknown_outputs: t.recipe = eqn
  _, unknown_outputs = split_list(res_ref_unknown_outputs, [num_res])
  unknown_outputs, _ = partition_list(known_used, unknown_outputs)
  return merge_lists(in_unknowns, known_outputs, unknown_outputs)
pe.custom_partial_eval_rules[for_p] = _for_partial_eval

def _for_partial_eval_custom(saveable, in_unknowns, in_inst, eqn):
  jaxpr, nsteps, reverse, which_linear, unroll = split_dict(
      eqn.params, ["jaxpr", "nsteps", "reverse", "which_linear", "unroll"])
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
  out_unknowns, out_inst =  in_unknowns, in_inst
  for _ in range(num_inputs):
    jaxpr_in_unknowns = [False] * len(discharged_consts) + [False, *in_unknowns]
    _, _, out_unknowns, out_inst, _, = pe.partial_eval_jaxpr_custom(
        discharged_jaxpr, jaxpr_in_unknowns, True,
          ensure_out_unknowns=in_unknowns, ensure_out_inst=True,
          saveable=saveable)
    out_unknowns = list(out_unknowns)
    if out_unknowns == in_unknowns:
      break
    in_unknowns = map(operator.or_, in_unknowns, out_unknowns)
  else:
    if num_inputs > 0: raise Exception("Invalid fixpoint")
  del out_unknowns # Redundant since it's the same as `in_unknowns`
  new_inst = [x for x, inst in zip(eqn.invars, in_inst)
              if type(x) is core.Var and not inst]
  in_inst = [True] * len(eqn.invars)

  # We use `partial_eval_jaxpr_custom` here because it won't remove effectful
  # primitives like `get`/`set`.
  jaxpr_known_resout, jaxpr_staged_resin_, _, _, num_res = \
        pe.partial_eval_jaxpr_custom(jaxpr, [False, *in_unknowns],
            [True, *in_inst], [], [], saveable)

  # `partial_eval_jaxpr_custom` will give us jaxprs that have hybrid `Ref` and
  # non-Ref input/outputs. However, we'd like to bind these jaxprs to a
  # `for`, which expects only `Ref` inputs and no output. We need to convert
  # both of these jaxprs into ones that are compatible with `for`.
  # TODO(sharadmv,mattjj): implement "passthrough" optimization.
  # TODO(sharadmv,mattjj): rematerialize loop-dependent values instead of
  # passing the loop index as a residual

  # `jaxpr_known_resout` is a jaxpr that maps from all the input `Refs`
  # to output residual values (none of them should be `Ref`s). We'll need to
  # convert the output residual values into `Ref`s that are initially empty
  # `Ref`s that are written to at the end of the jaxpr.

  # # Loop-invariant residual optimization
  # Here we are interested in finding out which of the residuals are *not*
  # dependent on the loop index. If a residual is not dependent on the loop
  # index, we don't need add an extra loop dimension we're reading from when we
  # convert it from an output into a write.
  loop_invar_res = _loop_invariant_outputs(jaxpr_known_resout)

  jaxpr_known, res_avals = _convert_outputs_to_writes(nsteps,
                                                      jaxpr_known_resout,
                                                      loop_invar_res)

  known_invars, _ = partition_list(in_unknowns, eqn.invars)
  known_outvars, _ = partition_list(in_unknowns, eqn.outvars)
  newvar = core.gensym()
  resvars = map(newvar, res_avals)

  @lu.wrap_init
  def known(*known_vals):
    empty_res = map(ad_util.zeros_like_aval, res_avals)
    jaxpr_known_args = [*known_vals, *empty_res]
    jaxpr_known_which_linear = (False,) * len(jaxpr_known_args)
    return for_p.bind(*jaxpr_known_args, jaxpr=jaxpr_known, nsteps=nsteps,
                      reverse=reverse, which_linear=jaxpr_known_which_linear,
                      unroll=unroll)
  call_jaxpr_, _, call_jaxpr_consts, () = pe.trace_to_jaxpr_dynamic(
      known, [v.aval for v in known_invars])
  call_jaxpr = core.ClosedJaxpr(call_jaxpr_, call_jaxpr_consts)
  eqn_known = pe.new_jaxpr_eqn(known_invars, [*known_outvars, *resvars],
                               core.closed_call_p, dict(call_jaxpr=call_jaxpr),
                               call_jaxpr.effects, eqn.source_info)

  jaxpr_staged = _convert_inputs_to_reads(nsteps, len(res_avals),
                                          jaxpr_staged_resin_,
                                          loop_invar_res)
  which_linear_unknown = (False,) * num_res + tuple(which_linear)
  params_staged = dict(eqn.params, jaxpr=jaxpr_staged, reverse=reverse,
                                   nsteps=nsteps,
                                   which_linear=which_linear_unknown,
                                   unroll=unroll)

  @lu.wrap_init
  def staged(*res_and_refs):
    out_flat = for_p.bind(*res_and_refs, **params_staged)
    _, ans = split_list(out_flat, [num_res])
    _, ans = partition_list(out_inst, ans)
    return ans
  call_jaxpr_, _, call_jaxpr_consts, () = pe.trace_to_jaxpr_dynamic(
      staged, [v.aval for v in [*resvars, *eqn.invars]])
  assert len(jaxpr_staged.invars) - 1 == len(call_jaxpr_.invars)
  call_jaxpr = core.ClosedJaxpr(call_jaxpr_, call_jaxpr_consts)
  _, outvars = partition_list(out_inst, eqn.outvars)
  eqn_staged = pe.new_jaxpr_eqn([*resvars, *eqn.invars], outvars,
                               core.closed_call_p, dict(call_jaxpr=call_jaxpr),
                               call_jaxpr.effects, eqn.source_info)
  new_vars = [*new_inst, *resvars]
  return eqn_known, eqn_staged, in_unknowns, out_inst, new_vars

pe.partial_eval_jaxpr_custom_rules[for_p] = _for_partial_eval_custom

def _convert_outputs_to_writes(
    nsteps: int, jaxpr: core.Jaxpr, loop_invar_res: Sequence[bool]
    ) -> tuple[core.Jaxpr, list[core.ShapedArray]]:
  assert not jaxpr.constvars, "Jaxpr shouldn't have constvars."

  in_avals = [v.aval for v in jaxpr.invars]  # [i, *orig_ref_avals]
  @lu.wrap_init
  def eval_jaxpr(i, *refs):
    # We split the refs into the original input refs and the dummy residual
    # refs.
    orig_refs, residual_refs = split_list(refs, [len(in_avals) - 1])
    residual_vals = core.eval_jaxpr(jaxpr, (), i, *orig_refs)
    for res_ref, res_val, loop_invar in zip(residual_refs, residual_vals,
                                            loop_invar_res):
      if loop_invar:
        res_ref[()] = res_val
      else:
        res_ref[i] = res_val
    return []
  # TODO(mattjj, sharadmv): better handling of tokens, which don't have shape/dtype
  res_ref_avals: list[core.AbstractValue] = [
      AbstractRef(v.aval) if loop_invar else  # pytype: disable=attribute-error
      AbstractRef(core.ShapedArray((nsteps, *v.aval.shape),  # pytype: disable=attribute-error
                  v.aval.dtype))  # pytype: disable=attribute-error
      for v, loop_invar in zip(jaxpr.outvars, loop_invar_res)]
  jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(
      eval_jaxpr, [*in_avals, *res_ref_avals])
  assert not consts
  return jaxpr, [core.ShapedArray(a.shape, a.dtype) for a in res_ref_avals]  # pytype: disable=attribute-error

def _convert_inputs_to_reads(
    nsteps: int, num_res: int, jaxpr: core.Jaxpr,
    loop_invar_res: Sequence[bool]) -> core.Jaxpr:
  assert not jaxpr.constvars, "Jaxpr should not have constvars"

  @lu.wrap_init
  def eval_jaxpr(i, *refs):
    residual_refs, orig_refs = split_list(refs, [num_res])
    residual_vals = [r[()] if loop_invar else r[i] for r, loop_invar
                     in zip(residual_refs, loop_invar_res)]
    () = core.eval_jaxpr(jaxpr, (), *residual_vals, i, *orig_refs)
    return []

  res_val_avals, (i_aval,), orig_ref_avals = \
      split_list([v.aval for v in jaxpr.invars], [num_res, 1])
  res_ref_avals: list[core.AbstractValue] = [
      AbstractRef(aval) if loop_invar else  # pytype: disable=attribute-error
      AbstractRef(core.ShapedArray((nsteps, *aval.shape),  # pytype: disable=attribute-error
                  aval.dtype))  # pytype: disable=attribute-error
      for aval, loop_invar in zip(res_val_avals, loop_invar_res)]

  jaxpr, _, (), () = pe.trace_to_jaxpr_dynamic(
      eval_jaxpr, [i_aval, *res_ref_avals, *orig_ref_avals])
  return jaxpr

def transpose_jaxpr(jaxpr: core.Jaxpr, which_linear: list[bool]) -> core.Jaxpr:
  def trans(i, *args):
    # First we want to run the computation to read all the residual refs. We can
    # do that by using partial evaluation with all linear inputs unknown.
    res_jaxpr, tangent_jaxpr_, *_ = \
        _partial_eval_jaxpr_custom(jaxpr, [False, *which_linear],
                                   _save_everything)
    res_args = [x for x, lin in zip(args, which_linear) if not lin]
    res = core.eval_jaxpr(res_jaxpr, (), i, *res_args)

    # Now that we have residual values, we run the tangent jaxpr. It takes as
    # input the residuals, the loop index, and all the refs (at least, the ones
    # that are used in the body). Luckily, `tangent_jaxpr_` has all known and
    # unknown inputs!
    tangent_jaxpr, used = pe.dce_jaxpr(tangent_jaxpr_, [])
    used_res, (used_i,), used_ct = split_list(used, [len(res), 1])
    primals_args = [*(r for u, r in zip(used_res, res) if u)]
    if used_i:
      primals_args = [*primals_args, i]
    ct_args = [x for x, u in zip(args, used_ct) if u]
    ad.backward_pass(tangent_jaxpr, False, (), (*primals_args, *ct_args), ())
    return []
  jaxpr_trans, _, _, () = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(trans), [v.aval for v in jaxpr.invars])
  return jaxpr_trans

def _for_transpose(in_cts, *args, jaxpr, nsteps, reverse, which_linear, unroll):
  # if any in_ct is nonzero, we definitely want it in args_ (and the
  # corresponding x in args could be an undefined primal, but doesn't have to be)
  # for non-res stuff:
  #   getting and setting => (nonzero ct, UndefinedPrimal arg)
  #   just setting =>        (nonzero ct, not UndefinedPrimal, dummy value)
  #   just getting =>        (zero ct   , UndefinedPrimal arg)
  # for res stuff:
  #                          (zero ct   , not UndefinedPrimal)
  args_ = []
  which_linear_transpose = []
  for x, ct in zip(args, in_cts):
    if   type(ct) is     ad_util.Zero and not ad.is_undefined_primal(x):
      # this is a residual, take x!
      args_.append(x)
      which_linear_transpose.append(False)
    elif type(ct) is     ad_util.Zero and     ad.is_undefined_primal(x):
      # the loop was 'just getting', plug in a zero
      args_.append(ad_util.zeros_like_aval(x.aval))
      which_linear_transpose.append(False)
    elif type(ct) is not ad_util.Zero and not ad.is_undefined_primal(x):
      # the loop was 'just setting', grab that cotangent! x is dummy
      args_.append(ct)
      which_linear_transpose.append(False)
    elif type(ct) is not ad_util.Zero and     ad.is_undefined_primal(x):
      # the loop was 'getting and setting', grab that cotangent!
      args_.append(ct)
      which_linear_transpose.append(True)

  jaxpr_transpose = transpose_jaxpr(jaxpr, which_linear)
  assert len(args_) == len(jaxpr_transpose.invars) - 1
  all_outs = for_p.bind(*args_, jaxpr=jaxpr_transpose, nsteps=nsteps,
                        reverse=not reverse,
                        which_linear=tuple(which_linear_transpose),
                        unroll=unroll)
  ct_outs = [ct if ad.is_undefined_primal(x) else None
             for x, ct in zip(args, all_outs)]
  return ct_outs
ad.primitive_transposes[for_p] = _for_transpose

### Testing utility

def discharged_for_loop(nsteps, body, init_state, *, reverse: bool = False):
  """A `for_loop` implementation that discharges its body right away.

  Potentially useful for testing and benchmarking.
  """
  flat_state, state_tree = tree_flatten(init_state)
  state_avals = map(state_utils.val_to_ref_aval, flat_state)
  idx_aval = core.ShapedArray((), dtypes.canonicalize_dtype(np.int64))
  jaxpr, consts, out_tree = _trace_to_jaxpr_with_refs(
      body, state_tree, [idx_aval, *state_avals])
  if out_tree != tree_structure(None):
    raise Exception("`body` should not return anything.")
  discharged_jaxpr, discharged_consts = discharge_state(jaxpr, consts)

  def fori_body(i, carry):
    i = lax.convert_element_type(i, dtypes.canonicalize_dtype(np.int64))
    if reverse:
      i = nsteps - i - 1
    out_flat = core.eval_jaxpr(discharged_jaxpr, discharged_consts,
                               i, *carry)
    return out_flat
  out_flat = loops.fori_loop(0, nsteps, fori_body, flat_state)
  return tree_unflatten(state_tree, out_flat)

def run_state(f, init_state):
  @functools.wraps(f)
  def wrapped_body(_, *args):
    return f(*args)
  return for_loop(1, wrapped_body, init_state)
