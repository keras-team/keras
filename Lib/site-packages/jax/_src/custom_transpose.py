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

from __future__ import annotations

from collections.abc import Callable
import functools
from typing import Any

from jax._src import ad_util
from jax._src import api_util
from jax._src import core
from jax._src import custom_api_util
from jax._src import linear_util as lu
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import util
from jax._src.interpreters import ad
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import xla
from jax._src.tree_util import (tree_flatten, tree_leaves, tree_map,
                                tree_structure, treedef_tuple, tree_unflatten)


source_info_util.register_exclusion(__file__)
traceback_util.register_exclusion(__file__)


map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


### bespoke linear_util and api_util deviations

class StoreEqual(lu.Store):
  """Stores an unchanging value. Checks empty reads and unequal overwrites."""
  def store(self, val):
    if self._val is not lu._EMPTY_STORE_VALUE and val != self._val:
      raise lu.StoreException(
          f"Store assignment mismatch, from {self._val} to {val}")
    self._val = val

@util.curry
def transformation_with_aux(
    gen, fun: lu.WrappedFun, *gen_static_args) -> tuple[lu.WrappedFun, Any]:
  out_store = StoreEqual()
  out_thunk = lambda: out_store.val
  return fun.wrap(gen, gen_static_args, out_store), out_thunk

flatten_fun_nokwargs = transformation_with_aux(
    api_util.flatten_fun_nokwargs.args[0])


### api

@custom_api_util.register_custom_decorator_type
class custom_transpose:
  fun: Callable
  transpose: Callable | None = None

  def __init__(self, fun: Callable):
    functools.update_wrapper(self, fun)
    self.fun = fun

  __getattr__ = custom_api_util.forward_attr

  def def_transpose(self, transpose: Callable):
    self.transpose = transpose
    return transpose

  @traceback_util.api_boundary
  def __call__(self, out_types, res_arg, lin_arg):
    _, res_tree = tree_flatten(res_arg)
    _, lin_tree = tree_flatten(lin_arg)
    args_flat, in_tree = tree_flatten((res_arg, lin_arg))

    # TODO(frostig,mattjj): check that out_trees match
    # TODO(frostig,mattjj): could, and should, we avoid flattening
    # self.fun at this point?

    flat_fun, out_tree2 = flatten_fun_nokwargs(lu.wrap_init(self.fun), in_tree)
    out_types_flat, out_tree = tree_flatten(out_types)
    out_flat = custom_transpose_p.bind(flat_fun, *args_flat,
                                       transpose=self.transpose,
                                       out_types=out_types_flat,
                                       lin_tree=lin_tree,
                                       res_tree=res_tree,
                                       out_tree=out_tree)
    return tree_unflatten(out_tree, out_flat)


### utils

def tree_fill(x, treedef):
  return tree_unflatten(treedef, [x] * treedef.num_leaves)

def tree_fill_like(x, tree):
  return tree_fill(x, tree_structure(tree))

def tree_broadcast(full_treedef, tree, is_leaf=None):
  full_tree = tree_fill(0, full_treedef)
  return tree_map(tree_fill_like, tree, full_tree, is_leaf=is_leaf)

def is_treedef_prefix(entire, prefix):
  entire = tree_fill(0, entire)
  prefix = tree_fill(0, prefix)
  try:
    tree_map(lambda x, y: x, prefix, entire)
  except ValueError:
    return False
  return True

def rule_name(rule):
  return getattr(rule, '__name__', '<unnamed transpose rule>')

def check_transpose_rule_trees(rule, lin_tree, rule_out_tree):
  if not is_treedef_prefix(lin_tree, rule_out_tree):
    if hasattr(rule, '_transpose_type_error'):
      raise rule._transpose_type_error(lin_tree, rule_out_tree)
    else:
      raise TypeError(
          'structure of custom transpose rule\'s output does not prefix-match '
          'structure of primal function\'s linear inputs under '
          f'custom transpose rule ({rule_name(rule)}).\n'
          f'Transpose rule output: {rule_out_tree}\n'
          f'Linear primal inputs: {lin_tree}')

def make_transpose_from_thunk(thunk, lin_tree):
  transpose_jaxpr, transpose_consts = thunk()
  transpose_jaxpr = core.ClosedJaxpr(
      pe.convert_constvars_jaxpr(transpose_jaxpr), ())
  def transpose(res_arg, ct_out):
    args_flat = tree_leaves((res_arg, ct_out))
    ct_ins = core.jaxpr_as_fun(transpose_jaxpr)(*transpose_consts, *args_flat)
    return tree_unflatten(lin_tree, ct_ins)
  return transpose


### custom_transpose primitive and rules

class CustomTransposePrimitive(core.Primitive):
  call_primitive = False
  map_primitive = False
  multiple_results = True

  def bind_with_trace(self, trace, call_args, params):
    call, tracers = call_args[0], call_args[1:]
    return trace.process_custom_transpose(self, call, tracers, **params)

  # TODO(frostig,mattjj): consider keeping `call` as a named parameter
  # instead of following this "call primitive" convention.
  def get_bind_params(self, params):
    assert 'call_jaxpr' in params
    assert 'transpose_jaxpr_thunk' in params
    new_params = dict(params)
    new_params['transpose'] = make_transpose_from_thunk(
        new_params.pop('transpose_jaxpr_thunk'),
        new_params['lin_tree'])
    call = lu.wrap_init(core.jaxpr_as_fun(new_params.pop('call_jaxpr')))
    return [call], new_params


# TODO(frostig,mattjj): reinstate checks
def custom_transpose_typecheck(_, *in_atoms, out_types, **params):
  del in_atoms, params
  return out_types, core.no_effects


def custom_transpose_transpose_rule(
    cts, *args, out_types, res_tree, lin_tree, out_tree, **params):

  if 'transpose_jaxpr_thunk' in params:
    assert 'call_jaxpr' in params
    transpose = make_transpose_from_thunk(
        params['transpose_jaxpr_thunk'], lin_tree)
  else:
    assert 'call' in params
    transpose = params['transpose']

  call_in_tree = treedef_tuple((res_tree, lin_tree))

  # TODO(frostig,mattjj): `lin_arg` indicates the inputs with respect
  # to which we are transposing (via `ad.is_undefined_primal`).
  # Consider passing this information to the custom transpose rule?

  res_arg, lin_arg = tree_unflatten(call_in_tree, args)
  del lin_arg
  assert all(not ad.is_undefined_primal(x) for x in tree_leaves(res_arg))

  cts = [ad_util.zeros_like_aval(ct.aval) if type(ct) is ad_util.Zero else ct
         for ct in cts]
  ct_out = tree_unflatten(out_tree, cts)
  ct_lin = transpose(res_arg, ct_out)
  check_transpose_rule_trees(transpose, lin_tree, tree_structure(ct_lin))
  ct_lin_flat, _ = tree_flatten(
      tree_broadcast(lin_tree, ct_lin, is_leaf=lambda x: x is None),
      is_leaf=lambda x: x is None)
  return [None] * len(tree_leaves(res_arg)) + ct_lin_flat


def custom_transpose_lowering(*args, call_jaxpr, **params):
  return core.jaxpr_as_fun(call_jaxpr)(*args)


custom_transpose_p = CustomTransposePrimitive('custom_transpose_call')
core.custom_typechecks[custom_transpose_p] = custom_transpose_typecheck
ad.primitive_transposes[custom_transpose_p] = custom_transpose_transpose_rule
mlir.register_lowering(
    custom_transpose_p,
    mlir.lower_fun(custom_transpose_lowering, multiple_results=True))
xla.register_initial_style_primitive(custom_transpose_p)
