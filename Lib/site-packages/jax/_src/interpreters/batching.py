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

import collections
from collections.abc import Callable, Sequence
import dataclasses
from functools import partial
from typing import Any, Union

import numpy as np

import jax
from jax._src import config
from jax._src import core
from jax._src import source_info_util
from jax._src import linear_util as lu
from jax._src.ad_util import (Zero, instantiate, SymbolicZero,
                              replace_rule_output_symbolic_zeros,
                              add_jaxvals, add_jaxvals_p)
from jax._src.core import Trace, Tracer, TraceTag, AxisName
from jax._src.interpreters import partial_eval as pe
from jax._src.tree_util import (tree_unflatten, tree_flatten,
                                register_pytree_node)
from jax._src.typing import Array
from jax._src.util import (unzip2, safe_map, safe_zip, split_list,
                           canonicalize_axis, moveaxis, as_hashable_function,
                           curry, memoize, weakref_lru_cache)


map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip


# Jumbles

# i:(Fin 3) => f32[[3, 1, 4].i]
@dataclasses.dataclass(frozen=True)
class JumbleTy:
  binder: core.Var
  length: int | Tracer | core.Var
  elt_ty: core.DShapedArray
  def __repr__(self) -> str:
    return f'Var{id(self.binder)}:{self.length} => {self.elt_ty}'
  replace = dataclasses.replace

# [3, 1, 4].i
@dataclasses.dataclass(frozen=True)
class IndexedAxisSize:
  idx: core.Var
  lengths: Array | core.Var | Tracer
  def __repr__(self) -> str:
    return f'{self.lengths}.Var{id(self.idx)}'
  replace = dataclasses.replace

# Jumble(aval=a:3 => f32[[3 1 4].a],
#        data=Array([0., 1., 2., 0., 0., 1., 2., 3.], dtype=float32))
@dataclasses.dataclass(frozen=True)
class Jumble:
  aval: JumbleTy
  data: Array

# To vmap over a jumble, one must specify the axis as JumbleAxis.
class JumbleAxis: pass
jumble_axis = JumbleAxis()

# As a temporary measure before we have more general JITable / ADable interfaces
# (analogues to vmappable), to enable Jumbles to be used with other
# transformations and higher-order primitives (primarily jit, though also grad
# with allow_int=True) we register them as pytrees.
# TODO(mattjj): add JITable / ADable interfaces, remove this pytree registration
def _jumble_flatten(jumble):
  lengths = []
  new_shape = [lengths.append(d.lengths) or d.replace(lengths=len(lengths))
               if type(d) is IndexedAxisSize else d
               for d in jumble.aval.elt_ty.shape]
  elt_ty = jumble.aval.elt_ty.update(shape=tuple(new_shape))
  aval = jumble.aval.replace(elt_ty=elt_ty)
  return (lengths, jumble.data), aval


def _ragged_axis_parts(dim: RaggedAxis) -> tuple[int, int, int]:
  stacked_axis = dim.stacked_axis
  ragged_axes = dim.ragged_axes
  if len(ragged_axes) != 1:
    raise ValueError('Multiple ragged axes not yet implemented.')
  ragged_axis_dim = ragged_axes[0][0]
  ragged_axis_length = ragged_axes[0][1]
  return stacked_axis, ragged_axis_dim, ragged_axis_length


def _jumble_unflatten(aval, x):
  lengths, data = x
  new_shape = [d.replace(lengths=lengths[d.lengths - 1])
               if type(d) is IndexedAxisSize else d
               for d in aval.elt_ty.shape]
  elt_ty = aval.elt_ty.update(shape=tuple(new_shape))
  aval = aval.replace(elt_ty=elt_ty)
  return Jumble(aval, data)
register_pytree_node(Jumble, _jumble_flatten, _jumble_unflatten)

def _jumble_result(axis_size, stacked_axis, ragged_axes, x):
  binder = core.Var('', core.ShapedArray((), np.dtype('int32')))
  if stacked_axis != 0:
    raise NotImplementedError  # TODO Transpose x so the stacked axis is axis 0
  shape = list(x.shape)
  del shape[0]
  for ragged_axis, segment_lens in ragged_axes:
    shape[ragged_axis-1] = IndexedAxisSize(binder, segment_lens)
  elt_ty = core.DShapedArray(tuple(shape), x.dtype, x.weak_type)
  return Jumble(JumbleTy(binder, axis_size, elt_ty), x)


@dataclasses.dataclass(frozen=True)
class RaggedAxis:
  stacked_axis: int
  # For each axis, we store its index and the corresponding segment lengths.
  # For example, the jumble i:(Fin 3) => f32[lens1.i, 7, lens2.i]
  # would be represented with ragged_axes = [(1, lens1), (3, lens2)]
  ragged_axes: tuple[tuple[int, Any], ...]

  @property
  def size(self):
    # TODO(mattjj, axch): All the segment lengths arrays better be the
    # same length!
    return len(self.ragged_axes[0][1])

  def move_stacked_axis(self: RaggedAxis, dst: int) -> RaggedAxis:
    # Assumes that all stored and incoming axes are already canonicalized
    def move_axis(ax):
      if self.stacked_axis > ax and ax >= dst:
        return ax + 1
      if self.stacked_axis < ax and ax <= dst:
        return ax - 1
      return ax
    new_axes = tuple((move_axis(ax), sizes) for ax, sizes in self.ragged_axes)
    return RaggedAxis(dst, new_axes)


def transpose_ragged_axes(dim: RaggedAxis, perm: tuple[int, ...]) -> RaggedAxis:
  new_ragged_axes = []
  for idx, old_idx in enumerate(perm):
    for ax, size in dim.ragged_axes:
      if old_idx == ax:
        new_ragged_axes.append((idx, size))
        break
  return _sorted_ragged_axis(dim.stacked_axis, new_ragged_axes)

def _sorted_ragged_axis(stacked_axis, ragged_axes):
  return RaggedAxis(stacked_axis, tuple(sorted(ragged_axes, key=lambda p: p[0])))

def make_batch_axis(
    ndim: int,
    stacked_axis: int,
    ragged_axes: list[tuple[int, Array | core.Var]],
) -> int | RaggedAxis:
  if ragged_axes:
    canonical = [(canonicalize_axis(ax, ndim), sz) for ax, sz in ragged_axes]
    return _sorted_ragged_axis(canonicalize_axis(stacked_axis, ndim), canonical)
  else:
    return canonicalize_axis(stacked_axis, ndim)

def bdim_as_shape(
    bdim: int | RaggedAxis, data_shape: core.Shape) -> core.Shape:
  if isinstance(bdim, RaggedAxis):
    result = list(data_shape)
    binder = core.Var('', core.ShapedArray((), np.dtype('int32')))
    for ragged_axis, segment_lens in bdim.ragged_axes:
      result[ragged_axis] = IndexedAxisSize(binder, segment_lens)
    return tuple(result)
  else:
    return data_shape

def shape_as_bdim(
    stacked_axis: int, data_shape: core.Shape) -> int | RaggedAxis:
  # This assumes that there is only one binder in the data_shape.
  ragged_axes = [(i, size.lengths) for i, size in enumerate(data_shape)
                 if isinstance(size, IndexedAxisSize)]
  return make_batch_axis(len(data_shape), stacked_axis, ragged_axes)


def _update_annotation(
    f: lu.WrappedFun, orig_type: core.InputType | None,
    axis_size: core.AxisSize, axis_name: AxisName,
    explicit_in_dims: Sequence[int | RaggedAxis | None],
    segment_lens: Sequence[Array],
  ) -> lu.WrappedFun:
  if orig_type is None: return f
  # By convention, `explicit_in_dims` only accounts for explicit arguments.
  assert len(explicit_in_dims) == sum(explicit for _, explicit in orig_type)
  # We need to:
  #  * if `axis_size` is dynamic, add a new implicit binder (type) for it;
  #  * for each element of `segment_lengths`, add a new explicit binder for it;
  #  * drop other implicit binders, replacing DBIdx which refer to them with
  #    Name objects;
  #  * for each (aval, in_dim) pair: if int-valued in_dim, add batch axis (int
  #    size if `axis_size` is int, otherwise Name); if RaggedAxis-valued in_dim,
  #    add batch axis (int if corresponding segment_lengths is concrete, Name if
  #    not);
  #  * generate full in_type with implicit args too.

  class Name:
    def __init__(self, a): self.a = a
  names = [Name(a) for a, _  in orig_type]
  avals = [a.update(shape=tuple(names[d.val] if type(d) is pe.DBIdx else d
                                for d in a.shape))
           if type(a) is core.DShapedArray else a for a, e in orig_type if e]

  new_avals = [core.get_aval(s) for s in segment_lens]
  sz = Name(axis_size.aval) if isinstance(axis_size, Tracer) else axis_size
  for a, d in zip(avals, explicit_in_dims):
    if isinstance(d, RaggedAxis):
      raise NotImplementedError
    else:
      new_avals.append(core.unmapped_aval(sz, axis_name, d, a))  # type: ignore

  mentioned = {d for a in new_avals if type(a) is core.DShapedArray
               for d in a.shape if type(d) is Name}
  expl_names = set(map(Name, new_avals))
  impl_names = mentioned - expl_names  # type: ignore
  impl_part = [(n.a, False) for n in impl_names]  # type: ignore
  name_map = {n: pe.DBIdx(i) for i, n in enumerate((*impl_names, *expl_names))}
  expl_part = [(a.update(shape=tuple(name_map.get(d, d) for d in a.shape))
                if type(a) is core.DShapedArray else a, True) for a in new_avals]
  return lu.annotate(f, (*impl_part, *expl_part))

### vmappable typeclass

Vmappable = Any
Elt = Any
MapSpec = Any
AxisSize = Any
GetIdx = Callable[[], Tracer]  # TODO(mattjj): revise this laziness
ToEltHandler = Callable[[Callable, GetIdx, Vmappable, MapSpec], Elt]
FromEltHandler = Callable[[Callable, AxisSize, Elt, MapSpec], Vmappable]
MakeIotaHandler = Callable[[AxisSize], Array]

def to_elt(trace: Trace, get_idx: GetIdx, x: Vmappable, spec: MapSpec) -> Elt:
  handler = to_elt_handlers.get(type(x))
  if handler:
    return handler(partial(to_elt, trace, get_idx), get_idx, x, spec)
  elif type(x) is Jumble:
    if spec is not jumble_axis:
      raise TypeError("jumble input without using jumble_axis in_axes spec")
    ias: IndexedAxisSize  # Not present in the AxisSize union in core.py
    (d, ias), = ((i, sz)  # type: ignore
                 for i, sz in enumerate(x.aval.elt_ty.shape)
                 if type(sz) is IndexedAxisSize)
    batch_axis = make_batch_axis(x.data.ndim, 0, [(d+1, ias.lengths)])
    return BatchTracer(trace, x.data, batch_axis)
  elif isinstance(spec, int) or spec is None:
    spec = spec and canonicalize_axis(spec, len(np.shape(x)))
    return (BatchTracer(trace, x, spec, source_info_util.current())
            if spec is not None else x)
  else:
    if isinstance(trace, BatchTrace) and isinstance(spec, JumbleAxis):
      # TODO(mvoz): A vaguely questionable assumption that it is always
      # sound to have a 0 axis here. This is true for the current use cases
      # and comes from how we handle intermediary products of jumbles in
      # vmap.
      return BatchTracer(trace, x, 0, source_info_util.current())
    # TODO(mvoz): This is a terrible place to fall into if you pass
    # a non jumble type in, make it clearer what went wrong.
    assert False, f'Unexpected type in ELT? {type(x)}'


to_elt_handlers: dict[type, ToEltHandler] = {}

def from_elt(trace: BatchTrace, axis_size: AxisSize, i: int,
             x: Elt, spec: MapSpec) -> Vmappable:
  handler = from_elt_handlers.get(type(x))
  if handler:
    def _cont(axis_size, elt, axis):
      return from_elt(trace, axis_size, i, elt, axis)
    return handler(_cont, axis_size, x, spec)
  val, bdim = trace.to_batch_info(x)
  if type(bdim) is RaggedAxis:
    if spec is not jumble_axis:
      # TODO(mattjj): improve this error message
      raise TypeError("ragged output without using jumble_axis out_axes spec")
    return _jumble_result(axis_size, bdim.stacked_axis, bdim.ragged_axes, val)
  else:
    try:
      return matchaxis(trace.axis_data.name, axis_size, bdim, spec, val)
    except SpecMatchError:
      raise SpecMatchError(i, x.batch_dim, spec) from None
from_elt_handlers: dict[type, FromEltHandler] = {}

def make_iota(axis_size: AxisSize) -> Array:
  handler = make_iota_handlers.get(type(axis_size))
  if handler:
    return handler(axis_size)
  else:
    return jax.lax.iota('int32', int(axis_size))
make_iota_handlers: dict[type, MakeIotaHandler] = {}

def register_vmappable(data_type: type, spec_type: type, axis_size_type: type,
                       to_elt: Callable, from_elt: Callable,
                       make_iota: Callable | None):
  vmappables[data_type] = (spec_type, axis_size_type)
  spec_types.add(spec_type)
  to_elt_handlers[data_type] = to_elt
  from_elt_handlers[data_type] = from_elt
  if make_iota: make_iota_handlers[axis_size_type] = make_iota
vmappables: dict[type, tuple[type, type]] = {}
spec_types: set[type] = {JumbleAxis}

def unregister_vmappable(data_type: type) -> None:
  spec_type, axis_size_type = vmappables.pop(data_type)
  spec_types.remove(spec_type)
  del to_elt_handlers[data_type]
  del from_elt_handlers[data_type]
  if axis_size_type in make_iota_handlers:
    del make_iota_handlers[axis_size_type]

def is_vmappable(x: Any) -> bool:
  return type(x) is Jumble or type(x) in vmappables

@lu.transformation_with_aux2
def flatten_fun_for_vmap(f, store, in_tree, *args_flat):
  py_args, py_kwargs = tree_unflatten(in_tree, args_flat)
  ans = f(*py_args, **py_kwargs)
  ans, out_tree = tree_flatten(ans, is_leaf=is_vmappable)
  store.store(out_tree)
  return ans

# Propagate ragged masking rules from invars to outvars
# rule([params], [raggedness_per_invar], outvars) ->
#   [raggedness_per_invar, raggedness_per_outvar]
RaggedMaskingRule = Callable[
    [list[Any], list[Any], list[Any]], tuple[list[Any], list[Any]]
]

ragged_prop_rules: dict[core.Primitive, RaggedMaskingRule] = {}


def ragged_mask_elementwise_rule(eqn_params, invar_raggedness, outvars):
  # TODO(mvoz): A util for getting the ragged representations
  first_invar_raggedness = invar_raggedness[0]
  for other_invar_raggedness in invar_raggedness[1:]:
    if other_invar_raggedness != first_invar_raggedness:
      raise ValueError(f'{other_invar_raggedness} != {first_invar_raggedness}')

  outvar_raggedness = [first_invar_raggedness] * len(outvars)
  return invar_raggedness, outvar_raggedness


def ragged_mask_assert_no_op_rule(eqn_params, invar_raggedness, outvars):
  if any(invar_raggedness):
    raise ValueError(f'unexpected invar_raggedness: {invar_raggedness}')
  return invar_raggedness, [None] * len(outvars)


def ragged_mask_no_op_rule(eqn_params, invar_raggedness, outvars):
  return invar_raggedness, [None] * len(outvars)


def ragged_mask_transfer_identity(
    eqn_params, invar_raggedness, outvar_raggedness
):
  assert len(invar_raggedness) == 1, invar_raggedness
  outvar_raggedness = invar_raggedness
  return invar_raggedness, outvar_raggedness


### tracer

# TODO(mattjj): use a special sentinel type rather than None
NotMapped = type(None)
not_mapped = None


class BatchTracer(Tracer):
  __slots__ = ['val', 'batch_dim', 'source_info']

  def __init__(self, trace, val, batch_dim: NotMapped | int | RaggedAxis,
               source_info: source_info_util.SourceInfo | None = None):
    if config.enable_checks.value:
      assert type(batch_dim) in (NotMapped, int, RaggedAxis)
      if type(batch_dim) is int:
        aval = core.get_aval(val)
        assert 0 <= batch_dim < len(aval.shape)
    self._trace = trace
    self.val = val
    self.batch_dim = batch_dim
    self.source_info = source_info

  @property
  def aval(self):
    aval = core.get_aval(self.val)
    if self.batch_dim is not_mapped:
      return aval
    elif type(self.batch_dim) is int:
      return core.mapped_aval(aval.shape[self.batch_dim], self.batch_dim, aval)
    elif type(self.batch_dim) is RaggedAxis:
      new_aval = core.mapped_aval(
        aval.shape[self.batch_dim.stacked_axis], self.batch_dim.stacked_axis, aval)
      shape = list(new_aval.shape)  # pytype: disable=attribute-error
      for ragged_axis, segment_lengths in self.batch_dim.ragged_axes:
        size_tracer = BatchTracer(self._trace, segment_lengths, 0)
        if self.batch_dim.stacked_axis < ragged_axis:
          ragged_axis -= 1
        shape[ragged_axis] = size_tracer
      return core.DShapedArray(shape=tuple(shape), dtype=aval.dtype,
                               weak_type=aval.weak_type)

  def full_lower(self):
    if self.batch_dim is not_mapped:
      return core.full_lower(self.val)
    else:
      return self

  def _origin_msg(self):
    if self.source_info is None:
      return ""
    return (f"\nThis BatchTracer with object id {id(self)} was created on line:"
            f"\n  {source_info_util.summarize(self.source_info)}")

  def _contents(self):
    return [('val', self.val), ('batch_dim', self.batch_dim)]

  def get_referent(self):
    if self.batch_dim is None or type(self.batch_dim) is int:
      return core.get_referent(self.val)
    else:  # TODO(mattjj): could handle the RaggedAxis case?
      return self

@dataclasses.dataclass(frozen=True)
class AxisData:
  name : Any
  size : Any
  spmd_name : Any


class BatchTrace(Trace):

  def __init__(self, parent_trace, tag, axis_data):
    self.parent_trace = parent_trace
    assert isinstance(axis_data, AxisData)
    self.axis_data = axis_data
    self.tag = tag

  def to_batch_info(self, val):
    if isinstance(val, BatchTracer) and val._trace.tag is self.tag:
      return val.val, val.batch_dim
    else:
      return val, not_mapped

  def process_primitive(self, p, tracers, params):
    if config.dynamic_shapes.value:
      p.abstract_eval(*(map(core.get_aval, tracers)), **params)
    vals_in, dims_in = unzip2(map(self.to_batch_info, tracers))
    args_not_mapped = all(bdim is not_mapped for bdim in dims_in)
    if p in fancy_primitive_batchers:
      if (args_not_mapped
          and p in skippable_batchers
          and not any(self.axis_data.name == axis_name
                      for axis_name in skippable_batchers[p](params))):
        # no-op shortcut
        return p.bind_with_trace(self.parent_trace, vals_in, params)
      else:
        with core.set_current_trace(self.parent_trace):
          val_out, dim_out = fancy_primitive_batchers[p](self.axis_data, vals_in, dims_in, **params)
    elif args_not_mapped:
      # no-op shortcut
      return p.bind_with_trace(self.parent_trace, vals_in, params)
    elif p in primitive_batchers:
      with core.set_current_trace(self.parent_trace):
        val_out, dim_out = primitive_batchers[p](vals_in, dims_in, **params)
    else:
      raise NotImplementedError("Batching rule for '{}' not implemented".format(p))
    src = source_info_util.current()
    if p.multiple_results:
      with core.set_current_trace(self.parent_trace):  # val_out may be lazy map
        return [BatchTracer(self, x, d, src) if d is not not_mapped else x
                for x, d in zip(val_out, dim_out)]
    else:
      return (BatchTracer(self, val_out, dim_out, src)
              if dim_out is not not_mapped else val_out)

  def process_call(self, call_primitive, f, tracers, params):
    assert call_primitive.multiple_results
    params = dict(params, name=params.get('name', f.__name__))
    vals, dims = unzip2(map(self.to_batch_info, tracers))
    segment_lens, dims = indirectify_ragged_axes(dims)
    f_, dims_out = batch_subtrace(f, self.tag, self.axis_data, tuple(dims))
    f_ = _update_annotation(
        f_, f.in_type, self.axis_data.size, self.axis_data.name, dims, segment_lens)

    with core.set_current_trace(self.parent_trace):
      vals_out = call_primitive.bind(f_, *segment_lens, *vals, **params)
    vals_out, dims_out = resolve_ragged_axes(vals_out, dims_out())
    src = source_info_util.current()
    return [BatchTracer(self, v, d, src) for v, d in zip(vals_out, dims_out)]

  def process_map(self, map_primitive, f: lu.WrappedFun, tracers, params):
    vals, dims = unzip2(map(self.to_batch_info, tracers))
    # The logic for the dimension math below is as follows:
    # ╔═════════════╦════════════════════════════════════════╦═══════════╗
    # ║ d / in_axis ║ None                                   ║ int       ║
    # ╠═════════════╬════════════════════════════════════════╩═══════════╣
    # ║ None        ║ No extra axis, so in_axis unaffected               ║
    # ╠═════════════╬════════════════════════════════════════╦═══════════╣
    # ║ int         ║ Not mapped, so batching dim unaffected ║ See below ║
    # ╚═════════════╩════════════════════════════════════════╩═══════════╝
    # When both d and in_axis are defined then:
    # - If `d <= in_axis`, we have to move the `in_axis` one dimension further;
    # - If `d >  in_axis`, we have to decrement `d` (as `in_axis` will get removed).
    def both_mapped(in_out_axis, d):
      return in_out_axis is not None and d is not not_mapped
    new_in_axes = tuple(
      in_axis + 1 if both_mapped(in_axis, d) and d <= in_axis else in_axis
      for d, in_axis in zip(dims, params['in_axes']))
    new_dims = tuple(
      d - 1 if both_mapped(in_axis, d) and in_axis < d else d
      for d, in_axis in zip(dims, params['in_axes']))
    f, dims_out = batch_subtrace(f, self.tag, self.axis_data, new_dims)
    out_axes_thunk = params['out_axes_thunk']
    # NOTE: This assumes that the choice of the dimensions over which outputs
    #       are batched is entirely dependent on the function and not e.g. on the
    #       data or its shapes.
    @as_hashable_function(closure=out_axes_thunk)
    def new_out_axes_thunk():
      return tuple(out_axis + 1 if both_mapped(out_axis, d) and d < out_axis else out_axis
                    for out_axis, d in zip(out_axes_thunk(), dims_out()))
    new_params = dict(params, in_axes=new_in_axes, out_axes_thunk=new_out_axes_thunk)
    with core.set_current_trace(self.parent_trace):
      vals_out = map_primitive.bind(f, *vals, **new_params)
    dims_out_ = [d + 1 if both_mapped(out_axis, d) and out_axis <= d else d
                  for d, out_axis in zip(dims_out(), out_axes_thunk())]
    src = source_info_util.current()
    return [BatchTracer(self, v, d, src) for v, d in zip(vals_out, dims_out_)]

  def process_custom_jvp_call(self, prim, fun, jvp, tracers, *, symbolic_zeros):
    in_vals, in_dims = unzip2(map(self.to_batch_info, tracers))
    fun, out_dims1 = batch_subtrace(fun, self.tag, self.axis_data, in_dims)
    jvp, out_dims2 = batch_custom_jvp_subtrace(jvp, self.tag, self.axis_data, in_dims)
    out_vals = prim.bind_with_trace(self.parent_trace, (fun, jvp) + tuple(in_vals),
                                    dict(symbolic_zeros=symbolic_zeros))
    fst, out_dims = lu.merge_linear_aux(out_dims1, out_dims2)
    if not fst:
      assert out_dims == out_dims[:len(out_dims) // 2] * 2
      out_dims = out_dims[:len(out_dims) // 2]
    src = source_info_util.current()
    return [BatchTracer(self, v, d, src) for v, d in zip(out_vals, out_dims)]

  def process_custom_vjp_call(self, prim, fun, fwd, bwd, tracers, *, out_trees,
                              symbolic_zeros):  # pytype: disable=signature-mismatch
    in_vals, in_dims = unzip2(map(self.to_batch_info, tracers))
    fwd_in_dims = [d for in_dim in in_dims for d in [in_dim, not_mapped]]

    fun, out_dims1 = batch_subtrace(fun, self.tag, self.axis_data, in_dims)
    fwd, out_dims2 = batch_subtrace(fwd, self.tag, self.axis_data, fwd_in_dims)

    bwd = batch_custom_vjp_bwd(bwd, self.tag, self.axis_data, out_dims2, in_dims)
    out_vals = prim.bind_with_trace(self.parent_trace,
                                    (fun, fwd, bwd) + tuple(in_vals),
                                    dict(out_trees=out_trees, symbolic_zeros=symbolic_zeros))
    fst, out_dims = lu.merge_linear_aux(out_dims1, out_dims2)
    if not fst:
      _, res_tree = out_trees()
      _, out_dims = split_list(out_dims, [res_tree.num_leaves])
    src = source_info_util.current()
    return [BatchTracer(self, v, d, src) for v, d in zip(out_vals, out_dims)]

### API for batching callables with vmappable inputs and outputs

def batch(fun: lu.WrappedFun, axis_data,
          in_dims, out_dim_dests) -> lu.WrappedFun:
  # we split up _batch_inner and _batch_outer for the leak checker
  f = _batch_inner(fun, axis_data, out_dim_dests)
  return _batch_outer(f, axis_data, in_dims)

@lu.transformation2
def _batch_outer(f, axis_data, in_dims, *in_vals):
  tag = TraceTag()
  with source_info_util.transform_name_stack('vmap'):
    outs, trace = f(tag, in_dims, *in_vals)
  with core.ensure_no_leaks(trace): del trace
  return outs

@lu.transformation2
def _batch_inner(f, axis_data, out_dim_dests, tag, in_dims, *in_vals):
  in_dims = in_dims() if callable(in_dims) else in_dims
  with core.take_current_trace() as parent_trace:
    trace = BatchTrace(parent_trace, tag, axis_data)
    idx = memoize(lambda: BatchTracer(trace, make_iota(axis_data.size), 0,
                                      source_info_util.current()))
    in_tracers = map(partial(to_elt, trace, idx), in_vals, in_dims)
    with (core.set_current_trace(trace),
          core.extend_axis_env_nd([(axis_data.name, axis_data.size)]),
          core.add_spmd_axis_names(axis_data.spmd_name)):
      outs = f(*in_tracers)

  out_dim_dests = out_dim_dests() if callable(out_dim_dests) else out_dim_dests
  out_vals = map(partial(from_elt, trace, axis_data.size), range(len(outs)),
                 outs, out_dim_dests)

  return out_vals, trace

# NOTE: This divides the in_axes by the tile_size and multiplies the out_axes by it.
def vtile(f_flat: lu.WrappedFun,
          in_axes_flat: tuple[int | None, ...],
          out_axes_flat: tuple[int | None, ...],
          tile_size: int | None,
          axis_name: AxisName):
  @curry
  def tile_axis(arg, axis: int | None, tile_size):
    if axis is None:
      return arg
    shape = list(arg.shape)
    shape[axis:axis+1] = [tile_size, shape[axis] // tile_size]
    return arg.reshape(shape)

  def untile_axis(out, axis: int | None):
    if axis is None:
      return out
    shape = list(out.shape)
    shape[axis:axis+2] = [shape[axis] * shape[axis+1]]
    return out.reshape(shape)

  @lu.transformation2
  def _map_to_tile(f, *args_flat):
    sizes = (x.shape[i] for x, i in safe_zip(args_flat, in_axes_flat) if i is not None)
    tile_size_ = tile_size or next(sizes, None)
    assert tile_size_ is not None, "No mapped arguments?"
    outputs_flat = f(*map(tile_axis(tile_size=tile_size_), args_flat, in_axes_flat))
    return map(untile_axis, outputs_flat, out_axes_flat)

  axis_data = AxisData(axis_name, tile_size, None)
  return _map_to_tile(batch(f_flat, axis_data, in_axes_flat, out_axes_flat))

### API for batching functions with jaxpr type inputs and outputs

@lu.transformation_with_aux2
def batch_subtrace(f, store, tag, axis_data, in_dims, *in_vals):
  with core.take_current_trace() as parent_trace:
    trace = BatchTrace(parent_trace, tag, axis_data)
    with core.set_current_trace(trace):
      in_dims = in_dims() if callable(in_dims) else in_dims
      in_vals, in_dims = resolve_ragged_axes(in_vals, in_dims)
      in_tracers = [BatchTracer(trace, x, dim, source_info_util.current())
                    if dim is not None else x for x, dim in zip(in_vals, in_dims)]
      outs = f(*in_tracers)
    out_vals, out_dims = unzip2(map(trace.to_batch_info, outs))
    segment_lens, out_dims = indirectify_ragged_axes(out_dims)
  store.store(out_dims)
  return (*segment_lens, *out_vals)

def indirectify_ragged_axes(dims):
  if not any(type(d) is RaggedAxis for d in dims):
    return [], dims
  axis_map : dict[int, tuple[Array, pe.DBIdx]] = collections.OrderedDict()
  def canonicalize_segment_lengths(d: RaggedAxis) -> RaggedAxis:
    new_ragged_axes = []
    for ragged_axis, segment_lengths in d.ragged_axes:
      _, dbidx = axis_map.setdefault(
          id(core.get_referent(segment_lengths)),
          (segment_lengths, pe.DBIdx(len(axis_map))))
      new_ragged_axes.append((ragged_axis, dbidx))
    return RaggedAxis(d.stacked_axis, tuple(new_ragged_axes))
  new_dims = [canonicalize_segment_lengths(d)
              if isinstance(d, RaggedAxis) else d for d in dims]
  segment_lens = [s for s, _ in axis_map.values()]
  return segment_lens, new_dims

def indirectify_ragged_axes_against_inputs_outputs(dims, in_vals, out_vals):
  def canonicalize_segment_lengths(d: RaggedAxis) -> RaggedAxis:
    new_ragged_axes = []
    for ragged_axis, segment_lengths in d.ragged_axes:
      key = id(core.get_referent(segment_lengths))
      value = _locate_value(key, in_vals, out_vals)
      new_ragged_axes.append((ragged_axis, value))
    return RaggedAxis(d.stacked_axis, tuple(new_ragged_axes))
  new_dims = [canonicalize_segment_lengths(d)
              if isinstance(d, RaggedAxis) else d for d in dims]
  return new_dims

def _locate_value(key, in_vals, out_vals):
  for ix, candidate in enumerate(in_vals):
    if key == id(candidate):
      return pe.InDBIdx(ix)
  for ix, candidate in enumerate(out_vals):
    if key == id(candidate):
      return pe.OutDBIdx(ix)
  assert False, "Could not find segment lengths"

def resolve_ragged_axes(vals, dims):
  idxs = {lengths_idx.val for d in dims if isinstance(d, RaggedAxis)
          for (_, lengths_idx) in d.ragged_axes}
  dims = [RaggedAxis(d.stacked_axis,
                     tuple((ragged_axis, vals[lengths_idx.val])
                           for ragged_axis, lengths_idx in d.ragged_axes))
          if isinstance(d, RaggedAxis) else d for d in dims]
  vals = [x for i, x in enumerate(vals) if i not in idxs]
  return vals, dims

def resolve_ragged_axes_against_inputs_outputs(in_vals, out_vals, dims):
  def fetch(idx):
    if isinstance(idx, pe.InDBIdx):
      return in_vals[idx.val]
    else:
      assert isinstance(idx, pe.OutDBIdx)
      return out_vals[idx.val]

  dims = [RaggedAxis(d.stacked_axis,
                     tuple((ragged_axis, fetch(lengths_idx))
                           for ragged_axis, lengths_idx in d.ragged_axes))
          if isinstance(d, RaggedAxis) else d for d in dims]
  return dims

### API for batching jaxprs

# TODO(axch): parameterize RaggedAxis annotations by a type parameter so as to
# indicate whether we're dealing with instances that contain Arrays or DBIdx.
# Can reuse same pattern for all dynamic shape stuff.
def batch_jaxpr2(
    closed_jaxpr: core.ClosedJaxpr,
    axis_data,
    in_axes: tuple[int | NotMapped | RaggedAxis, ...],
  ) -> tuple[core.ClosedJaxpr, tuple[int | NotMapped | RaggedAxis, ...]]:
  # This is only ever used in pjit.  The difference vs batch_jaxpr is that
  # batch_jaxpr2 lets the callee decide which outputs are batched and what
  # their batch axes are; whereas batch_jaxpr has to obey caller-imposed
  # consistency constraints, such as type-agreement across arms of a
  # `lax.cond`, or input-output agreement for the body of a `lax.scan`.
  return _batch_jaxpr2(closed_jaxpr, axis_data, tuple(in_axes))

@weakref_lru_cache
def _batch_jaxpr2(
    closed_jaxpr: core.ClosedJaxpr,
    axis_data,
    in_axes: tuple[int | NotMapped | RaggedAxis, ...],
  ) -> tuple[core.ClosedJaxpr, tuple[int | NotMapped, ...]]:
  f = lu.wrap_init(core.jaxpr_as_fun(closed_jaxpr))
  f, out_axes = _batch_jaxpr_inner(f, axis_data)
  f = _batch_jaxpr_outer(f, axis_data, in_axes)
  in_axes2, avals_in = unzip2([
      handle_ragged(closed_jaxpr.in_avals, dim, aval)
      if isinstance(dim, RaggedAxis) else (dim, aval)
      for dim, aval in zip(in_axes, closed_jaxpr.in_avals)])
  avals_in2 = [core.unmapped_aval(axis_data.size, axis_data.name, b, aval)
               if b is not not_mapped else aval
               for aval, b in unsafe_zip(avals_in, in_axes2)]
  jaxpr_out, _, consts, () = pe.trace_to_jaxpr_dynamic(f, avals_in2)
  return core.ClosedJaxpr(jaxpr_out, consts), out_axes()

def handle_ragged(in_avals: list[core.AbstractValue], dim: RaggedAxis,
                  aval: core.ShapedArray) -> tuple[int, core.ShapedArray]:
  new_shape = list(aval.shape)
  for i, dbi in dim.ragged_axes:
    new_shape[i - (dim.stacked_axis < i)] = in_avals[dbi.val].dtype.bound
  new_aval = aval.update(shape=tuple(new_shape))
  return dim.stacked_axis, new_aval

def batch_jaxpr(closed_jaxpr, axis_data, in_batched, instantiate):
  inst = tuple(instantiate) if isinstance(instantiate, list) else instantiate
  return _batch_jaxpr(closed_jaxpr, axis_data, tuple(in_batched), inst)

def _batch_jaxpr(closed_jaxpr, axis_data, in_batched, instantiate):
  assert (isinstance(instantiate, bool) or
          isinstance(instantiate, (list, tuple)) and
          all(isinstance(b, bool) for b in instantiate))
  if isinstance(instantiate, bool):
    instantiate = [instantiate] * len(closed_jaxpr.out_avals)
  in_axes = [0 if b else not_mapped for b in in_batched]
  out_axes_dest = [0 if inst else zero_if_mapped for inst in instantiate]
  return batch_jaxpr_axes(closed_jaxpr, axis_data, in_axes, out_axes_dest)

def batch_jaxpr_axes(closed_jaxpr, axis_data, in_axes, out_axes_dest):
  return _batch_jaxpr_axes(closed_jaxpr, axis_data, tuple(in_axes), tuple(out_axes_dest))

@weakref_lru_cache
def _batch_jaxpr_axes(closed_jaxpr, axis_data, in_axes, out_axes_dest):
  f = lu.wrap_init(core.jaxpr_as_fun(closed_jaxpr))
  f, out_axes = _batch_jaxpr_inner(f, axis_data)
  f, out_batched = _match_axes_jaxpr(f, axis_data, out_axes_dest, out_axes)
  f = _batch_jaxpr_outer(f, axis_data, in_axes)
  avals_in = [core.unmapped_aval(axis_data.size, axis_data.name, b, aval) if b is not not_mapped
              else aval for aval, b in unsafe_zip(closed_jaxpr.in_avals, in_axes)]
  jaxpr_out, _, consts, () = pe.trace_to_jaxpr_dynamic(f, avals_in)
  return core.ClosedJaxpr(jaxpr_out, consts), out_batched()

@lu.transformation_with_aux2
def _batch_jaxpr_inner(f, store, axis_data, tag, in_axes, *in_vals):
  with core.take_current_trace() as parent_trace:
    trace = BatchTrace(parent_trace, tag, axis_data)
    _, in_axes = resolve_ragged_axes(in_vals, in_axes)
    in_tracers = [BatchTracer(trace, val, dim) if dim is not None else val
                  for val, dim in zip(in_vals, in_axes)]
    with (core.set_current_trace(trace),
          core.extend_axis_env_nd([(axis_data.name, axis_data.size)]),
          core.add_spmd_axis_names(axis_data.spmd_name)):
      outs = f(*in_tracers)
    out_vals, out_axes = unzip2(map(trace.to_batch_info, outs))
    new_out_axes = indirectify_ragged_axes_against_inputs_outputs(
        out_axes, in_vals, out_vals)
  store.store(new_out_axes)
  return out_vals

@lu.transformation_with_aux2
def _match_axes_jaxpr(f, store, axis_data, out_axes_dest, out_axes, trace, in_axes,
                      *in_vals):
  out_vals = f(trace, in_axes, *in_vals)
  out_axes = out_axes()
  out_axes_dest = [(None if src is not_mapped else 0)
                   if dst is zero_if_mapped else dst
                   for src, dst in unsafe_zip(out_axes, out_axes_dest)]
  if len(out_axes_dest) != len(out_axes):
    out_axis_dest, = out_axes_dest
    out_axes_dest = [out_axis_dest] * len(out_axes)
  out_vals = map(partial(matchaxis, axis_data.name, axis_data.size),
                 out_axes, out_axes_dest, out_vals)
  out_batched = [dst is not None for dst in out_axes_dest]
  store.store(out_batched)
  return out_vals

@lu.transformation2
def _batch_jaxpr_outer(f, axis_data, in_dims, *in_vals):
  in_dims = in_dims() if callable(in_dims) else in_dims
  in_dims = [canonicalize_axis(ax, np.ndim(x)) if isinstance(ax, int)
             else ax for x, ax in unsafe_zip(in_vals, in_dims)]
  tag = TraceTag()
  return f(tag, in_dims, *in_vals)

def _merge_bdims(x, y):
  if x == y:
    return x
  elif x is not_mapped:
    return y
  elif y is not_mapped:
    return x
  else:
    return x  # arbitrary

class ZeroIfMapped: pass
zero_if_mapped = ZeroIfMapped()

### functions for handling custom_vjp

@lu.transformation_with_aux2
def batch_custom_jvp_subtrace(f, store, tag, axis_data, in_dims, *in_vals):
  size = axis_data.size
  with core.take_current_trace() as parent_trace:
    trace = BatchTrace(parent_trace, tag, axis_data)
    in_tracers = [val if dim is None else
                  SymbolicZero(core.mapped_aval(size, dim, val.aval))
                  if type(val) is SymbolicZero else BatchTracer(trace, val, dim)
                  for val, dim in zip(in_vals, in_dims * 2)]
    with core.set_current_trace(trace):
      outs = f(*in_tracers)
      # TODO(mattjj,frostig): instantiating any SymbolicZero output is easy, but can
      # be wasteful in the rare case it actually triggers; handle symbolically!
      outs = [instantiate(replace_rule_output_symbolic_zeros(x)) for x in outs]

  out_vals, out_dims = unzip2(map(trace.to_batch_info, outs))
  out_primals, out_tangents = split_list(out_vals, [len(out_vals) // 2])
  out_primal_bds, out_tangent_bds = split_list(out_dims, [len(out_vals) // 2])
  out_dims = map(_merge_bdims, out_primal_bds, out_tangent_bds)
  out_primals  = map(partial(matchaxis, trace.axis_data.name, size),
                     out_primal_bds, out_dims,  out_primals)
  out_tangents = map(partial(matchaxis, trace.axis_data.name, size),
                     out_tangent_bds, out_dims, out_tangents)
  store.store(out_dims * 2)
  return out_primals + out_tangents

def batch_custom_vjp_bwd(bwd, tag, axis_data, in_dims, out_dim_dests):
  axis_size = axis_data.size
  axis_name = axis_data.name
  def new_bwd(*args):
    in_dims_ = in_dims() if callable(in_dims) else in_dims
    args = [SymbolicZero(core.mapped_aval(axis_size, dim, x.aval))
            if type(x) is SymbolicZero else x
            for x, dim in zip(args, in_dims_)]
    in_dims_ = [None if type(x) is SymbolicZero else d
                for x, d in zip(args, in_dims_)]
    bwd_, out_dims_thunk = batch_subtrace(lu.wrap_init(bwd), tag, axis_data, in_dims_)
    bwd_ = _match_axes_and_sum(bwd_, axis_size, axis_name, out_dims_thunk,
                               out_dim_dests)
    return bwd_.call_wrapped(*args)
  return new_bwd

@lu.transformation2
def _match_axes_and_sum(f, axis_size, axis_name, out_dims_thunk, out_dim_dests, *in_vals):
  # this is like _match_axes, but we do reduce-sums as needed
  out_vals = f(*in_vals)
  return map(partial(_matchaxis_symbolic_zeros, axis_name, axis_size, axis_name,
                    sum_match=True), out_dims_thunk(), out_dim_dests, out_vals)

def _matchaxis_symbolic_zeros(axis_name, sz, name, src, dst, x, sum_match=False):
  # Just like `matchaxis`, but handles symbolic zeros using ad_util.py
  # TODO(mattjj): dedup with matchaxis
  if isinstance(x, (Zero, SymbolicZero)):
    if src == dst:
      return x
    elif type(src) == type(dst) == int:
      aval = core.mapped_aval(sz, src, x.aval)
      return Zero(core.unmapped_aval(sz, name, dst, aval))
    elif src is not_mapped and dst is not not_mapped:
      return Zero(core.unmapped_aval(sz, name, dst, x.aval))
    elif dst is not_mapped and sum_match:
      return Zero(core.mapped_aval(sz, src, x.aval))
    else:
      raise ValueError((axis_name, x, src, dst))
  else:
    return matchaxis(axis_name, sz, src, dst, x, sum_match=sum_match)


### utilities for defining primitives' batching rules

BatchingRule = Callable[
    ...,
    tuple[Any, Union[int, None, tuple[Union[int, None], ...]]]
]
primitive_batchers : dict[core.Primitive, BatchingRule] = {}
# "fancy" primitive batchers just take a extra leading `AxisData` and "trace type" args
fancy_primitive_batchers: dict[core.Primitive, Callable] = {}

# backwards compat shim. TODO: delete
class AxisPrimitiveBatchersProxy:
  def __setitem__(self, prim, batcher):
    def wrapped(axis_data, vals, dims, **params):
      return batcher(axis_data.size, axis_data.name, None, vals, dims, **params)
    fancy_primitive_batchers[prim] = wrapped

axis_primitive_batchers = AxisPrimitiveBatchersProxy()


# Presence in this table allows fancy batchers to be skipped by batch traces for
# irrelevant axes. The Callable takes the params and returns a list of relevant
# axes.
skippable_batchers : dict[core.Primitive, Callable] = {}

def defvectorized(prim):
  primitive_batchers[prim] = partial(vectorized_batcher, prim)

def vectorized_batcher(prim, batched_args, batch_dims, **params):
  assert all(batch_dims[0] == bd for bd in batch_dims[1:]), batch_dims
  return prim.bind(*batched_args, **params), batch_dims[0]

def defbroadcasting(prim):
  primitive_batchers[prim] = partial(broadcast_batcher, prim)

def broadcast_batcher(prim, args, dims, **params):
  """Process a primitive with built-in broadcasting.

  Args:
    args: the possibly-batched arguments
    dims: list or tuple of the same length as `args`, where each
      entry indicates the batching state of the corresponding entry to `args`:
      either an int indicating the batch dimension, or else `not_mapped`
      indicating no batching.
  """
  assert len(args) > 1
  shape, dim = next((x.shape, d) for x, d in zip(args, dims)
                    if d is not not_mapped)
  if all(core.definitely_equal_shape(shape, x.shape) and d == dim
         for x, d in zip(args, dims) if np.ndim(x)):
    # if there's only agreeing batch dims and scalars, just call the primitive
    out = prim.bind(*args, **params)
    return (out, (dim,) * len(out)) if prim.multiple_results else (out, dim)
  else:
    # We pass size of 1 here because (1) at least one argument has a real batch
    # dimension and (2) all unmapped axes can have a singleton axis inserted and
    # then rely on the primitive's built-in broadcasting.
    args = [bdim_at_front(x, d, 1) if np.ndim(x) else x
            for x, d in zip(args, dims)]
    ndim = max(np.ndim(x) for x in args)  # special-case scalar broadcasting
    args = [_handle_scalar_broadcasting(ndim, x, d) for x, d in zip(args, dims)]
    out = prim.bind(*args, **params)
    return (out, (0,) * len(out)) if prim.multiple_results else (out, 0)

def _handle_scalar_broadcasting(nd, x, d):
  if d is not_mapped or nd == np.ndim(x):
    return x
  else:
    return jax.lax.expand_dims(x, tuple(range(np.ndim(x), nd)))

def defreducer(prim, ident):
  primitive_batchers[prim] = partial(reducer_batcher, prim, ident)

def reducer_batcher(prim, ident, batched_args, batch_dims, axes, **params):
  def out_axis(axes, axis):
    return int(list(np.delete(np.arange(operand.ndim), axes)).index(axis))
  operand, = batched_args
  bdim, = batch_dims
  if isinstance(bdim, int):
    axes = tuple(np.where(np.less(axes, bdim), axes, np.add(axes, 1)))
    bdim_out = out_axis(axes, bdim)
    if 'input_shape' in params:
      params = dict(params, input_shape=operand.shape)
    return prim.bind(operand, axes=axes, **params), bdim_out
  elif isinstance(bdim, RaggedAxis):
    assert ident is not None, "TODO Ragged batching a reduction requires an identity"
    axes = tuple(np.where(np.less(axes, bdim.stacked_axis), axes, np.add(axes, 1)))
    bdim_out = out_axis(axes, bdim.stacked_axis)
    # For each ragged_axis, we either mask the operand there or append
    # it to the set of axes that will be ragged in the result.
    axes_to_mask = []
    ragged_axes_out = []
    for ragged_axis, segment_lengths in bdim.ragged_axes:
      if ragged_axis in axes:
        axes_to_mask.append((ragged_axis, segment_lengths))
      else:
        ragged_axes_out.append((out_axis(axes, ragged_axis), segment_lengths))
    operand = mask_ragged_axes(
        operand, ident, RaggedAxis(bdim.stacked_axis, tuple(axes_to_mask)))
    result = prim.bind(operand, axes=axes, **params)
    return result, make_batch_axis(operand.ndim, bdim_out, ragged_axes_out)
  else:
    assert False

def mask_ragged_axes(operand: Array, ident, axis_spec: RaggedAxis) -> Array:
  # TODO(mattjj, axch) Can we mask multiple axes more efficiently at
  # once, rather than one at a time?
  for ragged_axis, segment_lengths in axis_spec.ragged_axes:
    this_axis_spec = RaggedAxis(
        axis_spec.stacked_axis, ((ragged_axis, segment_lengths),))
    operand = _mask_one_ragged_axis(operand, ident, this_axis_spec)
  return operand

def _mask_one_ragged_axis(
    operand: Array, ident, axis_spec: RaggedAxis) -> Array:
  assert len(axis_spec.ragged_axes) == 1, "Mask just one ragged axis at a time"
  ragged_axis, segment_lengths = axis_spec.ragged_axes[0]
  value = ident(operand.dtype)
  positions = jax.lax.broadcasted_iota('int32', operand.shape, ragged_axis)
  # TODO(mattjj, axch) can't get ._data, need to convert it
  # lengths = jax.lax.convert_element_type(segment_lengths._data, 'int32')
  lengths = jax.lax.convert_element_type(segment_lengths, 'int32')
  limits = jax.lax.broadcast_in_dim(
      lengths, operand.shape, [axis_spec.stacked_axis])
  mask = positions < limits
  return jax.lax.select(mask, operand, jax.lax.broadcast(value, operand.shape))

def move_stacked_axis(operand, bdim, dst):
  dst = canonicalize_axis(dst, operand.ndim)
  if isinstance(bdim, int):
    return moveaxis(operand, bdim, dst), dst
  elif isinstance(bdim, RaggedAxis):
    result = moveaxis(operand, bdim.stacked_axis, dst)
    return result, bdim.move_stacked_axis(dst)
  else:
    raise TypeError(f"Unrecognized batch dimension type {bdim}")

### general utilities for manipulating axes on jaxpr types (not vmappables)

def broadcast(x, sz, axis):
  shape = list(np.shape(x))
  shape.insert(axis, sz)
  broadcast_dims = tuple(np.delete(np.arange(len(shape)), axis))
  return jax.lax.broadcast_in_dim(x, shape, broadcast_dims)

def matchaxis(axis_name, sz, src, dst, x, sum_match=False):
  if dst == jumble_axis:
    x = bdim_at_front(x, src, sz)
    elt_ty = x.aval.update(shape=x.shape[1:])
    aval = JumbleTy(core.Var('', core.ShapedArray((), np.dtype('int32'))),
                    x.shape[0], elt_ty)
    return Jumble(aval, x)
  try:
    _ = core.get_aval(x)
  except TypeError as e:
    raise TypeError(f"Output from batched function {x!r} with type "
                    f"{type(x)} is not a valid JAX type") from e
  if src == dst:
    return x
  elif type(src) == type(dst) == int:
    return moveaxis(x, src, dst)
  elif src is not_mapped and dst is not not_mapped:
    return broadcast(x, sz, canonicalize_axis(dst, np.ndim(x) + 1))
  elif dst is not_mapped and sum_match:
    return x.sum(src)
  else:
    if (not isinstance(axis_name, core._TempAxisName) and
        axis_name is not core.no_axis_name):
      raise ValueError(f'vmap has mapped output ({axis_name=}) but out_axes is {dst}')
    else:
      raise SpecMatchError(None, None, None)

class SpecMatchError(Exception):
  def __init__(self, leaf_idx, src, dst):
    self.leaf_idx = leaf_idx
    self.src = src
    self.dst = dst

def bdim_at_front(x, bdim, size):
  if bdim is not_mapped:
    return broadcast(x, size, 0)
  else:
    return moveaxis(x, bdim, 0)


def add_batched(batched_args, batch_dims):
  bdx, bdy = batch_dims
  x, y = batched_args
  if bdx == bdy:
    return add_jaxvals(x, y), bdx
  elif bdx is not_mapped:
    x = broadcast(x, y.shape[bdy], bdy)
    return add_jaxvals(x, y), bdy
  elif bdy is not_mapped:
    y = broadcast(y, x.shape[bdx], bdx)
    return add_jaxvals(x, y), bdx
  else:
    x = moveaxis(x, bdx, bdy)
    return add_jaxvals(x, y), bdy
primitive_batchers[add_jaxvals_p] = add_batched
