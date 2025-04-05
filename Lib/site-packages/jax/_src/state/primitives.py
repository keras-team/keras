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
"""Module for state primitives."""
from __future__ import annotations

from functools import partial
import types
from typing import Any, Union

from jax._src import ad_util
from jax._src import core
from jax._src import dispatch
from jax._src import pretty_printer as pp
from jax._src import traceback_util
from jax._src import tree_util
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lax import lax
from jax._src.state import indexing
from jax._src.state.types import (
    AbstractRef,
    AccumEffect,
    ReadEffect,
    RefBitcaster,
    RefReshaper,
    Transform,
    TransformedRef,
    WriteEffect,
)
from jax._src.typing import Array
from jax._src.util import safe_map, safe_zip
import numpy as np


## General utilities

## JAX utilities

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip
traceback_util.register_exclusion(__file__)

## get/swap/addupdate implementations

# `get` reads a value from a `Ref` type, a.k.a.:
# a = get_p.bind(x)
# or we can read using indices:
# a = get_p.bind(x, 0, 1)
# Staging out `a = get_p.bind(x)` where the aval of `x` is
# `Ref((3,), np.dtype('float32'))` leads to a jaxpr eqn printed like
#   a:f32[3] <- x[]
get_p = core.Primitive("get")
get_p.def_impl(partial(dispatch.apply_primitive, get_p))
batching.ragged_prop_rules[get_p] = batching.ragged_mask_transfer_identity

Indexer = Union[int, slice, Array, types.EllipsisType]


def get_ref_and_transforms(
    ref_or_view: Any,
    idx: Indexer | tuple[Indexer, ...] | None,
    function_name: str,
    force_trailing_indexer: bool = True,  # TODO(apaszke): Clean this up.
) -> tuple[Any, tuple[Transform, ...]]:
  if isinstance(ref_or_view, TransformedRef):
    ref, transforms = ref_or_view.ref, ref_or_view.transforms
  else:
    ref, transforms = ref_or_view, ()
  ref_aval = core.get_aval(ref)
  if not isinstance(ref_aval, AbstractRef):
    raise ValueError(f"Can only call `{function_name}` on a `Ref`: {ref}.")
  if not isinstance(ref_aval.inner_aval, core.ShapedArray):
    return ref, ()

  if idx is None or idx is Ellipsis:
    idx = ()
  elif not isinstance(idx, tuple):
    idx = (idx,)

  if not idx and not force_trailing_indexer:
    return ref, transforms
  if not idx and transforms and isinstance(transforms[-1], indexing.NDIndexer):
    return ref, transforms
  nd_indexer = indexing.NDIndexer.from_indices_shape(idx, ref_or_view.shape)
  return ref, (*transforms, nd_indexer)


def ref_get(
    ref_or_view: Any, idx: Indexer | tuple[Indexer, ...] | None = None
) -> Array:
  """Reads a value from a `Ref`, a.k.a. value <- ref[idx]."""
  ref, transforms = get_ref_and_transforms(ref_or_view, idx, "ref_get")
  flat_transforms, tree = tree_util.tree_flatten(transforms)
  return get_p.bind(ref, *flat_transforms, tree=tree)


# `swap` mutates a `Ref`, setting its value and returns its previous value.
# b = swap_p.bind(x, a)
# It generalizes the setting operation for a `Ref` as we can ignore the return
# value:
# _ = swap_p.bind(x, a)
# `swap_p` also takes in index arguments following the value, i.e.:
# _ = swap_p.bind(x, a, 0, 1)
# Staging out `b = swap_p.bind(x, a)` where the aval of `x` is
# `Ref((3,), np.dtype('float32'))` and the aval of `a` is
# `ShapedArray((3,), np.dtype('float32'))` leads to a jaxpr eqn printed like
#   b:f32[3], x:Ref{f32[3]} <- x, a
# Staging out `_ = swap_p.bind(x, a, i, j)` where the aval of `x` is
# `Ref((3,), np.dtype('float32'))` , the aval of `a` is
# `ShapedArray((3,), np.dtype('float32'))`, and the avals of both `i` and `j`
# are `ShapedArray((), np.dtype('int32'))` leads to a jaxpr eqn printed like
#   x:Ref{f32[3]}[i, j] <- a
swap_p = core.Primitive("swap")
swap_p.def_impl(partial(dispatch.apply_primitive, swap_p))


def swap_ragged_prop_rule(eqn_params, invar_raggedness, outvars):
  assert len(invar_raggedness) == 2
  invar_raggedness_lhs = invar_raggedness[0]
  invar_raggedness_rhs = invar_raggedness[1]

  return [invar_raggedness_rhs, invar_raggedness_lhs], [None]


batching.ragged_prop_rules[swap_p] = swap_ragged_prop_rule

def ref_swap(
    ref_or_view: AbstractRef | TransformedRef,
    idx: Indexer | tuple[Indexer, ...] | None,
    value: Array,
    _function_name: str = "ref_swap",
) -> Array:
  """Sets a `Ref`'s value and returns the original value."""
  ref, transforms = get_ref_and_transforms(ref_or_view, idx, _function_name)
  flat_transforms, tree = tree_util.tree_flatten(transforms)
  return swap_p.bind(ref, value, *flat_transforms, tree=tree)


def ref_set(
    ref_or_view: AbstractRef | TransformedRef,
    idx: Indexer | tuple[Indexer, ...] | None,
    value: Array,
) -> None:
  """Sets a `Ref`'s value, a.k.a. ref[idx] <- value."""
  ref_swap(ref_or_view, idx, value, _function_name="ref_set")


# `addupdate_p` mutates a `Ref`, adding a value to its existing value.
# Semantically,
# ```
# addupdate ref a *idx
# ```
# is equivalent to
# ```
# b = get ref *idx
# c = add b x
# _ = swap ref c *idx
# ```
addupdate_p = core.Primitive('addupdate')
addupdate_p.multiple_results = True
addupdate_p.def_impl(partial(dispatch.apply_primitive, addupdate_p))


def ref_addupdate(
    ref_or_view: AbstractRef,
    idx: Indexer | tuple[Indexer, ...] | None,
    x: Array,
) -> None:
  """Mutates a ref with an additive update i.e. `ref[idx] += x`."""
  ref, transforms = get_ref_and_transforms(ref_or_view, idx, "ref_addupdate")
  flat_transforms, tree = tree_util.tree_flatten(transforms)
  return addupdate_p.bind(ref, x, *flat_transforms, tree=tree)


## get/set/addupdate abstract evaluation rules


def _shape_after_transforming(
    shape: tuple[int | Array, ...], transforms: tuple[Transform, ...]
) -> tuple[int | Array, ...]:
  for transform in transforms:
    shape = transform.transform_shape(shape)  # type: ignore
  assert shape is not None
  return shape


def _dtype_after_transforming(
    dtype: Any, transforms: tuple[Transform, ...]
) -> Any:
  for transform in transforms:
    dtype = transform.transform_dtype(dtype)
  assert dtype is not None
  return dtype


def _get_abstract_eval(ref_aval: AbstractRef, *args,
                       tree):
  transforms = tree_util.tree_unflatten(tree, args)
  if not isinstance(ref_aval, AbstractRef):
    raise ValueError(f"`get` must be called on `Ref` types: {ref_aval}.")
  if isinstance(ref_aval.inner_aval, core.ShapedArray):
    out_shape = _shape_after_transforming(ref_aval.shape, transforms)
    out_dtype = _dtype_after_transforming(ref_aval.dtype, transforms)
    # TODO(yashkatariya): Transform the sharding too instead of setting it to
    # None.
    out_aval = ref_aval.inner_aval.update(shape=out_shape, dtype=out_dtype,
                                          sharding=None)
  else:
    if transforms:
      raise ValueError("Cannot index non-shaped array with nontrivial indices.")
    out_aval = ref_aval.inner_aval
  return (out_aval, {ReadEffect(0)})
get_p.def_effectful_abstract_eval(_get_abstract_eval)

def _swap_abstract_eval(ref_aval: AbstractRef,
                        val_aval: core.AbstractValue,
                        *args: Any, tree):
  transforms = tree_util.tree_unflatten(tree, args)
  out_aval: core.AbstractValue
  if not isinstance(ref_aval, AbstractRef):
    raise ValueError(f"`swap` must be called on `Ref` types: {ref_aval}.")
  if isinstance(ref_aval.inner_aval, core.ShapedArray):
    assert isinstance(val_aval, core.ShapedArray)
    expected_out_shape = _shape_after_transforming(ref_aval.shape, transforms)
    expected_out_dtype = _dtype_after_transforming(ref_aval.dtype, transforms)
    if expected_out_shape != val_aval.shape:
      raise ValueError("Invalid shape for `swap`. "
                       f"Ref shape: {ref_aval.shape}. "
                       f"Expected shape: {expected_out_shape}. "
                       f"Value shape: {val_aval.shape}. "
                       f"Transforms: {transforms}. ")
    if expected_out_dtype != val_aval.dtype and not val_aval.weak_type:
      raise ValueError(
          "Invalid dtype for `swap`. "
          f"Ref dtype: {expected_out_dtype}. "
          f"Value dtype: {val_aval.dtype}. "
      )
    out_aval = core.ShapedArray(expected_out_shape, expected_out_dtype)
  else:
    if transforms:
      raise ValueError("Cannot index non-shaped array with nontrivial indices.")
    out_aval = ref_aval.inner_aval
  return (out_aval, {WriteEffect(0)})
swap_p.def_effectful_abstract_eval(_swap_abstract_eval)


def _addupdate_abstract_eval(ref_aval: AbstractRef,
                             val_aval: core.AbstractValue,
                             *args: Any, tree):
  transforms = tree_util.tree_unflatten(tree, args)
  if not isinstance(ref_aval, AbstractRef):
    raise ValueError(f"`addupdate` must be called on `Ref` types: {ref_aval}.")
  if isinstance(ref_aval.inner_aval, core.ShapedArray):
    out_shape = _shape_after_transforming(ref_aval.shape, transforms)
    out_dtype = _dtype_after_transforming(ref_aval.dtype, transforms)
    assert isinstance(val_aval, core.ShapedArray)
    if out_shape != val_aval.shape:
      raise ValueError(
          "Invalid shape for `addupdate`. "
          f"Ref shape: {ref_aval.shape}. "
          f"Expected shape: {out_shape}. "
          f"Value shape: {val_aval.shape}. "
          f"Transforms: {transforms}. "
      )
    if out_dtype != val_aval.dtype:
      raise ValueError("Invalid dtype for `addupdate`. "
                       f"Ref dtype: {ref_aval.dtype}. "
                       f"Value shape: {val_aval.dtype}. ")
  else:
    # Check that the transforms are valid
    if transforms:
      raise ValueError("Cannot index non-shaped array with nontrivial indices.")
  return [], {AccumEffect(0)}
addupdate_p.def_effectful_abstract_eval(_addupdate_abstract_eval)

## Pretty printing for `get` and `swap` in jaxprs

pp_ref_var = partial(pp.color, intensity=pp.Intensity.NORMAL,
                 foreground=pp.Color.GREEN)

def _pp_slice(context: core.JaxprPpContext, dim, slc: indexing.Slice
              ) -> str:
  start, size = slc.start, slc.size
  if isinstance(start, core.Var):
    start_str = core.pp_var(start, context)
    size_str = (
        core.pp_var(size, context)
        if isinstance(size, core.Var)
        else str(size)
    )
    return f'{start_str}:{start_str}+{size_str}'
  else:
    start_str = str(start)
    if start == 0:
      start_str = ''
    if isinstance(size, core.Var):
      size_str = core.pp_var(size, context)
      if start_str:
        return f'{start_str}:{start_str}+{size_str}'
      else:
        return f':{size_str}'
    else:
      end = start + size
      end_str = '' if end == dim else str(end)
      return f'{start_str}:{end_str}'

def pp_indexer(context: core.JaxprPpContext,indexer: indexing.NDIndexer
                ) -> pp.Doc:
  indices = []
  for idx, dim in zip(indexer.indices, indexer.shape):
    if isinstance(idx, indexing.Slice):
      indices.append(_pp_slice(context, dim, idx))
    else:
      indices.append(core.pp_var(idx, context))  # type: ignore
  return pp.concat([pp.text("["), pp.text(','.join(indices)), pp.text("]")])


def pp_bitcaster(
    context: core.JaxprPpContext, bitcaster: RefBitcaster
) -> pp.Doc:
  del context
  return pp.text(
      f"[bitcast({bitcaster.dtype}[{','.join(str(d) for d in bitcaster.shape)}])]"
  )


def pp_reshaper(context: core.JaxprPpContext, reshaper: RefReshaper) -> pp.Doc:
  del context
  return pp.text(
      f"[reshape({reshaper.dtype}[{','.join(str(d) for d in reshaper.shape)}])]"
  )


def pp_transform(context: core.JaxprPpContext, transform: Transform) -> pp.Doc:
  match transform:
    case indexing.NDIndexer():
      return pp_indexer(context, transform)
    case RefBitcaster():
      return pp_bitcaster(context, transform)
    case RefReshaper():
      return pp_reshaper(context, transform)
    case _:
      return pp.text(f"[{transform}]")


def _pp_transforms(
    context: core.JaxprPpContext,
    transforms: tuple[Transform, ...],
):
  if not transforms:
    return pp.text("[...]")
  return pp.concat(
      [pp_transform(context, transform) for transform in transforms]
  )


def pp_ref_transforms(context: core.JaxprPpContext, ref, transforms):
  return pp_ref_var(
      pp.concat([
          pp.text(core.pp_var(ref, context)),
          _pp_transforms(context, transforms),
      ])
  )


def _get_pp_rule(eqn, context, settings) -> pp.Doc:
  # Pretty prints `a = get x i` as `x[i] <- a`
  y, = eqn.outvars
  x, *flat_idx = eqn.invars
  transforms = tree_util.tree_unflatten(eqn.params["tree"], flat_idx)
  lhs = core.pp_vars([y], context, print_shapes=settings.print_shapes)
  return pp.concat(
      [lhs, pp.text(" <- "), pp_ref_transforms(context, x, transforms)]
  )
core.pp_eqn_rules[get_p] = _get_pp_rule

def _swap_pp_rule(eqn, context, settings) -> pp.Doc:
  y, = eqn.outvars
  x, v, *flat_idx = eqn.invars
  transforms = tree_util.tree_unflatten(eqn.params["tree"], flat_idx)
  if type(y) is core.DropVar:
    # In the case of a set (ignored return value),
    # pretty print `_ = swap x v i` as `x[i] <- v`
    del y
    return pp.concat([
        pp_ref_transforms(context, x, transforms),
        pp.text(" <- "),
        pp.text(core.pp_var(v, context)),
    ])
  else:
    # pretty-print `y:T = swap x v i` as `y:T, x[i] <- x[i], v`
    x_i = pp_ref_transforms(context, x, transforms)
    y = core.pp_vars([y], context, print_shapes=settings.print_shapes)
    return pp.concat([y, pp.text(', '), x_i, pp.text(' <- '),
                      x_i, pp.text(', '),
                      pp.text(core.pp_var(v, context))])
core.pp_eqn_rules[swap_p] = _swap_pp_rule

def _addupdate_pp_rule(eqn, context, settings) -> pp.Doc:
  del settings
  # pretty-print ` = addupdate x i v` as `x[i] += v`
  () = eqn.outvars
  x, v, *flat_idx = eqn.invars
  transforms = tree_util.tree_unflatten(eqn.params["tree"], flat_idx)
  return pp.concat([
      pp_ref_transforms(context, x, transforms),
      pp.text(" += "),
      pp.text(core.pp_var(v, context)),
  ])
core.pp_eqn_rules[addupdate_p] = _addupdate_pp_rule

## get/swap/addupdate JVP rules

def _get_jvp(primals: list[Any], tangents: list[Any], **params: Any):
  ref_primal, *idx = primals
  assert isinstance(ref_primal.aval, AbstractRef)
  ref_tangent, *_ = tangents
  assert isinstance(ref_tangent.aval, AbstractRef)
  return (get_p.bind(ref_primal, *idx, **params),
          get_p.bind(ref_tangent, *idx, **params))
ad.primitive_jvps[get_p] = _get_jvp

def _swap_jvp(primals: list[Any], tangents: list[Any], **params: Any):
  ref_primal, x_primal, *idx = primals
  assert isinstance(ref_primal.aval, AbstractRef)
  ref_tangent, x_tangent, *_ = tangents
  assert isinstance(ref_tangent.aval, AbstractRef)
  x_tangent = ad_util.instantiate(x_tangent)
  return (swap_p.bind(ref_primal, x_primal, *idx, **params),
          swap_p.bind(ref_tangent, x_tangent, *idx, **params))
ad.primitive_jvps[swap_p] = _swap_jvp

def addupdate_jvp_rule(primals: list[Any], tangents: list[Any], **params: Any):
  ref_primal, x_primal, *idx = primals
  ref_tangent, x_tangent, *_ = tangents
  x_tangent = ad_util.instantiate(x_tangent)
  addupdate_p.bind(ref_primal, x_primal, *idx, **params)
  addupdate_p.bind(ref_tangent, x_tangent, *idx, **params)
  return [], []
ad.primitive_jvps[addupdate_p] = addupdate_jvp_rule

##  get/swap/addupdate transpose rules

def _get_transpose(g, ref, *idx, **params):
  # get transpose is addupdate
  if type(g) is not ad_util.Zero:
    addupdate_p.bind(ref, g, *idx, **params)
  return [None] + [None] * len(idx)
ad.primitive_transposes[get_p] = _get_transpose

def _swap_transpose(g, ref, x, *idx, **params):
  del x  # old value doesn't matter anymore
  # swap transpose is swap
  x_bar = swap_p.bind(ref, ad_util.instantiate(g), *idx, **params)
  return [None, x_bar] + [None] * len(idx)
ad.primitive_transposes[swap_p] = _swap_transpose

def addupdate_transpose(cts_in, ref, x, *idx, **params):
  # addupdate transpose is get
  del cts_in, x
  g = get_p.bind(ref, *idx, **params)
  return [None, g] + [None] * len(idx)
ad.primitive_transposes[addupdate_p] = addupdate_transpose

## get/swap/addupdate partial_eval_custom rules

def _state_partial_eval_custom(prim, saveable, unks_in, inst_in, eqn):
  if any(unks_in):
    res = [v for v, inst in zip(eqn.invars, inst_in) if not inst]
    return None, eqn, [True] * len(eqn.outvars), [True] * len(eqn.outvars), res
  elif saveable(prim, *[var.aval for var in eqn.invars], **eqn.params):
    return eqn, None, [False] * len(eqn.outvars), [False] * len(eqn.outvars), []
  res = [v for v, inst in zip(eqn.invars, inst_in) if not inst]
  return eqn, eqn, [False] * len(eqn.outvars), [True] * len(eqn.outvars), res

pe.partial_eval_jaxpr_custom_rules[get_p] = partial(_state_partial_eval_custom,
                                                    get_p)
pe.partial_eval_jaxpr_custom_rules[swap_p] = partial(_state_partial_eval_custom,
                                                     swap_p)
pe.partial_eval_jaxpr_custom_rules[addupdate_p] = partial(
    _state_partial_eval_custom, addupdate_p)

##  get/swap/addupdate batching rules

def _batch_indexer(indexer: indexing.NDIndexer, dims,
                   axis_size: int,
                   ref_shape: tuple[int, ...],
                   ref_dim: int | batching.NotMapped,
                   idx_is_batched: bool) -> indexing.NDIndexer:
  indices = indexer.indices
  indices_dims = dims.indices
  new_indices: list[Array | indexing.Slice | int] = []
  new_integer_indexer_shape = (axis_size, *indexer.int_indexer_shape)
  for idx, dim in zip(indices, indices_dims):
    if idx_is_batched:
      # If at least one of the idx is batched, we broadcast them all and move the
      # batch dim to the front.
      if isinstance(idx, indexing.Slice):
        # size is static, but start can be dynamic
        # Check if start is static (which it can be)
        is_static_slice = len(tree_util.tree_leaves(idx)) == 0
        if is_static_slice:
          new_indices.append(idx)
          continue
        dim = dim.start
        if dim is batching.not_mapped:
          # Broadcasting the slice is free (the start index stays the same)
          new_indices.append(idx)
        else:
          raise NotImplementedError(
              f"No support for vmapping over nontrivial slices just yet: {idx}")
      else:
        # Check if we are indexing with a scalar or not. If we are indexing
        # with a scalar and we are not batched, we can avoid broadcasting it.
        assert hasattr(idx, "shape")
        if not idx.shape:
          if dim is not batching.not_mapped:
            assert idx.shape == (axis_size,)
            idx = lax.broadcast_in_dim(idx, new_integer_indexer_shape, (0,))
          new_indices.append(idx)
        else:
          if dim is batching.not_mapped:
            bcast_dims = tuple(range(1, np.ndim(idx) + 1))
            idx = lax.broadcast_in_dim(idx, new_integer_indexer_shape,
                                       bcast_dims)
          else:
            idx = batching.moveaxis(idx, dim, 0)
          new_indices.append(idx)
    else:
      if ref_dim is not batching.not_mapped:
        if not isinstance(idx, indexing.Slice):
          assert hasattr(idx, "shape")
          if idx.shape:
            bcast_dims = tuple(range(1, np.ndim(idx) + 1))
            idx = lax.broadcast_in_dim(idx, new_integer_indexer_shape,
                                      bcast_dims)
      new_indices.append(idx)
  if ref_dim is not batching.not_mapped:
    iota = lax.broadcasted_iota(np.dtype('int32'), new_integer_indexer_shape, 0)
    new_indices.insert(ref_dim, iota)
  return indexing.NDIndexer(tuple(new_indices), ref_shape,
                            new_integer_indexer_shape,
                            validate=True)

def _get_vmap(batched_args, batched_dims, *, tree):
  axis_size, = {x.shape[d] for x, d in zip(batched_args, batched_dims)
                if d is not batching.not_mapped}
  ref, *flat_idxs = batched_args
  ref_dim, *flat_idx_dims = batched_dims
  indexers = tree_util.tree_unflatten(tree, flat_idxs)
  indexers_dims = tree_util.tree_unflatten(tree, flat_idx_dims)

  idx_is_batched = any(i_dim is not batching.not_mapped
                       for i_dim in flat_idx_dims)
  if len(indexers) > 1:
    raise NotImplementedError("Batching with multiple indexers not supported.")
  # TODO(sharadmv): handle vmap of multiple indexers
  indexers = tuple(_batch_indexer(indexer, dims, axis_size,
                                  ref.shape, ref_dim, idx_is_batched)
                     for indexer, dims in zip(indexers, indexers_dims))
  flat_indexers, tree = tree_util.tree_flatten(indexers)
  return get_p.bind(ref, *flat_indexers, tree=tree), 0
batching.primitive_batchers[get_p] = _get_vmap

def _swap_vmap(batched_args, batched_dims, *, tree):
  axis_size, = {x.shape[d] for x, d in zip(batched_args, batched_dims)
                if d is not batching.not_mapped}
  ref, val, *flat_idxs = batched_args
  ref_dim, val_dim, *flat_idx_dims = batched_dims
  indexers = tree_util.tree_unflatten(tree, flat_idxs)
  indexers_dims = tree_util.tree_unflatten(tree, flat_idx_dims)

  ref_is_batched = ref_dim is not batching.not_mapped
  val_is_batched = val_dim is not batching.not_mapped
  idx_is_batched = any(i_dim is not batching.not_mapped
                       for i_dim in flat_idx_dims)
  if len(indexers) > 1:
    raise NotImplementedError("Batching with multiple indexers not supported.")
  # TODO(sharadmv): handle vmap of multiple indexers
  indexers = tuple(_batch_indexer(indexer, dims, axis_size,
                                  ref.shape, ref_dim, idx_is_batched)
                     for indexer, dims in zip(indexers, indexers_dims))
  flat_indexers, tree = tree_util.tree_flatten(indexers)
  if (ref_is_batched or idx_is_batched) and not val_is_batched:
    val = batching.broadcast(val, axis_size, 0)
  if val_is_batched:
    val = batching.moveaxis(val, val_dim, 0)
  return swap_p.bind(ref, val, *flat_indexers, tree=tree), 0
batching.primitive_batchers[swap_p] = _swap_vmap

def _addupdate_vmap(batched_args, batched_dims, *, tree):
  axis_size, = {x.shape[d] for x, d in zip(batched_args, batched_dims)
                if d is not batching.not_mapped}
  ref, val, *flat_idxs = batched_args
  ref_dim, val_dim, *flat_idx_dims = batched_dims
  indexers = tree_util.tree_unflatten(tree, flat_idxs)
  indexers_dims = tree_util.tree_unflatten(tree, flat_idx_dims)

  ref_is_batched = ref_dim is not batching.not_mapped
  val_is_batched = val_dim is not batching.not_mapped
  idx_is_batched = any(i_dim is not batching.not_mapped
                       for i_dim in flat_idx_dims)
  if len(indexers) > 1:
    raise NotImplementedError("Batching with multiple indexers not supported.")
  # TODO(sharadmv): handle vmap of multiple indexers
  indexers = tuple(_batch_indexer(indexer, dims, axis_size,
                                  ref.shape, ref_dim, idx_is_batched)
                     for indexer, dims in zip(indexers, indexers_dims))
  flat_indexers, tree = tree_util.tree_flatten(indexers)
  if (ref_is_batched or idx_is_batched) and not val_is_batched:
    val = batching.broadcast(val, axis_size, 0)
  if val_is_batched:
    val = batching.moveaxis(val, val_dim, 0)
  return addupdate_p.bind(ref, val, *flat_indexers, tree=tree), []
batching.primitive_batchers[addupdate_p] = _addupdate_vmap

# Currently, JAX doesn't have a primitive that does an equal-rank broadcast.
# We could use `jnp.broadcast_to` but that lowers to squeezing,
# then broadcast_in_dim. Triton has an equal-rank broadcast (`tl.broadcast_to`)
# so in the lowering, we have to expand out those squeezed dimensions again.
# Having a simple `broadcast_to` primitive allows us to lower directly
# to `tl.broadcast_to`.
broadcast_to_p = core.Primitive('broadcast_to')

def broadcast_to(a: Array, shape: tuple[int, ...]) -> Array:
  import jax.numpy as jnp
  a = jnp.asarray(a)
  if a.shape == shape:
    return a
  return broadcast_to_p.bind(a, shape=shape)

@broadcast_to_p.def_impl
def _broadcast_to_impl(a, *, shape):
  import jax.numpy as jnp
  return jnp.broadcast_to(a, shape)

@broadcast_to_p.def_abstract_eval
def _broadcast_to_abstract_eval(aval, *, shape):
  return core.ShapedArray(shape, aval.dtype)

mlir.register_lowering(
    broadcast_to_p, mlir.lower_fun(_broadcast_to_impl, False)
)

# === AD rules for mutable arrays ===

ad.defjvp(core.mutable_array_p, lambda g, _: core.mutable_array(g))
ad.defjvp(core.freeze_p, lambda g, _: core.freeze(g))
