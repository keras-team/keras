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
from functools import partial
import warnings

from jax import tree_util
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import util
from jax._src.core import ShapedArray
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lax import convolution
from jax._src.lax import lax
from jax._src.lax import slicing
from jax._src.lax.other import logaddexp
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.typing import Array
import numpy as np
from jax._src.core import ClosedJaxpr
from jax._src.core import jaxpr_as_fun
from jax._src.interpreters.ad import jvp_jaxpr
from jax._src import ad_util

map = util.safe_map
zip = util.safe_zip


def _reduce_window(
    operand,
    init_value,
    computation,
    window_dimensions: core.Shape,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    base_dilation: Sequence[int] | None = None,
    window_dilation: Sequence[int] | None = None,
):
  flat_operands, operand_tree = tree_util.tree_flatten(operand)
  flat_init_values, init_value_tree = tree_util.tree_flatten(init_value)
  if operand_tree != init_value_tree:
    raise ValueError(
        "Operands must have the same tree structure as "
        f"init_values: {operand_tree} vs. {init_value_tree}"
    )
  if len(flat_operands) != len(flat_init_values):
    raise ValueError(
        "Must have same total number of operands as init_values: "
        f" {len(flat_operands)} vs. {len(flat_init_values)}"
    )

  if len(flat_operands) == 0:
    raise ValueError("reduce_window must have at least one operand.")
  if isinstance(padding, str):
    dilated_window_dims = (
        window_dimensions if window_dilation is None else
        lax._dilate_shape(window_dimensions, window_dilation))
    padding = tuple(lax.padtype_to_pads(
        flat_operands[0].shape, dilated_window_dims, window_strides, padding))
  else:
    padding = tuple(padding)
  if base_dilation is None:
    base_dilation = (1,) * len(window_dimensions)
  if window_dilation is None:
    window_dilation = (1,) * len(window_dimensions)
  monoid_reducer = _get_monoid_window_reducer(computation, flat_init_values)
  if monoid_reducer:
    return monoid_reducer(operand, window_dimensions, window_strides, padding,
                          base_dilation, window_dilation)
  else:
    flat_init_avals = map(core.get_aval, flat_init_values)
    jaxpr, out_tree = lax._variadic_reduction_jaxpr(
        computation, tuple(flat_init_avals), init_value_tree
    )
    if operand_tree != out_tree:
      raise ValueError(
        'reduce_window output must have the same tree structure as the operands'
        f' {operand_tree} vs. {out_tree}')
    out_flat = reduce_window_p.bind(
        *flat_operands,
        *flat_init_values,
        jaxpr=jaxpr.jaxpr,
        consts=tuple(jaxpr.consts),
        window_dimensions=tuple(window_dimensions),
        window_strides=tuple(window_strides),
        padding=padding,
        base_dilation=tuple(base_dilation),
        window_dilation=tuple(window_dilation),
    )
    return tree_util.tree_unflatten(out_tree, out_flat)



def reduce_window(
    operand,
    init_value,
    computation: Callable,
    window_dimensions: core.Shape,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    base_dilation: Sequence[int] | None = None,
    window_dilation: Sequence[int] | None = None,
) -> Array:
  """Wraps XLA's `ReduceWindowWithGeneralPadding
  <https://www.tensorflow.org/xla/operation_semantics#reducewindow>`_
  operator.
  """
  return _reduce_window(
      operand,
      init_value,
      computation,
      window_dimensions,
      window_strides,
      padding,
      base_dilation,
      window_dilation,
  )


def _get_monoid_window_reducer(
    monoid_op, xs: Sequence[Array]
) -> Callable | None:
  if len(xs) != 1:
    return None
  x, = xs
  aval = core.get_aval(x)
  if core.is_concrete(x) and aval.shape == ():
    val = core.to_concrete_value(x)
    if monoid_op is lax.add:
      return val == 0 and _reduce_window_sum
    elif monoid_op is lax.max:
      return (val == lax._get_max_identity(aval.dtype)
              and _reduce_window_max)
    elif monoid_op is lax.min:
      return (val == lax._get_min_identity(aval.dtype)
              and _reduce_window_min)
  return None


def _reduce_window_sum(operand: Array, window_dimensions: core.Shape,
                       window_strides: Sequence[int],
                       padding: Sequence[tuple[int, int]],
                       base_dilation: Sequence[int] | None = None,
                       window_dilation: Sequence[int] | None = None) -> Array:
  if base_dilation is None:
    base_dilation = (1,) * len(window_dimensions)
  if window_dilation is None:
    window_dilation = (1,) * len(window_dimensions)
  return reduce_window_sum_p.bind(
      operand, window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding),
      base_dilation=tuple(base_dilation),
      window_dilation=tuple(window_dilation))

def _reduce_window_prod(operand: Array, window_dimensions: core.Shape,
                        window_strides: Sequence[int],
                        padding: Sequence[tuple[int, int]],
                        base_dilation: Sequence[int] | None = None,
                        window_dilation: Sequence[int] | None = None) -> Array:
  init_value = lax._const(operand, 1)
  jaxpr, consts = lax._reduction_jaxpr(lax.mul, core.get_aval(init_value))
  if base_dilation is None:
    base_dilation = (1,) * len(window_dimensions)
  if window_dilation is None:
    window_dilation = (1,) * len(window_dimensions)
  out, = reduce_window_p.bind(
      operand, init_value, jaxpr=jaxpr, consts=consts,
      window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding),
      base_dilation=tuple(base_dilation),
      window_dilation=tuple(window_dilation))
  return out

def _reduce_window_max(operand: Array, window_dimensions: core.Shape,
                       window_strides: Sequence[int],
                       padding: Sequence[tuple[int, int]],
                       base_dilation: Sequence[int] | None = None,
                       window_dilation: Sequence[int] | None = None) -> Array:
  if base_dilation is None:
    base_dilation = (1,) * len(window_dimensions)
  if window_dilation is None:
    window_dilation = (1,) * len(window_dimensions)
  return reduce_window_max_p.bind(
      operand, window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding),
      base_dilation=tuple(base_dilation),
      window_dilation=tuple(window_dilation))

def _reduce_window_min(operand: Array, window_dimensions: core.Shape,
                       window_strides: Sequence[int],
                       padding: Sequence[tuple[int, int]],
                       base_dilation: Sequence[int] | None = None,
                       window_dilation: Sequence[int] | None = None) -> Array:
  if base_dilation is None:
    base_dilation = (1,) * len(window_dimensions)
  if window_dilation is None:
    window_dilation = (1,) * len(window_dimensions)
  return reduce_window_min_p.bind(
      operand, window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding),
      base_dilation=tuple(base_dilation),
      window_dilation=tuple(window_dilation))

def _reduce_window_logaddexp(
    operand: Array, window_dimensions: core.Shape,
    window_strides: Sequence[int],
    padding: Sequence[tuple[int, int]],
    base_dilation: Sequence[int] | None = None,
    window_dilation: Sequence[int] | None = None) -> Array:
  init_value = lax._const(operand, -np.inf)
  jaxpr, consts = lax._reduction_jaxpr(logaddexp, core.get_aval(init_value))
  if base_dilation is None:
    base_dilation = (1,) * len(window_dimensions)
  if window_dilation is None:
    window_dilation = (1,) * len(window_dimensions)
  out, = reduce_window_p.bind(
      operand, init_value, jaxpr=jaxpr, consts=consts,
      window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding),
      base_dilation=tuple(base_dilation),
      window_dilation=tuple(window_dilation))
  return out

def _select_and_scatter(operand: Array, select: Callable,
                        window_dimensions: core.Shape,
                        window_strides: Sequence[int],
                        padding: Sequence[tuple[int, int]], source: Array,
                        init_value: Array, scatter: Callable) -> Array:
  select_jaxpr, select_consts = lax._reduction_jaxpr(
    select, core.get_aval(init_value))
  scatter_jaxpr, scatter_consts = lax._reduction_jaxpr(
    scatter, core.get_aval(init_value))
  return select_and_scatter_p.bind(
      operand, source, init_value, select_jaxpr=select_jaxpr,
      select_consts=select_consts, scatter_jaxpr=scatter_jaxpr,
      scatter_consts=scatter_consts, window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding))

def _select_and_scatter_add(source: Array, operand: Array,
                            select_prim: core.Primitive,
                            window_dimensions: core.Shape,
                            window_strides: Sequence[int],
                            padding: Sequence[tuple[int, int]]) -> Array:
  return select_and_scatter_add_p.bind(
      source, operand, select_prim=select_prim,
      window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding))

def _select_and_gather_add(tangents: Array, operand: Array,
                           select_prim: core.Primitive,
                           window_dimensions: core.Shape,
                           window_strides: Sequence[int],
                           padding: Sequence[tuple[int, int]],
                           base_dilation: Sequence[int],
                           window_dilation: Sequence[int]) -> Array:
  """Extracts the tangent corresponding to the minimum or maximum element in
  each window of the `operand` array.

  Wraps XLA's `ReduceWindow
  <https://www.tensorflow.org/xla/operation_semantics#reducewindow>`_
  operator, which applies a reduction function to all elements in each window of
  the input multi-dimensional array. In this case, the input multi-dimensional
  array is built by packing each element in the `operand` array with its
  corresponding element in the `tangents` array.

  Args:
    tangents: an array
    operand: an array with the same shape as `tangents`
    select_prim: a reduction function (restricted to `ge_p` and `le_p`)
    window_dimensions: an array of integers for window dimension values
    window_strides: an array of integers for window stride values
    base_dilation: an array of integers for base dilation values
    window_dilation: an array of integers for window dilation values

  Returns:
    An array containing the elements in `tangents` corresponding to the output
    of the reduction of `operand` fin each window.
  """
  return select_and_gather_add_p.bind(
      tangents, operand, select_prim=select_prim,
      window_dimensions=tuple(window_dimensions),
      window_strides=tuple(window_strides), padding=tuple(padding),
      base_dilation=tuple(base_dilation),
      window_dilation=tuple(window_dilation))


def _reduce_window_abstract_eval_rule(
    *avals,
    jaxpr,
    consts,
    window_dimensions,
    window_strides,
    padding,
    base_dilation,
    window_dilation,
):
  operand_avals, init_val_avals = util.split_list(avals, [len(avals) // 2])
  if any(
      o.dtype != iv.dtype for o, iv in zip(operand_avals, init_val_avals)
  ):
    msg = ("reduce_window got inconsistent dtypes for operands and init_values:"
           " got operand dtypes {} and init_value dtypes {}.")
    raise TypeError(msg.format([o.dtype for o in operand_avals],
                               [iv.dtype for iv in init_val_avals]))
  if any(len(v.shape) != 0 for v in init_val_avals):
    msg = ("reduce_window expected init_values to be scalars but init_values "
           "have shapes {}.")
    raise TypeError(msg.format([v.shape for v in init_val_avals]))
  out_shape = _common_reduce_window_shape_rule(
      operand_avals[0],
      window_dimensions,
      window_strides,
      padding,
      base_dilation,
      window_dilation,
  )
  return tuple(ShapedArray(out_shape, op.dtype) for op in operand_avals)


def _generic_reduce_window_batch_rule(
    batched_args,
    batch_dims,
    *,
    jaxpr,
    consts,
    window_dimensions,
    window_strides,
    padding,
    base_dilation,
    window_dilation,
):
  num_operands = len(batched_args) // 2
  operands, init_values = util.split_list(batched_args, [num_operands])
  operand_bdims, init_value_bdims = util.split_list(batch_dims, [num_operands])

  if any(init_bdim is not None for init_bdim in init_value_bdims):
    raise NotImplementedError("reduce_window batching is not implemented for "
                              "initial values")

  size = next(x.shape[ax] for x, ax in zip(operands, operand_bdims)
              if ax is not None)
  operands = [batching.bdim_at_front(arg, bdim, size)
              for arg, bdim in zip(operands, operand_bdims)]
  window_dimensions = (1,) + window_dimensions
  window_strides = (1,) + window_strides
  padding = ((0, 0),) + padding
  base_dilation = (1,) + base_dilation
  window_dilation = (1,) + window_dilation
  outs = reduce_window_p.bind(
      *(operands + init_values), jaxpr=jaxpr, consts=consts,
      window_dimensions=window_dimensions, window_strides=window_strides,
      padding=padding, base_dilation=base_dilation,
      window_dilation=window_dilation)
  return outs, (0,) * num_operands


reduce_window_p = core.Primitive('reduce_window')


def reduce_window_jvp(
    primals,
    tangents,
    window_dimensions,
    window_strides,
    padding,
    base_dilation,
    window_dilation,
    jaxpr,
    consts,
):

  reduction_jaxpr = jaxpr

  n = len(primals) // 2  # number of primal operands
  operand, init_value = util.split_list(primals, [n])
  operand_tangent, init_value_tangent = util.split_list(tangents, [n])
  if not all(isinstance(t, ad.Zero) for t in init_value_tangent):
    raise TypeError("reduce_window jvp does not support non-zero init_value_tangent.")

  init_value_tangent = map(ad_util.instantiate, init_value_tangent)
  c_reduction_jaxpr = ClosedJaxpr(reduction_jaxpr, consts)
  jvp_reduction = jvp_jaxpr(c_reduction_jaxpr, (True,) * len(tangents), [False] * len(init_value_tangent))[0]

  def wrapper(left, right):
    pl, tl = util.split_list(left, [n])
    pr, tr = util.split_list(right, [n])
    return jaxpr_as_fun(jvp_reduction)(*pl, *pr, *tl, *tr)

  jvp_primals_tangents = _reduce_window(
      operand=[*operand, *operand_tangent],
      init_value=[*init_value, *init_value_tangent],
      computation=wrapper,
      window_dimensions=window_dimensions,
      window_strides=window_strides,
      padding=padding,
      base_dilation=base_dilation,
      window_dilation=window_dilation,
  )
  primals, tangents = util.split_list(jvp_primals_tangents, [len(jvp_primals_tangents) // 2])
  return [*primals], [*tangents]

ad.primitive_jvps[reduce_window_p] = reduce_window_jvp
reduce_window_p.multiple_results = True
reduce_window_p.def_impl(partial(dispatch.apply_primitive, reduce_window_p))
reduce_window_p.def_abstract_eval(_reduce_window_abstract_eval_rule)
batching.primitive_batchers[reduce_window_p] = _generic_reduce_window_batch_rule


def _generic_reduce_window_lower(
    ctx,
    *args,
    jaxpr,
    consts,
    window_dimensions,
    window_strides,
    padding,
    base_dilation,
    window_dilation,
):
  operands, init_values = util.split_list(args, [len(args) // 2])
  _, init_value_avals = util.split_list(ctx.avals_in, [len(operands)])

  def reducer_body(reducer: ir.Block) -> Sequence[ir.Value]:
    if jaxpr.effects:
      raise NotImplementedError('Cannot lower effectful `reduce_window`.')
    out_nodes, _ = mlir.jaxpr_subcomp(ctx.module_context, jaxpr, ctx.name_stack,
        mlir.TokenSet(), consts, *reducer.arguments,  # type: ignore[misc]
        dim_var_values=ctx.dim_var_values)
    return mlir.flatten_ir_values(out_nodes)

  return mlir.reduce_window(
      ctx,
      reducer_name="generic_reduce_window_reducer",
      reducer_body=reducer_body,
      operands=operands,
      init_values=init_values,
      init_values_avals=init_value_avals,
      out_avals=ctx.avals_out,
      window_dimensions=window_dimensions,
      window_strides=window_strides,
      base_dilation=base_dilation,
      window_dilation=window_dilation,
      padding=padding,
  )


mlir.register_lowering(reduce_window_p, _generic_reduce_window_lower)


def _reduce_window_sum_shape_rule(operand, *, window_dimensions, window_strides,
                                  padding, base_dilation, window_dilation):
  if not dtypes.issubdtype(operand.dtype, np.number):
    msg = "operand to reduce_window_sum must have a number dtype, got {}"
    raise TypeError(msg.format(np.dtype(operand.dtype).name))
  return _common_reduce_window_shape_rule(operand, window_dimensions,
                                          window_strides, padding,
                                          base_dilation, window_dilation)

def _reduce_window_sum_transpose_rule(cotangent, operand, *, window_dimensions,
                                      window_strides, padding, base_dilation,
                                      window_dilation):
  assert ad.is_undefined_primal(operand)
  input_shape = operand.aval.shape
  pads = convolution._conv_general_vjp_lhs_padding(
      input_shape, window_dimensions, window_strides, cotangent.shape, padding,
      base_dilation, window_dilation)
  ones = [1] * len(input_shape)
  padding_config = [(lo, hi, stride - 1)
                    for (lo, hi), stride in zip(pads, window_strides)]
  pad_cotangent = lax.pad(cotangent, lax._zero(cotangent), padding_config)
  result = _reduce_window_sum(pad_cotangent, window_dimensions, base_dilation,
                              [(0, 0)] * len(input_shape),
                              base_dilation=ones,
                              window_dilation=window_dilation)
  assert result.shape == input_shape, (result.shape, input_shape)
  return [result]

def _reduce_window_batch_rule(reduce_window, batched_args, bdims, *,
                              window_dimensions, window_strides, padding,
                              base_dilation, window_dilation):
  operand, = batched_args
  bdim, = bdims

  if bdim is not None:
    window_dimensions = \
        window_dimensions[:bdim] + (1,) + window_dimensions[bdim:]
    window_strides = window_strides[:bdim] + (1,) + window_strides[bdim:]
    padding = padding[:bdim] + ((0, 0),) + padding[bdim:]
    base_dilation = base_dilation[:bdim] + (1,) + base_dilation[bdim:]
    window_dilation = window_dilation[:bdim] + (1,) + window_dilation[bdim:]

  operand = reduce_window(operand, window_dimensions, window_strides, padding,
                          base_dilation, window_dilation)
  return operand, bdim

reduce_window_sum_p = lax.standard_primitive(
    _reduce_window_sum_shape_rule, lax._input_dtype, 'reduce_window_sum')
ad.deflinear2(reduce_window_sum_p, _reduce_window_sum_transpose_rule)
batching.primitive_batchers[reduce_window_sum_p] = partial(
  _reduce_window_batch_rule, _reduce_window_sum)

def _reduce_window_chooser_jvp_rule(prim, g, operand, *, window_dimensions,
                                    window_strides, padding, base_dilation,
                                    window_dilation):
  assert prim is lax.max_p or prim is lax.min_p
  select_prim = lax.ge_p if prim is lax.max_p else lax.le_p
  return _select_and_gather_add(g, operand, select_prim, window_dimensions,
                                window_strides, padding, base_dilation,
                                window_dilation)


def _common_reduce_window_shape_rule(
    operand,
    window_dimensions,
    window_strides,
    padding,
    base_dilation,
    window_dilation,
):
  lax._check_shapelike("reduce_window", "window_dimensions", window_dimensions,
                       non_zero_shape=True)
  lax._check_shapelike("reduce_window", "window_strides", window_strides,
                       non_zero_shape=True)
  lax._check_shapelike("reduce_window", "base_dilation", base_dilation)
  lax._check_shapelike("reduce_window", "window_dilation", window_dilation)
  if operand.ndim != len(window_dimensions):
    msg = (
        "reduce_window got the wrong number of window_dimensions for "
        "operand: got operand shape {} with window_dimensions {}."
    )
    raise TypeError(msg.format(operand.shape, window_dimensions))
  if len(window_strides) != len(window_dimensions):
    msg = ("reduce_window got inconsistent window_strides and "
           "window_dimensions: got window_strides {} and window_dimensions {}.")
    raise TypeError(msg.format(window_strides, window_dimensions))
  if len(base_dilation) != len(window_dimensions):
    msg = ("reduce_window got inconsistent base_dilation and "
           "window_dimensions: got base_dilation {} and window_dimensions {}.")
    raise TypeError(msg.format(base_dilation, window_dimensions))
  if len(window_dilation) != len(window_dimensions):
    msg = ("reduce_window got inconsistent window_dilation and "
           "window_dimensions: got window_dilation {} and window_dimensions "
           "{}.")
    raise TypeError(msg.format(window_dilation, window_dimensions))

  return reduce_window_shape_tuple(operand.shape, window_dimensions,
                                   window_strides, padding, base_dilation,
                                   window_dilation)

def reduce_window_shape_tuple(operand_shape, window_dimensions, window_strides,
                              padding, base_dilation=None,
                              window_dilation=None):
  if base_dilation is not None:
    operand_shape = lax._dilate_shape(operand_shape, base_dilation)
  if window_dilation is not None:
    window_dimensions = lax._dilate_shape(window_dimensions, window_dilation)
  operand_padded = tuple(d + pl + ph for d, (pl, ph) in zip(operand_shape, padding))
  return tuple(map(core.stride_dim, operand_padded, window_dimensions, window_strides))


reduce_window_max_p = lax.standard_primitive(
    _common_reduce_window_shape_rule, lax._input_dtype, 'reduce_window_max')
ad.defjvp(reduce_window_max_p, partial(_reduce_window_chooser_jvp_rule,
                                       lax.max_p))
batching.primitive_batchers[reduce_window_max_p] = partial(
  _reduce_window_batch_rule, _reduce_window_max)

reduce_window_min_p = lax.standard_primitive(
    _common_reduce_window_shape_rule, lax._input_dtype, 'reduce_window_min')
ad.defjvp(reduce_window_min_p, partial(_reduce_window_chooser_jvp_rule,
                                       lax.min_p))

_reduce_window_min_batch_rule = partial(_reduce_window_batch_rule,
                                        _reduce_window_min)
batching.primitive_batchers[reduce_window_min_p] = partial(
  _reduce_window_batch_rule, _reduce_window_min)


def _reduce_window_lower(
    reduce_op,
    init_value,
    ctx,
    operand,
    *,
    window_dimensions,
    window_strides,
    padding,
    base_dilation,
    window_dilation,
):

  operand_aval, = ctx.avals_in
  scalar_aval = operand_aval.update(shape=())

  return mlir.reduce_window(
      ctx,
      reducer_name=f"reduce_window_{scalar_aval.dtype}_reducer",
      reducer_body=lambda reducer: [reduce_op(*reducer.arguments)],
      operands=[operand],
      init_values=[
          mlir.full_like_aval(ctx, init_value(scalar_aval.dtype), scalar_aval)
      ],
      init_values_avals=[scalar_aval],
      out_avals=ctx.avals_out,
      window_dimensions=window_dimensions,
      window_strides=window_strides,
      base_dilation=base_dilation,
      window_dilation=window_dilation,
      padding=padding,
  )


mlir.register_lowering(reduce_window_sum_p, partial(
    _reduce_window_lower, hlo.add, lambda _: 0))
mlir.register_lowering(reduce_window_min_p, partial(
    _reduce_window_lower, mlir.min_hlo, lax._get_min_identity))
mlir.register_lowering(reduce_window_max_p, partial(
    _reduce_window_lower, mlir.max_hlo, lax._get_max_identity))



def _select_and_scatter_shape_rule(
    operand, source, init_value, *, select_jaxpr, select_consts, scatter_jaxpr,
    scatter_consts, window_dimensions, window_strides, padding):
  lax._check_shapelike("select_and_scatter", "window_dimensions",
                       window_dimensions)
  lax._check_shapelike("select_and_scatter", "window_strides", window_strides)
  if len(window_dimensions) != len(window_strides):
    msg = ("select_and_scatter got inconsistent window_strides and "
           "window_dimensions: got window_strides {} and window_dimensions {}.")
    raise TypeError(msg.format(window_strides, window_dimensions))
  return operand.shape

select_and_scatter_p = lax.standard_primitive(
    _select_and_scatter_shape_rule, lax._input_dtype, 'select_and_scatter')

def _select_and_scatter_lower(
    ctx, operand, source, init_value, *, select_jaxpr,
    select_consts, scatter_jaxpr, scatter_consts, window_dimensions,
    window_strides, padding):
  operand_aval, source_aval, init_value_aval = ctx.avals_in
  aval_out, = ctx.avals_out
  scalar_aval = operand_aval.update(shape=())
  scalar_type = mlir.aval_to_ir_type(scalar_aval)
  op = hlo.SelectAndScatterOp(
      mlir.aval_to_ir_type(aval_out),
      operand,
      source,
      init_value,
      window_dimensions=mlir.dense_int_array(window_dimensions),
      window_strides=mlir.dense_int_array(window_strides),
      padding=ir.DenseIntElementsAttr.get(np.asarray(padding, np.int64),
                                          shape=(len(padding), 2)))
  select = op.select.blocks.append(scalar_type, scalar_type)
  with ir.InsertionPoint(select):
    if select_jaxpr.effects:
      raise NotImplementedError('Cannot lower effectful `select`.')
    out_nodes, _ = mlir.jaxpr_subcomp(ctx.module_context, select_jaxpr,
                                      ctx.name_stack,
                                      mlir.TokenSet(), select_consts,
                                      *select.arguments,
                                      dim_var_values=ctx.dim_var_values)
    hlo.return_(mlir.flatten_ir_values(out_nodes))
  scatter = op.scatter.blocks.append(scalar_type, scalar_type)
  with ir.InsertionPoint(scatter):
    if scatter_jaxpr.effects:
      raise NotImplementedError('Cannot lower effectful `scatter`.')
    out_nodes, _ = mlir.jaxpr_subcomp(ctx.module_context, scatter_jaxpr,
                                      ctx.name_stack,
                                      mlir.TokenSet(), scatter_consts,
                                      *scatter.arguments,
                                      dim_var_values=ctx.dim_var_values)
    hlo.return_(mlir.flatten_ir_values(out_nodes))
  return op.results

mlir.register_lowering(select_and_scatter_p, _select_and_scatter_lower)

def _select_and_scatter_add_shape_rule(
    source, operand, *, select_prim, window_dimensions, window_strides,
    padding):
  return operand.shape

def _select_and_scatter_add_jvp(
    primals, tangents, *, select_prim, window_dimensions, window_strides,
    padding):
  source, operand = primals
  g_source, g_operand = tangents
  val_out = _select_and_scatter_add(
      source, operand, select_prim, window_dimensions, window_strides,
      padding)
  del g_operand
  if type(g_source) is ad_util.Zero:
    tangent_out = ad_util.Zero.from_primal_value(val_out)
  else:
    tangent_out = _select_and_scatter_add(
        g_source, operand, select_prim, window_dimensions,
        window_strides, padding)
  return val_out, tangent_out

def _select_and_scatter_add_transpose(
    t, source, operand, *, select_prim, window_dimensions, window_strides,
    padding):
  assert ad.is_undefined_primal(source) and not ad.is_undefined_primal(operand)
  if type(t) is ad_util.Zero:
    return [ad_util.Zero(source.aval), None]
  ones = (1,) * len(window_dimensions)
  source_t = _select_and_gather_add(t, operand, select_prim, window_dimensions,
                                    window_strides, padding, ones, ones)
  return [source_t, None]

def _select_and_scatter_add_batch_rule(
    batched_args, batch_dims, *, select_prim, window_dimensions, window_strides,
    padding):
  source, operand = batched_args
  s_bdim, o_bdim = batch_dims
  size = next(a.shape[bdim] for a, bdim in zip(batched_args, batch_dims)
              if bdim is not None)
  source = batching.bdim_at_front(source, s_bdim, size)
  operand = batching.bdim_at_front(operand, o_bdim, size)

  window_dimensions = (1,) + window_dimensions
  window_strides = (1,) + window_strides
  padding = ((0, 0),) + padding
  out = _select_and_scatter_add(source, operand, select_prim, window_dimensions,
                                window_strides, padding)
  return out, 0

select_and_scatter_add_p = lax.standard_primitive(
    _select_and_scatter_add_shape_rule, lax._input_dtype,
    'select_and_scatter_add')

ad.primitive_transposes[select_and_scatter_add_p] = \
    _select_and_scatter_add_transpose
ad.primitive_jvps[select_and_scatter_add_p] = _select_and_scatter_add_jvp
batching.primitive_batchers[select_and_scatter_add_p] = \
    _select_and_scatter_add_batch_rule

def _select_and_scatter_add_impl(source, operand, *,
                                 select_prim, window_dimensions, window_strides,
                                 padding, expand_padding):
  dtype = source.dtype
  select = lambda x, y: select_prim.bind(x, y)
  scatter = lax.bitwise_or if dtype == np.bool_ else lax.add
  if expand_padding:
    operand_shape = operand.shape
    original_padding = padding
    identity = (lax._get_max_identity if select_prim is lax.ge_p
                else lax._get_min_identity)
    pads = [(lo, hi, 0) for (lo, hi) in padding]
    operand = lax.pad(operand, identity(dtype), pads)
    padding = [(0, 0) for _ in padding]
  out = _select_and_scatter(
      operand, select, window_dimensions, window_strides, padding, source,
      lax._zero(operand), scatter)
  if expand_padding:
    start_indices = [lo for (lo, hi) in original_padding]
    stop_indices = [lo + d for ((lo, hi), d) in zip(original_padding,
                                                    operand_shape)]
    out = slicing.slice(out, start_indices, stop_indices)
  return out

mlir.register_lowering(select_and_scatter_add_p, mlir.lower_fun(
    partial(_select_and_scatter_add_impl, expand_padding=False),
    multiple_results=False))
# TODO(b/161704903): workaround for XLA/CPU crash.
mlir.register_lowering(select_and_scatter_add_p, mlir.lower_fun(
    partial(_select_and_scatter_add_impl, expand_padding=True),
    multiple_results=False), platform='cpu')
# TODO(b/182390722): workaround for XLA/GPU crash.
mlir.register_lowering(select_and_scatter_add_p, mlir.lower_fun(
    partial(_select_and_scatter_add_impl, expand_padding=True),
    multiple_results=False), platform='gpu')


def _select_and_gather_add_shape_rule(
    tangents, operand, *, select_prim, window_dimensions, window_strides,
    padding, base_dilation, window_dilation):
  if tangents.shape != operand.shape:
    msg = ("select_and_gather_add tangents and operand shapes must match, "
           "got {} and {}.")
    raise TypeError(msg.format(tangents.shape, operand.shape))
  return _common_reduce_window_shape_rule(
    operand, window_dimensions, window_strides, padding, base_dilation,
    window_dilation)

def _select_and_gather_add_lowering(
    ctx: mlir.LoweringRuleContext,
    tangents, operand, *, select_prim,
    window_dimensions, window_strides, padding, base_dilation, window_dilation,
    max_bits=64):
  _, operand_aval, = ctx.avals_in
  out_aval, = ctx.avals_out
  assert isinstance(operand_aval, core.ShapedArray), operand_aval
  dtype = operand_aval.dtype
  etype = mlir.dtype_to_ir_type(dtype)
  nbits = dtypes.finfo(dtype).bits

  assert nbits <= max_bits
  double_word_reduction = nbits * 2 <= max_bits

  const = lambda dtype, x: mlir.ir_constant(np.array(x, dtype=dtype))

  def _broadcast_scalar_const(x, aval_out):
    return mlir.broadcast_in_dim(ctx, const(aval_out.dtype, x),
                                 aval_out,
                                 broadcast_dimensions=())

  if double_word_reduction:
    # TODO(b/73062247): XLA doesn't yet implement ReduceWindow on tuples, so
    # we implement a pair-wise ReduceWindow by packing two k-bit values into
    # 2k-bit unsigned integer using bit tricks.
    word_dtype = lax._UINT_DTYPES[nbits]
    double_word_dtype = lax._UINT_DTYPES[nbits * 2]
    word_type = mlir.dtype_to_ir_type(word_dtype)  # type: ignore
    # Packs two values into a double_word_type.
    def pack(a, b, ab_aval):
      word_type_ab_aval = ab_aval.update(dtype=word_dtype)
      double_word_type_ab_aval = ab_aval.update(dtype=double_word_dtype)
      a = hlo.bitcast_convert(mlir.aval_to_ir_type(word_type_ab_aval), a)
      b = hlo.bitcast_convert(mlir.aval_to_ir_type(word_type_ab_aval), b)
      a = hlo.convert(mlir.aval_to_ir_type(double_word_type_ab_aval), a)
      b = hlo.convert(mlir.aval_to_ir_type(double_word_type_ab_aval), b)
      a = hlo.shift_left(
          a, _broadcast_scalar_const(nbits, double_word_type_ab_aval))
      return hlo.or_(a, b)

    # Unpacks the first element of a double_word_type.
    def fst(t):
      assert not ir.RankedTensorType(t.type).shape
      st = hlo.shift_right_logical(t, const(double_word_dtype, nbits))
      return hlo.bitcast_convert(
          ir.RankedTensorType.get([], etype),
          hlo.convert(ir.RankedTensorType.get([], word_type), st))

    # Unpacks the second element of a double_word_type.
    def snd(t, t_aval):
      return hlo.bitcast_convert(
          mlir.aval_to_ir_type(t_aval.update(dtype=dtype)),
          hlo.convert(mlir.aval_to_ir_type(t_aval.update(dtype=word_dtype)), t))

  else:
    # The double-word trick above only works if we have a sufficiently large
    # type. As an alternative, we can pack two half words into a single word,
    # at the cost of precision.
    # TODO(b/73062247): add support for tuple reductions and remove this case.
    warnings.warn("Using reduced precision for gradient of reduce-window "
                  "min/max operator to work around missing XLA support for "
                  "pair-reductions. This is likely from a second or "
                  "higher derivative of a max-pooling operation.")
    r_nbits = nbits // 2
    # Drop/round the bottom mantissa bits.
    nexp = dtypes.finfo(dtype).nexp
    nmant = r_nbits - nexp - 1

    double_word_dtype = word_dtype = lax._UINT_DTYPES[nbits]

    # Packs two values into a double_word_type.
    def pack(a, b, ab_aval):
      word_type_ab_aval = ab_aval.update(dtype=word_dtype)
      a = hlo.reduce_precision(a, exponent_bits=mlir.i32_attr(nexp),
                                mantissa_bits=mlir.i32_attr(nmant))
      b = hlo.reduce_precision(b, exponent_bits=mlir.i32_attr(nexp),
                                mantissa_bits=mlir.i32_attr(nmant))
      a = hlo.bitcast_convert(mlir.aval_to_ir_type(word_type_ab_aval), a)
      b = hlo.bitcast_convert(mlir.aval_to_ir_type(word_type_ab_aval), b)
      b = hlo.shift_right_logical(
          b, _broadcast_scalar_const(r_nbits, word_type_ab_aval))
      return hlo.or_(a, b)

    # Unpacks the first element of a double_word_type.
    def fst(t):
      assert not ir.RankedTensorType(t.type).shape
      st = hlo.and_(t, const(word_dtype, ((1 << r_nbits) - 1) << r_nbits))
      return hlo.bitcast_convert(ir.RankedTensorType.get([], etype), st)

    # Unpacks the second element of a double_word_type.
    def snd(t, t_aval):
      return hlo.bitcast_convert(
          mlir.aval_to_ir_type(t_aval.update(dtype=dtype)),
          hlo.shift_left(t, _broadcast_scalar_const(r_nbits, t_aval.update(dtype=word_dtype))))

  assert select_prim is lax.ge_p or select_prim is lax.le_p, select_prim
  init = -np.inf if select_prim is lax.ge_p else np.inf
  double_word_out_aval = out_aval.update(dtype=double_word_dtype)

  def reducer_body(reducer: ir.Block) -> Sequence[ir.Value]:
    x: ir.Value
    y: ir.Value
    x, y = reducer.arguments  # type: ignore
    assert select_prim is lax.ge_p or select_prim is lax.le_p
    cmp_op = "GE" if select_prim is lax.ge_p else "LE"
    out = hlo.SelectOp(mlir.compare_hlo(fst(x), fst(y), cmp_op), x, y)
    return out

  res, = mlir.reduce_window(ctx,
      reducer_name="reduce_window_select_and_gather_add",
      reducer_body=reducer_body,
      operands=[pack(operand, tangents, operand_aval)],
      init_values=[pack(const(dtype, init), const(dtype, 0), core.ShapedArray((), dtype))],
      init_values_avals=[core.ShapedArray((), double_word_dtype)],
      out_avals=[double_word_out_aval],
      window_dimensions=window_dimensions,
      window_strides=window_strides,
      base_dilation=base_dilation,
      window_dilation=window_dilation,
      padding=padding)
  return [snd(res, double_word_out_aval)]

# TODO(phawkins): use this translation rule on all platforms.
def _select_and_gather_add_using_variadic_reducewindow(
    tangents, operand, *, select_prim, window_dimensions, window_strides,
    padding, base_dilation, window_dilation):
  def reducer(x, y):
    kx, vx = x
    ky, vy = y
    which = select_prim.bind(kx, ky)
    return (lax.select(which, kx, ky), lax.select(which, vx, vy))

  assert select_prim is lax.ge_p or select_prim is lax.le_p, select_prim
  init = -np.inf if select_prim is lax.ge_p else np.inf
  _, out = reduce_window(
    (operand, tangents),
    (np.array(init, dtype=operand.dtype), np.array(0, dtype=operand.dtype)),
    reducer, window_dimensions, window_strides, padding, base_dilation,
    window_dilation)
  return out

def _select_and_gather_add_jvp(
    primals, tangents, *, select_prim, window_dimensions, window_strides,
    padding, base_dilation, window_dilation):
  source, operand = primals
  g_source, g_operand = tangents
  val_out = _select_and_gather_add(
      source, operand, select_prim, window_dimensions, window_strides,
      padding, base_dilation, window_dilation)
  del g_operand
  if type(g_source) is ad_util.Zero:
    tangent_out = ad_util.Zero.from_primal_value(val_out)
  else:
    tangent_out = _select_and_gather_add(
        g_source, operand, select_prim, window_dimensions,
        window_strides, padding, base_dilation, window_dilation)
  return val_out, tangent_out

def _select_and_gather_add_transpose(
    t, tangents, operand, *, select_prim, window_dimensions, window_strides,
    padding, base_dilation, window_dilation):
  assert select_prim in (lax.le_p, lax.ge_p)
  assert (ad.is_undefined_primal(tangents) and
          not ad.is_undefined_primal(operand))
  if any(d != 1 for d in window_dilation):
    msg = ("VJP not implemented for select_and_gather (MaxPool) with window "
           "dilation, got window_dilation={}.")
    raise NotImplementedError(msg.format(window_dilation))
  if type(t) is ad_util.Zero:
    return [ad_util.Zero(tangents.aval), None]
  has_base_dilation = any(d != 1 for d in base_dilation)
  if has_base_dilation:
    select_identity = (lax._get_max_identity if select_prim is lax.ge_p
                       else lax._get_min_identity)
    operand = lax.pad(operand, select_identity(operand.dtype),
                      tuple((0, 0, d - 1) for d in base_dilation))
  result = _select_and_scatter_add(t, operand, select_prim, window_dimensions,
                                   window_strides, padding)
  if has_base_dilation:
    result = slicing.slice(result, (0,) * len(result.shape), result.shape,
                           base_dilation)
  return [result, None]

def _select_and_gather_add_batching_rule(
    batched_args, batch_dims, *, select_prim, window_dimensions, window_strides,
    padding, base_dilation, window_dilation):
  t, x = batched_args
  t_bdim, x_bdim = batch_dims
  size = next(a.shape[bdim] for a, bdim in zip(batched_args, batch_dims)
              if bdim is not None)
  t = batching.bdim_at_front(t, t_bdim, size)
  x = batching.bdim_at_front(x, x_bdim, size)
  window_dimensions = (1,) + window_dimensions
  window_strides = (1,) + window_strides
  padding = ((0, 0),) + padding
  base_dilation = (1,) + base_dilation
  window_dilation = (1,) + window_dilation
  out = _select_and_gather_add(t, x, select_prim, window_dimensions,
                               window_strides, padding, base_dilation,
                               window_dilation)
  return (out, 0)


select_and_gather_add_p = lax.standard_primitive(
    _select_and_gather_add_shape_rule, lax._input_dtype,
    'select_and_gather_add')
ad.primitive_jvps[select_and_gather_add_p] = _select_and_gather_add_jvp
ad.primitive_transposes[select_and_gather_add_p] = \
  _select_and_gather_add_transpose
batching.primitive_batchers[select_and_gather_add_p] = \
  _select_and_gather_add_batching_rule

mlir.register_lowering(select_and_gather_add_p, mlir.lower_fun(
    _select_and_gather_add_using_variadic_reducewindow,
    multiple_results=False))


# TODO(b/183233858): use variadic reducewindow on GPU, when implemented.
mlir.register_lowering(
    select_and_gather_add_p,
    _select_and_gather_add_lowering,
    platform="gpu")
