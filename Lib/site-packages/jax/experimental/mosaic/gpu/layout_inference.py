# Copyright 2024 The JAX Authors.
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

"""Layout inference pass for the MLIR Mosaic GPU dialect."""

from collections.abc import Callable
import enum
from functools import partial
from typing import cast

from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import func
from jax._src.lib.mlir.dialects import vector

from . import fragmented_array as fa
from . import layouts as layouts_lib

# mypy: ignore-errors

OptionalLayouts = tuple[list[ir.Attribute], list[ir.Attribute]] | None
LayoutInferenceRule = Callable[[ir.OpView], OptionalLayouts]
_layout_inference_rules: dict[str, LayoutInferenceRule] = {}


def _add_layout_inference_rule(op: type[ir.OpView], rule: LayoutInferenceRule):
  _layout_inference_rules[op.OPERATION_NAME] = rule  # pytype: disable=attribute-error


def _set_layout_attributes(
    op: ir.OpView,
    in_layouts: list[ir.Attribute],
    out_layouts: list[ir.Attribute],
):
  op.attributes["in_layouts"] = ir.ArrayAttr.get(in_layouts)
  op.attributes["out_layouts"] = ir.ArrayAttr.get(out_layouts)


def _choose_representative_layout(
    layouts: set[ir.Attribute],
) -> ir.Attribute | None:
  """Chooses an appropriate layout from a given set of possible layouts.

  Given the input set of possible layouts, this function extracts a single
  representative layout. Currently, this function only works with strided and
  splat fragmented layouts.

  Returns:
    A single layout that can be used to annotate the operation, or None if the
    input set is empty.
  """

  if not layouts:
    return None

  strided_layouts: list[fa.WGStridedFragLayout] = [
      layouts_lib.from_strided_fragmented_layout_attr(layout)
      for layout in layouts
      if layouts_lib.is_strided_fragmented_layout(layout)
  ]

  splat_layouts: list[fa.WGSplatFragLayout] = list(
      map(
          layouts_lib.from_splat_fragmented_layout_attr,
          filter(layouts_lib.is_splat_fragmented_layout, layouts),
      )
  )

  if len(splat_layouts) + len(strided_layouts) != len(layouts):
    raise ValueError(f"Expected only strided and splat layouts, got {layouts}")

  if len(splat_layouts) > 1:
    raise NotImplementedError(
        "Finding a representative layout for several distinct splat layouts "
        "is not supported."
    )

  if len(strided_layouts) > 1:
    raise NotImplementedError(
        "Finding a representative layout for several distinct strided layouts "
        "is not supported."
    )

  if not splat_layouts:
    return layouts_lib.to_strided_fragmented_layout_attr(strided_layouts[0])

  if not strided_layouts:
    return layouts_lib.to_splat_fragmented_layout_attr(splat_layouts[0])

  [strided_layout] = strided_layouts
  return layouts_lib.to_strided_fragmented_layout_attr(strided_layout)


def _in_layout_for_operand(
    op: ir.OpView,
    operand: ir.Value,
) -> ir.Attribute | None:
  """Returns the layout of the operand in the given operation if it is set.

  Raises:
    ValueError: If `operand` is not an operand of `op`, or if `operand` is not a
      Vector.
  """
  if not ir.VectorType.isinstance(operand.type):
    raise ValueError(f"{operand} is not a vector.")

  operand_number = [
      o for o in op.operands if ir.VectorType.isinstance(o.type)
  ].index(operand)

  if not layouts_lib.has_in_layouts_set(op):
    return None

  return layouts_lib.in_layouts(op)[operand_number]


def _value_layout(value: ir.Value) -> ir.Attribute | None:
  """Returns the layout for a given value as defined by its owner.

  Raises:
    ValueError: If `result` is not a Vector.
  """
  if not ir.VectorType.isinstance(value.type):
    raise ValueError(f"{value} is not a vector.")

  owner = value.owner
  if isinstance(owner, ir.Operation):
    if not layouts_lib.has_out_layouts_set(owner):
      return None
    value_result_number = [
        r for r in owner.results if ir.VectorType.isinstance(r.type)
    ].index(value)
    return layouts_lib.out_layouts(owner)[value_result_number]

  # Function block case, useful when attempting to derive layouts for ops
  # depending on function parameters.
  if isinstance(owner, ir.Block) and isinstance(owner.owner, func.FuncOp):
    func_op = owner.owner
    block = cast(ir.Block, owner)
    if not layouts_lib.has_in_layouts_set(func_op):
      return None
    value_arg_number = [
        r for r in block.arguments if ir.VectorType.isinstance(r.type)
    ].index(value)
    return layouts_lib.in_layouts(func_op)[value_arg_number]

  raise NotImplementedError(
      f"{owner} is not a function block nor an operation.")


def _infer_pointwise_op_layouts(op: ir.OpView) -> OptionalLayouts:

  def is_array(v: ir.Value) -> bool:
    return ir.VectorType.isinstance(v.type)

  num_vector_operands = len([o for o in op.operands if is_array(o)])
  num_vector_results = len([r for r in op.results if is_array(r)])

  if layouts_lib.has_in_layouts_set(op):
    op_in_layouts = layouts_lib.in_layouts(op)
    if op_in_layouts:
      layout = op_in_layouts[0]
      return (num_vector_operands * [layout], num_vector_results * [layout])

  if layouts_lib.has_out_layouts_set(op):
    op_out_layouts = layouts_lib.out_layouts(op)
    if op_out_layouts:
      layout = op_out_layouts[0]
      return (num_vector_operands * [layout], num_vector_results * [layout])

  layouts = set()

  # We can also try to infer layouts from the layout of producer and
  # consumer operations.
  #
  # We first look at producers; this enables e.g. propagating splat layouts as
  # far down as possible, until since we may be able to propagate splat layouts
  # further down before requiring a relayout in that way.
  for operand in op.operands:
    if not ir.VectorType.isinstance(operand.type):
      continue
    if (layout := _value_layout(operand)) is not None:
      layouts.add(layout)

  # We only look at consumers if we haven't found a possible layout yet. This is
  # to avoid propagating more complicated layouts up, to e.g. preserve splat
  # layouts as far down as possible.
  if not layouts:
    for op_result in op.results:
      if not ir.VectorType.isinstance(op_result.type):
        continue
      for op_operand_use in cast(ir.OpResult, op_result).uses:
        consumer = op_operand_use.owner
        op_user = consumer.operands[op_operand_use.operand_number]
        layout = _in_layout_for_operand(consumer, op_user)
        if layout is not None:
          layouts.add(layout)

  # TODO(bchetioui): when propagating up, the representative layout should be
  # chosen in the opposite way as when propagating down. E.g., when propagating
  # down, we should pick a strided layout over a splat layout; when propagating
  # up, we should pick a splat layout over a strided layout.
  # This is left for a future change, and currently we only do "down
  # propagation".
  layout = _choose_representative_layout(layouts)
  if layout is None:
    return None

  return (num_vector_operands * [layout], num_vector_results * [layout])


for op in (
    arith.AddFOp,
    arith.MulFOp,
    vector.LoadOp,
    vector.StoreOp,
):
  _add_layout_inference_rule(op, _infer_pointwise_op_layouts)


@partial(_add_layout_inference_rule, arith.ConstantOp)
def _infer_constant_op_layout(constant_op: arith.ConstantOp) -> OptionalLayouts:
  if not ir.VectorType.isinstance(constant_op.result.type):
    return None

  shaped_ty = cast(ir.ShapedType, constant_op.result.type)
  value = constant_op.value
  layout = None
  if (
      ir.DenseElementsAttr.isinstance(value)
      and ir.DenseElementsAttr(value).is_splat
  ):
    layout = layouts_lib.to_splat_fragmented_layout_attr(
        fa.WGSplatFragLayout(shape=shaped_ty.shape)
    )
  # If the constant is not a splat, there is no obvious good choice of layout.
  # We need to look at the consumers of the constant to find a layout that works
  # for them. If there are several users with N different layouts, we can
  # arbitrarily choose any one of them for the constant, since we expect
  # whichever choice we make to lead to N-1 relayouts, which all have the same
  # cost.
  #
  # We assign a strided layout if the constant has no user, for completeness.
  elif constant_op.result.uses:
    for use in cast(ir.OpResult, constant_op.result).uses:
      consumer = use.owner
      operand = consumer.operands[use.operand_number]
      layout = _in_layout_for_operand(consumer, operand)
      if layout is not None:
        break

  # If the constant is not a splat, has no user, or a layout could not be
  # determined from looking at the users, we assign a strided layout for
  # completeness.
  if layout is None:
    layout = layouts_lib.to_strided_fragmented_layout_attr(
        fa.WGStridedFragLayout.from_shaped_type(shaped_ty)
    )

  return [], [layout]


@partial(_add_layout_inference_rule, vector.SplatOp)
def _infer_splat_op_layout(splat_op: vector.SplatOp) -> OptionalLayouts:
  layout = layouts_lib.to_splat_fragmented_layout_attr(
      fa.WGSplatFragLayout(
          shape=cast(ir.ShapedType, splat_op.result.type).shape
      )
  )

  return [], [layout]


class TraversalOrder(enum.Enum):
  """Traversal orders with respect to the data flow for IR."""

  FORWARD = 1
  BACKWARDS = 2


def traverse_op(
    op: ir.OpView,
    callback: Callable[[ir.OpView], None],
    traversal_order: TraversalOrder = TraversalOrder.FORWARD,
):
  """Traverses the operation and applies the callback in the given order."""
  for region in op.operation.regions:
    for block in region:
      if traversal_order == TraversalOrder.FORWARD:
        ops_to_traverse = block
      else:
        ops_to_traverse = reversed(list(block))
      for block_op in ops_to_traverse:
        traverse_op(block_op, callback, traversal_order)
  callback(op)


def infer_layout(module: ir.Module):
  def inference_step(op: ir.Operation):
    if not layouts_lib.should_have_layout(op):
      return
    elif inference_rule := _layout_inference_rules.get(op.OPERATION_NAME, None):  # pytype: disable=attribute-error
      pass
    else:
      raise NotImplementedError(f"Can not infer layout for {op}")

    maybe_layouts = inference_rule(op)
    if maybe_layouts is None:
      return

    _set_layout_attributes(op, *maybe_layouts)

  # TODO(bchetioui): consider switching the order of the passes. This would
  # allow propagating "simpler" layouts further down in the computation, which
  # is more efficient when possible.
  #
  # We run two passes over the module, in order to make sure that layouts
  # defined in the middle of the computation are propagated wherever they need
  # to be propagated. We start with a backwards (root-to-parameters) pass to
  # propagate the information as far up as possible, and then a forward pass
  # (parameters-to-root).
  #
  # Backwards pass
  for op in module.body:
    traverse_op(op, inference_step, TraversalOrder.BACKWARDS)

  # Forward pass
  for op in module.body:
    traverse_op(op, inference_step, TraversalOrder.FORWARD)

  # At this point, layouts have been propagated as far as they could be
  # propagated. However, it is possible for some operations to remain
  # unannotated---for example, if there were no annotations on any operation in
  # the module at the start of this function. We annotate all the remaining ops
  # that should be annotated with a strided fragmented layout.
  def to_default_layout(ty: ir.Type) -> ir.Attribute | None:
    if not ir.VectorType.isinstance(ty):
      return None
    layout = fa.WGStridedFragLayout.from_shaped_type(ty)
    return layouts_lib.to_strided_fragmented_layout_attr(layout)

  def set_default_layout(op: ir.OpView):
    if (layouts_lib.should_have_layout(op) and
        not layouts_lib.has_any_layout_set(op)):
      in_layouts = []
      for operand in op.operands:
        if (layout := to_default_layout(operand.type)) is not None:
          in_layouts.append(layout)

      out_layouts = []
      for result in op.results:
        if (layout := to_default_layout(result.type)) is not None:
          out_layouts.append(layout)

      _set_layout_attributes(op, in_layouts, out_layouts)

  for op in module.body:
    traverse_op(op, set_default_layout)
