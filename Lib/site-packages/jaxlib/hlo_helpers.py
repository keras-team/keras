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

"""A small library of helpers for use in jaxlib to build MLIR operations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Union

import jaxlib.mlir.ir as ir
import jaxlib.mlir.dialects.stablehlo as hlo
import numpy as np


_dtype_to_ir_type_factory : dict[np.dtype, Callable[[], ir.Type]] = {
  np.dtype(np.bool_): partial(ir.IntegerType.get_signless, 1),
  np.dtype(np.int8): partial(ir.IntegerType.get_signless, 8),
  np.dtype(np.int16): partial(ir.IntegerType.get_signless, 16),
  np.dtype(np.int32): partial(ir.IntegerType.get_signless, 32),
  np.dtype(np.int64): partial(ir.IntegerType.get_signless, 64),
  np.dtype(np.uint8): partial(ir.IntegerType.get_unsigned, 8),
  np.dtype(np.uint16): partial(ir.IntegerType.get_unsigned, 16),
  np.dtype(np.uint32): partial(ir.IntegerType.get_unsigned, 32),
  np.dtype(np.uint64): partial(ir.IntegerType.get_unsigned, 64),
  np.dtype(np.float16): ir.F16Type.get,
  np.dtype(np.float32): ir.F32Type.get,
  np.dtype(np.float64): ir.F64Type.get,
  np.dtype(np.complex64): lambda: ir.ComplexType.get(ir.F32Type.get()),
  np.dtype(np.complex128): lambda: ir.ComplexType.get(ir.F64Type.get()),
}
def dtype_to_ir_type(dtype) -> ir.Type:
  return _dtype_to_ir_type_factory[np.dtype(dtype)]()


def shape_dtype_to_ir_type(shape: Sequence[int], dtype) -> ir.Type:
  return ir.RankedTensorType.get(shape, dtype_to_ir_type(dtype))


# When we generate custom calls with dynamic shapes we have to pass
# both the result_types, with ir.ShapedType.get_dynamic_size in place of
# the dynamic dimensions, and also result_shapes, which are ir.Value
# representing 1D int32 tensors. If all the shapes are static we can use
# result_shapes=None. We first construct for each result a pair with the shape
# and element type, the shape containing either integer or ir.Value.
DimensionSize = Union[int, ir.Value]  # an ir.Value if not static dimension
ShapeTypePair = tuple[Sequence[DimensionSize], ir.Type]

def mk_result_types_and_shapes(
    shape_type_pairs: Sequence[ShapeTypePair]
) -> tuple[list[ir.Type], list[ir.Value] | None]:
  result_types: list[ir.Type] = []
  result_shapes: list[ir.Value] = []
  has_dynamic_shapes = any(
      any(not isinstance(d, int) for d in rshape)
      for rshape, _ in shape_type_pairs)
  for (rshape, rtype) in shape_type_pairs:
    if has_dynamic_shapes:
      result_shapes.append(shape_tensor(rshape))
    result_types.append(
        ir.RankedTensorType.get(
            [d if isinstance(d, int) else ir.ShapedType.get_dynamic_size()
             for d in rshape],
            rtype))
  return (result_types,
          result_shapes if has_dynamic_shapes else None)

# TODO(necula): share this with mlir.shape_tensor
def shape_tensor(sizes: Sequence[int | ir.Value]) -> ir.Value:
  int1d = shape_dtype_to_ir_type((1,), np.int32)
  i32_type = shape_dtype_to_ir_type((), np.int32)
  def dim_to_i32x1(d):
    if type(d) is int:
      return hlo_const(np.array([d], dtype=np.int32))
    else:
      if d.type != i32_type:
        d = hlo.convert(i32_type, d)
      return hlo.reshape(int1d, d)
  ds = [dim_to_i32x1(sz) for sz in sizes]
  if not ds:
    return hlo_const(np.array([], np.int32))
  elif len(ds) == 1:
    return ds[0]
  else:
    return hlo.concatenate(
        ds, ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0))

def hlo_const(x: np.ndarray) -> ir.Value:
  assert isinstance(x, np.ndarray)
  return hlo.constant(
      ir.DenseElementsAttr.get(x, type=dtype_to_ir_type(x.dtype)))

def hlo_u8(x: int):
  return hlo_const(np.array(x, dtype=np.uint8))
def hlo_s32(x: int):
  return hlo_const(np.array(x, dtype=np.int32))

def ensure_hlo_s32(x: DimensionSize):
  return hlo_s32(x) if isinstance(x, int) else x

def dense_int_array(xs) -> ir.DenseI64ArrayAttr:
  return ir.DenseI64ArrayAttr.get(np.asarray(xs, np.int64))

def hlo_min(x: DimensionSize, y: DimensionSize) -> DimensionSize:
  if type(x) is int:
    if type(y) is int:
      return min(x, y)
    x = hlo_s32(x)
  if type(y) is int:
    y = hlo_s32(y)
  return hlo.minimum(x, y)


def hlo_add(x: DimensionSize, y: DimensionSize) -> DimensionSize:
  if type(x) is int:
    if type(y) is int:
      return x + y
    x = hlo_s32(x)
  if type(y) is int:
    y = hlo_s32(y)
  return hlo.add(x, y)


# TODO(necula): this is identical with mlir.custom_call, but meant for use
# in jaxlib. Find a way to share these implementations.
def custom_call(
    call_target_name: str,
    *,
    result_types: Sequence[ir.Type],
    operands: Sequence[ir.Value],
    backend_config: str | bytes | dict[str, ir.Attribute] = "",
    has_side_effect: bool = False,
    result_shapes: Sequence[ir.Value] | None = None,
    called_computations: Sequence[str] = (),
    api_version: int = 2,
    operand_output_aliases: dict[int, int] | None = None,
    operand_layouts: Sequence[Sequence[int]] | None = None,
    result_layouts: Sequence[Sequence[int]] | None = None,
    extra_attributes: dict[str, ir.Attribute] | None = None,
) -> ir.Operation:
  """Helper function for building an hlo.CustomCall.

  Args:
    call_target_name: the name of the custom call target
    result_types: the MLIR types of the results of the custom call
    operands: the MLIR IR values that are arguments to the custom call
    backend_config: an opaque string passed to the custom call kernel
    has_side_effect: if True, marks the custom call as effectful
    result_shapes: tensors that represent the result shapes, to be used when
      the results have dynamic shapes. If not-None, its length must match the
      number of the results.
    called_computations: the list of function names called by the custom call.
    api_version: the ABI contract version of the custom call
    operand_output_aliases: a dict mapping operand numbers to outputs they alias
    operand_layouts: a sequence of layouts (dimension orders) for each operand
    result_layouts: a sequence of layouts (dimension orders) for each result
    extra_attributes: additional IR attributes to apply to the custom_call.
  """
  operands = list(operands)

  if backend_config is None:
    backend_config_attr = ir.StringAttr.get("")
  elif isinstance(backend_config, (str, bytes)):
    backend_config_attr = ir.StringAttr.get(backend_config)
  elif isinstance(backend_config, dict):
    # TODO(necula): it seems that the CustomCallOp constructor requires that
    # backend_config_attr be a string attribute, even though in some cases we
    # need it to be a DictAttr, e.g., for ApproxTopK on TPU.
    # "Verification failed: 'stablehlo.custom_call' op attribute 'backend_config' failed to satisfy constraint: string attribute"
    # To workaround this limitation we first set it to the empty string and we
    # use an unregistered attribute mhlo.backend_config to hold the DictAttr.
    # We must also use api_version=1 to ensure that mhlo.backend_config is
    # handled properly.
    backend_config_attr = ir.StringAttr.get("")
    api_version = 1
  else:
    raise ValueError("custom_call backend_config unexpected type: " + str(backend_config))
  attributes = dict(
      call_target_name=ir.StringAttr.get(call_target_name),
      has_side_effect=ir.BoolAttr.get(has_side_effect),
      backend_config=backend_config_attr,
      api_version=ir.IntegerAttr.get(
          ir.IntegerType.get_signless(32), api_version),
      called_computations=ir.ArrayAttr.get(
          [ir.FlatSymbolRefAttr.get(name) for name in called_computations]
      ),
  )
  if operand_output_aliases is not None:
    attributes["output_operand_aliases"] = ir.ArrayAttr.get([
      hlo.OutputOperandAlias.get(
          # if len(result_types) == 1 then the aliasing refers implicitly to
          # the only output.
          output_tuple_indices=[output_idx] if len(result_types) > 1 else [],
          operand_index=input_idx,
          operand_tuple_indices=[],
      )
      for input_idx, output_idx in (operand_output_aliases.items() or ())
    ])

  if extra_attributes is not None:
    attributes.update(extra_attributes)

  if result_shapes is not None:
    # We add the result_shapes at the end of the operands, and must pass
    # the indices_of_output_operands attribute. This attribute is not yet
    # accepted by the CustomCall constructor, so we use build_generic
    attributes["indices_of_shape_operands"] = ir.DenseIntElementsAttr.get(
        np.asarray(list(range(len(operands), len(operands) + len(result_shapes))),
                   dtype=np.int64))
    if operand_layouts is not None:
      assert len(operand_layouts) == len(operands), (operand_layouts, operands)
      operand_layouts = list(operand_layouts) + [(0,)] * len(result_shapes)
    operands = list(operands) + list(result_shapes)

  if operand_layouts is not None:
    attributes["operand_layouts"] = ir.ArrayAttr.get([
        ir.DenseIntElementsAttr.get(
            np.atleast_1d(np.asarray(l, dtype=np.int64)),
            type=ir.IndexType.get()) for l in operand_layouts
    ])
  if result_layouts is not None:
    assert result_layouts is not None
    assert len(result_layouts) == len(result_types), (
        result_layouts, result_types)
    attributes["result_layouts"] = ir.ArrayAttr.get([
        ir.DenseIntElementsAttr.get(
            np.atleast_1d(np.asarray(l, dtype=np.int64)),
            type=ir.IndexType.get()) for l in result_layouts
    ])

  op = hlo.CustomCallOp.build_generic(results=result_types, operands=operands,
                                      attributes=attributes)
  if isinstance(backend_config, dict):
    backend_config_attr = ir.DictAttr.get(backend_config)
    op.operation.attributes["mhlo.backend_config"] = backend_config_attr
  return op
