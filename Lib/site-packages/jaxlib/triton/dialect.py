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

# ruff: noqa

"""Python bindings for the MLIR Triton dialect."""

from __future__ import annotations

from collections.abc import Sequence

from jaxlib.mlir._mlir_libs._triton_ext import (
    PointerType as PointerType,
    register_dialect as register_dialect,
    infer_reduce_op_encoding as _infer_reduce_op_encoding,
)
from jaxlib.mlir import ir

from ._triton_enum_gen import *  # pylint: disable=wildcard-import
from ._triton_ops_gen import *  # pylint: disable=wildcard-import


class ReduceOp(ReduceOp):  # type: ignore

  def __init__(
      self,
      operands: Sequence[ir.Value],
      axis: int,
      *,
      loc: ir.Location | None = None,
      ip: ir.InsertionPoint | None = None,
  ):
    return_types = _infer_reduce_op_return_types(operands, axis)
    super().__init__(return_types, operands, axis, loc=loc, ip=ip)


# TODO(slebedev): Consider overriding instead.
del reduce


class ScanOp(ScanOp):  # type: ignore

  def __init__(
      self,
      operands: Sequence[ir.Value],
      axis: int,
      reverse: bool = False,
      *,
      loc: ir.Location | None = None,
      ip: ir.InsertionPoint | None = None,
  ):
    return_types = [op.type for op in operands]
    super().__init__(return_types, operands, axis, reverse, loc=loc, ip=ip)


# TODO(slebedev): Consider overriding instead.
del scan


# The following reimplements return type inference for some Triton operations.
# We cannot avoid doing that atm, because MLIR Python bindings do not support
# neither
# * transparent return type inference for operations with regions; nor
# * manual return type inference for dialects with usePropertiesForAttributes.


def _infer_reduce_op_return_types(
    operands: Sequence[ir.Value], axis: int
) -> Sequence[ir.Type]:
  return_types = []
  for op in operands:
    op_type = ir.RankedTensorType(op.type)
    shape = list(op_type.shape)
    del shape[axis]
    if not shape:
      return_types.append(op_type.element_type)
    elif op_encoding := op_type.encoding:
      encoding = _infer_reduce_op_encoding(op_encoding, axis)
      if encoding is not None:
        raise RuntimeError("Failed to infer return type encoding for ReduceOp")
      return_types.append(
          ir.RankedTensorType.get(shape, op_type.element_type, encoding)
      )
    else:
      return_types.append(ir.RankedTensorType.get(shape, op_type.element_type))
  return return_types
