/* Copyright 2022 The IREE Authors
   Copyright 2023 OpenXLA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Utils for lowering of the StableHLO dialect to the Linalg dialect.

#ifndef STABLEHLO_CONVERSIONS_LINALG_TRANSFORMS_LEGALIZE_TO_LINALG_UTILS_H
#define STABLEHLO_CONVERSIONS_LINALG_TRANSFORMS_LEGALIZE_TO_LINALG_UTILS_H

#include <algorithm>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/conversions/linalg/transforms/MapStablehloToScalarOp.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::stablehlo {

/// Returns an ArrayAttr that contains `nLoops` attributes. All the attributes
/// are "parallel" except the last `nReduction` elements, where are "reduction"
/// attributes.
SmallVector<utils::IteratorType, 3> getParallelAndReductionIterators(
    unsigned nLoops, unsigned nReduction);

/// Returns an ArrayAttr that contains `nParallelLoops` "parallel" attributes.
SmallVector<utils::IteratorType, 3> getNParallelLoopsAttrs(
    unsigned nParallelLoops);

/// Generates an init sparse tensor.
Value getEmptySparseTensor(OpBuilder &b, Location loc, ShapedType type,
                           ArrayRef<Value> dynSizes);

/// Generates a tensor.empty op.
Value getEmptyTensor(OpBuilder &b, Location loc, ShapedType type,
                     ArrayRef<Value> dynSizes);

/// Generates an empty tensor for the result of the operation, which could be a
/// dense tensor or a sparse tensor.
Value getEmptyTensorFor(OpBuilder &b, Location loc, ShapedType resultType,
                        Operation *op, ValueRange operands);

/// Ensures a tensor has the same shape (not including the element type) as
/// another.
Value coerceTensorShape(OpBuilder &builder, Location loc,
                        TypedValue<ShapedType> value, ShapedType targetType);

/// Fills |tensor| with a zero constant of the matching type. Returns the new
/// value.
Value fillTensorWithZeros(OpBuilder &builder, Location loc, Value tensor);

/// Sparsifies a (block of) operation(s) that cannot be handled directly
/// by the sparse compiler but has well-known semi-ring semantics.
///
/// This yields something of the following form:
///
///   %result = sparse_tensor.unary %values[0]
///     present={
///       ^bb1(%val):
///         ... codegen proceeds here using %val ....
///         sparse_tensor.yield
///     }
///     absent={}
///   linalg.yield %result
Value preSparsify(Operation *op, llvm::SmallVector<Value, 2> &values, Type rtp,
                  OpBuilder *b);

/// Finalizes sparse semi-ring construction.
Value postSparsify(Operation *op, Value semiring, Value result, OpBuilder *b);

/// Returns true if all operands are tensors with rank 0.
bool allOperandsAreScalarTensors(Operation *op);

/// Returns true if parent op is linalg.
bool isInBodyOfLinalgOps(Operation *op);

/// Extracts integer values from the attribute |elements|.
SmallVector<int64_t> extract1DVector(DenseIntElementsAttr elements);

/// Returns true if the given |attr| is a splat of the given |value|.
inline bool isSplatValue(DenseIntElementsAttr attr, uint64_t value) {
  return attr.isSplat() && attr.getSplatValue<uint64_t>() == value;
}

}  // namespace mlir::stablehlo

#endif  // STABLEHLO_CONVERSIONS_LINALG_TRANSFORMS_LEGALIZE_TO_LINALG_UTILS_H
