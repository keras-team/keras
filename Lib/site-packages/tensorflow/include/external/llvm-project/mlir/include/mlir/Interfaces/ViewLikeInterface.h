//===- ViewLikeInterface.h - View-like operations interface ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation interface for view-like operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_VIEWLIKEINTERFACE_H_
#define MLIR_INTERFACES_VIEWLIKEINTERFACE_H_

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

class OffsetSizeAndStrideOpInterface;

namespace detail {

LogicalResult verifyOffsetSizeAndStrideOp(OffsetSizeAndStrideOpInterface op);

bool sameOffsetsSizesAndStrides(
    OffsetSizeAndStrideOpInterface a, OffsetSizeAndStrideOpInterface b,
    llvm::function_ref<bool(OpFoldResult, OpFoldResult)> cmp);

/// Helper method to compute the number of dynamic entries of `staticVals`,
/// up to `idx`.
unsigned getNumDynamicEntriesUpToIdx(ArrayRef<int64_t> staticVals,
                                     unsigned idx);

} // namespace detail
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interfaces/ViewLikeInterface.h.inc"

namespace mlir {

/// Pattern to rewrite dynamic offsets/sizes/strides of view/slice-like ops as
/// constant arguments. This pattern assumes that the op has a suitable builder
/// that takes a result type, a "source" operand and mixed offsets, sizes and
/// strides.
///
/// `OpType` is the type of op to which this pattern is applied. `ResultTypeFn`
/// returns the new result type of the op, based on the new offsets, sizes and
/// strides. `CastOpFunc` is used to generate a cast op if the result type of
/// the op has changed.
template <typename OpType, typename ResultTypeFn, typename CastOpFunc>
class OpWithOffsetSizesAndStridesConstantArgumentFolder final
    : public OpRewritePattern<OpType> {
public:
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpFoldResult> mixedOffsets(op.getMixedOffsets());
    SmallVector<OpFoldResult> mixedSizes(op.getMixedSizes());
    SmallVector<OpFoldResult> mixedStrides(op.getMixedStrides());

    // No constant operands were folded, just return;
    if (failed(foldDynamicIndexList(mixedOffsets, /*onlyNonNegative=*/true)) &&
        failed(foldDynamicIndexList(mixedSizes, /*onlyNonNegative=*/true)) &&
        failed(foldDynamicIndexList(mixedStrides)))
      return failure();

    // Create the new op in canonical form.
    auto resultType =
        ResultTypeFn()(op, mixedOffsets, mixedSizes, mixedStrides);
    if (!resultType)
      return failure();
    auto newOp =
        rewriter.create<OpType>(op.getLoc(), resultType, op.getSource(),
                                mixedOffsets, mixedSizes, mixedStrides);
    CastOpFunc()(rewriter, op, newOp);

    return success();
  }
};

/// Printer hook for custom directive in assemblyFormat.
///
///   custom<DynamicIndexList>($values, $integers)
///   custom<DynamicIndexList>($values, $integers, type($values))
///
/// where `values` is of ODS type `Variadic<*>` and `integers` is of ODS
/// type `I64ArrayAttr`. Prints a list with either (1) the static integer value
/// in `integers` is `kDynamic` or (2) the next value otherwise. If `valueTypes`
/// is non-empty, it is expected to contain as many elements as `values`
/// indicating their types. This allows idiomatic printing of mixed value and
/// integer attributes in a list. E.g.
/// `[%arg0 : index, 7, 42, %arg42 : i32]`.
///
/// Indices can be scalable. For example, "4" in "[2, [4], 8]" is scalable.
/// This notation is similar to how scalable dims are marked when defining
/// Vectors. For each value in `integers`, the corresponding `bool` in
/// `scalables` encodes whether it's a scalable index. If `scalableVals` is
/// empty then assume that all indices are non-scalable.
void printDynamicIndexList(
    OpAsmPrinter &printer, Operation *op, OperandRange values,
    ArrayRef<int64_t> integers, ArrayRef<bool> scalables,
    TypeRange valueTypes = TypeRange(),
    AsmParser::Delimiter delimiter = AsmParser::Delimiter::Square);
inline void printDynamicIndexList(
    OpAsmPrinter &printer, Operation *op, OperandRange values,
    ArrayRef<int64_t> integers, TypeRange valueTypes = TypeRange(),
    AsmParser::Delimiter delimiter = AsmParser::Delimiter::Square) {
  return printDynamicIndexList(printer, op, values, integers, {}, valueTypes,
                               delimiter);
}

/// Parser hook for custom directive in assemblyFormat.
///
///   custom<DynamicIndexList>($values, $integers)
///   custom<DynamicIndexList>($values, $integers, type($values))
///
/// where `values` is of ODS type `Variadic<*>` and `integers` is of ODS
/// type `I64ArrayAttr`. Parse a mixed list with either (1) static integer
/// values or (2) SSA values. Fill `integers` with the integer ArrayAttr, where
/// `kDynamic` encodes the position of SSA values. Add the parsed SSA values
/// to `values` in-order. If `valueTypes` is non-null, fill it with types
/// corresponding to values; otherwise the caller must handle the types.
///
/// E.g. after parsing "[%arg0 : index, 7, 42, %arg42 : i32]":
///   1. `result` is filled with the i64 ArrayAttr "[`kDynamic`, 7, 42,
///   `kDynamic`]"
///   2. `ssa` is filled with "[%arg0, %arg1]".
///
/// Indices can be scalable. For example, "4" in "[2, [4], 8]" is scalable.
/// This notation is similar to how scalable dims are marked when defining
/// Vectors. For each value in `integers`, the corresponding `bool` in
/// `scalableVals` encodes whether it's a scalable index.
ParseResult parseDynamicIndexList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    DenseI64ArrayAttr &integers, DenseBoolArrayAttr &scalableVals,
    SmallVectorImpl<Type> *valueTypes = nullptr,
    AsmParser::Delimiter delimiter = AsmParser::Delimiter::Square);
inline ParseResult parseDynamicIndexList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    DenseI64ArrayAttr &integers, SmallVectorImpl<Type> *valueTypes = nullptr,
    AsmParser::Delimiter delimiter = AsmParser::Delimiter::Square) {
  DenseBoolArrayAttr scalableVals = {};
  return parseDynamicIndexList(parser, values, integers, scalableVals,
                               valueTypes, delimiter);
}
inline ParseResult parseDynamicIndexList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    DenseI64ArrayAttr &integers, SmallVectorImpl<Type> &valueTypes,
    AsmParser::Delimiter delimiter = AsmParser::Delimiter::Square) {
  DenseBoolArrayAttr scalableVals = {};
  return parseDynamicIndexList(parser, values, integers, scalableVals,
                               &valueTypes, delimiter);
}
inline ParseResult parseDynamicIndexList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    DenseI64ArrayAttr &integers, SmallVectorImpl<Type> &valueTypes,
    DenseBoolArrayAttr &scalableVals,
    AsmParser::Delimiter delimiter = AsmParser::Delimiter::Square) {

  return parseDynamicIndexList(parser, values, integers, scalableVals,
                               &valueTypes, delimiter);
}

/// Verify that a the `values` has as many elements as the number of entries in
/// `attr` for which `isDynamic` evaluates to true.
LogicalResult verifyListOfOperandsOrIntegers(Operation *op, StringRef name,
                                             unsigned expectedNumElements,
                                             ArrayRef<int64_t> attr,
                                             ValueRange values);

} // namespace mlir

#endif // MLIR_INTERFACES_VIEWLIKEINTERFACE_H_
