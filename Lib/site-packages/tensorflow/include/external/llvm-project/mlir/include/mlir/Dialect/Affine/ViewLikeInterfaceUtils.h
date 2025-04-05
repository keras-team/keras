//===- ViewLikeInterfaceUtils.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_VIEWLIKEINTERFACEUTILS_H
#define MLIR_DIALECT_AFFINE_VIEWLIKEINTERFACEUTILS_H

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
class RewriterBase;

namespace affine {

/// Fills the `combinedOffsets`, `combinedSizes` and `combinedStrides` to use
/// when combining a producer slice **into** a consumer slice.
///
/// This function performs the following computation:
/// - Combined offsets = producer_offsets * consumer_strides + consumer_offsets
/// - Combined sizes = consumer_sizes
/// - Combined strides = producer_strides * consumer_strides
// TODO: unify this API with resolveIndicesIntoOpWithOffsetsAndStrides or
// deprecate.
LogicalResult
mergeOffsetsSizesAndStrides(OpBuilder &builder, Location loc,
                            ArrayRef<OpFoldResult> producerOffsets,
                            ArrayRef<OpFoldResult> producerSizes,
                            ArrayRef<OpFoldResult> producerStrides,
                            const llvm::SmallBitVector &droppedProducerDims,
                            ArrayRef<OpFoldResult> consumerOffsets,
                            ArrayRef<OpFoldResult> consumerSizes,
                            ArrayRef<OpFoldResult> consumerStrides,
                            SmallVector<OpFoldResult> &combinedOffsets,
                            SmallVector<OpFoldResult> &combinedSizes,
                            SmallVector<OpFoldResult> &combinedStrides);

/// Fills the `combinedOffsets`, `combinedSizes` and `combinedStrides` to use
/// when combining a `producer` slice op **into** a `consumer` slice op.
// TODO: unify this API with resolveIndicesIntoOpWithOffsetsAndStrides or
// deprecate.
LogicalResult
mergeOffsetsSizesAndStrides(OpBuilder &builder, Location loc,
                            OffsetSizeAndStrideOpInterface producer,
                            OffsetSizeAndStrideOpInterface consumer,
                            const llvm::SmallBitVector &droppedProducerDims,
                            SmallVector<OpFoldResult> &combinedOffsets,
                            SmallVector<OpFoldResult> &combinedSizes,
                            SmallVector<OpFoldResult> &combinedStrides);

/// Given the 'consumerIndices' of a load/store operation operating on an op
/// with offsets and strides, return the combined indices.
///
/// For example, using `memref.load` and `memref.subview` as an illustration:
///
/// ```
///    %0 = ... : memref<12x42xf32>
///    %1 = memref.subview %0[%arg0, %arg1][...][%stride1, %stride2] :
///      memref<12x42xf32> to memref<4x4xf32, offset=?, strides=[?, ?]>
///    %2 = load %1[%i1, %i2] : memref<4x4xf32, offset=?, strides=[?, ?]>
/// ```
///
/// could be folded into:
///
/// ```
///    %2 = load %0[%arg0 + %i1 * %stride1][%arg1 + %i2 * %stride2] :
///         memref<12x42xf32>å
/// ```
void resolveIndicesIntoOpWithOffsetsAndStrides(
    RewriterBase &rewriter, Location loc,
    ArrayRef<OpFoldResult> mixedSourceOffsets,
    ArrayRef<OpFoldResult> mixedSourceStrides,
    const llvm::SmallBitVector &rankReducedDims,
    ArrayRef<OpFoldResult> consumerIndices,
    SmallVectorImpl<Value> &resolvedIndices);

inline void resolveIndicesIntoOpWithOffsetsAndStrides(
    RewriterBase &rewriter, Location loc,
    ArrayRef<OpFoldResult> mixedSourceOffsets,
    ArrayRef<OpFoldResult> mixedSourceStrides,
    const llvm::SmallBitVector &rankReducedDims, ValueRange consumerIndices,
    SmallVectorImpl<Value> &resolvedIndices) {
  return resolveIndicesIntoOpWithOffsetsAndStrides(
      rewriter, loc, mixedSourceOffsets, mixedSourceStrides, rankReducedDims,
      getAsOpFoldResult(consumerIndices), resolvedIndices);
}

/// Given `sourceSizes`, `destSizes` and information about which dimensions are
/// dropped by the source: `rankReducedSourceDims`, compute the resolved sizes
/// that correspond to dest_op(source_op).
/// In practice, this amounts to filtering by `rankReducedSourceDims` and taking
/// from `sourceSizes` if a dimension is dropped, otherwise taking from
/// `destSizes`.
void resolveSizesIntoOpWithSizes(
    ArrayRef<OpFoldResult> sourceSizes, ArrayRef<OpFoldResult> destSizes,
    const llvm::SmallBitVector &rankReducedSourceDims,
    SmallVectorImpl<OpFoldResult> &resolvedSizes);

} // namespace affine
} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_VIEWLIKEINTERFACEUTILS_H
