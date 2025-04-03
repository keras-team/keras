//===- Transforms.h - Tensor Transformation Patterns ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TENSOR_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_TENSOR_TRANSFORMS_TRANSFORMS_H

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {

struct TilingResult;

namespace tensor {

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

/// Method to swap an `tensor.extract_slice` with its producer when the
/// producer implements the `TilingInterface`. The pattern itself does not
/// provide a mechanism to control where the application happens. With use of
/// transform dialect that control is done within the transform dialect. Other
/// use cases can inherit from this pattern and add necessary controls.
FailureOr<TilingResult> replaceExtractSliceWithTiledProducer(
    OpBuilder &builder, tensor::ExtractSliceOp sliceOp, OpResult producerOp);

/// Method to swap an `tensor.insert_slice` with its consumer when the
/// consumer implements the `TilingInterface`.
FailureOr<TilingResult>
replaceInsertSliceWithTiledConsumer(OpBuilder &builder,
                                    OffsetSizeAndStrideOpInterface sliceOp,
                                    OpOperand &consumerOp);

//===----------------------------------------------------------------------===//
// Populate functions.
//===----------------------------------------------------------------------===//

/// Appends patterns for folding tensor subset ops into consumer load/store
/// ops into `patterns`. (This includes patterns for folding tensor subset ops
/// into vector transfer ops.)
void populateFoldTensorSubsetOpPatterns(RewritePatternSet &patterns);

/// Appends patterns for folding tensor subset ops into vector transfer ops.
void populateFoldTensorSubsetIntoVectorTransferPatterns(
    RewritePatternSet &patterns);

/// Collects patterns to merge consecutive tensor.insert_slice/extract_slice
/// into one. These patterns are in this separate entry point because the
/// bufferization is sensitive to IR structure, particularly those
/// tensor.extract_slice and tensor.insert_slice ops for creating the slices.
void populateMergeConsecutiveInsertExtractSlicePatterns(
    RewritePatternSet &patterns);

/// Populates `patterns` with patterns that drop redundant tensor.insert_slice
/// rank expansions.
void populateDropRedundantInsertSliceRankExpansionPatterns(
    RewritePatternSet &patterns);

/// Populates `patterns` with patterns that fold `tensor.expand_shape` and
/// `tensor.collapse_shape` into other ops.
void populateReassociativeReshapeFoldingPatterns(RewritePatternSet &patterns);

/// Populates `patterns` with patterns that bubble up `tensor.expand_shape`
/// through `tensor.collapse_shape` ops.
void populateBubbleUpExpandShapePatterns(RewritePatternSet &patterns);

/// Populates `patterns` with patterns that fold tensor.empty with its
/// consumers.
///
/// If `singleUseOnly` is set to "true", only tensor.empty ops with a single
/// use are folded.
void populateFoldTensorEmptyPatterns(RewritePatternSet &patterns,
                                     bool foldSingleUseOnly = false);

/// Populates `patterns` with patterns that decompose `tensor.concat` into
/// `tensor.empty` of a tensor of the concatenated size, followed by a chain
/// of `tensor.insert_slice` operations on the inputs. This is intended to be
/// used as a fallback tensor -> tensor lowering that decomposes concat such
/// that it can be bufferized into a sequence of copies.
void populateDecomposeTensorConcatPatterns(RewritePatternSet &patterns);

/// Populates `patterns` with patterns that simplify `tensor.pack` and
/// `tensor.unpack` operations.
void populateSimplifyPackAndUnpackPatterns(RewritePatternSet &patterns);

/// Populates `patterns` with patterns that fold operations like `tensor.pad`
/// and `tensor.extract_slice` into `tensor.pack` and `tensor.unpack` operations
/// respectively.
void populateFoldIntoPackAndUnpackPatterns(RewritePatternSet &patterns);

using ControlFoldFn = std::function<bool(OpOperand *)>;

/// Populates `patterns` with patterns that replace tensor ops (such as
/// tensor.generate) with constants when possible.
void populateRewriteAsConstantPatterns(RewritePatternSet &patterns,
                                       const ControlFoldFn &controlFn);

//===----------------------------------------------------------------------===//
// Transform helpers
//===----------------------------------------------------------------------===//

/// Build a new tensor::PadOp with low/high padding that is independent of all
/// given independencies. If the op is already independent of all
/// independencies, the same PadOp result is returned.
///
/// Failure indicates the no suitable upper bound for low/high padding could be
/// found.
///
/// Example:
/// scf.for %iv = %lb to %ub step %step {
///   %high = affine.apply affine_map<(d0)[s0] -> (s0 - d0)> (%i)[%ub]
///   %p = tensor.pad %t low[5] high[%high] ...
///   ...
/// }
///
/// The function builds IR such as:
/// %high_new = affine.apply affine_map<()[s0, s1] -> (-s0 + s1)> ()[%lb, %ub]
/// %p_hoistable = tensor.pad %t low[5] high[%high_new]
/// %dim = tensor.dim %t, %c0
/// %size = affine.apply affine_map<(d0)[s0, s1] -> (-d0 + s0 + s1 + 5)>
///     (%iv)[%ub, %dim]
/// %slice = tensor.extract_slice %p_hoistable [0] [%size] [1]
///
/// The slice is returned.
FailureOr<Value> buildIndependentOp(OpBuilder &b, tensor::PadOp padOp,
                                    ValueRange independencies);

/// Build a new tensor::EmptyOp who's dynamic sizes are independent of all
/// given independencies. If the op is already independent of all
/// independencies, the same EmptyOp result is returned.
///
/// Failure indicates the no suitable upper bound for the dynamic sizes could be
/// found.
FailureOr<Value> buildIndependentOp(OpBuilder &b, tensor::EmptyOp emptyOp,
                                    ValueRange independencies);

} // namespace tensor
} // namespace mlir

#endif // MLIR_DIALECT_TENSOR_TRANSFORMS_TRANSFORMS_H
