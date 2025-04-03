//===- LoweringPatterns.h - Vector rewrite patterns --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_TRANSFORMS_LOWERINGPATTERNS_H
#define MLIR_DIALECT_VECTOR_TRANSFORMS_LOWERINGPATTERNS_H

#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"

namespace mlir {
class RewritePatternSet;

namespace vector {

//===----------------------------------------------------------------------===//
// Lowering pattern populate functions
//===----------------------------------------------------------------------===//

/// Populate the pattern set with the following patterns:
///
/// [OuterProductOpLowering]
/// Progressively lower a `vector.outerproduct` to linearized
/// `vector.extract` + `vector.fma` + `vector.insert`.
///
/// [ContractionOpLowering]
/// Progressive lowering of ContractionOp.
/// One:
///   %x = vector.contract with at least one free/batch dimension
/// is replaced by:
///   %a = vector.contract with one less free/batch dimension
///   %b = vector.contract with one less free/batch dimension
///
/// [ContractionOpToMatmulOpLowering]
/// Progressively lower a `vector.contract` with row-major matmul semantics to
/// linearized `vector.shape_cast` + `vector.matmul` on the way to
/// `llvm.matrix.multiply`.
///
/// [ContractionOpToDotLowering]
/// Progressively lower a `vector.contract` with row-major matmul semantics to
/// linearized `vector.extract` + `vector.reduce` + `vector.insert`.
///
/// [ContractionOpToOuterProductOpLowering]
/// Progressively lower a `vector.contract` with row-major matmul semantics to
/// linearized `vector.extract` + `vector.outerproduct` + `vector.insert`.
void populateVectorContractLoweringPatterns(
    RewritePatternSet &patterns, VectorTransformsOptions options,
    PatternBenefit benefit = 1, bool disableOuterProductLowering = false);

/// Populate the pattern set with the following patterns:
///
/// [OuterProductOpLowering]
/// Progressively lower a `vector.outerproduct` to linearized
/// `vector.extract` + `vector.fma` + `vector.insert`.
void populateVectorOuterProductLoweringPatterns(RewritePatternSet &patterns,
                                                PatternBenefit benefit = 1);

/// Collect a set of patterns to convert vector.multi_reduction op into
/// a sequence of vector.reduction ops. The patterns comprise:
///
/// [InnerOuterDimReductionConversion]
/// Rewrites vector.multi_reduction such that all reduction dimensions are
/// either innermost or outermost, by adding the proper vector.transpose
/// operations.
///
/// [ReduceMultiDimReductionRank]
/// Once in innermost or outermost reduction
/// form, rewrites n-D vector.multi_reduction into 2-D vector.multi_reduction,
/// by introducing vector.shape_cast ops to collapse + multi-reduce + expand
/// back.
///
/// [TwoDimMultiReductionToElementWise]
/// Once in 2-D vector.multi_reduction form, with an **outermost** reduction
/// dimension, unroll the outer dimension to obtain a sequence of 1-D vector
/// ops. This also has an opportunity for tree-reduction (in the future).
///
/// [TwoDimMultiReductionToReduction]
/// Once in 2-D vector.multi_reduction form, with an **innermost** reduction
/// dimension, unroll the outer dimension to obtain a sequence of extract +
/// vector.reduction + insert. This can further lower to horizontal reduction
/// ops.
///
/// [OneDimMultiReductionToTwoDim]
/// For cases that reduce to 1-D vector<k> reduction (and are thus missing
/// either a parallel or a reduction), we lift them back up to 2-D with a simple
/// vector.shape_cast to vector<1xk> so that the other patterns can kick in,
/// thus fully exiting out of the vector.multi_reduction abstraction.
void populateVectorMultiReductionLoweringPatterns(
    RewritePatternSet &patterns, VectorMultiReductionLowering options,
    PatternBenefit benefit = 1);

/// Populate the pattern set with the following patterns:
///
/// [TransferReadToVectorLoadLowering]
/// Progressive lowering of BroadcastOp to ExtractOp + InsertOp + lower-D
/// BroadcastOp until dim 1.
void populateVectorBroadcastLoweringPatterns(RewritePatternSet &patterns,
                                             PatternBenefit benefit = 1);

/// Populate the pattern set with the following patterns:
///
/// [CreateMaskOp]
/// Progressive lowering of CreateMaskOp to lower-D CreateMaskOp until dim 1.
///
/// [ConstantMaskOp]
/// Progressive lowering of ConstantMaskOp to lower-D ConstantMaskOp until
/// dim 1.
void populateVectorMaskOpLoweringPatterns(RewritePatternSet &patterns,
                                          PatternBenefit benefit = 1);

/// Collects patterns that lower scalar vector transfer ops to memref loads and
/// stores when beneficial. If `allowMultipleUses` is set to true, the patterns
/// are applied to vector transfer reads with any number of uses. Otherwise,
/// only vector transfer reads with a single use will be lowered.
void populateScalarVectorTransferLoweringPatterns(RewritePatternSet &patterns,
                                                  PatternBenefit benefit,
                                                  bool allowMultipleUses);

/// Populate the pattern set with the following patterns:
///
/// [ShapeCastOp2DDownCastRewritePattern]
/// ShapeOp 2D -> 1D downcast serves the purpose of flattening 2-D to 1-D
/// vectors progressively.
///
/// [ShapeCastOp2DUpCastRewritePattern]
/// ShapeOp 1D -> 2D upcast serves the purpose of unflattening 2-D from 1-D
/// vectors progressively.
///
/// [ShapeCastOpRewritePattern]
/// Reference lowering to fully unrolled sequences of single element ExtractOp +
/// InsertOp. Note that applying this pattern can almost always be considered a
/// performance bug.
void populateVectorShapeCastLoweringPatterns(RewritePatternSet &patterns,
                                             PatternBenefit benefit = 1);

/// Populate the pattern set with the following patterns:
///
/// [TransposeOpLowering]
///
/// [TransposeOp2DToShuffleLowering]
///
void populateVectorTransposeLoweringPatterns(RewritePatternSet &patterns,
                                             VectorTransformsOptions options,
                                             PatternBenefit benefit = 1);

/// Populate the pattern set with the following patterns:
///
/// [TransferReadToVectorLoadLowering]
/// Progressive lowering of transfer_read.This pattern supports lowering of
/// `vector.transfer_read` to a combination of `vector.load` and
/// `vector.broadcast`
///
/// [TransferWriteToVectorStoreLowering]
/// Progressive lowering of transfer_write. This pattern supports lowering of
/// `vector.transfer_write` to `vector.store`
///
/// [VectorLoadToMemrefLoadLowering]
/// Replace a 0-d vector.load with a memref.load + vector.broadcast.
///
/// [VectorStoreToMemrefStoreLowering]
/// Replace a 0-d vector.store with a vector.extractelement + memref.store.
///
/// These patterns lower transfer ops to simpler ops like `vector.load`,
/// `vector.store` and `vector.broadcast`. Only transfers with a transfer rank
/// of a most `maxTransferRank` are lowered. This is useful when combined with
/// VectorToSCF, which reduces the rank of vector transfer ops.
void populateVectorTransferLoweringPatterns(
    RewritePatternSet &patterns,
    std::optional<unsigned> maxTransferRank = std::nullopt,
    PatternBenefit benefit = 1);

/// Collect a set of transfer read/write lowering patterns that simplify the
/// permutation map (e.g., converting it to a minor identity map) by inserting
/// broadcasts and transposes. More specifically:
///
/// [TransferReadPermutationLowering]
/// Lower transfer_read op with permutation into a transfer_read with a
/// permutation map composed of leading zeros followed by a minor identity +
/// vector.transpose op.
/// Ex:
///     vector.transfer_read ...
///         permutation_map: (d0, d1, d2) -> (0, d1)
/// into:
///     %v = vector.transfer_read ...
///         permutation_map: (d0, d1, d2) -> (d1, 0)
///     vector.transpose %v, [1, 0]
///
///     vector.transfer_read ...
///         permutation_map: (d0, d1, d2, d3) -> (0, 0, 0, d1, d3)
/// into:
///     %v = vector.transfer_read ...
///         permutation_map: (d0, d1, d2, d3) -> (0, 0, d1, 0, d3)
///     vector.transpose %v, [0, 1, 3, 2, 4]
/// Note that an alternative is to transform it to linalg.transpose +
/// vector.transfer_read to do the transpose in memory instead.
///
/// [TransferWritePermutationLowering]
/// Lower transfer_write op with permutation into a transfer_write with a
/// minor identity permutation map. (transfer_write ops cannot have broadcasts.)
/// Ex:
///     vector.transfer_write %v ...
///         permutation_map: (d0, d1, d2) -> (d2, d0, d1)
/// into:
///     %tmp = vector.transpose %v, [2, 0, 1]
///     vector.transfer_write %tmp ...
///         permutation_map: (d0, d1, d2) -> (d0, d1, d2)
///
///     vector.transfer_write %v ...
///         permutation_map: (d0, d1, d2, d3) -> (d3, d2)
/// into:
///     %tmp = vector.transpose %v, [1, 0]
///     %v = vector.transfer_write %tmp ...
///         permutation_map: (d0, d1, d2, d3) -> (d2, d3)
///
/// [TransferOpReduceRank]
/// Lower transfer_read op with broadcast in the leading dimensions into
/// transfer_read of lower rank + vector.broadcast.
/// Ex: vector.transfer_read ...
///         permutation_map: (d0, d1, d2, d3) -> (0, d1, 0, d3)
/// into:
///     %v = vector.transfer_read ...
///         permutation_map: (d0, d1, d2, d3) -> (d1, 0, d3)
///     vector.broadcast %v
void populateVectorTransferPermutationMapLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit = 1);

/// Populate the pattern set with the following patterns:
///
/// [ScanToArithOps]
/// Convert vector.scan op into arith ops and vector.insert_strided_slice /
/// vector.extract_strided_slice.
void populateVectorScanLoweringPatterns(RewritePatternSet &patterns,
                                        PatternBenefit benefit = 1);

/// Populate the pattern set with the following patterns:
///
/// [FlattenGather]
/// Flattens 2 or more dimensional `vector.gather` ops by unrolling the
/// outermost dimension.
///
/// [Gather1DToConditionalLoads]
/// Turns 1-d `vector.gather` into a scalarized sequence of `vector.loads` or
/// `tensor.extract`s. To avoid out-of-bounds memory accesses, these
/// loads/extracts are made conditional using `scf.if` ops.
void populateVectorGatherLoweringPatterns(RewritePatternSet &patterns,
                                          PatternBenefit benefit = 1);

/// Populates instances of `MaskOpRewritePattern` to lower masked operations
/// with `vector.mask`. Patterns should rewrite the `vector.mask` operation and
/// not its nested `MaskableOpInterface`.
void populateVectorMaskLoweringPatternsForSideEffectingOps(
    RewritePatternSet &patterns);

/// Populate the pattern set with the following patterns:
///
/// [VectorMaskedLoadOpConverter]
/// Turns vector.maskedload to scf.if + memref.load
///
/// [VectorMaskedStoreOpConverter]
/// Turns vector.maskedstore to scf.if + memref.store
void populateVectorMaskedLoadStoreEmulationPatterns(RewritePatternSet &patterns,
                                                    PatternBenefit benefit = 1);

/// Populate the pattern set with the following patterns:
///
/// [UnrollInterleaveOp]
/// A one-shot unrolling of InterleaveOp to (one or more) ExtractOp +
/// InterleaveOp (of `targetRank`) + InsertOp.
void populateVectorInterleaveLoweringPatterns(RewritePatternSet &patterns,
                                              int64_t targetRank = 1,
                                              PatternBenefit benefit = 1);

void populateVectorInterleaveToShufflePatterns(RewritePatternSet &patterns,
                                               PatternBenefit benefit = 1);

/// Populates the pattern set with the following patterns:
///
/// [UnrollBitCastOp]
/// A one-shot unrolling of BitCastOp to (one or more) ExtractOp +
/// BitCastOp (of `targetRank`) + InsertOp.
void populateVectorBitCastLoweringPatterns(RewritePatternSet &patterns,
                                           int64_t targetRank = 1,
                                           PatternBenefit benefit = 1);

} // namespace vector
} // namespace mlir
#endif // MLIR_DIALECT_VECTOR_TRANSFORMS_LOWERINGPATTERNS_H
