//===- VectorRewritePatterns.h - Vector rewrite patterns --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORREWRITEPATTERNS_H
#define MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORREWRITEPATTERNS_H

#include <optional>
#include <utility>

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Dialect/Vector/Transforms/VectorTransformsEnums.h.inc"

namespace mlir {
class ConversionTarget;
class RewritePatternSet;
class TypeConverter;

namespace arith {
class AndIOp;
class NarrowTypeEmulationConverter;
class TruncIOp;
} // namespace arith

namespace vector {
struct VectorTransformsOptions;

/// Options that control the vector unrolling.
struct UnrollVectorOptions {
  using FilterConstraintFnType = std::function<LogicalResult(Operation *op)>;
  /// Callback function that indicates whether vector unrolling should be
  /// attempted on the operation.
  FilterConstraintFnType filterConstraint = nullptr;
  UnrollVectorOptions &setFilterConstraint(FilterConstraintFnType constraint) {
    filterConstraint = std::move(constraint);
    return *this;
  }

  using NativeShapeFnType =
      std::function<std::optional<SmallVector<int64_t>>(Operation *op)>;
  /// Function that returns the shape of the vector to unroll to for a given
  /// operation. The unrolling is aborted if the function returns
  /// `std::nullopt`.
  NativeShapeFnType nativeShape = nullptr;
  UnrollVectorOptions &setNativeShapeFn(NativeShapeFnType fn) {
    nativeShape = std::move(fn);
    return *this;
  }

  /// Set the native shape to use for unrolling.
  UnrollVectorOptions &setNativeShape(ArrayRef<int64_t> shape) {
    SmallVector<int64_t> tsShape(shape);
    nativeShape = [=](Operation *) -> std::optional<SmallVector<int64_t>> {
      return tsShape;
    };
    return *this;
  }

  /// Function that returns the traversal order (in terms of "for loop order",
  /// i.e. slowest varying dimension to fastest varying dimension) that should
  /// be used when unrolling the given operation into units of the native vector
  /// size.
  using UnrollTraversalOrderFnType =
      std::function<std::optional<SmallVector<int64_t>>(Operation *op)>;
  UnrollTraversalOrderFnType traversalOrderCallback = nullptr;
  UnrollVectorOptions &
  setUnrollTraversalOrderFn(UnrollTraversalOrderFnType traversalOrderFn) {
    traversalOrderCallback = std::move(traversalOrderFn);
    return *this;
  }
};

/// Canonicalization of a `vector.contraction %a, %b, %c` with row-major matmul
/// semantics to a contraction with MMT semantics (matrix matrix multiplication
/// with the RHS transposed). This specific form is meant to have the vector
/// operands are organized such that the reduction dimension is contiguous.
/// Example:
/// ```
/// vector.contract {indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
///                                   affine_map<(m, n, k) -> (n, k)>,
///                                   affine_map<(m, n, k) -> (m, n)>],
///                  iterator_types = ["parallel", "parallel", "reduction"],
///                  kind = #vector.kind<add>} %a, %b, %c : ...
/// ```
///
///  The `constraint` predicate is used to decide which `vector.contraction` ops
///  to filter out.
void populateVectorContractCanonicalizeMatmulToMMT(
    RewritePatternSet &patterns,
    std::function<LogicalResult(vector::ContractionOp)> constraint =
        [](vector::ContractionOp) { return success(); },
    PatternBenefit = 1);

/// Collect patterns to convert reduction op to vector.contract and fold
/// transpose/broadcast ops into the contract.
void populateVectorReductionToContractPatterns(RewritePatternSet &patterns,
                                               PatternBenefit benefit = 1);

/// Populate `patterns` with the following patterns.
///
///   - VectorTransferFullPartialRewriter
///
/// Split a vector.transfer operation into an in-bounds (i.e., no out-of-bounds
/// masking) fast path and a slow path.
///
/// Example (a 2-D vector.transfer_read):
/// ```
///    %1 = vector.transfer_read %0[...], %pad : memref<A...>, vector<...>
/// ```
/// is transformed into:
/// ```
///    %1:3 = scf.if (%inBounds) {
///      // fast path, direct cast
///      memref.cast %A: memref<A...> to compatibleMemRefType
///      scf.yield %view : compatibleMemRefType, index, index
///    } else {
///      // slow path, not in-bounds vector.transfer or linalg.copy.
///      memref.cast %alloc: memref<B...> to compatibleMemRefType
///      scf.yield %4 : compatibleMemRefType, index, index
//     }
///    %0 = vector.transfer_read %1#0[%1#1, %1#2] {in_bounds = [true ... true]}
/// ```
/// where `alloc` is a top of the function alloca'ed buffer of one vector.
///
/// Preconditions:
///  1. `xferOp.permutation_map()` must be a minor identity map
///  2. the rank of the `xferOp.memref()` and the rank of the `xferOp.vector()`
///  must be equal. This will be relaxed in the future but requires
///  rank-reducing subviews.
void populateVectorTransferFullPartialPatterns(
    RewritePatternSet &patterns, const VectorTransformsOptions &options);

/// Collect a set of patterns to reduce the rank of the operands of vector
/// transfer ops to operate on the largest contigious vector.
/// These patterns are useful when lowering to dialects with 1d vector type
/// such as llvm and it will result fewer memory reads.
void populateVectorTransferCollapseInnerMostContiguousDimsPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit = 1);

/// Patterns that remove redundant Vector Ops by re-ordering them with
/// e.g. elementwise Ops:
/// ```
/// %at = vector.transpose %a, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
/// %bt = vector.transpose %b, [1, 0]: vector<4x2xf32> to vector<2x4xf32>
/// %r = arith.addf %at, %bt : vector<2x4xf32>
/// ```
/// gets converted to:
/// ```
/// %0 = arith.addf %a, %b : vector<4x2xf32>
/// %r = vector.transpose %0, [1, 0] : vector<2x4xf32>
/// ```
/// At the moment, these patterns are limited to vector.broadcast and
/// vector.transpose.
void populateSinkVectorOpsPatterns(RewritePatternSet &patterns,
                                   PatternBenefit benefit = 1);

/// Patterns that fold chained vector reductions. These patterns assume that
/// elementwise operations (e.g., `arith.addf` with vector operands) are
/// cheaper than vector reduction.
/// Note that these patterns change the order of reduction which may not always
/// produce bit-identical results on some floating point inputs.
///
/// Example:
/// ```
/// %a = vector.reduction <add> %x, %acc
/// %b = vector.reduction <add> %y, %a
/// ```
/// is transformed into:
/// ```
/// %a = arith.addf %x, %y
/// %b = vector.reduction <add> %a, %acc
/// ```
void populateChainedVectorReductionFoldingPatterns(RewritePatternSet &patterns,
                                                   PatternBenefit benefit = 1);

/// Patterns to break down vector reductions into a series of arith reductions
/// over vector elements. This is intended to be simplify code with reductions
/// over small vector types and avoid more specialized reduction lowering when
/// possible.
///
/// Example:
/// ```
/// %a = vector.reduction <add> %x : vector<2xf32> into f32
/// ```
/// is transformed into:
/// ```
/// %y = vector.extract %x[0] : f32 from vector<2xf32>
/// %z = vector.extract %x[1] : f32 from vector<2xf32>
/// %a = arith.addf %y, %z : f32
/// ```
void populateBreakDownVectorReductionPatterns(
    RewritePatternSet &patterns, unsigned maxNumElementsToExtract = 2,
    PatternBenefit benefit = 1);

/// Populate `patterns` with the following patterns.
///
/// [DecomposeDifferentRankInsertStridedSlice]
/// ==========================================
/// RewritePattern for InsertStridedSliceOp where source and destination vectors
/// have different ranks.
///
/// When ranks are different, InsertStridedSlice needs to extract a properly
/// ranked vector from the destination vector into which to insert. This pattern
/// only takes care of this extraction part and forwards the rest to
/// [VectorInsertStridedSliceOpSameRankRewritePattern].
///
/// For a k-D source and n-D destination vector (k < n), we emit:
///   1. ExtractOp to extract the (unique) (n-1)-D subvector into which to
///      insert the k-D source.
///   2. k-D -> (n-1)-D InsertStridedSlice op
///   3. InsertOp that is the reverse of 1.
///
/// [DecomposeNDExtractStridedSlice]
/// ================================
/// For such cases, we can rewrite it to ExtractOp/ExtractElementOp + lower
/// rank ExtractStridedSliceOp + InsertOp/InsertElementOp for the n-D case.
void populateVectorInsertExtractStridedSliceDecompositionPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit = 1);

/// Populate `patterns` with a pattern to breaks down 1-D extract_strided_slice
/// ops into a chain of Extract ops to extract each element from the source, and
/// then a chain of Insert ops to insert to the target vector.
///
/// If `controlFn` is not nullptr, the pattern will only be invoked on ops that
/// `controlFn` returns true. Otherwise runs on ops.
void populateVectorExtractStridedSliceToExtractInsertChainPatterns(
    RewritePatternSet &patterns,
    std::function<bool(ExtractStridedSliceOp)> controlFn = nullptr,
    PatternBenefit benefit = 1);

/// Populate `patterns` with a pattern to break down 1-D vector.bitcast ops
/// based on the destination vector shape. Bitcasts from a lower bitwidth
/// element type to a higher bitwidth one are extracted from the lower bitwidth
/// based on the native destination vector shape and inserted based on the ratio
/// of the bitwidths.
///
/// This acts as a last resort way to break down vector.bitcast ops to smaller
/// vector sizes. Because this pattern composes until it is bitcasting to a
/// single element of the higher bitwidth, the is an optional control function.
/// If `controlFn` is not nullptr, the pattern will only apply to ops where
/// `controlFn` returns true, otherwise applies to all bitcast ops.
void populateBreakDownVectorBitCastOpPatterns(
    RewritePatternSet &patterns,
    std::function<bool(BitCastOp)> controlFn = nullptr,
    PatternBenefit benefit = 1);

/// Populate `patterns` with the following patterns.
///
/// Patterns in populateVectorInsertExtractStridedSliceDecompositionPatterns();
///
/// [ConvertSameRankInsertStridedSliceIntoShuffle]
/// ==============================================
/// RewritePattern for InsertStridedSliceOp where source and destination vectors
/// have the same rank. For each outermost index in the slice:
///   begin    end             stride
/// [offset : offset+size*stride : stride]
///   1. ExtractOp one (k-1)-D source subvector and one (n-1)-D dest subvector.
///   2. InsertStridedSlice (k-1)-D into (n-1)-D
///   3. the destination subvector is inserted back in the proper place
///   3. InsertOp that is the reverse of 1.
///
/// [Convert1DExtractStridedSliceIntoShuffle]
/// =========================================
/// For such cases, we can lower it to a ShuffleOp.
void populateVectorInsertExtractStridedSliceTransforms(
    RewritePatternSet &patterns, PatternBenefit benefit = 1);

/// Collect a set of pattern to unroll vector operations to a smaller shapes.
/// `options` structure controls which operations are unrolled and the target
/// shape.
/// `op` is unrolled to the `targetShape` as follows, for each of its operands:
///   1. the unrolled type `unrolledVectorType` and number of unrolled instances
///   `numUnrolledInstances` are computed from the `targetShape`. For now it is
///   assumed the unrolling factors divide the vector sizes.
///   2. ExtractStridedSlice are created to break-up the vector operands.
///   3. the original op is cloned `numUnrolledInstances` times, once for each
///   result.
///   4. InsertStridedSlice are inserted to re-assemble the slices into the
///   original vectore shape.
///
/// Example:
///
///    opA(operand0, operand1)  // numUnrolledInstances = 3
///
///            operand0                   operand1
///               |                          |
///             fork                       fork
///        <----------gather all fork ops --------->
///              /|\                        /|\
///          f00 f01 f02                f10 f11 f12
///        <---------- clone op 3 times --------->
///          opA0(f00, f10), opA1(f01, f11), opA2(f02, f12)
///                 \            |            /
///      <-------------------- join ------------------------->
///
/// Other local patterns then kick in iteratively (including DCE) and compose
/// to combine the ExtractStridedSlice/InsertStridedSlice.
void populateVectorUnrollPatterns(RewritePatternSet &patterns,
                                  const UnrollVectorOptions &options,
                                  PatternBenefit benefit = 1);

/// Collect a set of vector.shape_cast folding patterns.
void populateShapeCastFoldingPatterns(RewritePatternSet &patterns,
                                      PatternBenefit benefit = 1);

/// Collect a set of leading one dimension removal patterns.
///
/// These patterns insert vector.shape_cast to remove leading one dimensions
/// to expose more canonical forms of read/write/insert/extract operations.
/// With them, there are more chances that we can cancel out extract-insert
/// pairs or forward write-read pairs.
void populateCastAwayVectorLeadingOneDimPatterns(RewritePatternSet &patterns,
                                                 PatternBenefit benefit = 1);

/// Collect a set of one dimension removal patterns.
///
/// These patterns insert rank-reducing memref.subview ops to remove one
/// dimensions. With them, there are more chances that we can avoid
/// potentially expensive vector.shape_cast operations.
void populateVectorTransferDropUnitDimsPatterns(RewritePatternSet &patterns,
                                                PatternBenefit benefit = 1);

/// Collect a set of patterns that use vector.shape_cast to help fold unit dims.
///
/// These patterns use vector.shape_cast to remove unit dims from e.g.
/// arithmetic operations on Vectors. The newly inserted shape_casts will either
/// cancel each other out or will be folded away when combined with other
/// patterns.
void populateDropUnitDimWithShapeCastPatterns(RewritePatternSet &patterns,
                                              PatternBenefit benefit = 1);

/// Collect a set of patterns to flatten n-D vector transfers on contiguous
/// memref.
///
/// These patterns insert memref.collapse_shape + vector.shape_cast patterns
/// to transform multiple small n-D transfers into a larger 1-D transfer where
/// the memref contiguity properties allow it.
///
/// Flattening is only applied if the bitwidth of the trailing vector dimension
/// is smaller or equal to `targetVectorBitwidth`.
void populateFlattenVectorTransferPatterns(
    RewritePatternSet &patterns,
    unsigned targetVectorBitwidth = std::numeric_limits<unsigned>::max(),
    PatternBenefit benefit = 1);

/// Collect a set of patterns that bubble up/down bitcast ops.
///
/// These patterns move vector.bitcast ops to be before insert ops or after
/// extract ops where suitable. With them, bitcast will happen on smaller
/// vectors and there are more chances to share extract/insert ops.
void populateBubbleVectorBitCastOpPatterns(RewritePatternSet &patterns,
                                           PatternBenefit benefit = 1);

/// These patterns materialize masks for various vector ops such as transfers.
void populateVectorMaskMaterializationPatterns(RewritePatternSet &patterns,
                                               bool force32BitVectorIndices,
                                               PatternBenefit benefit = 1);

/// Appends patterns for emulating vector operations over narrow types with ops
/// over wider types.
void populateVectorNarrowTypeEmulationPatterns(
    arith::NarrowTypeEmulationConverter &typeConverter,
    RewritePatternSet &patterns);

/// Rewrite a vector `bitcast(trunci)` to use a more efficient sequence of
/// vector operations comprising `shuffle` and `bitwise` ops.
/// Warning: these patterns currently only work for little endian targets.
FailureOr<Value> rewriteBitCastOfTruncI(RewriterBase &rewriter,
                                        vector::BitCastOp bitCastOp,
                                        arith::TruncIOp truncOp,
                                        vector::BroadcastOp maybeBroadcastOp);

/// Rewrite a vector `ext(bitcast)` to use a more efficient sequence of
/// vector operations comprising `shuffle` and `bitwise` ops.
/// Warning: these patterns currently only work for little endian targets.
FailureOr<Value> rewriteExtOfBitCast(RewriterBase &rewriter, Operation *extOp,
                                     vector::BitCastOp bitCastOp,
                                     vector::BroadcastOp maybeBroadcastOp);

/// Appends patterns for rewriting vector operations over narrow types with
/// ops over wider types.
/// Warning: these patterns currently only work for little endian targets.
void populateVectorNarrowTypeRewritePatterns(RewritePatternSet &patterns,
                                             PatternBenefit benefit = 1);

/// Appends patterns for emulating a sub-byte vector transpose.
void populateVectorTransposeNarrowTypeRewritePatterns(
    RewritePatternSet &patterns, PatternBenefit benefit = 1);

/// Populates patterns for ND vectors (N >= 2) linearization and sets up the
/// provided ConversionTarget with the appropriate legality configuration for
/// the ops to get converted properly.
void populateVectorLinearizeTypeConversionsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, unsigned targetBitWidth);

/// Populates patterns for linearizing ND (N >= 2) vector operations to 1D
/// vector shuffle operations.
void populateVectorLinearizeShuffleLikeOpsPatterns(TypeConverter &typeConverter,
                                                   RewritePatternSet &patterns,
                                                   ConversionTarget &target,
                                                   unsigned targetBitWidth);

} // namespace vector
} // namespace mlir

#endif // MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORREWRITEPATTERNS_H
