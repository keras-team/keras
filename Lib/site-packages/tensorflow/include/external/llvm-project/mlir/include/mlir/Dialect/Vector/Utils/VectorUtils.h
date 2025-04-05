//===- VectorUtils.h - Vector Utilities -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_UTILS_VECTORUTILS_H_
#define MLIR_DIALECT_VECTOR_UTILS_VECTORUTILS_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {

// Forward declarations.
class AffineMap;
class Block;
class Location;
class OpBuilder;
class Operation;
class ShapedType;
class Value;
class VectorType;
class VectorTransferOpInterface;

namespace affine {
class AffineApplyOp;
class AffineForOp;
} // namespace affine

namespace vector {
/// Helper function that creates a memref::DimOp or tensor::DimOp depending on
/// the type of `source`.
Value createOrFoldDimOp(OpBuilder &b, Location loc, Value source, int64_t dim);

/// Returns two dims that are greater than one if the transposition is applied
/// on a 2D slice. Otherwise, returns a failure.
FailureOr<std::pair<int, int>> isTranspose2DSlice(vector::TransposeOp op);

/// Return true if `vectorType` is a contiguous slice of `memrefType`.
///
/// Only the N = vectorType.getRank() trailing dims of `memrefType` are
/// checked (the other dims are not relevant). Note that for `vectorType` to be
/// a contiguous slice of `memrefType`, the trailing dims of the latter have
/// to be contiguous - this is checked by looking at the corresponding strides.
///
/// There might be some restriction on the leading dim of `VectorType`:
///
/// Case 1. If all the trailing dims of `vectorType` match the trailing dims
///         of `memrefType` then the leading dim of `vectorType` can be
///         arbitrary.
///
///        Ex. 1.1 contiguous slice, perfect match
///          vector<4x3x2xi32> from memref<5x4x3x2xi32>
///        Ex. 1.2 contiguous slice, the leading dim does not match (2 != 4)
///          vector<2x3x2xi32> from memref<5x4x3x2xi32>
///
/// Case 2. If an "internal" dim of `vectorType` does not match the
///         corresponding trailing dim in `memrefType` then the remaining
///         leading dims of `vectorType` have to be 1 (the first non-matching
///         dim can be arbitrary).
///
///        Ex. 2.1 non-contiguous slice, 2 != 3 and the leading dim != <1>
///          vector<2x2x2xi32> from memref<5x4x3x2xi32>
///        Ex. 2.2  contiguous slice, 2 != 3 and the leading dim == <1>
///          vector<1x2x2xi32> from memref<5x4x3x2xi32>
///        Ex. 2.3. contiguous slice, 2 != 3 and the leading dims == <1x1>
///          vector<1x1x2x2xi32> from memref<5x4x3x2xi32>
///        Ex. 2.4. non-contiguous slice, 2 != 3 and the leading dims != <1x1>
///         vector<2x1x2x2xi32> from memref<5x4x3x2xi32>)
bool isContiguousSlice(MemRefType memrefType, VectorType vectorType);

/// Returns an iterator for all positions in the leading dimensions of `vType`
/// up to the `targetRank`. If any leading dimension before the `targetRank` is
/// scalable (so cannot be unrolled), it will return an iterator for positions
/// up to the first scalable dimension.
///
/// If no leading dimensions can be unrolled an empty optional will be returned.
///
/// Examples:
///
///   For vType = vector<2x3x4> and targetRank = 1
///
///   The resulting iterator will yield:
///     [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]
///
///   For vType = vector<3x[4]x5> and targetRank = 0
///
///   The scalable dimension blocks unrolling so the iterator yields only:
///     [0], [1], [2]
///
std::optional<StaticTileOffsetRange>
createUnrollIterator(VectorType vType, int64_t targetRank = 1);

/// Returns a functor (int64_t -> Value) which returns a constant vscale
/// multiple.
///
/// Example:
/// ```c++
/// auto createVscaleMultiple = makeVscaleConstantBuilder(rewriter, loc);
/// auto c4Vscale = createVscaleMultiple(4); // 4 * vector.vscale
/// ```
inline auto makeVscaleConstantBuilder(PatternRewriter &rewriter, Location loc) {
  Value vscale = nullptr;
  return [loc, vscale, &rewriter](int64_t multiplier) mutable {
    if (!vscale)
      vscale = rewriter.create<vector::VectorScaleOp>(loc);
    return rewriter.create<arith::MulIOp>(
        loc, vscale, rewriter.create<arith::ConstantIndexOp>(loc, multiplier));
  };
}

/// Returns a range over the dims (size and scalability) of a VectorType.
inline auto getDims(VectorType vType) {
  return llvm::zip_equal(vType.getShape(), vType.getScalableDims());
}

/// A wrapper for getMixedSizes for vector.transfer_read and
/// vector.transfer_write Ops (for source and destination, respectively).
///
/// Tensor and MemRef types implement their own, very similar version of
/// getMixedSizes. This method will call the appropriate version (depending on
/// `hasTensorSemantics`). It will also automatically extract the operand for
/// which to call it on (source for "read" and destination for "write" ops).
SmallVector<OpFoldResult> getMixedSizesXfer(bool hasTensorSemantics,
                                            Operation *xfer,
                                            RewriterBase &rewriter);

/// A pattern for ops that implement `MaskableOpInterface` and that _might_ be
/// masked (i.e. inside `vector.mask` Op region). In particular:
///   1. Matches `SourceOp` operation, Op.
///   2.1. If Op is masked, retrieves the masking Op, maskOp, and updates the
///     insertion point to avoid inserting new ops into the `vector.mask` Op
///     region (which only allows one Op).
///   2.2 If Op is not masked, this step is skipped.
///   3. Invokes `matchAndRewriteMaskableOp` on Op and optionally maskOp if
///     found in step 2.1.
///
/// This wrapper frees patterns from re-implementing the logic to update the
/// insertion point when a maskable Op is masked. Such patterns are still
/// responsible for providing an updated ("rewritten") version of:
///   a. the source Op when mask _is not_ present,
///   b. the source Op and the masking Op when mask _is_ present.
/// To use this pattern, implement `matchAndRewriteMaskableOp`. Note that
/// the return value will depend on the case above.
template <class SourceOp>
struct MaskableOpRewritePattern : OpRewritePattern<SourceOp> {
  using OpRewritePattern<SourceOp>::OpRewritePattern;

private:
  LogicalResult matchAndRewrite(SourceOp sourceOp,
                                PatternRewriter &rewriter) const final {
    auto maskableOp = dyn_cast<MaskableOpInterface>(sourceOp.getOperation());
    if (!maskableOp)
      return failure();

    Operation *rootOp = sourceOp;

    // If this Op is masked, update the insertion point to avoid inserting into
    // the vector.mask Op region.
    OpBuilder::InsertionGuard guard(rewriter);
    MaskingOpInterface maskOp;
    if (maskableOp.isMasked()) {
      maskOp = maskableOp.getMaskingOp();
      rewriter.setInsertionPoint(maskOp);
      rootOp = maskOp;
    }

    FailureOr<Value> newOp =
        matchAndRewriteMaskableOp(sourceOp, maskOp, rewriter);
    if (failed(newOp))
      return failure();

    // Rewriting succeeded but there are no values to replace.
    if (rootOp->getNumResults() == 0) {
      rewriter.eraseOp(rootOp);
    } else {
      assert(*newOp != Value() &&
             "Cannot replace an op's use with an empty value.");
      rewriter.replaceOp(rootOp, *newOp);
    }
    return success();
  }

public:
  // Matches `sourceOp` that can potentially be masked with `maskingOp`. If the
  // latter is present, returns a replacement for `maskingOp`. Otherwise,
  // returns a replacement for `sourceOp`.
  virtual FailureOr<Value>
  matchAndRewriteMaskableOp(SourceOp sourceOp, MaskingOpInterface maskingOp,
                            PatternRewriter &rewriter) const = 0;
};

/// Returns true if the input Vector type can be linearized.
///
/// Linearization is meant in the sense of flattening vectors, e.g.:
///   * vector<NxMxKxi32> -> vector<N*M*Kxi32>
/// In this sense, Vectors that are either:
///   * already linearized, or
///   * contain more than 1 scalable dimensions,
/// are not linearizable.
bool isLinearizableVector(VectorType type);

/// Create a TransferReadOp from `source` with static shape `readShape`. If the
/// vector type for the read is not the same as the type of `source`, then a
/// mask is created on the read, if use of mask is specified or the bounds on a
/// dimension are different.
///
/// `useInBoundsInsteadOfMasking` if false, the inBoundsVal values are set
/// properly, based on
///   the rank dimensions of the source and destination tensors. And that is
///   what determines if masking is done.
///
/// Note that the internal `vector::TransferReadOp` always read at indices zero
/// for each dimension of the passed in tensor.
Value createReadOrMaskedRead(OpBuilder &builder, Location loc, Value source,
                             ArrayRef<int64_t> readShape, Value padValue,
                             bool useInBoundsInsteadOfMasking);

/// Returns success if `inputVectorSizes` is a valid masking configuraion for
/// given `shape`, i.e., it meets:
///   1. The numbers of elements in both array are equal.
///   2. `inputVectorSizes` does not have dynamic dimensions.
///   3. All the values in `inputVectorSizes` are greater than or equal to
///      static sizes in `shape`.
LogicalResult isValidMaskedInputVector(ArrayRef<int64_t> shape,
                                       ArrayRef<int64_t> inputVectorSizes);
} // namespace vector

/// Constructs a permutation map of invariant memref indices to vector
/// dimension.
///
/// If no index is found to be invariant, 0 is added to the permutation_map and
/// corresponds to a vector broadcast along that dimension.
///
/// The implementation uses the knowledge of the mapping of loops to
/// vector dimension. `loopToVectorDim` carries this information as a map with:
///   - keys representing "vectorized enclosing loops";
///   - values representing the corresponding vector dimension.
/// Note that loopToVectorDim is a whole function map from which only enclosing
/// loop information is extracted.
///
/// Prerequisites: `indices` belong to a vectorizable load or store operation
/// (i.e. at most one invariant index along each AffineForOp of
/// `loopToVectorDim`). `insertPoint` is the insertion point for the vectorized
/// load or store operation.
///
/// Example 1:
/// The following MLIR snippet:
///
/// ```mlir
///    affine.for %i3 = 0 to %0 {
///      affine.for %i4 = 0 to %1 {
///        affine.for %i5 = 0 to %2 {
///          %a5 = load %arg0[%i4, %i5, %i3] : memref<?x?x?xf32>
///    }}}
/// ```
///
/// may vectorize with {permutation_map: (d0, d1, d2) -> (d2, d1)} into:
///
/// ```mlir
///    affine.for %i3 = 0 to %0 step 32 {
///      affine.for %i4 = 0 to %1 {
///        affine.for %i5 = 0 to %2 step 256 {
///          %4 = vector.transfer_read %arg0, %i4, %i5, %i3
///               {permutation_map: (d0, d1, d2) -> (d2, d1)} :
///               (memref<?x?x?xf32>, index, index) -> vector<32x256xf32>
///    }}}
/// ```
///
/// Meaning that vector.transfer_read will be responsible for reading the slice:
/// `%arg0[%i4, %i5:%15+256, %i3:%i3+32]` into vector<32x256xf32>.
///
/// Example 2:
/// The following MLIR snippet:
///
/// ```mlir
///    %cst0 = arith.constant 0 : index
///    affine.for %i0 = 0 to %0 {
///      %a0 = load %arg0[%cst0, %cst0] : memref<?x?xf32>
///    }
/// ```
///
/// may vectorize with {permutation_map: (d0) -> (0)} into:
///
/// ```mlir
///    affine.for %i0 = 0 to %0 step 128 {
///      %3 = vector.transfer_read %arg0, %c0_0, %c0_0
///           {permutation_map: (d0, d1) -> (0)} :
///           (memref<?x?xf32>, index, index) -> vector<128xf32>
///    }
/// ````
///
/// Meaning that vector.transfer_read will be responsible of reading the slice
/// `%arg0[%c0, %c0]` into vector<128xf32> which needs a 1-D vector broadcast.
///
AffineMap
makePermutationMap(Block *insertPoint, ArrayRef<Value> indices,
                   const DenseMap<Operation *, unsigned> &loopToVectorDim);
AffineMap
makePermutationMap(Operation *insertPoint, ArrayRef<Value> indices,
                   const DenseMap<Operation *, unsigned> &loopToVectorDim);

namespace matcher {

/// Matches vector.transfer_read, vector.transfer_write and ops that return a
/// vector type that is a multiple of the sub-vector type. This allows passing
/// over other smaller vector types in the function and avoids interfering with
/// operations on those.
/// This is a first approximation, it can easily be extended in the future.
/// TODO: this could all be much simpler if we added a bit that a vector type to
/// mark that a vector is a strict super-vector but it still does not warrant
/// adding even 1 extra bit in the IR for now.
bool operatesOnSuperVectorsOf(Operation &op, VectorType subVectorType);

} // namespace matcher
} // namespace mlir

#endif // MLIR_DIALECT_VECTOR_UTILS_VECTORUTILS_H_
