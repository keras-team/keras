//===- VectorTransforms.h - Vector transformations as patterns --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORTRANSFORMS_H
#define MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORTRANSFORMS_H

#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir {
class MLIRContext;
class VectorTransferOpInterface;
class RewritePatternSet;
class RewriterBase;

namespace scf {
class IfOp;
} // namespace scf

namespace vector {

//===----------------------------------------------------------------------===//
// Vector transformation options exposed as auxiliary structs.
//===----------------------------------------------------------------------===//

/// Structure to control the behavior of vector transform patterns.
struct VectorTransformsOptions {
  /// Option to control the lowering of vector.contract.
  VectorContractLowering vectorContractLowering = VectorContractLowering::Dot;
  VectorTransformsOptions &
  setVectorTransformsOptions(VectorContractLowering opt) {
    vectorContractLowering = opt;
    return *this;
  }
  /// Option to control the lowering of vector.multi_reduction.
  VectorMultiReductionLowering vectorMultiReductionLowering =
      VectorMultiReductionLowering::InnerParallel;
  VectorTransformsOptions &
  setVectorMultiReductionLowering(VectorMultiReductionLowering opt) {
    vectorMultiReductionLowering = opt;
    return *this;
  }
  /// Option to control the lowering of vector.transpose.
  VectorTransposeLowering vectorTransposeLowering =
      VectorTransposeLowering::EltWise;
  VectorTransformsOptions &
  setVectorTransposeLowering(VectorTransposeLowering opt) {
    vectorTransposeLowering = opt;
    return *this;
  }
  /// Option to control the splitting of vector transfers.
  VectorTransferSplit vectorTransferSplit = VectorTransferSplit::None;
  VectorTransformsOptions &setVectorTransferSplit(VectorTransferSplit opt) {
    vectorTransferSplit = opt;
    return *this;
  }
};

//===----------------------------------------------------------------------===//
// Standalone transformations and helpers.
//===----------------------------------------------------------------------===//

/// Split a vector.transfer operation into an in-bounds (i.e., no
/// out-of-bounds masking) fastpath and a slowpath. If `ifOp` is not null and
/// the result is `success, the `ifOp` points to the newly created conditional
/// upon function return. To accomodate for the fact that the original
/// vector.transfer indexing may be arbitrary and the slow path indexes
/// @[0...0] in the temporary buffer, the scf.if op returns a view and values
/// of type index. At this time, only vector.transfer_read case is
/// implemented.
///
/// Example (a 2-D vector.transfer_read):
/// ```
///    %1 = vector.transfer_read %0[...], %pad : memref<A...>, vector<...>
/// ```
/// is transformed into:
/// ```
///    %1:3 = scf.if (%inBounds) {
///      // fastpath, direct cast
///      memref.cast %A: memref<A...> to compatibleMemRefType
///      scf.yield %view : compatibleMemRefType, index, index
///    } else {
///      // slowpath, not in-bounds vector.transfer or linalg.copy.
///      memref.cast %alloc: memref<B...> to compatibleMemRefType
///      scf.yield %4 : compatibleMemRefType, index, index
//     }
///    %0 = vector.transfer_read %1#0[%1#1, %1#2] {in_bounds = [true ...
///    true]}
/// ```
/// where `alloc` is a top of the function alloca'ed buffer of one vector.
///
/// Preconditions:
///  1. `xferOp.permutation_map()` must be a minor identity map
///  2. the rank of the `xferOp.memref()` and the rank of the
///  `xferOp.vector()` must be equal. This will be relaxed in the future but
///  requires rank-reducing subviews.
LogicalResult splitFullAndPartialTransfer(
    RewriterBase &b, VectorTransferOpInterface xferOp,
    VectorTransformsOptions options = VectorTransformsOptions(),
    scf::IfOp *ifOp = nullptr);

/// Implements transfer op write to read forwarding and dead transfer write
/// optimizations.
void transferOpflowOpt(RewriterBase &rewriter, Operation *rootOp);

/// Cast away the leading unit dim, if exists, for the given contract op.
/// Return success if the transformation applies; return failure otherwise.
FailureOr<Value>
castAwayContractionLeadingOneDim(vector::ContractionOp contractOp,
                                 MaskingOpInterface maskingOp,
                                 RewriterBase &rewriter);

// Structure to hold the range of `vector.vscale`.
struct VscaleRange {
  unsigned vscaleMin;
  unsigned vscaleMax;
};

/// Attempts to eliminate redundant vector masks by replacing them with all-true
/// constants at the top of the function (which results in the masks folding
/// away). Note: Currently, this only runs for vector.create_mask ops and
/// requires `vscaleRange`. If `vscaleRange` is not provided this transform does
/// nothing. This is because these redundant masks are much more likely for
/// scalable code which requires memref/tensor dynamic sizes, whereas fixed-size
/// code has static sizes, so simpler folds remove the masks.
void eliminateVectorMasks(IRRewriter &rewriter, FunctionOpInterface function,
                          std::optional<VscaleRange> vscaleRange = {});

} // namespace vector
} // namespace mlir

#endif // MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORTRANSFORMS_H
