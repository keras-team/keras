//===- TileUsingInterface.h - Tiling ops using TilingInterface --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_TRANSFORMS_TILEUSINGINTERFACE_H
#define MLIR_DIALECT_SCF_TRANSFORMS_TILEUSINGINTERFACE_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include <deque>

namespace mlir {
class Operation;
class RewriterBase;
class TilingInterface;
} // namespace mlir

namespace mlir {
namespace scf {

using SCFTileSizeComputationFunction =
    std::function<SmallVector<OpFoldResult>(OpBuilder &, Operation *)>;

/// Options to use to control tiling.
struct SCFTilingOptions {
  /// Computation function that returns the tile sizes to use for each loop.
  /// Returning a tile size of zero implies no tiling for that loop. If the
  /// size of the returned vector is smaller than the number of loops, the inner
  /// loops are not tiled. If the size of the returned vector is larger, then
  /// the vector is truncated to number of loops.
  SCFTileSizeComputationFunction tileSizeComputationFunction = nullptr;

  SCFTilingOptions &
  setTileSizeComputationFunction(SCFTileSizeComputationFunction fun) {
    tileSizeComputationFunction = std::move(fun);
    return *this;
  }
  /// Convenience function to set the `tileSizeComputationFunction` to a
  /// function that computes tile sizes at the point they are needed. Allows
  /// proper interaction with folding.
  SCFTilingOptions &setTileSizes(ArrayRef<OpFoldResult> tileSizes);

  /// Computation function that returns the number of threads to use for
  /// each loop. Returning a num threads of zero implies no tiling for that
  /// loop. If the size of the returned vector is smaller than the number of
  /// loops, the inner loops are not tiled. If the size of the returned vector
  /// is larger, then the vector is truncated to number of loops. Note: This
  /// option is only supported with loopType set to `LoopType::ForallOp`. If the
  /// tile size function is not specified while the num threads computation is,
  /// then the tile size is determined automatically to map at most one tile per
  /// thread.
  SCFTileSizeComputationFunction numThreadsComputationFunction = nullptr;

  SCFTilingOptions &
  setNumThreadsComputationFunction(SCFTileSizeComputationFunction fun) {
    numThreadsComputationFunction = std::move(fun);
    return *this;
  }
  /// Convenience function to set the `numThreadsComputationFunction` to a
  /// function that computes num threads at the point they are needed.
  SCFTilingOptions &setNumThreads(ArrayRef<OpFoldResult> numThreads);

  /// The interchange vector to reorder the tiled loops.
  SmallVector<int64_t> interchangeVector = {};
  SCFTilingOptions &setInterchange(ArrayRef<int64_t> interchange) {
    interchangeVector = llvm::to_vector(interchange);
    return *this;
  }

  /// Specify which loop construct to use for tile and fuse.
  enum class LoopType { ForOp, ForallOp };
  LoopType loopType = LoopType::ForOp;
  SCFTilingOptions &setLoopType(LoopType type) {
    loopType = type;
    return *this;
  }

  /// Specify mapping of loops to devices. This is only respected when the loop
  /// constructs support such a mapping (like `scf.forall`). Will be ignored
  /// when using loop constructs that dont support such a mapping (like
  /// `scf.for`)
  SmallVector<Attribute> mappingVector = {};
  SCFTilingOptions &setMapping(ArrayRef<Attribute> mapping) {
    mappingVector = llvm::to_vector(mapping);
    return *this;
  }
};

/// Transformation information returned after tiling.
struct SCFTilingResult {
  /// Tiled operations that are generated during tiling. The order does not
  /// matter except the last op. The replacements are expected to be the results
  /// of the last op.
  SmallVector<Operation *> tiledOps;
  /// The `scf.for` operations that iterate over the tiles.
  SmallVector<LoopLikeOpInterface> loops;
  /// Values to use as replacements for the untiled op. Is the same size as the
  /// number of results of the untiled op.
  SmallVector<Value> replacements;
  /// Slices generated after tiling that can be used for fusing with the tiled
  /// producer.
  SmallVector<Operation *> generatedSlices;
};

/// Method to tile an op that implements the `TilingInterface` using
/// `scf.for` for iterating over the tiles.
FailureOr<SCFTilingResult> tileUsingSCF(RewriterBase &rewriter,
                                        TilingInterface op,
                                        const SCFTilingOptions &options);

/// Options used to control tile + fuse.
struct SCFTileAndFuseOptions {
  /// The tiling options used to control the tiling of the consumer.
  SCFTilingOptions tilingOptions;
  SCFTileAndFuseOptions &setTilingOptions(SCFTilingOptions options) {
    tilingOptions = options;
    return *this;
  }

  /// Control function to check if a slice needs to be fused or not,
  /// The control function receives
  /// 1) the slice along which fusion is to be done,
  /// 2) the producer value that is to be fused
  /// 3) a boolean value set to `true` if the fusion is from
  ///    a destination operand.
  /// The control function returns an `std::optiona<ControlFnResult>`.
  /// If the return value is `std::nullopt`, that implies no fusion
  /// is to be performed along that slice.
  struct ControlFnResult {
    /// Set to true if the loop nest has to return a replacement value
    /// for the fused producer.
    bool yieldProducerReplacement = false;
  };
  using ControlFnTy = std::function<std::optional<ControlFnResult>(
      tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
      bool isDestinationOperand)>;
  /// The default control function implements greedy fusion without yielding
  /// a replacement for any of the fused results.
  ControlFnTy fusionControlFn = [](tensor::ExtractSliceOp, OpResult,
                                   bool) -> std::optional<ControlFnResult> {
    return ControlFnResult{};
  };
  SCFTileAndFuseOptions &setFusionControlFn(ControlFnTy controlFn) {
    fusionControlFn = controlFn;
    return *this;
  }
};

/// Fuse the producer of the source of `candidateSliceOp` by computing the
/// required slice of the producer in-place.  Note that the method
/// replaces the uses of `candidateSliceOp` with the tiled and fused producer
/// value but does not delete the slice operation.
struct SCFFuseProducerOfSliceResult {
  OpResult origProducer;       // Original untiled producer.
  Value tiledAndFusedProducer; // Tile and fused producer value.
  SmallVector<Operation *> tiledOps;
  SmallVector<Operation *> generatedSlices;
};
std::optional<SCFFuseProducerOfSliceResult>
tileAndFuseProducerOfSlice(RewriterBase &rewriter,
                           tensor::ExtractSliceOp candidateSliceOp,
                           MutableArrayRef<LoopLikeOpInterface> loops);

/// Reconstruct the fused producer from within the tiled-and-fused code. Based
/// on the slice of the producer computed in place it is possible that within
/// the loop nest same slice of the producer is computed multiple times. It is
/// in general not possible to recompute the value of the fused producer from
/// the tiled loop code in such cases. For the cases where no slice of the
/// producer is computed in a redundant fashion it is possible to reconstruct
/// the value of the original producer from within the tiled loop. It is upto
/// the caller to ensure that the producer is not computed redundantly within
/// the tiled loop nest. For example, consider
///
/// ```mlir
/// %0 = linalg.matmul ins(...) outs(...) -> tensor<?x?xf32>
/// %1 = linalg.matmul ins(%0, ..) outs(...) -> tensor<?x?x?f32>
/// ```
///
/// If `%1` is tiled in a 2D fashion and `%0` is fused with it, the resulting IR
/// is,
///
/// ```mlir
/// %t1_0 = scf.for .... iter_args(%arg0 = ...) {
///   %t1_1 = scf.for ... iter_args(%arg1 = %arg0) {
///     ...
///     %t1_2 = linalg.matmul ins(...) outs(...) -> tensor<?x?xf32>
///     %t1_3 = linalg.matmul ins(%t1_2, ...)
///     %t1_4 = tensor.insert_slice %t1_3 into %arg1 ...
///     scf.yield %t1_4
///   }
///   scf.yield %t1_1
/// }
/// ```
///
/// Here `%t1_2` is the same for all iterations of the inner `scf.for`. Instead
/// if `%1` were tiled only along the rows, the resultant code would be
///
/// ```mlir
/// %t2_0 = scf.for .... iter_args(%arg0 = ...) {
///   ...
///   %t2_1 = linalg.matmul ins(...) outs(...) -> tensor<?x?xf32>
///   %t2_2 = linalg.matmul ins(%t2_1, ...)
///   %t2_3 = tensor.insert_slice %t2_2 into %arg0 ...
///   scf.yield %t2_3
/// }
/// ```
///
/// Here there is no intersection in the different slices of `%t2_1` computed
/// across iterations of the `scf.for`. In such cases, the value of the original
/// `%0` can be reconstructed from within the loop body. This is useful in cases
/// where `%0` had other uses as well. If not reconstructed from within the loop
/// body, uses of `%0` could not be replaced, making it still live and the
/// fusion immaterial.
///
/// The @param `yieldResultNumber` decides which result would be yield. If not
/// given, yield all `opResult` of fused producer.
///
/// The method returns the list of new slices added during the process (which
/// can be used to fuse along).
FailureOr<SmallVector<Operation *>> yieldReplacementForFusedProducer(
    RewriterBase &rewriter, tensor::ExtractSliceOp sliceOp,
    scf::SCFFuseProducerOfSliceResult fusedProducerInfo,
    MutableArrayRef<LoopLikeOpInterface> loops,
    ArrayRef<unsigned> yieldResultNumber = ArrayRef<unsigned>{});

/// Transformation information returned after tile and fuse.
struct SCFTileAndFuseResult {
  /// List of untiled operations that were fused with the tiled consumer.
  llvm::SetVector<Operation *> fusedProducers;
  /// List of tiled and fused operations generated. The first one in this list
  /// is guaranteed to be the tiled operations generated during tiling of the
  /// generated operation.
  llvm::SetVector<Operation *> tiledAndFusedOps;
  /// The `scf.for` operations that iterate over the tiles.
  SmallVector<LoopLikeOpInterface> loops;
  /// The replacement values to use for the tiled and fused operations.
  llvm::DenseMap<Value, Value> replacements;
};

/// Method to tile and fuse a sequence of operations, by tiling the consumer
/// and fusing its producers. Note that this assumes that it is valid to
/// tile+fuse the producer into the innermost tiled loop. Its up to the caller
/// to ensure that the tile sizes provided make this fusion valid.
///
/// For example, for the following sequence
///
/// ```mlir
/// %0 =
/// %1 = linalg.fill ... outs(%0 : ... )
/// %2 = linalg.matmul ... outs(%1 : ...) ...
/// ```
///
/// it is legal to fuse the fill with the matmul only if the matmul is tiled
/// along the parallel dimensions and not the reduction dimension, i.e. the tile
/// size for the reduction dimension should be 0. The resulting fused
/// transformation is
///
/// ```mlir
/// %1 = scf.for ... iter_args(%arg0 = %0)
///   %2 = tensor.extract_slice %arg0
///   %3 = linalg.fill .. outs(%2 : ... )
///   %4 = linalg.matmul .. outs(%3 : ...)
/// }
/// ```
FailureOr<SCFTileAndFuseResult>
tileConsumerAndFuseProducersUsingSCF(RewriterBase &rewriter,
                                     TilingInterface consumer,
                                     const SCFTileAndFuseOptions &options);

/// Fuse the consumer of the source of `candidateSliceOp` by computing the
/// required slice of the consumer in-place.  Note that the method
/// replaces the uses of `candidateSliceOp` with the tiled and fused consumer
/// value but does not delete the slice operation.
struct SCFFuseConsumerOfSliceResult {
  OpOperand *origConsumerOperand; // Original untiled consumer's operand.
  OpOperand
      *tiledAndFusedConsumerOperand; // Tiled and fused consumer's operand.
  SmallVector<Operation *> tiledOps;
};
FailureOr<scf::SCFFuseConsumerOfSliceResult>
tileAndFuseConsumerOfSlice(RewriterBase &rewriter, Operation *candidateSliceOp);

/// Method to lower an `op` that implements the `TilingInterface` to
/// loops/scalars.
FailureOr<SmallVector<scf::ForOp>>
lowerToLoopsUsingSCFForOp(RewriterBase &rewriter, TilingInterface op);

/// Transformation information returned after reduction tiling.
struct SCFReductionTilingResult {
  /// The partial reduction tiled op generated.
  SmallVector<Operation *> parallelTiledOps;
  /// The final reduction operation merging all the partial reductions.
  SmallVector<Operation *> mergeOps;
  /// Initial values used for reduction.
  SmallVector<Value> initialValues;
  /// The loop operations that iterate over the tiles.
  SmallVector<LoopLikeOpInterface> loops;
  /// The replacements to use for the results of the tiled operation.
  SmallVector<Value> replacements;
};

/// Method to tile a reduction and generate a parallel op within a serial loop.
/// Each of the partial reductions are calculated in parallel. Then after the
/// loop all the partial reduction are merged into a final reduction.
/// For example for the following sequence
///
/// ```mlir
/// %0 = linalg.generic %in ["parallel", "reduction"]
///   : tensor<7x9xf32> -> tensor<7xf32>
/// ```
///
/// into:
///
/// ```mlir
/// %0 = linalg.fill ... : tensor<7x4xf32>
/// %1 = scf.for ... iter_args(%arg0 = %0)
///   %2 = tensor.extract_slice %arg0 : tensor<7x4xf32> -> tensor<7x?xf32>
///   %3 = tensor.extract_slice %in : tensor<7x9xf32> -> tensor<7x?xf32>
///   %4 = linalg.generic %2, %3 ["parallel", "parallel"]
///     : tensor<7x?xf32> -> tensor<7x?xf32>
///   %5 = tensor.insert_slice %3, %0[0, 0] : tensor<7x4xf32>
/// }
/// %6 = linalg.generic %1 ["parallel", "reduction"]
///   : tensor<7x4xf32> -> tensor<7xf32>
/// ```
FailureOr<scf::SCFReductionTilingResult>
tileReductionUsingScf(RewriterBase &b, PartialReductionOpInterface op,
                      ArrayRef<OpFoldResult> tileSize);

} // namespace scf
} // namespace mlir

#endif // MLIR_DIALECT_SCF_TRANSFORMS_TILEUSINGINTERFACE_H
