//===- VectorDistribution.h - Vector distribution patterns --*- C++------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORDISTRIBUTION_H_
#define MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORDISTRIBUTION_H_

#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir {
class RewritePatternSet;
namespace vector {

struct WarpExecuteOnLane0LoweringOptions {
  /// Lamdba function to let users allocate memory needed for the lowering of
  /// WarpExecuteOnLane0Op.
  /// The function needs to return an allocation that the lowering can use as
  /// temporary memory. The allocation needs to match the shape of the type (the
  /// type may be VectorType or a scalar) and be availble for the current warp.
  /// If there are several warps running in parallel the allocation needs to be
  /// split so that each warp has its own allocation.
  using WarpAllocationFn =
      std::function<Value(Location, OpBuilder &, WarpExecuteOnLane0Op, Type)>;
  WarpAllocationFn warpAllocationFn = nullptr;

  /// Lamdba function to let user emit operation to syncronize all the thread
  /// within a warp. After this operation all the threads can see any memory
  /// written before the operation.
  using WarpSyncronizationFn =
      std::function<void(Location, OpBuilder &, WarpExecuteOnLane0Op)>;
  WarpSyncronizationFn warpSyncronizationFn = nullptr;
};

void populateWarpExecuteOnLane0OpToScfForPattern(
    RewritePatternSet &patterns,
    const WarpExecuteOnLane0LoweringOptions &options,
    PatternBenefit benefit = 1);

using DistributionMapFn = std::function<AffineMap(Value)>;

/// Distribute transfer_write ops based on the affine map returned by
/// `distributionMapFn`. Writes of size more than `maxNumElementToExtract`
/// will not be distributed (it should be less than the warp size).
///
/// Example:
/// ```
/// %0 = vector.warp_execute_on_lane_0(%id){
///   ...
///   vector.transfer_write %v, %A[%c0] : vector<32xf32>, memref<128xf32>
///   vector.yield
/// }
/// ```
/// To
/// ```
/// %r:3 = vector.warp_execute_on_lane_0(%id) -> (vector<1xf32>) {
///   ...
///   vector.yield %v : vector<32xf32>
/// }
/// vector.transfer_write %v, %A[%id] : vector<1xf32>, memref<128xf32>
///
/// When applied at the same time as the vector propagation patterns,
/// distribution of `vector.transfer_write` is expected to have the highest
/// priority (pattern benefit). By making propagation of `vector.transfer_read`
/// be the lowest priority pattern, it will be the last vector operation to
/// distribute, meaning writes should propagate first.
void populateDistributeTransferWriteOpPatterns(
    RewritePatternSet &patterns, const DistributionMapFn &distributionMapFn,
    unsigned maxNumElementsToExtract, PatternBenefit benefit = 2);

/// Move scalar operations with no dependency on the warp op outside of the
/// region.
void moveScalarUniformCode(WarpExecuteOnLane0Op op);

/// Lambda signature to compute a warp shuffle of a given value of a given lane
/// within a given warp size.
using WarpShuffleFromIdxFn =
    std::function<Value(Location, OpBuilder &b, Value, Value, int64_t)>;

/// Collect patterns to propagate warp distribution. `distributionMapFn` is used
/// to decide how a value should be distributed when this cannot be inferred
/// from its uses.
///
/// The separate control over the `vector.transfer_read` op pattern benefit
/// is given to ensure the order of reads/writes before and after distribution
/// is consistent. As noted above, writes are expected to have the highest
/// priority for distribution, but are only ever distributed if adjacent to the
/// yield. By making reads the lowest priority pattern, it will be the last
/// vector operation to distribute, meaning writes should propagate first. This
/// is relatively brittle when ops fail to distribute, but that is a limitation
/// of these propagation patterns when there is a dependency not modeled by SSA.
void populatePropagateWarpVectorDistributionPatterns(
    RewritePatternSet &pattern, const DistributionMapFn &distributionMapFn,
    const WarpShuffleFromIdxFn &warpShuffleFromIdxFn,
    PatternBenefit benefit = 1, PatternBenefit readBenefit = 0);

/// Lambda signature to compute a reduction of a distributed value for the given
/// reduction kind and size.
using DistributedReductionFn =
    std::function<Value(Location, OpBuilder &, Value, CombiningKind, uint32_t)>;

/// Collect patterns to distribute vector reduction ops using given lamdba to
/// distribute reduction op.
void populateDistributeReduction(
    RewritePatternSet &pattern,
    const DistributedReductionFn &distributedReductionFn,
    PatternBenefit benefit = 1);

} // namespace vector
} // namespace mlir
#endif // MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORDISTRIBUTION_H_
