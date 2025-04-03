//===- LoopFusionUtils.h - Loop fusion utilities ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various loop fusion utility
// methods: these are not passes by themselves but are used either by passes,
// optimization sequences, or in turn by other transformation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_LOOPFUSIONUTILS_H
#define MLIR_DIALECT_AFFINE_LOOPFUSIONUTILS_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Operation;

namespace affine {
class AffineForOp;
struct ComputationSliceState;

struct FusionResult {
  enum ResultEnum {
    Success,
    FailPrecondition,     // Failed precondition for fusion. (e.g. same block).
    FailBlockDependence,  // Fusion would violate another dependence in block.
    FailFusionDependence, // Fusion would reverse dependences between loops.
    FailComputationSlice, // Unable to compute src loop computation slice.
    FailIncorrectSlice,   // Slice is computed, but it is incorrect.
  } value;
  FusionResult(ResultEnum v) : value(v) {}
};

/// Describes the fusion strategy to be used in the Affine loop fusion
/// utilities. Currently, it is used to specialized the loop fusion utilities
/// with the assumptions made in the AffineLoopFusion pass for producer-consumer
/// and sibling fusion, while sharing a single implementation. The latter
/// strategies are also limited to scenarios where a single memref is involved
/// in the producer-consume or sibling relationship between the candidate
/// loops. We use 'memref' to keep track of such a memref.
// TODO: Generalize utilities so that producer-consumer and sibling fusion
// strategies can be used without the assumptions made in the AffineLoopFusion
// pass.
class FusionStrategy {
public:
  enum StrategyEnum {
    // Generic loop fusion: Arbitrary loops are considered for fusion. No
    // assumptions about a specific fusion strategy from AffineLoopFusion pass
    // are made.
    // TODO: Generic fusion is not fully implemented by fusion utilities yet.
    // It should only be used for testing.
    Generic,
    // Producer-consumer fusion: Only loops with a producer-consumer
    // memref dependence are considered for fusion. Currently, assumptions from
    // the producer-consumer fusion implementation in AffineLoopFusion pass are
    // made. See pass for specific details.
    ProducerConsumer,
    // Sibling fusion: Only sibling loops with no producer-consumer memref
    // dependences are considered for fusion. Memref reuse is taken into account
    // for profitability. Currently, assumptions from the sibling fusion
    // implementation in AffineLoopFusion pass are made. See pass for specific
    // details.
    Sibling
  };

  /// Construct a generic or producer-consumer fusion strategy.
  FusionStrategy(StrategyEnum strategy) : strategy(strategy) {
    assert(strategy != Sibling &&
           "Sibling fusion strategy requires a specific memref");
  }

  /// Construct a sibling fusion strategy targeting 'memref'. This construct
  /// should only be used for sibling fusion.
  FusionStrategy(Value memref) : strategy(Sibling), memref(memref) {}

  /// Returns the fusion strategy.
  StrategyEnum getStrategy() const { return strategy; };

  /// Returns the memref attached to this sibling fusion strategy.
  Value getSiblingFusionMemRef() const {
    assert(strategy == Sibling && "Memref is only valid for sibling fusion");
    return memref;
  }

private:
  /// Fusion strategy.
  StrategyEnum strategy;

  /// Target memref for this fusion transformation. Only used for sibling
  /// fusion.
  Value memref;
};

/// Checks the feasibility of fusing the loop nest rooted at 'srcForOp' into the
/// loop nest rooted at 'dstForOp' at 'dstLoopDepth'. Returns FusionResult
/// 'Success' if fusion of the src/dst loop nests is feasible (i.e. they are
/// in the same block and dependences would not be violated). Otherwise
/// returns a FusionResult explaining why fusion is not feasible.
/// NOTE: This function is not feature complete and should only be used in
/// testing.
FusionResult
canFuseLoops(AffineForOp srcForOp, AffineForOp dstForOp, unsigned dstLoopDepth,
             ComputationSliceState *srcSlice,
             FusionStrategy fusionStrategy = FusionStrategy::Generic);

/// Fuses 'srcForOp' into 'dstForOp' with destination loop block insertion
/// point and source slice loop bounds specified in 'srcSlice'.
/// `isInnermostSiblingInsertionFusion` enables cleanup of `srcForOp that is a
/// single-iteration reduction loop being sibling-fused into a 'dstForOp'.
void fuseLoops(AffineForOp srcForOp, AffineForOp dstForOp,
               const ComputationSliceState &srcSlice,
               bool isInnermostSiblingInsertionFusion = false);

/// LoopNestStats aggregates various per-loop statistics (eg. loop trip count
/// and operation count) for a loop nest up until (and including) the innermost
/// loop body.
struct LoopNestStats {
  /// Map from AffineForOp to immediate child AffineForOps in its loop body.
  DenseMap<Operation *, SmallVector<AffineForOp, 2>> loopMap;
  /// Map from AffineForOp to count of operations in its loop body.
  DenseMap<Operation *, uint64_t> opCountMap;
  /// Map from AffineForOp to its constant trip count.
  DenseMap<Operation *, uint64_t> tripCountMap;
};

/// Collect loop nest statistics (eg. loop trip count and operation count)
/// in 'stats' for loop nest rooted at 'forOp'. Returns true on success,
/// returns false otherwise.
// TODO: Consider moving this to LoopUtils.
bool getLoopNestStats(AffineForOp forOp, LoopNestStats *stats);

/// Computes the total cost of the loop nest rooted at 'forOp' using 'stats'.
/// Currently, the total cost is computed by counting the total operation
/// instance count (i.e. total number of operations in the loop body * loop
/// trip count) for the entire loop nest.
int64_t getComputeCost(AffineForOp forOp, LoopNestStats &stats);

/// Computes and returns in 'computeCost', the total compute cost of fusing the
/// 'slice' of the loop nest rooted at 'srcForOp' into 'dstForOp'. Currently,
/// the total cost is computed by counting the total operation instance count
/// (i.e. total number of operations in the loop body * loop trip count) for
/// the entire loop nest.
/// Returns true on success, failure otherwise (e.g. non-constant trip counts).
bool getFusionComputeCost(AffineForOp srcForOp, LoopNestStats &srcStats,
                          AffineForOp dstForOp, LoopNestStats &dstStats,
                          const ComputationSliceState &slice,
                          int64_t *computeCost);

/// Returns in 'producerConsumerMemrefs' the memrefs involved in a
/// producer-consumer dependence between write ops in 'srcOps' and read ops in
/// 'dstOps'.
void gatherProducerConsumerMemrefs(ArrayRef<Operation *> srcOps,
                                   ArrayRef<Operation *> dstOps,
                                   DenseSet<Value> &producerConsumerMemrefs);

} // namespace affine
} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_LOOPFUSIONUTILS_H
