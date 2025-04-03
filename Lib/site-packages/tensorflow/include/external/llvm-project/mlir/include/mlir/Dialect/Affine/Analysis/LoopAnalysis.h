//===- LoopAnalysis.h - loop analysis methods -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for methods to analyze loops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_ANALYSIS_LOOPANALYSIS_H
#define MLIR_DIALECT_AFFINE_ANALYSIS_LOOPANALYSIS_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include <optional>

namespace mlir {
class AffineExpr;
class AffineMap;
class BlockArgument;
class MemRefType;
class Operation;
class Value;

namespace affine {
class AffineForOp;
class NestedPattern;

/// Returns the trip count of the loop as an affine map with its corresponding
/// operands if the latter is expressible as an affine expression, and nullptr
/// otherwise. This method always succeeds as long as the lower bound is not a
/// multi-result map. The trip count expression is simplified before returning.
/// This method only utilizes map composition to construct lower and upper
/// bounds before computing the trip count expressions
void getTripCountMapAndOperands(AffineForOp forOp, AffineMap *map,
                                SmallVectorImpl<Value> *operands);

/// Returns the trip count of the loop if it's a constant, std::nullopt
/// otherwise. This uses affine expression analysis and is able to determine
/// constant trip count in non-trivial cases.
std::optional<uint64_t> getConstantTripCount(AffineForOp forOp);

/// Returns the greatest known integral divisor of the trip count. Affine
/// expression analysis is used (indirectly through getTripCount), and
/// this method is thus able to determine non-trivial divisors.
uint64_t getLargestDivisorOfTripCount(AffineForOp forOp);

/// Checks if an affine read or write operation depends on `forOp`'s IV, i.e.,
/// if the memory access is invariant on `forOp`.
template <typename LoadOrStoreOp>
bool isInvariantAccess(LoadOrStoreOp memOp, AffineForOp forOp);

/// Given an induction variable `iv` of type AffineForOp and `indices` of type
/// IndexType, returns the set of `indices` that are independent of `iv`.
///
/// Prerequisites (inherited from `isAccessInvariant` above):
///   1. `iv` and `indices` of the proper type;
///   2. at most one affine.apply is reachable from each index in `indices`;
///
/// Emits a note if it encounters a chain of affine.apply and conservatively
///  those cases.
DenseSet<Value, DenseMapInfo<Value>>
getInvariantAccesses(Value iv, ArrayRef<Value> indices);

/// Given:
///   1. an induction variable `iv` of type AffineForOp;
///   2. a `memoryOp` of type const LoadOp& or const StoreOp&;
/// determines whether `memoryOp` has a contiguous access along `iv`. Contiguous
/// is defined as either invariant or varying only along a unique MemRef dim.
/// Upon success, the unique MemRef dim is written in `memRefDim` (or -1 to
/// convey the memRef access is invariant along `iv`).
///
/// Prerequisites:
///   1. `memRefDim` ~= nullptr;
///   2. `iv` of the proper type;
///   3. the MemRef accessed by `memoryOp` has no layout map or at most an
///      identity layout map.
///
/// Currently only supports no layout map or identity layout map in the memref.
/// Returns false if the memref has a non-identity layoutMap. This behavior is
/// conservative.
template <typename LoadOrStoreOp>
bool isContiguousAccess(Value iv, LoadOrStoreOp memoryOp, int *memRefDim);

using VectorizableLoopFun = std::function<bool(AffineForOp)>;

/// Checks whether the loop is structurally vectorizable; i.e.:
///   1. no conditionals are nested under the loop;
///   2. all nested load/stores are to scalar MemRefs.
/// TODO: relax the no-conditionals restriction
bool isVectorizableLoopBody(AffineForOp loop,
                            NestedPattern &vectorTransferMatcher);

/// Checks whether the loop is structurally vectorizable and that all the LoadOp
/// and StoreOp matched have access indexing functions that are either:
///   1. invariant along the loop induction variable created by 'loop';
///   2. varying along at most one memory dimension. If such a unique dimension
///      is found, it is written into `memRefDim`.
bool isVectorizableLoopBody(AffineForOp loop, int *memRefDim,
                            NestedPattern &vectorTransferMatcher);

/// Checks where SSA dominance would be violated if a for op's body
/// operations are shifted by the specified shifts. This method checks if a
/// 'def' and all its uses have the same shift factor.
// TODO: extend this to check for memory-based dependence violation when we have
// the support.
bool isOpwiseShiftValid(AffineForOp forOp, ArrayRef<uint64_t> shifts);

/// Checks whether hyper-rectangular loop tiling of the nest represented by
/// `loops` is valid. The validity condition is from Irigoin and Triolet,
/// which states that two tiles cannot depend on each other. We simplify such
/// condition to just checking whether there is any negative dependence
/// direction, since we have the prior knowledge that the tiling results will be
/// hyper-rectangles, which are scheduled in the lexicographically increasing
/// order on the vector of loop indices. This function will return failure when
/// any dependence component is negative along any of `loops`.
bool isTilingValid(ArrayRef<AffineForOp> loops);

} // namespace affine
} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_ANALYSIS_LOOPANALYSIS_H
