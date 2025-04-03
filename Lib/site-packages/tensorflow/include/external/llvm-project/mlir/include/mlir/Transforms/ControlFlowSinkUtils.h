//===- ControlFlowSinkUtils.h - ControlFlow Sink Utils ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_CONTROLFLOWSINKUTILS_H
#define MLIR_TRANSFORMS_CONTROLFLOWSINKUTILS_H

#include "mlir/Support/LLVM.h"

namespace mlir {

class DominanceInfo;
class Operation;
class Region;
class RegionBranchOpInterface;
class RegionRange;

/// Given a list of regions, perform control flow sinking on them. For each
/// region, control-flow sinking moves operations that dominate the region but
/// whose only users are in the region into the regions so that they aren't
/// executed on paths where their results are not needed.
///
/// TODO: For the moment, this is a *simple* control-flow sink, i.e., no
/// duplicating of ops. It should be made to accept a cost model to determine
/// whether duplicating a particular op is profitable.
///
/// Example:
///
/// ```mlir
/// %0 = arith.addi %arg0, %arg1
/// scf.if %cond {
///   scf.yield %0
/// } else {
///   scf.yield %arg2
/// }
/// ```
///
/// After control-flow sink:
///
/// ```mlir
/// scf.if %cond {
///   %0 = arith.addi %arg0, %arg1
///   scf.yield %0
/// } else {
///   scf.yield %arg2
/// }
/// ```
///
/// Users must supply a callback `shouldMoveIntoRegion` that determines whether
/// the given operation that only has users in the given operation should be
/// moved into that region. If this returns true, `moveIntoRegion` is called on
/// the same operation and region.
///
/// `moveIntoRegion` must move the operation into the region such that dominance
/// of the operation is preserved; for example, by moving the operation to the
/// start of the entry block. This ensures the preservation of SSA dominance of
/// the operation's results.
///
/// Returns the number of operations sunk.
size_t
controlFlowSink(RegionRange regions, DominanceInfo &domInfo,
                function_ref<bool(Operation *, Region *)> shouldMoveIntoRegion,
                function_ref<void(Operation *, Region *)> moveIntoRegion);

/// Populates `regions` with regions of the provided region branch op that are
/// executed at most once at that are reachable given the current operands of
/// the op. These regions can be passed to `controlFlowSink` to perform sinking
/// on the regions of the operation.
void getSinglyExecutedRegionsToSink(RegionBranchOpInterface branch,
                                    SmallVectorImpl<Region *> &regions);

} // namespace mlir

#endif // MLIR_TRANSFORMS_CONTROLFLOWSINKUTILS_H
