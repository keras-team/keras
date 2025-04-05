//===- CFGToSCF.h - Control Flow Graph to Structured Control Flow *- C++ -*===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines a generic `transformCFGToSCF` function that can be
// used to lift any dialect operations implementing control flow graph
// operations to any dialect implementing structured control flow operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_CFGTOSCF_H
#define MLIR_TRANSFORMS_CFGTOSCF_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"

namespace mlir {

/// Interface that should be implemented by any caller of `transformCFGToSCF`.
/// The transformation requires the caller to 1) create switch-like control
/// flow operations for intermediate transformations and 2) to create
/// the desired structured control flow ops.
class CFGToSCFInterface {
public:
  virtual ~CFGToSCFInterface() = default;

  /// Creates a structured control flow operation branching to one of `regions`.
  /// It replaces `controlFlowCondOp` and must have `resultTypes` as results.
  /// `regions` contains the list of branch regions corresponding to each
  /// successor of `controlFlowCondOp`. Their bodies must simply be taken and
  /// left as is.
  /// Returns failure if incapable of converting the control flow graph
  /// operation.
  virtual FailureOr<Operation *> createStructuredBranchRegionOp(
      OpBuilder &builder, Operation *controlFlowCondOp, TypeRange resultTypes,
      MutableArrayRef<Region> regions) = 0;

  /// Creates a return-like terminator for a branch region of the op returned
  /// by `createStructuredBranchRegionOp`. `branchRegionOp` is the operation
  /// returned by `createStructuredBranchRegionOp`.
  /// `replacedControlFlowOp` is the control flow op being replaced by the
  /// terminator or nullptr if the terminator is not replacing any existing
  /// control flow op. `results` are the values that should be returned by the
  /// branch region.
  virtual LogicalResult createStructuredBranchRegionTerminatorOp(
      Location loc, OpBuilder &builder, Operation *branchRegionOp,
      Operation *replacedControlFlowOp, ValueRange results) = 0;

  /// Creates a structured control flow operation representing a do-while loop.
  /// The do-while loop is expected to have the exact same result types as the
  /// types of the iteration values.
  /// `loopBody` is the body of the loop. The implementation of this
  /// function must create a suitable terminator op at the end of the last block
  /// in `loopBody` which continues the loop if `condition` is 1 and exits the
  /// loop if 0. `loopValuesNextIter` are the values that have to be passed as
  /// the iteration values for the next iteration if continuing, or the result
  /// of the loop if exiting.
  /// `condition` is guaranteed to be of the same type as values returned by
  /// `getCFGSwitchValue` with either 0 or 1 as value.
  ///
  /// `loopValuesInit` are the values used to initialize the iteration
  /// values of the loop.
  /// Returns failure if incapable of creating a loop op.
  virtual FailureOr<Operation *> createStructuredDoWhileLoopOp(
      OpBuilder &builder, Operation *replacedOp, ValueRange loopValuesInit,
      Value condition, ValueRange loopValuesNextIter, Region &&loopBody) = 0;

  /// Creates a constant operation with a result representing `value` that is
  /// suitable as flag for `createCFGSwitchOp`.
  virtual Value getCFGSwitchValue(Location loc, OpBuilder &builder,
                                  unsigned value) = 0;

  /// Creates a switch CFG branch operation branching to one of
  /// `caseDestinations` or `defaultDest`. This is used by the transformation
  /// for intermediate transformations before lifting to structured control
  /// flow. The switch op branches based on `flag` which is guaranteed to be of
  /// the same type as values returned by `getCFGSwitchValue`. The insertion
  /// block of the builder is guaranteed to have its predecessors already set
  /// to create an equivalent CFG after this operation.
  /// Note: `caseValues` and other related ranges may be empty to represent an
  /// unconditional branch.
  virtual void createCFGSwitchOp(Location loc, OpBuilder &builder, Value flag,
                                 ArrayRef<unsigned> caseValues,
                                 BlockRange caseDestinations,
                                 ArrayRef<ValueRange> caseArguments,
                                 Block *defaultDest,
                                 ValueRange defaultArgs) = 0;

  /// Creates a constant operation returning an undefined instance of `type`.
  /// This is required by the transformation as the lifting process might create
  /// control-flow paths where an SSA-value is undefined.
  virtual Value getUndefValue(Location loc, OpBuilder &builder, Type type) = 0;

  /// Creates a return-like terminator indicating unreachable.
  /// This is required when the transformation encounters a statically known
  /// infinite loop. Since structured control flow ops are not terminators,
  /// after lifting an infinite loop, a terminator has to be placed after to
  /// possibly satisfy the terminator requirement of the region originally
  /// passed to `transformCFGToSCF`.
  ///
  /// `region` is guaranteed to be the region originally passed to
  /// `transformCFGToSCF` and the op is guaranteed to always be an op in a block
  /// directly nested under `region` after the transformation.
  ///
  /// Returns failure if incapable of creating an unreachable terminator.
  virtual FailureOr<Operation *>
  createUnreachableTerminator(Location loc, OpBuilder &builder,
                              Region &region) = 0;

  /// Helper function to create an unconditional branch using
  /// `createCFGSwitchOp`.
  void createSingleDestinationBranch(Location loc, OpBuilder &builder,
                                     Value dummyFlag, Block *destination,
                                     ValueRange arguments) {
    createCFGSwitchOp(loc, builder, dummyFlag, {}, {}, {}, destination,
                      arguments);
  }

  /// Helper function to create a conditional branch using
  /// `createCFGSwitchOp`.
  void createConditionalBranch(Location loc, OpBuilder &builder,
                               Value condition, Block *trueDest,
                               ValueRange trueArgs, Block *falseDest,
                               ValueRange falseArgs) {
    createCFGSwitchOp(loc, builder, condition, {0}, {falseDest}, {falseArgs},
                      trueDest, trueArgs);
  }
};

/// Transformation lifting any dialect implementing control flow graph
/// operations to a dialect implementing structured control flow operations.
/// `region` is the region that should be transformed.
/// The implementation of `interface` is responsible for the conversion of the
/// control flow operations to the structured control flow operations.
///
/// If the region contains only a single kind of return-like operation, all
/// control flow graph operations will be converted successfully.
/// Otherwise a single control flow graph operation branching to one block
/// per return-like operation kind remains.
///
/// The transformation currently requires that all control flow graph operations
/// have no side effects, implement the BranchOpInterface and does not have any
/// operation produced successor operands.
/// Returns failure if any of the preconditions are violated or if any of the
/// methods of `interface` failed. The IR is left in an unspecified state.
///
/// Otherwise, returns true or false if any changes to the IR have been made.
FailureOr<bool> transformCFGToSCF(Region &region, CFGToSCFInterface &interface,
                                  DominanceInfo &dominanceInfo);

} // namespace mlir

#endif // MLIR_TRANSFORMS_CFGTOSCF_H
