//===- DenseAnalysis.h - Dense data-flow analysis -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements dense data-flow analysis using the data-flow analysis
// framework. The analysis is forward and conditional and uses the results of
// dead code analysis to prune dead code during the analysis.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DENSEDATAFLOWANALYSIS_H
#define MLIR_ANALYSIS_DENSEDATAFLOWANALYSIS_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

namespace mlir {
namespace dataflow {

//===----------------------------------------------------------------------===//
// CallControlFlowAction
//===----------------------------------------------------------------------===//

/// Indicates whether the control enters, exits, or skips over the callee (in
/// the case of external functions).
enum class CallControlFlowAction { EnterCallee, ExitCallee, ExternalCallee };

//===----------------------------------------------------------------------===//
// AbstractDenseLattice
//===----------------------------------------------------------------------===//

/// This class represents a dense lattice. A dense lattice is attached to
/// operations to represent the program state after their execution or to blocks
/// to represent the program state at the beginning of the block. A dense
/// lattice is propagated through the IR by dense data-flow analysis.
class AbstractDenseLattice : public AnalysisState {
public:
  /// A dense lattice can only be created for operations and blocks.
  using AnalysisState::AnalysisState;

  /// Join the lattice across control-flow or callgraph edges.
  virtual ChangeResult join(const AbstractDenseLattice &rhs) {
    return ChangeResult::NoChange;
  }

  virtual ChangeResult meet(const AbstractDenseLattice &rhs) {
    return ChangeResult::NoChange;
  }
};

//===----------------------------------------------------------------------===//
// AbstractDenseForwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// Base class for dense forward data-flow analyses. Dense data-flow analysis
/// attaches a lattice between the execution of operations and implements a
/// transfer function from the lattice before each operation to the lattice
/// after. The lattice contains information about the state of the program at
/// that point.
///
/// In this implementation, a lattice attached to an operation represents the
/// state of the program after its execution, and a lattice attached to block
/// represents the state of the program right before it starts executing its
/// body.
class AbstractDenseForwardDataFlowAnalysis : public DataFlowAnalysis {
public:
  using DataFlowAnalysis::DataFlowAnalysis;

  /// Initialize the analysis by visiting every program point whose execution
  /// may modify the program state; that is, every operation and block.
  LogicalResult initialize(Operation *top) override;

  /// Visit a program point that modifies the state of the program. If this is a
  /// block, then the state is propagated from control-flow predecessors or
  /// callsites. If this is a call operation or region control-flow operation,
  /// then the state after the execution of the operation is set by control-flow
  /// or the callgraph. Otherwise, this function invokes the operation transfer
  /// function.
  LogicalResult visit(ProgramPoint point) override;

protected:
  /// Propagate the dense lattice before the execution of an operation to the
  /// lattice after its execution.
  virtual LogicalResult visitOperationImpl(Operation *op,
                                           const AbstractDenseLattice &before,
                                           AbstractDenseLattice *after) = 0;

  /// Get the dense lattice after the execution of the given lattice anchor.
  virtual AbstractDenseLattice *getLattice(LatticeAnchor anchor) = 0;

  /// Get the dense lattice after the execution of the given program point and
  /// add it as a dependency to a lattice anchor. That is, every time the
  /// lattice after anchor is updated, the dependent program point must be
  /// visited, and the newly triggered visit might update the lattice after
  /// dependent.
  const AbstractDenseLattice *getLatticeFor(ProgramPoint dependent,
                                            LatticeAnchor anchor);

  /// Set the dense lattice at control flow entry point and propagate an update
  /// if it changed.
  virtual void setToEntryState(AbstractDenseLattice *lattice) = 0;

  /// Join a lattice with another and propagate an update if it changed.
  void join(AbstractDenseLattice *lhs, const AbstractDenseLattice &rhs) {
    propagateIfChanged(lhs, lhs->join(rhs));
  }

  /// Visit an operation. If this is a call operation or region control-flow
  /// operation, then the state after the execution of the operation is set by
  /// control-flow or the callgraph. Otherwise, this function invokes the
  /// operation transfer function.
  virtual LogicalResult processOperation(Operation *op);

  /// Propagate the dense lattice forward along the control flow edge from
  /// `regionFrom` to `regionTo` regions of the `branch` operation. `nullopt`
  /// values correspond to control flow branches originating at or targeting the
  /// `branch` operation itself. Default implementation just joins the states,
  /// meaning that operations implementing `RegionBranchOpInterface` don't have
  /// any effect on the lattice that isn't already expressed by the interface
  /// itself.
  virtual void visitRegionBranchControlFlowTransfer(
      RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
      std::optional<unsigned> regionTo, const AbstractDenseLattice &before,
      AbstractDenseLattice *after) {
    join(after, before);
  }

  /// Propagate the dense lattice forward along the call control flow edge,
  /// which can be either entering or exiting the callee. Default implementation
  /// for enter and exit callee actions just meets the states, meaning that
  /// operations implementing `CallOpInterface` don't have any effect on the
  /// lattice that isn't already expressed by the interface itself. Default
  /// implementation for the external callee action additionally sets the
  /// "after" lattice to the entry state.
  virtual void visitCallControlFlowTransfer(CallOpInterface call,
                                            CallControlFlowAction action,
                                            const AbstractDenseLattice &before,
                                            AbstractDenseLattice *after) {
    join(after, before);
    // Note that `setToEntryState` may be a "partial fixpoint" for some
    // lattices, e.g., lattices that are lists of maps of other lattices will
    // only set fixpoint for "known" lattices.
    if (action == CallControlFlowAction::ExternalCallee)
      setToEntryState(after);
  }

  /// Visit a program point within a region branch operation with predecessors
  /// in it. This can either be an entry block of one of the regions of the
  /// parent operation itself.
  void visitRegionBranchOperation(ProgramPoint point,
                                  RegionBranchOpInterface branch,
                                  AbstractDenseLattice *after);

private:
  /// Visit a block. The state at the start of the block is propagated from
  /// control-flow predecessors or callsites.
  void visitBlock(Block *block);

  /// Visit an operation for which the data flow is described by the
  /// `CallOpInterface`.
  void visitCallOperation(CallOpInterface call,
                          const AbstractDenseLattice &before,
                          AbstractDenseLattice *after);
};

//===----------------------------------------------------------------------===//
// DenseForwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// A dense forward data-flow analysis for propagating lattices before and
/// after the execution of every operation across the IR by implementing
/// transfer functions for operations.
///
/// `LatticeT` is expected to be a subclass of `AbstractDenseLattice`.
template <typename LatticeT>
class DenseForwardDataFlowAnalysis
    : public AbstractDenseForwardDataFlowAnalysis {
  static_assert(
      std::is_base_of<AbstractDenseLattice, LatticeT>::value,
      "analysis state class expected to subclass AbstractDenseLattice");

public:
  using AbstractDenseForwardDataFlowAnalysis::
      AbstractDenseForwardDataFlowAnalysis;

  /// Visit an operation with the dense lattice before its execution. This
  /// function is expected to set the dense lattice after its execution and
  /// trigger change propagation in case of change.
  virtual LogicalResult visitOperation(Operation *op, const LatticeT &before,
                                       LatticeT *after) = 0;

  /// Hook for customizing the behavior of lattice propagation along the call
  /// control flow edges. Two types of (forward) propagation are possible here:
  ///   - `action == CallControlFlowAction::Enter` indicates that:
  ///     - `before` is the state before the call operation;
  ///     - `after` is the state at the beginning of the callee entry block;
  ///   - `action == CallControlFlowAction::Exit` indicates that:
  ///     - `before` is the state at the end of a callee exit block;
  ///     - `after` is the state after the call operation.
  /// By default, the `after` state is simply joined with the `before` state.
  /// Concrete analyses can override this behavior or delegate to the parent
  /// call for the default behavior. Specifically, if the `call` op may affect
  /// the lattice prior to entering the callee, the custom behavior can be added
  /// for `action == CallControlFlowAction::Enter`. If the `call` op may affect
  /// the lattice post exiting the callee, the custom behavior can be added for
  /// `action == CallControlFlowAction::Exit`.
  virtual void visitCallControlFlowTransfer(CallOpInterface call,
                                            CallControlFlowAction action,
                                            const LatticeT &before,
                                            LatticeT *after) {
    AbstractDenseForwardDataFlowAnalysis::visitCallControlFlowTransfer(
        call, action, before, after);
  }

  /// Hook for customizing the behavior of lattice propagation along the control
  /// flow edges between regions and their parent op. The control flows from
  /// `regionFrom` to `regionTo`, both of which may be `nullopt` to indicate the
  /// parent op. The lattice is propagated forward along this edge. The lattices
  /// are as follows:
  ///   - `before:`
  ///     - if `regionFrom` is a region, this is the lattice at the end of the
  ///       block that exits the region; note that for multi-exit regions, the
  ///       lattices are equal at the end of all exiting blocks, but they are
  ///       associated with different program points.
  ///     - otherwise, this is the lattice before the parent op.
  ///   - `after`:
  ///     - if `regionTo` is a region, this is the lattice at the beginning of
  ///       the entry block of that region;
  ///     - otherwise, this is the lattice after the parent op.
  /// By default, the `after` state is simply joined with the `before` state.
  /// Concrete analyses can override this behavior or delegate to the parent
  /// call for the default behavior. Specifically, if the `branch` op may affect
  /// the lattice before entering any region, the custom behavior can be added
  /// for `regionFrom == nullopt`. If the `branch` op may affect the lattice
  /// after all terminated, the custom behavior can be added for `regionTo ==
  /// nullptr`. The behavior can be further refined for specific pairs of "from"
  /// and "to" regions.
  virtual void visitRegionBranchControlFlowTransfer(
      RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
      std::optional<unsigned> regionTo, const LatticeT &before,
      LatticeT *after) {
    AbstractDenseForwardDataFlowAnalysis::visitRegionBranchControlFlowTransfer(
        branch, regionFrom, regionTo, before, after);
  }

protected:
  /// Get the dense lattice on this lattice anchor.
  LatticeT *getLattice(LatticeAnchor anchor) override {
    return getOrCreate<LatticeT>(anchor);
  }

  /// Set the dense lattice at control flow entry point and propagate an update
  /// if it changed.
  virtual void setToEntryState(LatticeT *lattice) = 0;
  void setToEntryState(AbstractDenseLattice *lattice) override {
    setToEntryState(static_cast<LatticeT *>(lattice));
  }

  /// Type-erased wrappers that convert the abstract dense lattice to a derived
  /// lattice and invoke the virtual hooks operating on the derived lattice.
  LogicalResult visitOperationImpl(Operation *op,
                                   const AbstractDenseLattice &before,
                                   AbstractDenseLattice *after) final {
    return visitOperation(op, static_cast<const LatticeT &>(before),
                          static_cast<LatticeT *>(after));
  }
  void visitCallControlFlowTransfer(CallOpInterface call,
                                    CallControlFlowAction action,
                                    const AbstractDenseLattice &before,
                                    AbstractDenseLattice *after) final {
    visitCallControlFlowTransfer(call, action,
                                 static_cast<const LatticeT &>(before),
                                 static_cast<LatticeT *>(after));
  }
  void visitRegionBranchControlFlowTransfer(RegionBranchOpInterface branch,
                                            std::optional<unsigned> regionFrom,
                                            std::optional<unsigned> regionTo,
                                            const AbstractDenseLattice &before,
                                            AbstractDenseLattice *after) final {
    visitRegionBranchControlFlowTransfer(branch, regionFrom, regionTo,
                                         static_cast<const LatticeT &>(before),
                                         static_cast<LatticeT *>(after));
  }
};

//===----------------------------------------------------------------------===//
// AbstractDenseBackwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// Base class for dense backward dataflow analyses. Such analyses attach a
/// lattice between the execution of operations and implement a transfer
/// function from the lattice after the operation ot the lattice before it, thus
/// propagating backward.
///
/// In this implementation, a lattice attached to an operation represents the
/// state of the program before its execution, and a lattice attached to a block
/// represents the state of the program before the end of the block, i.e., after
/// its terminator.
class AbstractDenseBackwardDataFlowAnalysis : public DataFlowAnalysis {
public:
  /// Construct the analysis in the given solver. Takes a symbol table
  /// collection that is used to cache symbol resolution in interprocedural part
  /// of the analysis. The symbol table need not be prefilled.
  AbstractDenseBackwardDataFlowAnalysis(DataFlowSolver &solver,
                                        SymbolTableCollection &symbolTable)
      : DataFlowAnalysis(solver), symbolTable(symbolTable) {}

  /// Initialize the analysis by visiting every program point whose execution
  /// may modify the program state; that is, every operation and block.
  LogicalResult initialize(Operation *top) override;

  /// Visit a program point that modifies the state of the program. The state is
  /// propagated along control flow directions for branch-, region- and
  /// call-based control flow using the respective interfaces. For other
  /// operations, the state is propagated using the transfer function
  /// (visitOperation).
  ///
  /// Note: the transfer function is currently *not* invoked for operations with
  /// region or call interface, but *is* invoked for block terminators.
  LogicalResult visit(ProgramPoint point) override;

protected:
  /// Propagate the dense lattice after the execution of an operation to the
  /// lattice before its execution.
  virtual LogicalResult visitOperationImpl(Operation *op,
                                           const AbstractDenseLattice &after,
                                           AbstractDenseLattice *before) = 0;

  /// Get the dense lattice before the execution of the lattice anchor. That is,
  /// before the execution of the given operation or after the execution of the
  /// block.
  virtual AbstractDenseLattice *getLattice(LatticeAnchor anchor) = 0;

  /// Get the dense lattice before the execution of the program point in
  /// `anchor` and declare that the `dependent` program point must be updated
  /// every time `point` is.
  const AbstractDenseLattice *getLatticeFor(ProgramPoint dependent,
                                            LatticeAnchor anchor);

  /// Set the dense lattice before at the control flow exit point and propagate
  /// the update if it changed.
  virtual void setToExitState(AbstractDenseLattice *lattice) = 0;

  /// Meet a lattice with another lattice and propagate an update if it changed.
  void meet(AbstractDenseLattice *lhs, const AbstractDenseLattice &rhs) {
    propagateIfChanged(lhs, lhs->meet(rhs));
  }

  /// Visit an operation. Dispatches to specialized methods for call or region
  /// control-flow operations. Otherwise, this function invokes the operation
  /// transfer function.
  virtual LogicalResult processOperation(Operation *op);

  /// Propagate the dense lattice backwards along the control flow edge from
  /// `regionFrom` to `regionTo` regions of the `branch` operation. `nullopt`
  /// values correspond to control flow branches originating at or targeting the
  /// `branch` operation itself. Default implementation just meets the states,
  /// meaning that operations implementing `RegionBranchOpInterface` don't have
  /// any effect on the lattice that isn't already expressed by the interface
  /// itself.
  virtual void visitRegionBranchControlFlowTransfer(
      RegionBranchOpInterface branch, RegionBranchPoint regionFrom,
      RegionBranchPoint regionTo, const AbstractDenseLattice &after,
      AbstractDenseLattice *before) {
    meet(before, after);
  }

  /// Propagate the dense lattice backwards along the call control flow edge,
  /// which can be either entering or exiting the callee. Default implementation
  /// for enter and exit callee action just meets the states, meaning that
  /// operations implementing `CallOpInterface` don't have any effect on the
  /// lattice that isn't already expressed by the interface itself. Default
  /// implementation for external callee action additional sets the result to
  /// the exit (fixpoint) state.
  virtual void visitCallControlFlowTransfer(CallOpInterface call,
                                            CallControlFlowAction action,
                                            const AbstractDenseLattice &after,
                                            AbstractDenseLattice *before) {
    meet(before, after);

    // Note that `setToExitState` may be a "partial fixpoint" for some lattices,
    // e.g., lattices that are lists of maps of other lattices will only
    // set fixpoint for "known" lattices.
    if (action == CallControlFlowAction::ExternalCallee)
      setToExitState(before);
  }

private:
  /// Visit a block. The state and the end of the block is propagated from
  /// control-flow successors of the block or callsites.
  void visitBlock(Block *block);

  /// Visit a program point within a region branch operation with successors
  /// (from which the state is propagated) in or after it. `regionNo` indicates
  /// the region that contains the successor, `nullopt` indicating the successor
  /// of the branch operation itself.
  void visitRegionBranchOperation(ProgramPoint point,
                                  RegionBranchOpInterface branch,
                                  RegionBranchPoint branchPoint,
                                  AbstractDenseLattice *before);

  /// Visit an operation for which the data flow is described by the
  /// `CallOpInterface`. Performs inter-procedural data flow as follows:
  ///
  ///   - find the callable (resolve via the symbol table),
  ///   - get the entry block of the callable region,
  ///   - take the state before the first operation if present or at block end
  ///     otherwise,
  ///   - meet that state with the state before the call-like op, or use the
  ///     custom logic if overridden by concrete analyses.
  void visitCallOperation(CallOpInterface call,
                          const AbstractDenseLattice &after,
                          AbstractDenseLattice *before);

  /// Symbol table for call-level control flow.
  SymbolTableCollection &symbolTable;
};

//===----------------------------------------------------------------------===//
// DenseBackwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// A dense backward dataflow analysis propagating lattices after and before the
/// execution of every operation across the IR by implementing transfer
/// functions for opreations.
///
/// `LatticeT` is expected to be a subclass of `AbstractDenseLattice`.
template <typename LatticeT>
class DenseBackwardDataFlowAnalysis
    : public AbstractDenseBackwardDataFlowAnalysis {
  static_assert(std::is_base_of_v<AbstractDenseLattice, LatticeT>,
                "analysis state expected to subclass AbstractDenseLattice");

public:
  using AbstractDenseBackwardDataFlowAnalysis::
      AbstractDenseBackwardDataFlowAnalysis;

  /// Transfer function. Visits an operation with the dense lattice after its
  /// execution. This function is expected to set the dense lattice before its
  /// execution and trigger propagation in case of change.
  virtual LogicalResult visitOperation(Operation *op, const LatticeT &after,
                                       LatticeT *before) = 0;

  /// Hook for customizing the behavior of lattice propagation along the call
  /// control flow edges. Two types of (back) propagation are possible here:
  ///   - `action == CallControlFlowAction::Enter` indicates that:
  ///     - `after` is the state at the top of the callee entry block;
  ///     - `before` is the state before the call operation;
  ///   - `action == CallControlFlowAction::Exit` indicates that:
  ///     - `after` is the state after the call operation;
  ///     - `before` is the state of exit blocks of the callee.
  /// By default, the `before` state is simply met with the `after` state.
  /// Concrete analyses can override this behavior or delegate to the parent
  /// call for the default behavior. Specifically, if the `call` op may affect
  /// the lattice prior to entering the callee, the custom behavior can be added
  /// for `action == CallControlFlowAction::Enter`. If the `call` op may affect
  /// the lattice post exiting the callee, the custom behavior can be added for
  /// `action == CallControlFlowAction::Exit`.
  virtual void visitCallControlFlowTransfer(CallOpInterface call,
                                            CallControlFlowAction action,
                                            const LatticeT &after,
                                            LatticeT *before) {
    AbstractDenseBackwardDataFlowAnalysis::visitCallControlFlowTransfer(
        call, action, after, before);
  }

  /// Hook for customizing the behavior of lattice propagation along the control
  /// flow edges between regions and their parent op. The control flows from
  /// `regionFrom` to `regionTo`, both of which may be `nullopt` to indicate the
  /// parent op. The lattice is propagated back along this edge. The lattices
  /// are as follows:
  ///   - `after`:
  ///     - if `regionTo` is a region, this is the lattice at the beginning of
  ///       the entry block of that region;
  ///     - otherwise, this is the lattice after the parent op.
  ///   - `before:`
  ///     - if `regionFrom` is a region, this is the lattice at the end of the
  ///       block that exits the region; note that for multi-exit regions, the
  ///       lattices are equal at the end of all exiting blocks, but they are
  ///       associated with different program points.
  ///     - otherwise, this is the lattice before the parent op.
  /// By default, the `before` state is simply met with the `after` state.
  /// Concrete analyses can override this behavior or delegate to the parent
  /// call for the default behavior. Specifically, if the `branch` op may affect
  /// the lattice before entering any region, the custom behavior can be added
  /// for `regionFrom == nullopt`. If the `branch` op may affect the lattice
  /// after all terminated, the custom behavior can be added for `regionTo ==
  /// nullptr`. The behavior can be further refined for specific pairs of "from"
  /// and "to" regions.
  virtual void visitRegionBranchControlFlowTransfer(
      RegionBranchOpInterface branch, RegionBranchPoint regionFrom,
      RegionBranchPoint regionTo, const LatticeT &after, LatticeT *before) {
    AbstractDenseBackwardDataFlowAnalysis::visitRegionBranchControlFlowTransfer(
        branch, regionFrom, regionTo, after, before);
  }

protected:
  /// Get the dense lattice at the given lattice anchor.
  LatticeT *getLattice(LatticeAnchor anchor) override {
    return getOrCreate<LatticeT>(anchor);
  }

  /// Set the dense lattice at control flow exit point (after the terminator)
  /// and propagate an update if it changed.
  virtual void setToExitState(LatticeT *lattice) = 0;
  void setToExitState(AbstractDenseLattice *lattice) final {
    setToExitState(static_cast<LatticeT *>(lattice));
  }

  /// Type-erased wrappers that convert the abstract dense lattice to a derived
  /// lattice and invoke the virtual hooks operating on the derived lattice.
  LogicalResult visitOperationImpl(Operation *op,
                                   const AbstractDenseLattice &after,
                                   AbstractDenseLattice *before) final {
    return visitOperation(op, static_cast<const LatticeT &>(after),
                          static_cast<LatticeT *>(before));
  }
  void visitCallControlFlowTransfer(CallOpInterface call,
                                    CallControlFlowAction action,
                                    const AbstractDenseLattice &after,
                                    AbstractDenseLattice *before) final {
    visitCallControlFlowTransfer(call, action,
                                 static_cast<const LatticeT &>(after),
                                 static_cast<LatticeT *>(before));
  }
  void visitRegionBranchControlFlowTransfer(
      RegionBranchOpInterface branch, RegionBranchPoint regionForm,
      RegionBranchPoint regionTo, const AbstractDenseLattice &after,
      AbstractDenseLattice *before) final {
    visitRegionBranchControlFlowTransfer(branch, regionForm, regionTo,
                                         static_cast<const LatticeT &>(after),
                                         static_cast<LatticeT *>(before));
  }
};

} // end namespace dataflow
} // end namespace mlir

#endif // MLIR_ANALYSIS_DENSEDATAFLOWANALYSIS_H
