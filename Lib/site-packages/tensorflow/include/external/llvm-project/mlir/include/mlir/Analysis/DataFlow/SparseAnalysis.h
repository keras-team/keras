//===- SparseAnalysis.h - Sparse data-flow analysis -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements sparse data-flow analysis using the data-flow analysis
// framework. The analysis is forward and conditional and uses the results of
// dead code analysis to prune dead code during the analysis.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DATAFLOW_SPARSEANALYSIS_H
#define MLIR_ANALYSIS_DATAFLOW_SPARSEANALYSIS_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace dataflow {

//===----------------------------------------------------------------------===//
// AbstractSparseLattice
//===----------------------------------------------------------------------===//

/// This class represents an abstract lattice. A lattice contains information
/// about an SSA value and is what's propagated across the IR by sparse
/// data-flow analysis.
class AbstractSparseLattice : public AnalysisState {
public:
  /// Lattices can only be created for values.
  AbstractSparseLattice(Value value) : AnalysisState(value) {}

  /// Return the value this lattice is located at.
  Value getAnchor() const { return AnalysisState::getAnchor().get<Value>(); }

  /// Join the information contained in 'rhs' into this lattice. Returns
  /// if the value of the lattice changed.
  virtual ChangeResult join(const AbstractSparseLattice &rhs) {
    return ChangeResult::NoChange;
  }

  /// Meet (intersect) the information in this lattice with 'rhs'. Returns
  /// if the value of the lattice changed.
  virtual ChangeResult meet(const AbstractSparseLattice &rhs) {
    return ChangeResult::NoChange;
  }

  /// When the lattice gets updated, propagate an update to users of the value
  /// using its use-def chain to subscribed analyses.
  void onUpdate(DataFlowSolver *solver) const override;

  /// Subscribe an analysis to updates of the lattice. When the lattice changes,
  /// subscribed analyses are re-invoked on all users of the value. This is
  /// more efficient than relying on the dependency map.
  void useDefSubscribe(DataFlowAnalysis *analysis) {
    useDefSubscribers.insert(analysis);
  }

private:
  /// A set of analyses that should be updated when this lattice changes.
  SetVector<DataFlowAnalysis *, SmallVector<DataFlowAnalysis *, 4>,
            SmallPtrSet<DataFlowAnalysis *, 4>>
      useDefSubscribers;
};

//===----------------------------------------------------------------------===//
// Lattice
//===----------------------------------------------------------------------===//

/// This class represents a lattice holding a specific value of type `ValueT`.
/// Lattice values (`ValueT`) are required to adhere to the following:
///
///   * static ValueT join(const ValueT &lhs, const ValueT &rhs);
///     - This method conservatively joins the information held by `lhs`
///       and `rhs` into a new value. This method is required to be monotonic.
///   * bool operator==(const ValueT &rhs) const;
///
template <typename ValueT>
class Lattice : public AbstractSparseLattice {
public:
  using AbstractSparseLattice::AbstractSparseLattice;

  /// Return the value this lattice is located at.
  Value getAnchor() const { return anchor.get<Value>(); }

  /// Return the value held by this lattice. This requires that the value is
  /// initialized.
  ValueT &getValue() { return value; }
  const ValueT &getValue() const {
    return const_cast<Lattice<ValueT> *>(this)->getValue();
  }

  using LatticeT = Lattice<ValueT>;

  /// Join the information contained in the 'rhs' lattice into this
  /// lattice. Returns if the state of the current lattice changed.
  ChangeResult join(const AbstractSparseLattice &rhs) override {
    return join(static_cast<const LatticeT &>(rhs).getValue());
  }

  /// Meet (intersect) the information contained in the 'rhs' lattice with
  /// this lattice. Returns if the state of the current lattice changed.
  ChangeResult meet(const AbstractSparseLattice &rhs) override {
    return meet(static_cast<const LatticeT &>(rhs).getValue());
  }

  /// Join the information contained in the 'rhs' value into this
  /// lattice. Returns if the state of the current lattice changed.
  ChangeResult join(const ValueT &rhs) {
    // Otherwise, join rhs with the current optimistic value.
    ValueT newValue = ValueT::join(value, rhs);
    assert(ValueT::join(newValue, value) == newValue &&
           "expected `join` to be monotonic");
    assert(ValueT::join(newValue, rhs) == newValue &&
           "expected `join` to be monotonic");

    // Update the current optimistic value if something changed.
    if (newValue == value)
      return ChangeResult::NoChange;

    value = newValue;
    return ChangeResult::Change;
  }

  /// Trait to check if `T` provides a `meet` method. Needed since for forward
  /// analysis, lattices will only have a `join`, no `meet`, but we want to use
  /// the same `Lattice` class for both directions.
  template <typename T, typename... Args>
  using has_meet = decltype(&T::meet);
  template <typename T>
  using lattice_has_meet = llvm::is_detected<has_meet, T>;

  /// Meet (intersect) the information contained in the 'rhs' value with this
  /// lattice. Returns if the state of the current lattice changed.  If the
  /// lattice elements don't have a `meet` method, this is a no-op (see below.)
  template <typename VT,
            std::enable_if_t<lattice_has_meet<VT>::value> * = nullptr>
  ChangeResult meet(const VT &rhs) {
    ValueT newValue = ValueT::meet(value, rhs);
    assert(ValueT::meet(newValue, value) == newValue &&
           "expected `meet` to be monotonic");
    assert(ValueT::meet(newValue, rhs) == newValue &&
           "expected `meet` to be monotonic");

    // Update the current optimistic value if something changed.
    if (newValue == value)
      return ChangeResult::NoChange;

    value = newValue;
    return ChangeResult::Change;
  }

  template <typename VT,
            std::enable_if_t<!lattice_has_meet<VT>::value> * = nullptr>
  ChangeResult meet(const VT &rhs) {
    return ChangeResult::NoChange;
  }

  /// Print the lattice element.
  void print(raw_ostream &os) const override { value.print(os); }

private:
  /// The currently computed value that is optimistically assumed to be true.
  ValueT value;
};

//===----------------------------------------------------------------------===//
// AbstractSparseForwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// Base class for sparse forward data-flow analyses. A sparse analysis
/// implements a transfer function on operations from the lattices of the
/// operands to the lattices of the results. This analysis will propagate
/// lattices across control-flow edges and the callgraph using liveness
/// information.
class AbstractSparseForwardDataFlowAnalysis : public DataFlowAnalysis {
public:
  /// Initialize the analysis by visiting every owner of an SSA value: all
  /// operations and blocks.
  LogicalResult initialize(Operation *top) override;

  /// Visit a program point. If this is a block and all control-flow
  /// predecessors or callsites are known, then the arguments lattices are
  /// propagated from them. If this is a call operation or an operation with
  /// region control-flow, then its result lattices are set accordingly.
  /// Otherwise, the operation transfer function is invoked.
  LogicalResult visit(ProgramPoint point) override;

protected:
  explicit AbstractSparseForwardDataFlowAnalysis(DataFlowSolver &solver);

  /// The operation transfer function. Given the operand lattices, this
  /// function is expected to set the result lattices.
  virtual LogicalResult
  visitOperationImpl(Operation *op,
                     ArrayRef<const AbstractSparseLattice *> operandLattices,
                     ArrayRef<AbstractSparseLattice *> resultLattices) = 0;

  /// The transfer function for calls to external functions.
  virtual void visitExternalCallImpl(
      CallOpInterface call,
      ArrayRef<const AbstractSparseLattice *> argumentLattices,
      ArrayRef<AbstractSparseLattice *> resultLattices) = 0;

  /// Given an operation with region control-flow, the lattices of the operands,
  /// and a region successor, compute the lattice values for block arguments
  /// that are not accounted for by the branching control flow (ex. the bounds
  /// of loops).
  virtual void visitNonControlFlowArgumentsImpl(
      Operation *op, const RegionSuccessor &successor,
      ArrayRef<AbstractSparseLattice *> argLattices, unsigned firstIndex) = 0;

  /// Get the lattice element of a value.
  virtual AbstractSparseLattice *getLatticeElement(Value value) = 0;

  /// Get a read-only lattice element for a value and add it as a dependency to
  /// a program point.
  const AbstractSparseLattice *getLatticeElementFor(ProgramPoint point,
                                                    Value value);

  /// Set the given lattice element(s) at control flow entry point(s).
  virtual void setToEntryState(AbstractSparseLattice *lattice) = 0;
  void setAllToEntryStates(ArrayRef<AbstractSparseLattice *> lattices);

  /// Join the lattice element and propagate and update if it changed.
  void join(AbstractSparseLattice *lhs, const AbstractSparseLattice &rhs);

private:
  /// Recursively initialize the analysis on nested operations and blocks.
  LogicalResult initializeRecursively(Operation *op);

  /// Visit an operation. If this is a call operation or an operation with
  /// region control-flow, then its result lattices are set accordingly.
  /// Otherwise, the operation transfer function is invoked.
  LogicalResult visitOperation(Operation *op);

  /// Visit a block to compute the lattice values of its arguments. If this is
  /// an entry block, then the argument values are determined from the block's
  /// "predecessors" as set by `PredecessorState`. The predecessors can be
  /// region terminators or callable callsites. Otherwise, the values are
  /// determined from block predecessors.
  void visitBlock(Block *block);

  /// Visit a program point `point` with predecessors within a region branch
  /// operation `branch`, which can either be the entry block of one of the
  /// regions or the parent operation itself, and set either the argument or
  /// parent result lattices.
  void visitRegionSuccessors(ProgramPoint point, RegionBranchOpInterface branch,
                             RegionBranchPoint successor,
                             ArrayRef<AbstractSparseLattice *> lattices);
};

//===----------------------------------------------------------------------===//
// SparseForwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// A sparse forward data-flow analysis for propagating SSA value lattices
/// across the IR by implementing transfer functions for operations.
///
/// `StateT` is expected to be a subclass of `AbstractSparseLattice`.
template <typename StateT>
class SparseForwardDataFlowAnalysis
    : public AbstractSparseForwardDataFlowAnalysis {
  static_assert(
      std::is_base_of<AbstractSparseLattice, StateT>::value,
      "analysis state class expected to subclass AbstractSparseLattice");

public:
  explicit SparseForwardDataFlowAnalysis(DataFlowSolver &solver)
      : AbstractSparseForwardDataFlowAnalysis(solver) {}

  /// Visit an operation with the lattices of its operands. This function is
  /// expected to set the lattices of the operation's results.
  virtual LogicalResult visitOperation(Operation *op,
                                       ArrayRef<const StateT *> operands,
                                       ArrayRef<StateT *> results) = 0;

  /// Visit a call operation to an externally defined function given the
  /// lattices of its arguments.
  virtual void visitExternalCall(CallOpInterface call,
                                 ArrayRef<const StateT *> argumentLattices,
                                 ArrayRef<StateT *> resultLattices) {
    setAllToEntryStates(resultLattices);
  }

  /// Given an operation with possible region control-flow, the lattices of the
  /// operands, and a region successor, compute the lattice values for block
  /// arguments that are not accounted for by the branching control flow (ex.
  /// the bounds of loops). By default, this method marks all such lattice
  /// elements as having reached a pessimistic fixpoint. `firstIndex` is the
  /// index of the first element of `argLattices` that is set by control-flow.
  virtual void visitNonControlFlowArguments(Operation *op,
                                            const RegionSuccessor &successor,
                                            ArrayRef<StateT *> argLattices,
                                            unsigned firstIndex) {
    setAllToEntryStates(argLattices.take_front(firstIndex));
    setAllToEntryStates(argLattices.drop_front(
        firstIndex + successor.getSuccessorInputs().size()));
  }

protected:
  /// Get the lattice element for a value.
  StateT *getLatticeElement(Value value) override {
    return getOrCreate<StateT>(value);
  }

  /// Get the lattice element for a value and create a dependency on the
  /// provided program point.
  const StateT *getLatticeElementFor(ProgramPoint point, Value value) {
    return static_cast<const StateT *>(
        AbstractSparseForwardDataFlowAnalysis::getLatticeElementFor(point,
                                                                    value));
  }

  /// Set the given lattice element(s) at control flow entry point(s).
  virtual void setToEntryState(StateT *lattice) = 0;
  void setAllToEntryStates(ArrayRef<StateT *> lattices) {
    AbstractSparseForwardDataFlowAnalysis::setAllToEntryStates(
        {reinterpret_cast<AbstractSparseLattice *const *>(lattices.begin()),
         lattices.size()});
  }

private:
  /// Type-erased wrappers that convert the abstract lattice operands to derived
  /// lattices and invoke the virtual hooks operating on the derived lattices.
  LogicalResult visitOperationImpl(
      Operation *op, ArrayRef<const AbstractSparseLattice *> operandLattices,
      ArrayRef<AbstractSparseLattice *> resultLattices) override {
    return visitOperation(
        op,
        {reinterpret_cast<const StateT *const *>(operandLattices.begin()),
         operandLattices.size()},
        {reinterpret_cast<StateT *const *>(resultLattices.begin()),
         resultLattices.size()});
  }
  void visitExternalCallImpl(
      CallOpInterface call,
      ArrayRef<const AbstractSparseLattice *> argumentLattices,
      ArrayRef<AbstractSparseLattice *> resultLattices) override {
    visitExternalCall(
        call,
        {reinterpret_cast<const StateT *const *>(argumentLattices.begin()),
         argumentLattices.size()},
        {reinterpret_cast<StateT *const *>(resultLattices.begin()),
         resultLattices.size()});
  }
  void visitNonControlFlowArgumentsImpl(
      Operation *op, const RegionSuccessor &successor,
      ArrayRef<AbstractSparseLattice *> argLattices,
      unsigned firstIndex) override {
    visitNonControlFlowArguments(
        op, successor,
        {reinterpret_cast<StateT *const *>(argLattices.begin()),
         argLattices.size()},
        firstIndex);
  }
  void setToEntryState(AbstractSparseLattice *lattice) override {
    return setToEntryState(reinterpret_cast<StateT *>(lattice));
  }
};

//===----------------------------------------------------------------------===//
// AbstractSparseBackwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// Base class for sparse backward data-flow analyses. Similar to
/// AbstractSparseForwardDataFlowAnalysis, but walks bottom to top.
class AbstractSparseBackwardDataFlowAnalysis : public DataFlowAnalysis {
public:
  /// Initialize the analysis by visiting the operation and everything nested
  /// under it.
  LogicalResult initialize(Operation *top) override;

  /// Visit a program point. If this is a call operation or an operation with
  /// block or region control-flow, then operand lattices are set accordingly.
  /// Otherwise, invokes the operation transfer function (`visitOperationImpl`).
  LogicalResult visit(ProgramPoint point) override;

protected:
  explicit AbstractSparseBackwardDataFlowAnalysis(
      DataFlowSolver &solver, SymbolTableCollection &symbolTable);

  /// The operation transfer function. Given the result lattices, this
  /// function is expected to set the operand lattices.
  virtual LogicalResult visitOperationImpl(
      Operation *op, ArrayRef<AbstractSparseLattice *> operandLattices,
      ArrayRef<const AbstractSparseLattice *> resultLattices) = 0;

  /// The transfer function for calls to external functions.
  virtual void visitExternalCallImpl(
      CallOpInterface call, ArrayRef<AbstractSparseLattice *> operandLattices,
      ArrayRef<const AbstractSparseLattice *> resultLattices) = 0;

  // Visit operands on branch instructions that are not forwarded.
  virtual void visitBranchOperand(OpOperand &operand) = 0;

  // Visit operands on call instructions that are not forwarded.
  virtual void visitCallOperand(OpOperand &operand) = 0;

  /// Set the given lattice element(s) at control flow exit point(s).
  virtual void setToExitState(AbstractSparseLattice *lattice) = 0;

  /// Set the given lattice element(s) at control flow exit point(s).
  void setAllToExitStates(ArrayRef<AbstractSparseLattice *> lattices);

  /// Get the lattice element for a value.
  virtual AbstractSparseLattice *getLatticeElement(Value value) = 0;

  /// Get the lattice elements for a range of values.
  SmallVector<AbstractSparseLattice *> getLatticeElements(ValueRange values);

  /// Join the lattice element and propagate and update if it changed.
  void meet(AbstractSparseLattice *lhs, const AbstractSparseLattice &rhs);

private:
  /// Recursively initialize the analysis on nested operations and blocks.
  LogicalResult initializeRecursively(Operation *op);

  /// Visit an operation. If this is a call operation or an operation with
  /// region control-flow, then its operand lattices are set accordingly.
  /// Otherwise, the operation transfer function is invoked.
  LogicalResult visitOperation(Operation *op);

  /// Visit a block.
  void visitBlock(Block *block);

  /// Visit an op with regions (like e.g. `scf.while`)
  void visitRegionSuccessors(RegionBranchOpInterface branch,
                             ArrayRef<AbstractSparseLattice *> operands);

  /// Visit a `RegionBranchTerminatorOpInterface` to compute the lattice values
  /// of its operands, given its parent op `branch`. The lattice value of an
  /// operand is determined based on the corresponding arguments in
  /// `terminator`'s region successor(s).
  void visitRegionSuccessorsFromTerminator(
      RegionBranchTerminatorOpInterface terminator,
      RegionBranchOpInterface branch);

  /// Get the lattice element for a value, and also set up
  /// dependencies so that the analysis on the given ProgramPoint is re-invoked
  /// if the value changes.
  const AbstractSparseLattice *getLatticeElementFor(ProgramPoint point,
                                                    Value value);

  /// Get the lattice elements for a range of values, and also set up
  /// dependencies so that the analysis on the given ProgramPoint is re-invoked
  /// if any of the values change.
  SmallVector<const AbstractSparseLattice *>
  getLatticeElementsFor(ProgramPoint point, ValueRange values);

  SymbolTableCollection &symbolTable;
};

//===----------------------------------------------------------------------===//
// SparseBackwardDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// A sparse (backward) data-flow analysis for propagating SSA value lattices
/// backwards across the IR by implementing transfer functions for operations.
///
/// `StateT` is expected to be a subclass of `AbstractSparseLattice`.
template <typename StateT>
class SparseBackwardDataFlowAnalysis
    : public AbstractSparseBackwardDataFlowAnalysis {
public:
  explicit SparseBackwardDataFlowAnalysis(DataFlowSolver &solver,
                                          SymbolTableCollection &symbolTable)
      : AbstractSparseBackwardDataFlowAnalysis(solver, symbolTable) {}

  /// Visit an operation with the lattices of its results. This function is
  /// expected to set the lattices of the operation's operands.
  virtual LogicalResult visitOperation(Operation *op,
                                       ArrayRef<StateT *> operands,
                                       ArrayRef<const StateT *> results) = 0;

  /// Visit a call to an external function. This function is expected to set
  /// lattice values of the call operands. By default, calls `visitCallOperand`
  /// for all operands.
  virtual void visitExternalCall(CallOpInterface call,
                                 ArrayRef<StateT *> argumentLattices,
                                 ArrayRef<const StateT *> resultLattices) {
    (void)argumentLattices;
    (void)resultLattices;
    for (OpOperand &operand : call->getOpOperands()) {
      visitCallOperand(operand);
    }
  };

protected:
  /// Get the lattice element for a value.
  StateT *getLatticeElement(Value value) override {
    return getOrCreate<StateT>(value);
  }

  /// Set the given lattice element(s) at control flow exit point(s).
  virtual void setToExitState(StateT *lattice) = 0;
  void setToExitState(AbstractSparseLattice *lattice) override {
    return setToExitState(reinterpret_cast<StateT *>(lattice));
  }
  void setAllToExitStates(ArrayRef<StateT *> lattices) {
    AbstractSparseBackwardDataFlowAnalysis::setAllToExitStates(
        {reinterpret_cast<AbstractSparseLattice *const *>(lattices.begin()),
         lattices.size()});
  }

private:
  /// Type-erased wrappers that convert the abstract lattice operands to derived
  /// lattices and invoke the virtual hooks operating on the derived lattices.
  LogicalResult visitOperationImpl(
      Operation *op, ArrayRef<AbstractSparseLattice *> operandLattices,
      ArrayRef<const AbstractSparseLattice *> resultLattices) override {
    return visitOperation(
        op,
        {reinterpret_cast<StateT *const *>(operandLattices.begin()),
         operandLattices.size()},
        {reinterpret_cast<const StateT *const *>(resultLattices.begin()),
         resultLattices.size()});
  }

  void visitExternalCallImpl(
      CallOpInterface call, ArrayRef<AbstractSparseLattice *> operandLattices,
      ArrayRef<const AbstractSparseLattice *> resultLattices) override {
    visitExternalCall(
        call,
        {reinterpret_cast<StateT *const *>(operandLattices.begin()),
         operandLattices.size()},
        {reinterpret_cast<const StateT *const *>(resultLattices.begin()),
         resultLattices.size()});
  }
};

} // end namespace dataflow
} // end namespace mlir

#endif // MLIR_ANALYSIS_DATAFLOW_SPARSEANALYSIS_H
