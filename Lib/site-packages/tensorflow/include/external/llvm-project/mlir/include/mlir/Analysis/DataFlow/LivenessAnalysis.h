//===- LivenessAnalysis.h - Liveness analysis -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements liveness analysis using the sparse backward data-flow
// analysis framework. Theoretically, liveness analysis assigns liveness to each
// (value, program point) pair in the program and it is thus a dense analysis.
// However, since values are immutable in MLIR, a sparse analysis, which will
// assign liveness to each value in the program, suffices here.
//
// Liveness analysis has many applications. It can be used to avoid the
// computation of extraneous operations that have no effect on the memory or the
// final output of a program. It can also be used to optimize register
// allocation. Both of these applications help achieve one very important goal:
// reducing runtime.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DATAFLOW_LIVENESSANALYSIS_H
#define MLIR_ANALYSIS_DATAFLOW_LIVENESSANALYSIS_H

#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <optional>

namespace mlir::dataflow {

//===----------------------------------------------------------------------===//
// Liveness
//===----------------------------------------------------------------------===//

/// This lattice represents, for a given value, whether or not it is "live".
///
/// A value is considered "live" iff it:
///   (1) has memory effects OR
///   (2) is returned by a public function OR
///   (3) is used to compute a value of type (1) or (2).
/// It is also to be noted that a value could be of multiple types (1/2/3) at
/// the same time.
///
/// A value "has memory effects" iff it:
///   (1.a) is an operand of an op with memory effects OR
///   (1.b) is a non-forwarded branch operand and its branch op could take the
///   control to a block that has an op with memory effects OR
///   (1.c) is a non-forwarded call operand.
///
/// A value `A` is said to be "used to compute" value `B` iff `B` cannot be
/// computed in the absence of `A`. Thus, in this implementation, we say that
/// value `A` is used to compute value `B` iff:
///   (3.a) `B` is a result of an op with operand `A` OR
///   (3.b) `A` is used to compute some value `C` and `C` is used to compute
///   `B`.
struct Liveness : public AbstractSparseLattice {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Liveness)
  using AbstractSparseLattice::AbstractSparseLattice;

  void print(raw_ostream &os) const override;

  ChangeResult markLive();

  ChangeResult meet(const AbstractSparseLattice &other) override;

  // At the beginning of the analysis, everything is marked "not live" and as
  // the analysis progresses, values are marked "live" if they are found to be
  // live.
  bool isLive = false;
};

//===----------------------------------------------------------------------===//
// LivenessAnalysis
//===----------------------------------------------------------------------===//

/// An analysis that, by going backwards along the dataflow graph, annotates
/// each value with a boolean storing true iff it is "live".
class LivenessAnalysis : public SparseBackwardDataFlowAnalysis<Liveness> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op, ArrayRef<Liveness *> operands,
                               ArrayRef<const Liveness *> results) override;

  void visitBranchOperand(OpOperand &operand) override;

  void visitCallOperand(OpOperand &operand) override;

  void setToExitState(Liveness *lattice) override;
};

//===----------------------------------------------------------------------===//
// RunLivenessAnalysis
//===----------------------------------------------------------------------===//

/// Runs liveness analysis on the IR defined by `op`.
struct RunLivenessAnalysis {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RunLivenessAnalysis)

  RunLivenessAnalysis(Operation *op);

  const Liveness *getLiveness(Value val);

private:
  /// Stores the result of the liveness analysis that was run.
  DataFlowSolver solver;
};

} // end namespace mlir::dataflow

#endif // MLIR_ANALYSIS_DATAFLOW_LIVENESSANALYSIS_H
