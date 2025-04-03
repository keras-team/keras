//===- ConstantPropagationAnalysis.h - Constant propagation analysis ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements constant propagation analysis. In this file are defined
// the lattice value class that represents constant values in the program and
// a sparse constant propagation analysis that uses operation folders to
// speculate about constant values in the program.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DATAFLOW_CONSTANTPROPAGATIONANALYSIS_H
#define MLIR_ANALYSIS_DATAFLOW_CONSTANTPROPAGATIONANALYSIS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include <optional>

namespace mlir {
namespace dataflow {

//===----------------------------------------------------------------------===//
// ConstantValue
//===----------------------------------------------------------------------===//

/// This lattice value represents a known constant value of a lattice.
class ConstantValue {
public:
  /// Construct a constant value as uninitialized.
  explicit ConstantValue() = default;

  /// Construct a constant value with a known constant.
  explicit ConstantValue(Attribute constant, Dialect *dialect)
      : constant(constant), dialect(dialect) {}

  /// Get the constant value. Returns null if no value was determined.
  Attribute getConstantValue() const {
    assert(!isUninitialized());
    return *constant;
  }

  /// Get the dialect instance that can be used to materialize the constant.
  Dialect *getConstantDialect() const {
    assert(!isUninitialized());
    return dialect;
  }

  /// Compare the constant values.
  bool operator==(const ConstantValue &rhs) const {
    return constant == rhs.constant;
  }

  /// Print the constant value.
  void print(raw_ostream &os) const;

  /// The state where the constant value is uninitialized. This happens when the
  /// state hasn't been set during the analysis.
  static ConstantValue getUninitialized() { return ConstantValue{}; }

  /// Whether the state is uninitialized.
  bool isUninitialized() const { return !constant.has_value(); }

  /// The state where the constant value is unknown.
  static ConstantValue getUnknownConstant() {
    return ConstantValue{/*constant=*/nullptr, /*dialect=*/nullptr};
  }

  /// The union with another constant value is null if they are different, and
  /// the same if they are the same.
  static ConstantValue join(const ConstantValue &lhs,
                            const ConstantValue &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    if (lhs == rhs)
      return lhs;
    return getUnknownConstant();
  }

private:
  /// The constant value.
  std::optional<Attribute> constant;
  /// A dialect instance that can be used to materialize the constant.
  Dialect *dialect = nullptr;
};

//===----------------------------------------------------------------------===//
// SparseConstantPropagation
//===----------------------------------------------------------------------===//

/// This analysis implements sparse constant propagation, which attempts to
/// determine constant-valued results for operations using constant-valued
/// operands, by speculatively folding operations. When combined with dead-code
/// analysis, this becomes sparse conditional constant propagation (SCCP).
class SparseConstantPropagation
    : public SparseForwardDataFlowAnalysis<Lattice<ConstantValue>> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const Lattice<ConstantValue> *> operands,
                 ArrayRef<Lattice<ConstantValue> *> results) override;

  void setToEntryState(Lattice<ConstantValue> *lattice) override;
};

} // end namespace dataflow
} // end namespace mlir

#endif // MLIR_ANALYSIS_DATAFLOW_CONSTANTPROPAGATIONANALYSIS_H
