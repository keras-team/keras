//===- LocalAliasAnalysis.h - Local Stateless Alias Analysis ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of a local stateless alias analysis.
// This analysis walks from the values being compared to determine their
// potential for aliasing.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_ALIASANALYSIS_LOCALALIASANALYSIS_H_
#define MLIR_ANALYSIS_ALIASANALYSIS_LOCALALIASANALYSIS_H_

#include "mlir/Analysis/AliasAnalysis.h"

namespace mlir {
/// This class implements a local form of alias analysis that tries to identify
/// the underlying values addressed by each value and performs a few basic
/// checks to see if they alias.
class LocalAliasAnalysis {
public:
  virtual ~LocalAliasAnalysis() = default;

  /// Given two values, return their aliasing behavior.
  AliasResult alias(Value lhs, Value rhs);

  /// Return the modify-reference behavior of `op` on `location`.
  ModRefResult getModRef(Operation *op, Value location);

protected:
  /// Given the two values, return their aliasing behavior.
  virtual AliasResult aliasImpl(Value lhs, Value rhs);
};
} // namespace mlir

#endif // MLIR_ANALYSIS_ALIASANALYSIS_LOCALALIASANALYSIS_H_
