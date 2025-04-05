//===- SymbolTableAnalysis.h - Analysis for cached symbol tables --*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_SYMBOLTABLEANALYSIS_H
#define MLIR_ANALYSIS_SYMBOLTABLEANALYSIS_H

#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/AnalysisManager.h"

namespace mlir {
/// This is a simple analysis that contains a symbol table collection and, for
/// simplicity, a reference to the top-level symbol table. This allows symbol
/// tables to be preserved across passes. Most often, symbol tables are
/// automatically kept up-to-date via the `insert` and `erase` functions.
class SymbolTableAnalysis {
public:
  /// Create the symbol table analysis at the provided top-level operation and
  /// instantiate the symbol table of the top-level operation.
  SymbolTableAnalysis(Operation *op)
      : topLevelSymbolTable(symbolTables.getSymbolTable(op)) {}

  /// Get the symbol table collection.
  SymbolTableCollection &getSymbolTables() { return symbolTables; }

  /// Get the top-level symbol table.
  SymbolTable &getTopLevelSymbolTable() { return topLevelSymbolTable; }

  /// Get the top-level operation.
  template <typename OpT>
  OpT getTopLevelOp() {
    return cast<OpT>(topLevelSymbolTable.getOp());
  }

  /// Symbol tables are kept up-to-date by passes. Assume that the analysis
  /// remains valid.
  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa) {
    return false;
  }

private:
  /// The symbol table collection containing cached symbol tables for all nested
  /// symbol table operations.
  SymbolTableCollection symbolTables;
  /// The symbol table of the top-level operation.
  SymbolTable &topLevelSymbolTable;
};
} // namespace mlir

#endif // MLIR_ANALYSIS_SYMBOLTABLEANALYSIS_H
