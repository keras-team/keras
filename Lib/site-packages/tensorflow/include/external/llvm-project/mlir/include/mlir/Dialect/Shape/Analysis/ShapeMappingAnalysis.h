//===- ShapeMappingAnalysis.h - Preserve shape Info  ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SHAPE_ANALYSIS_SHAPEMAPPINGANALYSIS_H_
#define MLIR_DIALECT_SHAPE_ANALYSIS_SHAPEMAPPINGANALYSIS_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

namespace shape {

/// ShapeMappingValue works as the value of ShapeMappingAnalysis table, where
/// `funcSymbol` is the symbol of mapping function, and `inputs` are the actual
/// parameters for the function.
struct ShapeMappingValue {
  ShapeMappingValue() = default;
  ShapeMappingValue(FlatSymbolRefAttr symbol, llvm::SmallVector<Value> &&inps)
      : funcSymbol(symbol), inputs(inps) {}

  FlatSymbolRefAttr funcSymbol;
  llvm::SmallVector<Value> inputs;
};

/// ShapeMappingAnalysis is used together with OutlineShapeComputationPass to
/// preserve Value and corresponding shape function / arguments mapping
/// information
struct ShapeMappingAnalysis {
  ShapeMappingAnalysis(Operation *op) : operation(op) { (void)operation; }

  /// Dumps the shape mapping information to the given stream.
  void print(raw_ostream &os) const {
    os << "// ---- Shape Mapping Information -----\n";
    for (const auto &it : shapeMapping) {
      const ShapeMappingValue &mappingValue = it.second;
      os << "// Shape for " << it.first << " :: " << mappingValue.funcSymbol;
      llvm::interleaveComma(mappingValue.inputs, os << "(");
      os << ")\n";
    }
  }

  llvm::DenseMap<Value, ShapeMappingValue> shapeMapping;

private:
  Operation *operation;
};

} // namespace shape
} // namespace mlir

#endif // MLIR_DIALECT_SHAPE_ANALYSIS_SHAPEMAPPINGANALYSIS_H_
