//===- BufferizableOpInterfaceImpl.h - Impl. of BufferizableOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BUFFERIZATION_TRANSFORMS_FUNCBUFFERIZABLEOPINTERFACEIMPL_H
#define MLIR_BUFFERIZATION_TRANSFORMS_FUNCBUFFERIZABLEOPINTERFACEIMPL_H

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
class DialectRegistry;

namespace func {
class FuncOp;
} // namespace func

namespace bufferization {
namespace func_ext {
/// The state of analysis of a FuncOp.
enum class FuncOpAnalysisState { NotAnalyzed, InProgress, Analyzed };

using func::FuncOp;

/// Extra analysis state that is required for bufferization of function
/// boundaries.
struct FuncAnalysisState : public OneShotAnalysisState::Extension {
  FuncAnalysisState(OneShotAnalysisState &state)
      : OneShotAnalysisState::Extension(state) {}

  // Note: Function arguments and/or function return values may disappear during
  // bufferization. Functions and their CallOps are analyzed and bufferized
  // separately. To ensure that a CallOp analysis/bufferization can access an
  // already bufferized function's analysis results, we store bbArg/return value
  // indices instead of BlockArguments/OpOperand pointers.

  /// A set of block argument indices.
  using BbArgIndexSet = DenseSet<int64_t>;

  /// A mapping of indices to indices.
  using IndexMapping = DenseMap<int64_t, int64_t>;

  /// A mapping of indices to a list of indices.
  using IndexToIndexListMapping = DenseMap<int64_t, SmallVector<int64_t>>;

  /// A mapping of ReturnOp OpOperand indices to equivalent FuncOp BBArg
  /// indices.
  DenseMap<FuncOp, IndexMapping> equivalentFuncArgs;

  /// A mapping of FuncOp BBArg indices to aliasing ReturnOp OpOperand indices.
  DenseMap<FuncOp, IndexToIndexListMapping> aliasingReturnVals;

  /// A set of all read BlockArguments of FuncOps.
  DenseMap<FuncOp, BbArgIndexSet> readBbArgs;

  /// A set of all written-to BlockArguments of FuncOps.
  DenseMap<FuncOp, BbArgIndexSet> writtenBbArgs;

  /// Keep track of which FuncOps are fully analyzed or currently being
  /// analyzed.
  DenseMap<FuncOp, FuncOpAnalysisState> analyzedFuncOps;

  /// This function is called right before analyzing the given FuncOp. It
  /// initializes the data structures for the FuncOp in this state object.
  void startFunctionAnalysis(FuncOp funcOp);
};

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace func_ext
} // namespace bufferization
} // namespace mlir

#endif // MLIR_BUFFERIZATION_TRANSFORMS_FUNCBUFFERIZABLEOPINTERFACEIMPL_H
