//===- OneShotModuleBufferize.h - Bufferization across Func. Boundaries ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ONESHOTMODULEBUFFERIZE_H
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ONESHOTMODULEBUFFERIZE_H

namespace llvm {
struct LogicalResult;
} // namespace llvm

namespace mlir {
class ModuleOp;

namespace bufferization {
struct BufferizationStatistics;
class OneShotAnalysisState;
struct OneShotBufferizationOptions;

/// Analyze `moduleOp` and its nested ops. Bufferization decisions are stored in
/// `state`.
llvm::LogicalResult
analyzeModuleOp(ModuleOp moduleOp, OneShotAnalysisState &state,
                BufferizationStatistics *statistics = nullptr);

/// Bufferize `op` and its nested ops that implement `BufferizableOpInterface`.
///
/// Note: This function does not run One-Shot Analysis. No buffer copies are
/// inserted except two cases:
/// - `options.copyBeforeWrite` is set, in which case buffers are copied before
///   every write.
/// - `options.copyBeforeWrite` is not set and `options.noAnalysisFuncFilter`
///   is not empty. The FuncOps it contains were not analyzed. Buffer copies
///   will be inserted only to these FuncOps.
llvm::LogicalResult
bufferizeModuleOp(ModuleOp moduleOp, const OneShotBufferizationOptions &options,
                  BufferizationStatistics *statistics = nullptr);

/// Remove bufferization attributes on every FuncOp arguments in the ModuleOp.
void removeBufferizationAttributesInModule(ModuleOp moduleOp);

/// Run One-Shot Module Bufferization on the given module. Performs a simple
/// function call analysis to determine which function arguments are
/// inplaceable. Then analyzes and bufferizes FuncOps one-by-one with One-Shot
/// Bufferize.
llvm::LogicalResult runOneShotModuleBufferize(
    ModuleOp moduleOp,
    const bufferization::OneShotBufferizationOptions &options,
    BufferizationStatistics *statistics = nullptr);

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ONESHOTMODULEBUFFERIZE_H
