//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors in the loop
// transformation library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_PASSES_H
#define MLIR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/ViewOpGraph.h"
#include "llvm/Support/Debug.h"
#include <limits>
#include <memory>

namespace mlir {

class GreedyRewriteConfig;

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL_CANONICALIZER
#define GEN_PASS_DECL_CONTROLFLOWSINK
#define GEN_PASS_DECL_CSEPASS
#define GEN_PASS_DECL_INLINER
#define GEN_PASS_DECL_LOOPINVARIANTCODEMOTION
#define GEN_PASS_DECL_MEM2REG
#define GEN_PASS_DECL_PRINTIRPASS
#define GEN_PASS_DECL_PRINTOPSTATS
#define GEN_PASS_DECL_SROA
#define GEN_PASS_DECL_STRIPDEBUGINFO
#define GEN_PASS_DECL_SCCP
#define GEN_PASS_DECL_SYMBOLDCE
#define GEN_PASS_DECL_SYMBOLPRIVATIZE
#define GEN_PASS_DECL_TOPOLOGICALSORT
#define GEN_PASS_DECL_COMPOSITEFIXEDPOINTPASS
#include "mlir/Transforms/Passes.h.inc"

/// Creates an instance of the Canonicalizer pass, configured with default
/// settings (which can be overridden by pass options on the command line).
std::unique_ptr<Pass> createCanonicalizerPass();

/// Creates an instance of the Canonicalizer pass with the specified config.
/// `disabledPatterns` is a set of labels used to filter out input patterns with
/// a debug label or debug name in this set. `enabledPatterns` is a set of
/// labels used to filter out input patterns that do not have one of the labels
/// in this set. Debug labels must be set explicitly on patterns or when adding
/// them with `RewritePatternSet::addWithLabel`. Debug names may be empty, but
/// patterns created with `RewritePattern::create` have their default debug name
/// set to their type name.
std::unique_ptr<Pass>
createCanonicalizerPass(const GreedyRewriteConfig &config,
                        ArrayRef<std::string> disabledPatterns = std::nullopt,
                        ArrayRef<std::string> enabledPatterns = std::nullopt);

/// Creates a pass to perform control-flow sinking.
std::unique_ptr<Pass> createControlFlowSinkPass();

/// Creates a pass to perform common sub expression elimination.
std::unique_ptr<Pass> createCSEPass();

/// Creates a pass to print IR on the debug stream.
std::unique_ptr<Pass> createPrintIRPass(const PrintIRPassOptions & = {});

/// Creates a pass that generates IR to verify ops at runtime.
std::unique_ptr<Pass> createGenerateRuntimeVerificationPass();

/// Creates a loop invariant code motion pass that hoists loop invariant
/// instructions out of the loop.
std::unique_ptr<Pass> createLoopInvariantCodeMotionPass();

/// Creates a pass that hoists loop-invariant subset ops.
std::unique_ptr<Pass> createLoopInvariantSubsetHoistingPass();

/// Creates a pass to strip debug information from a function.
std::unique_ptr<Pass> createStripDebugInfoPass();

/// Creates a pass which prints the list of ops and the number of occurrences in
/// the module.
std::unique_ptr<Pass> createPrintOpStatsPass(raw_ostream &os = llvm::errs());

/// Creates a pass which prints the list of ops and the number of occurrences in
/// the module with the output format option.
std::unique_ptr<Pass> createPrintOpStatsPass(raw_ostream &os, bool printAsJSON);

/// Creates a pass which inlines calls and callable operations as defined by
/// the CallGraph.
std::unique_ptr<Pass> createInlinerPass();
/// Creates an instance of the inliner pass, and use the provided pass managers
/// when optimizing callable operations with names matching the key type.
/// Callable operations with a name not within the provided map will use the
/// default inliner pipeline during optimization.
std::unique_ptr<Pass>
createInlinerPass(llvm::StringMap<OpPassManager> opPipelines);
/// Creates an instance of the inliner pass, and use the provided pass managers
/// when optimizing callable operations with names matching the key type.
/// Callable operations with a name not within the provided map will use the
/// provided default pipeline builder.
std::unique_ptr<Pass>
createInlinerPass(llvm::StringMap<OpPassManager> opPipelines,
                  std::function<void(OpPassManager &)> defaultPipelineBuilder);

/// Creates an optimization pass to remove dead values.
std::unique_ptr<Pass> createRemoveDeadValuesPass();

/// Creates a pass which performs sparse conditional constant propagation over
/// nested operations.
std::unique_ptr<Pass> createSCCPPass();

/// Creates a pass which delete symbol operations that are unreachable. This
/// pass may *only* be scheduled on an operation that defines a SymbolTable.
std::unique_ptr<Pass> createSymbolDCEPass();

/// Creates a pass which marks top-level symbol operations as `private` unless
/// listed in `excludeSymbols`.
std::unique_ptr<Pass>
createSymbolPrivatizePass(ArrayRef<std::string> excludeSymbols = {});

/// Creates a pass that recursively sorts nested regions without SSA dominance
/// topologically such that, as much as possible, users of values appear after
/// their producers.
std::unique_ptr<Pass> createTopologicalSortPass();

/// Create composite pass, which runs provided set of passes until fixed point
/// or maximum number of iterations reached.
std::unique_ptr<Pass> createCompositeFixedPointPass(
    std::string name, llvm::function_ref<void(OpPassManager &)> populateFunc,
    int maxIterations = 10);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Transforms/Passes.h.inc"

} // namespace mlir

#endif // MLIR_TRANSFORMS_PASSES_H
