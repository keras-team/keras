/* Copyright 2022 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef STABLEHLO_TRANSFORMS_PASSES_H
#define STABLEHLO_TRANSFORMS_PASSES_H

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/Version.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "stablehlo/transforms/Passes.h.inc"

// Populates --stablehlo-canonicalize-dynamism patterns.
void populateStablehloCanonicalizeDynamismPatterns(RewritePatternSet *patterns,
                                                   MLIRContext *context);

// Populates --stablehlo-refine-shapes patterns.
void populateStablehloRefineShapesPatterns(RewritePatternSet *patterns,
                                           MLIRContext *context);

// Populates StableHLO ops to VHLO ops rewriting patterns.
void populateStablehloToVhloPatterns(RewritePatternSet *patterns,
                                     TypeConverter *converter,
                                     MLIRContext *context);

// Populates VHLO ops to StableHLO ops rewriting patterns.
void populateVhloToStablehloPatterns(RewritePatternSet *patterns,
                                     TypeConverter *converter,
                                     MLIRContext *context);

// Populates VHLO downgrade rewriting patterns.
void populateVhloToVersionPatterns(RewritePatternSet *patterns,
                                   TypeConverter *converter,
                                   MLIRContext *contexts);

/// Collection of rewrite patterns for lowering of CHLO ops to StableHLO and
/// Shape ops.
void populateChloToStablehloPatterns(MLIRContext *context,
                                     RewritePatternSet *patterns);

/// Collection of folding patterns for StableHLO.
void populateStablehloAggressiveFolderPatterns(RewritePatternSet *patterns,
                                               MLIRContext *context,
                                               bool foldFloat);

/// Collection of rewrite patterns for lowering quantized StableHLO operations
/// using uniform dequantize/quantize operations.
void populateStablehloLegalizeQuantizedOpToQDQPatterns(
    RewritePatternSet *patterns, MLIRContext *context,
    PatternBenefit benefit = 1);

/// Collection of rewrite patterns for composing quantized StableHLO operations
/// using unform dequantize/quantize operations.
void populateStablehloLegalizeQDQToQuantizedOpPatterns(
    RewritePatternSet *patterns, MLIRContext *context);

/// A subset of folding patterns for StableHLO that is necessary for shape
/// refinement.
void populateStablehloShapeFolderPatterns(RewritePatternSet *patterns,
                                          MLIRContext *context,
                                          bool foldFloat = false);

/// Collection of canonicalization patterns for StableHLO.
void populateStablehloCanonicalizationPatterns(MLIRContext *context,
                                               RewritePatternSet *patterns,
                                               PatternBenefit benefit = 1);

/// Collection of patterns to upgrade deprecated ops to long-term supported ops.
void populateStablehloLegalizeDeprecatedOpsPatterns(
    MLIRContext *context, RewritePatternSet *patterns);

/// Collection of shape dialect to StableHLO patterns.
void populateShapeToStablehloPatterns(MLIRContext *context,
                                      RewritePatternSet *patterns);

/// Collection of patterns to create compatibility expander for StableHLO
/// operations.
void populateStablehloCreateCompatibilityExpanderPatterns(
    RewritePatternSet *patterns, MLIRContext *context,
    vhlo::Version targetVersion);

//// Additional pass constructors ////

std::unique_ptr<OperationPass<ModuleOp>> createStablehloRefineArgumentsPass(
    TypeRange refinedTypes);

//// Pass pipelines ////

// StableHLO consumers can add this pipeline to convert portable artifacts to
// StableHLO programs. This pipeline will silently pass if programs are not
// portable artifacts.
//
// Uses vhlo-to-version and vhlo-legalize-to-stablehlo passes. Does not require
// an option to specify VHLO target version since it always converts VHLO to
// the current version in order to legalize to StableHLO.
void createStablehloDeserializePipeline(OpPassManager &pm);

// Creates a pipeline of StableHLO-specific MLIR passes to remove dynamism from
// the program. This is achieved via refining the "main" function's arguments
// and propagating new shapes throughout the program argument types and shapes
// within an MLIR module. The main function is either a function with name
// "main", if there are multiple functions, or the single function within the
// module.
//
// This pipeline focuses on:
//   1. Refining function argument types based on provided `refinedTypes`.
//   2. Refining shape information of operations within functions.
//   3. Replaces dynamic StableHLO ops with the corresponding static
//   counterparts if applicable.
void createStablehloRemoveDynamismPipeline(OpPassManager &pm,
                                           TypeRange refinedTypes);

// Decomposes quantized operations within a StableHLO module by
// applying a series of MLIR passes essentially breaking down the quantized
// operations into a primitive math operations.
void createStablehloLowerQuantPipeline(OpPassManager &pm);

// Adds `stablehlo-deserialize` pipeline as a registered pass pipeline
// for opt tools.
void registerPassPipelines();

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_TRANSFORMS_PASSES_H
