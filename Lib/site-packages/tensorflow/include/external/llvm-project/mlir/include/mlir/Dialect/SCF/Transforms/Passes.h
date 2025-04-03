//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_SCF_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"

/// Creates a pass that bufferizes the SCF dialect.
std::unique_ptr<Pass> createSCFBufferizePass();

/// Creates a pass that specializes for loop for unrolling and
/// vectorization.
std::unique_ptr<Pass> createForLoopSpecializationPass();

/// Creates a pass that peels for loops at their upper bounds for
/// better vectorization.
std::unique_ptr<Pass> createForLoopPeelingPass();

/// Creates a pass that canonicalizes affine.min and affine.max operations
/// inside of scf.for loops with known lower and upper bounds.
std::unique_ptr<Pass> createSCFForLoopCanonicalizationPass();

/// Creates a pass that transforms a single ParallelLoop over N induction
/// variables into another ParallelLoop over less than N induction variables.
std::unique_ptr<Pass> createTestSCFParallelLoopCollapsingPass();

/// Creates a loop fusion pass which fuses parallel loops.
std::unique_ptr<Pass> createParallelLoopFusionPass();

/// Creates a pass that specializes parallel loop for unrolling and
/// vectorization.
std::unique_ptr<Pass> createParallelLoopSpecializationPass();

/// Creates a pass which tiles innermost parallel loops.
/// If noMinMaxBounds, the upper bound of the inner loop will
/// be a same value among different outter loop iterations, and
/// an additional inbound check will be emitted inside the internal
/// loops.
std::unique_ptr<Pass>
createParallelLoopTilingPass(llvm::ArrayRef<int64_t> tileSize = {},
                             bool noMinMaxBounds = false);

/// Creates a pass which folds arith ops on induction variable into
/// loop range.
std::unique_ptr<Pass> createForLoopRangeFoldingPass();

/// Creates a pass that converts SCF forall loops to SCF for loops.
std::unique_ptr<Pass> createForallToForLoopPass();

/// Creates a pass that converts SCF forall loops to SCF parallel loops.
std::unique_ptr<Pass> createForallToParallelLoopPass();

// Creates a pass which lowers for loops into while loops.
std::unique_ptr<Pass> createForToWhileLoopPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_SCF_TRANSFORMS_PASSES_H_
