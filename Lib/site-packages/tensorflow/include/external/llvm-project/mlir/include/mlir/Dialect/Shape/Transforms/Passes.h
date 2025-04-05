//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors in the
// shape transformation library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SHAPE_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_SHAPE_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
class ConversionTarget;
class ModuleOp;
class TypeConverter;
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace mlir {

#define GEN_PASS_DECL
#include "mlir/Dialect/Shape/Transforms/Passes.h.inc"

/// Creates an instance of the ShapeToShapeLowering pass that legalizes Shape
/// dialect to be convertible to Arith. For example, `shape.num_elements` get
/// transformed to `shape.reduce`, which can be lowered to SCF and Arith.
std::unique_ptr<Pass> createShapeToShapeLowering();

/// Collects a set of patterns to rewrite ops within the Shape dialect.
void populateShapeRewritePatterns(RewritePatternSet &patterns);

// Collects a set of patterns to replace all constraints with passing witnesses.
// This is intended to then allow all ShapeConstraint related ops and data to
// have no effects and allow them to be freely removed such as through
// canonicalization and dead code elimination.
//
// After this pass, no cstr_ operations exist.
void populateRemoveShapeConstraintsPatterns(RewritePatternSet &patterns);
std::unique_ptr<OperationPass<func::FuncOp>> createRemoveShapeConstraintsPass();

/// Outline the shape computation part by adding shape.func and populate
/// conrresponding mapping infomation into ShapeMappingAnalysis.
std::unique_ptr<OperationPass<ModuleOp>> createOutlineShapeComputationPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Shape/Transforms/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_SHAPE_TRANSFORMS_PASSES_H_
