//===- Transforms.h - Mesh Transforms ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MESH_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_MESH_TRANSFORMS_TRANSFORMS_H

#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class RewritePatternSet;
class SymbolTableCollection;
class DialectRegistry;
class ImplicitLocOpBuilder;
namespace mesh {

void populateProcessMultiIndexOpLoweringPatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTableCollection);
void registerProcessMultiIndexOpLoweringDialects(DialectRegistry &registry);

void populateAllSliceOpLoweringPatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTableCollection);
void registerAllSliceOpLoweringDialects(DialectRegistry &registry);

void populateAllOpLoweringPatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTableCollection);
void registerAllOpLoweringDialects(DialectRegistry &registry);

TypedValue<IndexType>
createCollectiveProcessGroupSize(MeshOp mesh, ArrayRef<MeshAxis> axes,
                                 ImplicitLocOpBuilder &builder);

// Get process linear index along the given mesh axes.
TypedValue<IndexType> createProcessLinearIndex(StringRef mesh,
                                               ArrayRef<MeshAxis> meshAxes,
                                               ImplicitLocOpBuilder &builder);

} // namespace mesh
} // namespace mlir

#endif // MLIR_DIALECT_MESH_TRANSFORMS_TRANSFORMS_H
