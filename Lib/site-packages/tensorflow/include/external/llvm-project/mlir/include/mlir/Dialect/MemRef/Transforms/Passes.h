//===- Passes.h - MemRef Patterns and Passes --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares patterns and passes on MemRef operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

class AffineDialect;
class ModuleOp;

namespace func {
namespace arith {
class ArithDialect;
} // namespace arith
class FuncDialect;
} // namespace func
namespace scf {
class SCFDialect;
} // namespace scf
namespace tensor {
class TensorDialect;
} // namespace tensor
namespace vector {
class VectorDialect;
} // namespace vector

namespace memref {
//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"

/// Creates an instance of the ExpandOps pass that legalizes memref dialect ops
/// to be convertible to LLVM. For example, `memref.reshape` gets converted to
/// `memref_reinterpret_cast`.
std::unique_ptr<Pass> createExpandOpsPass();

/// Creates an operation pass to fold memref aliasing ops into consumer
/// load/store ops into `patterns`.
std::unique_ptr<Pass> createFoldMemRefAliasOpsPass();

/// Creates an interprocedural pass to normalize memrefs to have a trivial
/// (identity) layout map.
std::unique_ptr<OperationPass<ModuleOp>> createNormalizeMemRefsPass();

/// Creates an operation pass to resolve `memref.dim` operations with values
/// that are defined by operations that implement the
/// `ReifyRankedShapedTypeOpInterface`, in terms of shapes of its input
/// operands.
std::unique_ptr<Pass> createResolveRankedShapeTypeResultDimsPass();

/// Creates an operation pass to resolve `memref.dim` operations with values
/// that are defined by operations that implement the
/// `InferShapedTypeOpInterface` or the `ReifyRankedShapedTypeOpInterface`,
/// in terms of shapes of its input operands.
std::unique_ptr<Pass> createResolveShapedTypeResultDimsPass();

/// Creates an operation pass to expand some memref operation into
/// easier to reason about operations.
std::unique_ptr<Pass> createExpandStridedMetadataPass();

/// Creates an operation pass to expand `memref.realloc` operations into their
/// components.
std::unique_ptr<Pass> createExpandReallocPass(bool emitDeallocs = true);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"

} // namespace memref
} // namespace mlir

#endif // MLIR_DIALECT_MEMREF_TRANSFORMS_PASSES_H
