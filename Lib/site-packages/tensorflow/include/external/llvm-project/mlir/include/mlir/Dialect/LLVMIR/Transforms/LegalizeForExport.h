//===- LegalizeForExport.h - Prepare for translation to LLVM IR -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_TRANSFORMS_LEGALIZEFOREXPORT_H
#define MLIR_DIALECT_LLVMIR_TRANSFORMS_LEGALIZEFOREXPORT_H

#include <memory>

namespace mlir {
class Operation;
class Pass;

namespace LLVM {

#define GEN_PASS_DECL_LLVMLEGALIZEFOREXPORT
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"

/// Make argument-taking successors of each block distinct.  PHI nodes in LLVM
/// IR use the predecessor ID to identify which value to take. They do not
/// support different values coming from the same predecessor. If a block has
/// another block as a successor more than once with different values, insert
/// a new dummy block for LLVM PHI nodes to tell the sources apart.
void ensureDistinctSuccessors(Operation *op);

/// Creates a pass that legalizes the LLVM dialect operations so that they can
/// be translated to LLVM IR.
std::unique_ptr<Pass> createLegalizeForExportPass();

} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_TRANSFORMS_LEGALIZEFOREXPORT_H
