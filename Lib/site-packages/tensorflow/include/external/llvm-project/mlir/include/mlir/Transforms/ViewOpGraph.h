//===- ViewOpGraph.h - View/write op graphviz graphs ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines interface to produce Graphviz outputs of MLIR op within block.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_VIEWOPGRAPH_H_
#define MLIR_TRANSFORMS_VIEWOPGRAPH_H_

#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
class Pass;

#define GEN_PASS_DECL_VIEWOPGRAPH
#include "mlir/Transforms/Passes.h.inc"

/// Creates a pass to print op graphs.
std::unique_ptr<Pass> createPrintOpGraphPass(raw_ostream &os = llvm::errs());

} // namespace mlir

#endif // MLIR_TRANSFORMS_VIEWOPGRAPH_H_
