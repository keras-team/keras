//===- CSE.h - Common Subexpression Elimination -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares methods for eliminating common subexpressions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_CSE_H_
#define MLIR_TRANSFORMS_CSE_H_

namespace mlir {

class DominanceInfo;
class Operation;
class RewriterBase;

/// Eliminate common subexpressions within the given operation. This transform
/// looks for and deduplicates equivalent operations.
///
/// `changed` indicates whether the IR was modified or not.
void eliminateCommonSubExpressions(RewriterBase &rewriter,
                                   DominanceInfo &domInfo, Operation *op,
                                   bool *changed = nullptr);

} // namespace mlir

#endif // MLIR_TRANSFORMS_CSE_H_
