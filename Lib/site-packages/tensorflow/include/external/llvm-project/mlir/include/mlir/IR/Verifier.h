//===- Verifier.h - Verifier analysis for MLIR structures -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_VERIFIER_H
#define MLIR_IR_VERIFIER_H

#include "mlir/Support/LLVM.h"

namespace mlir {
class Operation;

/// Perform (potentially expensive) checks of invariants, used to detect
/// compiler bugs, on this operation and any nested operations. On error, this
/// reports the error through the MLIRContext and returns failure. If
/// `verifyRecursively` is false, this assumes that nested operations have
/// already been properly verified, and does not recursively invoke the verifier
/// on nested operations.
LogicalResult verify(Operation *op, bool verifyRecursively = true);

} // namespace mlir

#endif
