//===- ParallelCombiningOpInterface.h - Parallel combining op interface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation interface for ops that parallel combining
// operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_PARALLELCOMBININGOPINTERFACE_H_
#define MLIR_INTERFACES_PARALLELCOMBININGOPINTERFACE_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace detail {
// TODO: Single region single block interface on interfaces ?
LogicalResult verifyParallelCombiningOpInterface(Operation *op);
} // namespace detail
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interfaces/ParallelCombiningOpInterface.h.inc"

#endif // MLIR_INTERFACES_PARALLELCOMBININGOPINTERFACE_H_
