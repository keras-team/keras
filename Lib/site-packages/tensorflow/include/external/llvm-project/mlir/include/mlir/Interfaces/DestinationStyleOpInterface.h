//===- DestinationStyleOpInterface.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_DESTINATIONSTYLEOPINTERFACE_H_
#define MLIR_INTERFACES_DESTINATIONSTYLEOPINTERFACE_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace detail {
/// Verify that `op` conforms to the invariants of DestinationStyleOpInterface
LogicalResult verifyDestinationStyleOpInterface(Operation *op);
} // namespace detail
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interfaces/DestinationStyleOpInterface.h.inc"

#endif // MLIR_INTERFACES_DESTINATIONSTYLEOPINTERFACE_H_
