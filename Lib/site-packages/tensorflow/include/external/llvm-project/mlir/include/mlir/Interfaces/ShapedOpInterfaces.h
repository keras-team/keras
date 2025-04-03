//===- ShapedOpInterfaces.h - Interfaces for Shaped Ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces for ops that operate on shaped values.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_SHAPEDOPINTERFACES_H_
#define MLIR_INTERFACES_SHAPEDOPINTERFACES_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace detail {

/// Verify invariants of ops that implement the ShapedDimOpInterface.
LogicalResult verifyShapedDimOpInterface(Operation *op);

} // namespace detail
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interfaces/ShapedOpInterfaces.h.inc"

#endif // MLIR_INTERFACES_SHAPEDOPINTERFACES_H_
