//===- CastInterfaces.h - Cast Interfaces for MLIR --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the cast interfaces defined in
// `CastInterfaces.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_CASTINTERFACES_H
#define MLIR_INTERFACES_CASTINTERFACES_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
class DialectRegistry;

namespace impl {
/// Attempt to fold the given cast operation.
LogicalResult foldCastInterfaceOp(Operation *op,
                                  ArrayRef<Attribute> attrOperands,
                                  SmallVectorImpl<OpFoldResult> &foldResults);

/// Attempt to verify the given cast operation.
LogicalResult verifyCastInterfaceOp(Operation *op);
} // namespace impl

namespace builtin {
void registerCastOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace builtin
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interfaces/CastInterfaces.h.inc"

#endif // MLIR_INTERFACES_CASTINTERFACES_H
