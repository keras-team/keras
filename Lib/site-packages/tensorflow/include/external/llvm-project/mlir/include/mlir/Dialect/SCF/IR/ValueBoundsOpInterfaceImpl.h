//===- ValueBoundsOpInterfaceImpl.h - Impl. of ValueBoundsOpInterface -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_IR_VALUEBOUNDSOPINTERFACEIMPL_H
#define MLIR_DIALECT_SCF_IR_VALUEBOUNDSOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace scf {
void registerValueBoundsOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace scf
} // namespace mlir

#endif // MLIR_DIALECT_SCF_IR_VALUEBOUNDSOPINTERFACEIMPL_H
