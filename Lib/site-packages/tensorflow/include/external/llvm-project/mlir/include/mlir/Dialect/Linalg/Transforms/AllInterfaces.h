//===- AllInterfaces.h - ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a common entry point for registering all external
// interface implementations to the linalg dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMS_ALLINTERFACES_H
#define MLIR_DIALECT_LINALG_TRANSFORMS_ALLINTERFACES_H

namespace mlir {
class DialectRegistry;

namespace linalg {
void registerAllDialectInterfaceImplementations(DialectRegistry &registry);
} // namespace linalg

} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_ALLINTERFACES_H
