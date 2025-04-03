//===- AllExtensions.h - All Func Extensions --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a common entry point for registering all extensions to the
// func dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_FUNC_EXTENSIONS_ALLEXTENSIONS_H
#define MLIR_DIALECT_FUNC_EXTENSIONS_ALLEXTENSIONS_H

namespace mlir {
class DialectRegistry;

namespace func {
/// Register all extensions of the func dialect. This should generally only be
/// used by tools, or other use cases that really do want *all* extensions of
/// the dialect. All other cases should prefer to instead register the specific
/// extensions they intend to take advantage of.
void registerAllExtensions(DialectRegistry &registry);
} // namespace func

} // namespace mlir

#endif // MLIR_DIALECT_FUNC_EXTENSIONS_ALLEXTENSIONS_H
