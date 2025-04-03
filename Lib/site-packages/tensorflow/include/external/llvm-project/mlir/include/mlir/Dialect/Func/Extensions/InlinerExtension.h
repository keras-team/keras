//===- InlinerExtension.h - Func Inliner Extension 0000----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an extension for the func dialect that implements the
// interfaces necessary to support inlining.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_FUNC_EXTENSIONS_INLINEREXTENSION_H
#define MLIR_DIALECT_FUNC_EXTENSIONS_INLINEREXTENSION_H

namespace mlir {
class DialectRegistry;

namespace func {
/// Register the extension used to support inlining the func dialect.
void registerInlinerExtension(DialectRegistry &registry);
} // namespace func

} // namespace mlir

#endif // MLIR_DIALECT_FUNC_EXTENSIONS_INLINEREXTENSION_H
