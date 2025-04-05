//===- InlinerInterfaceImpl.h - Inlining for LLVM the dialect ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Allows registering the LLVM DialectInlinerInterface with the LLVM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_TRANSFORMS_INLINERINTERFACEIMPL_H
#define MLIR_DIALECT_LLVMIR_TRANSFORMS_INLINERINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace LLVM {

/// Register the `LLVMInlinerInterface` implementation of
/// `DialectInlinerInterface` with the LLVM dialect.
void registerInlinerInterface(DialectRegistry &registry);

} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_TRANSFORMS_INLINERINTERFACEIMPL_H
