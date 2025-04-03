//==- BuiltinToLLVMIRTranslation.h - Builtin Dialect to LLVM IR -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for builtin dialect to LLVM IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMIR_DIALECT_BUILTIN_BUILTINTOLLVMIRTRANSLATION_H
#define MLIR_TARGET_LLVMIR_DIALECT_BUILTIN_BUILTINTOLLVMIRTRANSLATION_H

namespace mlir {

class DialectRegistry;
class MLIRContext;

/// Register the translation from the builtin dialect to the LLVM IR in the
/// given registry.
void registerBuiltinDialectTranslation(DialectRegistry &registry);

/// Register the translation from the builtin dialect in the registry associated
/// with the given context.
void registerBuiltinDialectTranslation(MLIRContext &context);

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_DIALECT_BUILTIN_BUILTINTOLLVMIRTRANSLATION_H
