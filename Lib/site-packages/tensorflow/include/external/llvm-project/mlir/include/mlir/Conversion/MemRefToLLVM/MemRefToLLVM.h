//===- MemRefToLLVM.h - MemRef to LLVM dialect conversion -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MEMREFTOLLVM_MEMREFTOLLVM_H
#define MLIR_CONVERSION_MEMREFTOLLVM_MEMREFTOLLVM_H

#include <memory>

namespace mlir {
class DialectRegistry;
class Pass;
class LLVMTypeConverter;
class RewritePatternSet;

#define GEN_PASS_DECL_FINALIZEMEMREFTOLLVMCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"

/// Collect a set of patterns to convert memory-related operations from the
/// MemRef dialect to the LLVM dialect.
void populateFinalizeMemRefToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns);

void registerConvertMemRefToLLVMInterface(DialectRegistry &registry);

} // namespace mlir

#endif // MLIR_CONVERSION_MEMREFTOLLVM_MEMREFTOLLVM_H
