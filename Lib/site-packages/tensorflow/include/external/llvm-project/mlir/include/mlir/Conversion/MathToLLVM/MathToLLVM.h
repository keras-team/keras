//===- MathToLLVM.h - Math to LLVM dialect conversion -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MATHTOLLVM_MATHTOLLVM_H
#define MLIR_CONVERSION_MATHTOLLVM_MATHTOLLVM_H

#include <memory>

namespace mlir {

class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTMATHTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"

void populateMathToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns,
                                          bool approximateLog1p = true);

void registerConvertMathToLLVMInterface(DialectRegistry &registry);

} // namespace mlir

#endif // MLIR_CONVERSION_MATHTOLLVM_MATHTOLLVM_H
