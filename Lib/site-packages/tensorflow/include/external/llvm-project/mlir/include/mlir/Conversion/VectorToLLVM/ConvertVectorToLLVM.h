//===- ConvertVectorToLLVM.h - Utils to convert from the vector dialect ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVM_H_
#define MLIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVM_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class LLVMTypeConverter;

/// Collect a set of patterns to convert from Vector contractions to LLVM Matrix
/// Intrinsics. To lower to assembly, the LLVM flag -lower-matrix-intrinsics
/// will be needed when invoking LLVM.
void populateVectorToLLVMMatrixConversionPatterns(LLVMTypeConverter &converter,
                                                  RewritePatternSet &patterns);

/// Collect a set of patterns to convert from the Vector dialect to LLVM.
void populateVectorToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    bool reassociateFPReductions = false, bool force32BitVectorIndices = false);

} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVM_H_
