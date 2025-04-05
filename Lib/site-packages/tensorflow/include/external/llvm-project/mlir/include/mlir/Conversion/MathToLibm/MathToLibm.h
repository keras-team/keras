//===- MathToLibm.h - Utils to convert from the complex dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_MATHTOLIBM_MATHTOLIBM_H_
#define MLIR_CONVERSION_MATHTOLIBM_MATHTOLIBM_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir {
template <typename T>
class OperationPass;

#define GEN_PASS_DECL_CONVERTMATHTOLIBM
#include "mlir/Conversion/Passes.h.inc"

/// Populate the given list with patterns that convert from Math to Libm calls.
/// If log1pBenefit is present, use it instead of benefit for the Log1p op.
void populateMathToLibmConversionPatterns(RewritePatternSet &patterns);

/// Create a pass to convert Math operations to libm calls.
std::unique_ptr<OperationPass<ModuleOp>> createConvertMathToLibmPass();

} // namespace mlir

#endif // MLIR_CONVERSION_MATHTOLIBM_MATHTOLIBM_H_
