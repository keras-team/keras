//===- ConversionTarget.h - LLVM dialect conversion target ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LLVMCOMMON_CONVERSIONTARGET_H
#define MLIR_CONVERSION_LLVMCOMMON_CONVERSIONTARGET_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
/// Derived class that automatically populates legalization information for
/// different LLVM ops.
class LLVMConversionTarget : public ConversionTarget {
public:
  explicit LLVMConversionTarget(MLIRContext &ctx);
};
} // namespace mlir

#endif // MLIR_CONVERSION_LLVMCOMMON_CONVERSIONTARGET_H
