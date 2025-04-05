//===- Transforms.h - AMX Dialect Transformation Entrypoints ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AMX_TRANSFORMS_H
#define MLIR_DIALECT_AMX_TRANSFORMS_H

namespace mlir {

class LLVMConversionTarget;
class LLVMTypeConverter;
class RewritePatternSet;

/// Collect a set of patterns to lower AMX ops to ops that map to LLVM
/// intrinsics.
void populateAMXLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                              RewritePatternSet &patterns);

/// Configure the target to support lowering AMX ops to ops that map to LLVM
/// intrinsics.
void configureAMXLegalizeForExportTarget(LLVMConversionTarget &target);

} // namespace mlir

#endif // MLIR_DIALECT_AMX_TRANSFORMS_H
