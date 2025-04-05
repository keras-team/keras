//===- Transforms.h - ArmSVE Dialect Transformation Entrypoints -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARMSVE_TRANSFORMS_H
#define MLIR_DIALECT_ARMSVE_TRANSFORMS_H

namespace mlir {

class LLVMConversionTarget;
class LLVMTypeConverter;
class RewritePatternSet;

/// Collect a set of patterns to lower ArmSVE ops to ops that map to LLVM
/// intrinsics.
void populateArmSVELegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns);

/// Configure the target to support lowering ArmSVE ops to ops that map to LLVM
/// intrinsics.
void configureArmSVELegalizeForExportTarget(LLVMConversionTarget &target);

} // namespace mlir

#endif // MLIR_DIALECT_ARMSVE_TRANSFORMS_H
