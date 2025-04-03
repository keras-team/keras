//===- OneToNTypeFuncConversions.h - 1:N type conv. for Func ----*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_FUNC_TRANSFORMS_ONETONTYPEFUNCCONVERSIONS_H
#define MLIR_DIALECT_FUNC_TRANSFORMS_ONETONTYPEFUNCCONVERSIONS_H

namespace mlir {
class TypeConverter;
class RewritePatternSet;
} // namespace mlir

namespace mlir {

// Populates the provided pattern set with patterns that do 1:N type conversions
// on func ops. This is intended to be used with `applyPartialOneToNConversion`.
void populateFuncTypeConversionPatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_DIALECT_FUNC_TRANSFORMS_ONETONTYPEFUNCCONVERSIONS_H
