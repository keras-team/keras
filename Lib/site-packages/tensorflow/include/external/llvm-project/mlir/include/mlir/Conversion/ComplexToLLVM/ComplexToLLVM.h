//===- ComplexToLLVM.h - Utils to convert from the complex dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_COMPLEXTOLLVM_COMPLEXTOLLVM_H_
#define MLIR_CONVERSION_COMPLEXTOLLVM_COMPLEXTOLLVM_H_

#include "mlir/Conversion/LLVMCommon/StructBuilder.h"

namespace mlir {
class DialectRegistry;
class LLVMTypeConverter;
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTCOMPLEXTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"

class ComplexStructBuilder : public StructBuilder {
public:
  /// Construct a helper for the given complex number value.
  using StructBuilder::StructBuilder;
  /// Build IR creating an `undef` value of the complex number type.
  static ComplexStructBuilder undef(OpBuilder &builder, Location loc,
                                    Type type);

  // Build IR extracting the real value from the complex number struct.
  Value real(OpBuilder &builder, Location loc);
  // Build IR inserting the real value into the complex number struct.
  void setReal(OpBuilder &builder, Location loc, Value real);

  // Build IR extracting the imaginary value from the complex number struct.
  Value imaginary(OpBuilder &builder, Location loc);
  // Build IR inserting the imaginary value into the complex number struct.
  void setImaginary(OpBuilder &builder, Location loc, Value imaginary);
};

/// Populate the given list with patterns that convert from Complex to LLVM.
void populateComplexToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns);

void registerConvertComplexToLLVMInterface(DialectRegistry &registry);

} // namespace mlir

#endif // MLIR_CONVERSION_COMPLEXTOLLVM_COMPLEXTOLLVM_H_
