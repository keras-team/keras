//===- ToLLVMInterface.h - Conversion to LLVM iface ---*- C++ -*-=============//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CONVERTTOLLVM_TOLLVMINTERFACE_H
#define MLIR_CONVERSION_CONVERTTOLLVM_TOLLVMINTERFACE_H

#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
class ConversionTarget;
class LLVMTypeConverter;
class MLIRContext;
class Operation;
class RewritePatternSet;

/// Base class for dialect interfaces providing translation to LLVM IR.
/// Dialects that can be translated should provide an implementation of this
/// interface for the supported operations. The interface may be implemented in
/// a separate library to avoid the "main" dialect library depending on LLVM IR.
/// The interface can be attached using the delayed registration mechanism
/// available in DialectRegistry.
class ConvertToLLVMPatternInterface
    : public DialectInterface::Base<ConvertToLLVMPatternInterface> {
public:
  ConvertToLLVMPatternInterface(Dialect *dialect) : Base(dialect) {}

  /// Hook for derived dialect interface to load the dialects they
  /// target. The LLVMDialect is implicitly already loaded, but this
  /// method allows to load other intermediate dialects used in the
  /// conversion, or target dialects like NVVM for example.
  virtual void loadDependentDialects(MLIRContext *context) const {}

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  virtual void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const = 0;
};

/// Recursively walk the IR and collect all dialects implementing the interface,
/// and populate the conversion patterns.
void populateConversionTargetFromOperation(Operation *op,
                                           ConversionTarget &target,
                                           LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_CONVERTTOLLVM_TOLLVMINTERFACE_H
