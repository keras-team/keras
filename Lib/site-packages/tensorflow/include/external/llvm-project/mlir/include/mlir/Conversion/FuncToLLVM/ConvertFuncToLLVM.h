//===- ConvertFuncToLLVM.h - Convert Func to LLVM ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides a set of conversion patterns from the Func dialect to the LLVM IR
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_FUNCTOLLVM_CONVERTFUNCTOLLVM_H
#define MLIR_CONVERSION_FUNCTOLLVM_CONVERTFUNCTOLLVM_H

#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir {

namespace LLVM {
class LLVMFuncOp;
} // namespace LLVM

class ConversionPatternRewriter;
class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
class SymbolTable;

/// Convert input FunctionOpInterface operation to LLVMFuncOp by using the
/// provided LLVMTypeConverter. Return failure if failed to so.
FailureOr<LLVM::LLVMFuncOp>
convertFuncOpToLLVMFuncOp(FunctionOpInterface funcOp,
                          ConversionPatternRewriter &rewriter,
                          const LLVMTypeConverter &converter);

/// Collect the default pattern to convert a FuncOp to the LLVM dialect. If
/// `emitCWrappers` is set, the pattern will also produce functions
/// that pass memref descriptors by pointer-to-structure in addition to the
/// default unpacked form.
void populateFuncToLLVMFuncOpConversionPattern(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns);

/// Collect the patterns to convert from the Func dialect to LLVM. The
/// conversion patterns capture the LLVMTypeConverter and the LowerToLLVMOptions
/// by reference meaning the references have to remain alive during the entire
/// pattern lifetime.
///
/// The `symbolTable` parameter can be used to speed up function lookups in the
/// module. It's good to provide it, but only if we know that the patterns will
/// be applied to a single module and the symbols referenced by the symbol table
/// will not be removed and new symbols will not be added during the usage of
/// the patterns. If provided, the lookups will have O(calls) cumulative
/// runtime, otherwise O(calls * functions). The symbol table is currently not
/// needed if `converter.getOptions().useBarePtrCallConv` is `true`, but it's
/// not an error to provide it anyway.
void populateFuncToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    const SymbolTable *symbolTable = nullptr);

void registerConvertFuncToLLVMInterface(DialectRegistry &registry);

} // namespace mlir

#endif // MLIR_CONVERSION_FUNCTOLLVM_CONVERTFUNCTOLLVM_H
