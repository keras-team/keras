//===- FunctionCallUtils.h - Utilities for C function calls -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares helper functions to call common simple C functions in
// LLVMIR (e.g. among others to support printing and debugging).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_FUNCTIONCALLUTILS_H_
#define MLIR_DIALECT_LLVMIR_FUNCTIONCALLUTILS_H_

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include <optional>

namespace mlir {
class Location;
class ModuleOp;
class OpBuilder;
class Operation;
class Type;
class ValueRange;

namespace LLVM {
class LLVMFuncOp;

/// Helper functions to lookup or create the declaration for commonly used
/// external C function calls. The list of functions provided here must be
/// implemented separately (e.g. as part of a support runtime library or as part
/// of the libc).
LLVM::LLVMFuncOp lookupOrCreatePrintI64Fn(Operation *moduleOp);
LLVM::LLVMFuncOp lookupOrCreatePrintU64Fn(Operation *moduleOp);
LLVM::LLVMFuncOp lookupOrCreatePrintF16Fn(Operation *moduleOp);
LLVM::LLVMFuncOp lookupOrCreatePrintBF16Fn(Operation *moduleOp);
LLVM::LLVMFuncOp lookupOrCreatePrintF32Fn(Operation *moduleOp);
LLVM::LLVMFuncOp lookupOrCreatePrintF64Fn(Operation *moduleOp);
/// Declares a function to print a C-string.
/// If a custom runtime function is defined via `runtimeFunctionName`, it must
/// have the signature void(char const*). The default function is `printString`.
LLVM::LLVMFuncOp
lookupOrCreatePrintStringFn(Operation *moduleOp,
                            std::optional<StringRef> runtimeFunctionName = {});
LLVM::LLVMFuncOp lookupOrCreatePrintOpenFn(Operation *moduleOp);
LLVM::LLVMFuncOp lookupOrCreatePrintCloseFn(Operation *moduleOp);
LLVM::LLVMFuncOp lookupOrCreatePrintCommaFn(Operation *moduleOp);
LLVM::LLVMFuncOp lookupOrCreatePrintNewlineFn(Operation *moduleOp);
LLVM::LLVMFuncOp lookupOrCreateMallocFn(Operation *moduleOp, Type indexType);
LLVM::LLVMFuncOp lookupOrCreateAlignedAllocFn(Operation *moduleOp,
                                              Type indexType);
LLVM::LLVMFuncOp lookupOrCreateFreeFn(Operation *moduleOp);
LLVM::LLVMFuncOp lookupOrCreateGenericAllocFn(Operation *moduleOp,
                                              Type indexType);
LLVM::LLVMFuncOp lookupOrCreateGenericAlignedAllocFn(Operation *moduleOp,
                                                     Type indexType);
LLVM::LLVMFuncOp lookupOrCreateGenericFreeFn(Operation *moduleOp);
LLVM::LLVMFuncOp lookupOrCreateMemRefCopyFn(Operation *moduleOp, Type indexType,
                                            Type unrankedDescriptorType);

/// Create a FuncOp with signature `resultType`(`paramTypes`)` and name `name`.
LLVM::LLVMFuncOp lookupOrCreateFn(Operation *moduleOp, StringRef name,
                                  ArrayRef<Type> paramTypes = {},
                                  Type resultType = {}, bool isVarArg = false);

} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_FUNCTIONCALLUTILS_H_
