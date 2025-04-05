//===- PrintCallHelper.h - Helper to emit runtime print calls ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_PRINTCALLHELPER_H_
#define MLIR_DIALECT_LLVMIR_PRINTCALLHELPER_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace mlir {

class OpBuilder;
class LLVMTypeConverter;

namespace LLVM {

/// Generate IR that prints the given string to stdout.
/// If a custom runtime function is defined via `runtimeFunctionName`, it must
/// have the signature void(char const*). The default function is `printString`.
void createPrintStrCall(OpBuilder &builder, Location loc, ModuleOp moduleOp,
                        StringRef symbolName, StringRef string,
                        const LLVMTypeConverter &typeConverter,
                        bool addNewline = true,
                        std::optional<StringRef> runtimeFunctionName = {});
} // namespace LLVM

} // namespace mlir

#endif
