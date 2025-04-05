//===- LLVMInterfaces.h - LLVM Interfaces -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines op interfaces for the LLVM dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMINTERFACES_H_
#define MLIR_DIALECT_LLVMIR_LLVMINTERFACES_H_

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"

namespace mlir {
namespace LLVM {
namespace detail {

/// Verifies the access groups attribute of memory operations that implement the
/// access group interface.
LogicalResult verifyAccessGroupOpInterface(Operation *op);

/// Verifies the alias analysis attributes of memory operations that implement
/// the alias analysis interface.
LogicalResult verifyAliasAnalysisOpInterface(Operation *op);

} // namespace detail
} // namespace LLVM
} // namespace mlir

#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h.inc"

#endif // MLIR_DIALECT_LLVMIR_LLVMINTERFACES_H_
