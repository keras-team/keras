//===- LogicalResult.h - Stub aliasing to llvm/LogicalResult ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_LOGICALRESULT_H
#define MLIR_SUPPORT_LOGICALRESULT_H

#include "llvm/Support/LogicalResult.h"

// TODO: This header is a stop-gap to avoid breaking downstream, and is to be
// removed eventually.
namespace mlir {
using llvm::failed;
using llvm::failure;
using llvm::FailureOr;
using llvm::LogicalResult;
using llvm::ParseResult;
using llvm::succeeded;
using llvm::success;
} // namespace mlir

#endif // MLIR_SUPPORT_LOGICALRESULT_H
