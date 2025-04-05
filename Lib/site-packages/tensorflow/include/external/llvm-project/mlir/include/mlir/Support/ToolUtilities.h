//===- ToolUtilities.h - MLIR Tool Utilities --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares common utilities for implementing MLIR tools.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_TOOLUTILITIES_H
#define MLIR_SUPPORT_TOOLUTILITIES_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

namespace llvm {
class MemoryBuffer;
} // namespace llvm

namespace mlir {
using ChunkBufferHandler = function_ref<LogicalResult(
    std::unique_ptr<llvm::MemoryBuffer> chunkBuffer, raw_ostream &os)>;

extern inline const char *const kDefaultSplitMarker = "// -----";

/// Splits the specified buffer on a marker (`// -----` by default), processes
/// each chunk independently according to the normal `processChunkBuffer` logic,
/// and writes all results to `os`.
///
/// This is used to allow a large number of small independent tests to be put
/// into a single file. The input split marker is configurable. If it is empty,
/// merging is disabled, which allows for merging split and non-split code
/// paths. Output split markers (`//-----` by default) followed by a new line
/// character, respectively, are placed between each of the processed output
/// chunks. (The new line character is inserted even if the split marker is
/// empty.)
LogicalResult
splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer> originalBuffer,
                      ChunkBufferHandler processChunkBuffer, raw_ostream &os,
                      llvm::StringRef inputSplitMarker = kDefaultSplitMarker,
                      llvm::StringRef outputSplitMarker = "");
} // namespace mlir

#endif // MLIR_SUPPORT_TOOLUTILITIES_H
