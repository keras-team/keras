//===- BytecodeReader.h - MLIR Bytecode Reader ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines interfaces to read MLIR bytecode files/streams.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BYTECODE_BYTECODEREADER_H
#define MLIR_BYTECODE_BYTECODEREADER_H

#include "mlir/IR/AsmState.h"
#include "mlir/Support/LLVM.h"
#include <functional>
#include <memory>

namespace llvm {
class MemoryBufferRef;
class SourceMgr;
} // namespace llvm

namespace mlir {
/// The BytecodeReader allows to load MLIR bytecode files, while keeping the
/// state explicitly available in order to support lazy loading.
/// The `finalize` method must be called before destruction.
class BytecodeReader {
public:
  /// Create a bytecode reader for the given buffer. If `lazyLoad` is true,
  /// isolated regions aren't loaded eagerly.
  explicit BytecodeReader(
      llvm::MemoryBufferRef buffer, const ParserConfig &config, bool lazyLoad,
      const std::shared_ptr<llvm::SourceMgr> &bufferOwnerRef = {});
  ~BytecodeReader();

  /// Read the operations defined within the given memory buffer, containing
  /// MLIR bytecode, into the provided block. If the reader was created with
  /// `lazyLoad` enabled, isolated regions aren't loaded eagerly.
  /// The lazyOps call back is invoked for every ops that can be lazy-loaded.
  /// This let the client decide if the op should be materialized
  /// immediately or delayed.
  LogicalResult readTopLevel(
      Block *block, llvm::function_ref<bool(Operation *)> lazyOps =
                        [](Operation *) { return false; });

  /// Return the number of ops that haven't been materialized yet.
  int64_t getNumOpsToMaterialize() const;

  /// Return true if the provided op is materializable.
  bool isMaterializable(Operation *op);

  /// Materialize the provide operation. The provided operation must be
  /// materializable.
  /// The lazyOps call back is invoked for every ops that can be lazy-loaded.
  /// This let the client decide if the op should be materialized immediately or
  /// delayed.
  /// !! Using this materialize withing an IR walk() can be confusing: make sure
  /// to use a PreOrder traversal !!
  LogicalResult materialize(
      Operation *op, llvm::function_ref<bool(Operation *)> lazyOpsCallback =
                         [](Operation *) { return false; });

  /// Finalize the lazy-loading by calling back with every op that hasn't been
  /// materialized to let the client decide if the op should be deleted or
  /// materialized. The op is materialized if the callback returns true, deleted
  /// otherwise. The implementation of the callback must be thread-safe.
  LogicalResult finalize(function_ref<bool(Operation *)> shouldMaterialize =
                             [](Operation *) { return true; });

  class Impl;

private:
  std::unique_ptr<Impl> impl;
};

/// Returns true if the given buffer starts with the magic bytes that signal
/// MLIR bytecode.
bool isBytecode(llvm::MemoryBufferRef buffer);

/// Read the operations defined within the given memory buffer, containing MLIR
/// bytecode, into the provided block.
LogicalResult readBytecodeFile(llvm::MemoryBufferRef buffer, Block *block,
                               const ParserConfig &config);
/// An overload with a source manager whose main file buffer is used for
/// parsing. The lifetime of the source manager may be freely extended during
/// parsing such that the source manager is not destroyed before the parsed IR.
LogicalResult
readBytecodeFile(const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
                 Block *block, const ParserConfig &config);

} // namespace mlir

#endif // MLIR_BYTECODE_BYTECODEREADER_H
