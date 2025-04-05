//===- Parser.h - MLIR Parser Library Interface -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is contains a unified interface for parsing serialized MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PARSER_PARSER_H
#define MLIR_PARSER_PARSER_H

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OwningOpRef.h"
#include <cstddef>

namespace llvm {
class SourceMgr;
class SMDiagnostic;
class StringRef;
} // namespace llvm

namespace mlir {
namespace detail {

/// Given a block containing operations that have just been parsed, if the block
/// contains a single operation of `ContainerOpT` type then remove it from the
/// block and return it. If the block does not contain just that operation,
/// create a new operation instance of `ContainerOpT` and move all of the
/// operations within `parsedBlock` into the first block of the first region.
/// `ContainerOpT` is required to have a single region containing a single
/// block, and must implement the `SingleBlockImplicitTerminator` trait.
template <typename ContainerOpT>
inline OwningOpRef<ContainerOpT> constructContainerOpForParserIfNecessary(
    Block *parsedBlock, MLIRContext *context, Location sourceFileLoc) {

  // Check to see if we parsed a single instance of this operation.
  if (llvm::hasSingleElement(*parsedBlock)) {
    if (ContainerOpT op = dyn_cast<ContainerOpT>(&parsedBlock->front())) {
      op->remove();
      return op;
    }
  }

  // If not, then build a new top-level op if a concrete operation type was
  // specified.
  if constexpr (std::is_same_v<ContainerOpT, Operation *>) {
    (void)context;
    return emitError(sourceFileLoc)
               << "source must contain a single top-level operation, found: "
               << parsedBlock->getOperations().size(),
           nullptr;
  } else {
    static_assert(
        ContainerOpT::template hasTrait<OpTrait::OneRegion>() &&
            (ContainerOpT::template hasTrait<OpTrait::NoTerminator>() ||
             OpTrait::template hasSingleBlockImplicitTerminator<
                 ContainerOpT>::value),
        "Expected `ContainerOpT` to have a single region with a single "
        "block that has an implicit terminator or does not require one");

    OpBuilder builder(context);
    ContainerOpT op = builder.create<ContainerOpT>(sourceFileLoc);
    OwningOpRef<ContainerOpT> opRef(op);
    assert(op->getNumRegions() == 1 &&
           llvm::hasSingleElement(op->getRegion(0)) &&
           "expected generated operation to have a single region with a single "
           "block");
    Block *opBlock = &op->getRegion(0).front();
    opBlock->getOperations().splice(opBlock->begin(),
                                    parsedBlock->getOperations());

    // After splicing, verify just this operation to ensure it can properly
    // contain the operations inside of it.
    if (failed(op.verifyInvariants()))
      return OwningOpRef<ContainerOpT>();
    return opRef;
  }
}
} // namespace detail

/// This parses the file specified by the indicated SourceMgr and appends parsed
/// operations to the given block. If the block is non-empty, the operations are
/// placed before the current terminator. If parsing is successful, success is
/// returned. Otherwise, an error message is emitted through the error handler
/// registered in the context, and failure is returned. If `sourceFileLoc` is
/// non-null, it is populated with a file location representing the start of the
/// source file that is being parsed.
LogicalResult parseSourceFile(const llvm::SourceMgr &sourceMgr, Block *block,
                              const ParserConfig &config,
                              LocationAttr *sourceFileLoc = nullptr);
/// An overload with a source manager that may have references taken during the
/// parsing process, and whose lifetime can be freely extended (such that the
/// source manager is not destroyed before the parsed IR). This is useful, for
/// example, to avoid copying some large resources into the MLIRContext and
/// instead referencing the data directly from the input buffers.
LogicalResult parseSourceFile(const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
                              Block *block, const ParserConfig &config,
                              LocationAttr *sourceFileLoc = nullptr);

/// This parses the file specified by the indicated filename and appends parsed
/// operations to the given block. If the block is non-empty, the operations are
/// placed before the current terminator. If parsing is successful, success is
/// returned. Otherwise, an error message is emitted through the error handler
/// registered in the context, and failure is returned. If `sourceFileLoc` is
/// non-null, it is populated with a file location representing the start of the
/// source file that is being parsed.
LogicalResult parseSourceFile(llvm::StringRef filename, Block *block,
                              const ParserConfig &config,
                              LocationAttr *sourceFileLoc = nullptr);

/// This parses the file specified by the indicated filename using the provided
/// SourceMgr and appends parsed operations to the given block. If the block is
/// non-empty, the operations are placed before the current terminator. If
/// parsing is successful, success is returned. Otherwise, an error message is
/// emitted through the error handler registered in the context, and failure is
/// returned. If `sourceFileLoc` is non-null, it is populated with a file
/// location representing the start of the source file that is being parsed.
LogicalResult parseSourceFile(llvm::StringRef filename,
                              llvm::SourceMgr &sourceMgr, Block *block,
                              const ParserConfig &config,
                              LocationAttr *sourceFileLoc = nullptr);
/// An overload with a source manager that may have references taken during the
/// parsing process, and whose lifetime can be freely extended (such that the
/// source manager is not destroyed before the parsed IR). This is useful, for
/// example, to avoid copying some large resources into the MLIRContext and
/// instead referencing the data directly from the input buffers.
LogicalResult parseSourceFile(llvm::StringRef filename,
                              const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
                              Block *block, const ParserConfig &config,
                              LocationAttr *sourceFileLoc = nullptr);

/// This parses the IR string and appends parsed operations to the given block.
/// If the block is non-empty, the operations are placed before the current
/// terminator. If parsing is successful, success is returned. Otherwise, an
/// error message is emitted through the error handler registered in the
/// context, and failure is returned.
/// `sourceName` is used as the file name of the source; any IR without
/// locations will get a `FileLineColLoc` location with `sourceName` as the file
/// name. If `sourceFileLoc` is non-null, it is populated with a file location
/// representing the start of the source file that is being parsed.
LogicalResult parseSourceString(llvm::StringRef sourceStr, Block *block,
                                const ParserConfig &config,
                                StringRef sourceName = "",
                                LocationAttr *sourceFileLoc = nullptr);

namespace detail {
/// The internal implementation of the templated `parseSourceFile` methods
/// below, that simply forwards to the non-templated version.
template <typename ContainerOpT, typename... ParserArgs>
inline OwningOpRef<ContainerOpT> parseSourceFile(const ParserConfig &config,
                                                 ParserArgs &&...args) {
  LocationAttr sourceFileLoc;
  Block block;
  if (failed(parseSourceFile(std::forward<ParserArgs>(args)..., &block, config,
                             &sourceFileLoc)))
    return OwningOpRef<ContainerOpT>();
  return detail::constructContainerOpForParserIfNecessary<ContainerOpT>(
      &block, config.getContext(), sourceFileLoc);
}
} // namespace detail

/// This parses the file specified by the indicated SourceMgr. If the source IR
/// contained a single instance of `ContainerOpT`, it is returned. Otherwise, a
/// new instance of `ContainerOpT` is constructed containing all of the parsed
/// operations. If parsing was not successful, null is returned and an error
/// message is emitted through the error handler registered in the context, and
/// failure is returned. `ContainerOpT` is required to have a single region
/// containing a single block, and must implement the
/// `SingleBlockImplicitTerminator` trait.
template <typename ContainerOpT = Operation *>
inline OwningOpRef<ContainerOpT>
parseSourceFile(const llvm::SourceMgr &sourceMgr, const ParserConfig &config) {
  return detail::parseSourceFile<ContainerOpT>(config, sourceMgr);
}
/// An overload with a source manager that may have references taken during the
/// parsing process, and whose lifetime can be freely extended (such that the
/// source manager is not destroyed before the parsed IR). This is useful, for
/// example, to avoid copying some large resources into the MLIRContext and
/// instead referencing the data directly from the input buffers.
template <typename ContainerOpT = Operation *>
inline OwningOpRef<ContainerOpT>
parseSourceFile(const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
                const ParserConfig &config) {
  return detail::parseSourceFile<ContainerOpT>(config, sourceMgr);
}

/// This parses the file specified by the indicated filename. If the source IR
/// contained a single instance of `ContainerOpT`, it is returned. Otherwise, a
/// new instance of `ContainerOpT` is constructed containing all of the parsed
/// operations. If parsing was not successful, null is returned and an error
/// message is emitted through the error handler registered in the context, and
/// failure is returned. `ContainerOpT` is required to have a single region
/// containing a single block, and must implement the
/// `SingleBlockImplicitTerminator` trait.
template <typename ContainerOpT = Operation *>
inline OwningOpRef<ContainerOpT> parseSourceFile(StringRef filename,
                                                 const ParserConfig &config) {
  return detail::parseSourceFile<ContainerOpT>(config, filename);
}

/// This parses the file specified by the indicated filename using the provided
/// SourceMgr. If the source IR contained a single instance of `ContainerOpT`,
/// it is returned. Otherwise, a new instance of `ContainerOpT` is constructed
/// containing all of the parsed operations. If parsing was not successful, null
/// is returned and an error message is emitted through the error handler
/// registered in the context, and failure is returned. `ContainerOpT` is
/// required to have a single region containing a single block, and must
/// implement the `SingleBlockImplicitTerminator` trait.
template <typename ContainerOpT = Operation *>
inline OwningOpRef<ContainerOpT> parseSourceFile(llvm::StringRef filename,
                                                 llvm::SourceMgr &sourceMgr,
                                                 const ParserConfig &config) {
  return detail::parseSourceFile<ContainerOpT>(config, filename, sourceMgr);
}
/// An overload with a source manager that may have references taken during the
/// parsing process, and whose lifetime can be freely extended (such that the
/// source manager is not destroyed before the parsed IR). This is useful, for
/// example, to avoid copying some large resources into the MLIRContext and
/// instead referencing the data directly from the input buffers.
template <typename ContainerOpT = Operation *>
inline OwningOpRef<ContainerOpT>
parseSourceFile(llvm::StringRef filename,
                const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
                const ParserConfig &config) {
  return detail::parseSourceFile<ContainerOpT>(config, filename, sourceMgr);
}

/// This parses the provided string containing MLIR. If the source IR contained
/// a single instance of `ContainerOpT`, it is returned. Otherwise, a new
/// instance of `ContainerOpT` is constructed containing all of the parsed
/// operations. If parsing was not successful, null is returned and an error
/// message is emitted through the error handler registered in the context, and
/// failure is returned. `ContainerOpT` is required to have a single region
/// containing a single block, and must implement the
/// `SingleBlockImplicitTerminator` trait.
/// `sourceName` is used as the file name of the source; any IR without
/// locations will get a `FileLineColLoc` location with `sourceName` as the file
/// name.
template <typename ContainerOpT = Operation *>
inline OwningOpRef<ContainerOpT> parseSourceString(llvm::StringRef sourceStr,
                                                   const ParserConfig &config,
                                                   StringRef sourceName = "") {
  LocationAttr sourceFileLoc;
  Block block;
  if (failed(parseSourceString(sourceStr, &block, config, sourceName,
                               &sourceFileLoc)))
    return OwningOpRef<ContainerOpT>();
  return detail::constructContainerOpForParserIfNecessary<ContainerOpT>(
      &block, config.getContext(), sourceFileLoc);
}

} // namespace mlir

#endif // MLIR_PARSER_PARSER_H
