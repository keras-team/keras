//===- BytecodeWriter.h - MLIR Bytecode Writer ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines interfaces to write MLIR bytecode files/streams.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BYTECODE_BYTECODEWRITER_H
#define MLIR_BYTECODE_BYTECODEWRITER_H

#include "mlir/IR/AsmState.h"
#include "llvm/Config/llvm-config.h" // for LLVM_VERSION_STRING

namespace mlir {
class DialectBytecodeWriter;
class DialectVersion;
class Operation;

/// A class to interact with the attributes and types printer when emitting MLIR
/// bytecode.
template <class T>
class AttrTypeBytecodeWriter {
public:
  AttrTypeBytecodeWriter() = default;
  virtual ~AttrTypeBytecodeWriter() = default;

  /// Callback writer API used in IRNumbering, where groups are created and
  /// type/attribute components are numbered. At this stage, writer is expected
  /// to be a `NumberingDialectWriter`.
  virtual LogicalResult write(T entry, std::optional<StringRef> &name,
                              DialectBytecodeWriter &writer) = 0;

  /// Callback writer API used in BytecodeWriter, where groups are created and
  /// type/attribute components are numbered. Here, DialectBytecodeWriter is
  /// expected to be an actual writer. The optional stringref specified by
  /// the user is ignored, since the group was already specified when numbering
  /// the IR.
  LogicalResult write(T entry, DialectBytecodeWriter &writer) {
    std::optional<StringRef> dummy;
    return write(entry, dummy, writer);
  }

  /// Return an Attribute/Type printer implemented via the given callable, whose
  /// form should match that of the `write` function above.
  template <typename CallableT,
            std::enable_if_t<std::is_convertible_v<
                                 CallableT, std::function<LogicalResult(
                                                T, std::optional<StringRef> &,
                                                DialectBytecodeWriter &)>>,
                             bool> = true>
  static std::unique_ptr<AttrTypeBytecodeWriter<T>>
  fromCallable(CallableT &&writeFn) {
    struct Processor : public AttrTypeBytecodeWriter<T> {
      Processor(CallableT &&writeFn)
          : AttrTypeBytecodeWriter(), writeFn(std::move(writeFn)) {}
      LogicalResult write(T entry, std::optional<StringRef> &name,
                          DialectBytecodeWriter &writer) override {
        return writeFn(entry, name, writer);
      }

      std::decay_t<CallableT> writeFn;
    };
    return std::make_unique<Processor>(std::forward<CallableT>(writeFn));
  }
};

/// This class contains the configuration used for the bytecode writer. It
/// controls various aspects of bytecode generation, and contains all of the
/// various bytecode writer hooks.
class BytecodeWriterConfig {
public:
  /// `producer` is an optional string that can be used to identify the producer
  /// of the bytecode when reading. It has no functional effect on the bytecode
  /// serialization.
  BytecodeWriterConfig(StringRef producer = "MLIR" LLVM_VERSION_STRING);
  /// `map` is a fallback resource map, which when provided will attach resource
  /// printers for the fallback resources within the map.
  BytecodeWriterConfig(FallbackAsmResourceMap &map,
                       StringRef producer = "MLIR" LLVM_VERSION_STRING);
  ~BytecodeWriterConfig();

  /// An internal implementation class that contains the state of the
  /// configuration.
  struct Impl;

  /// Return an instance of the internal implementation.
  const Impl &getImpl() const { return *impl; }

  /// Set the desired bytecode version to emit. This method does not validate
  /// the desired version. The bytecode writer entry point will return failure
  /// if it cannot emit the desired version.
  void setDesiredBytecodeVersion(int64_t bytecodeVersion);

  /// Get the set desired bytecode version to emit.
  int64_t getDesiredBytecodeVersion() const;

  /// A map containing the dialect versions to emit.
  llvm::StringMap<std::unique_ptr<DialectVersion>> &
  getDialectVersionMap() const;

  /// Set a given dialect version to emit on the map.
  template <class T>
  void setDialectVersion(std::unique_ptr<DialectVersion> dialectVersion) const {
    return setDialectVersion(T::getDialectNamespace(),
                             std::move(dialectVersion));
  }
  void setDialectVersion(StringRef dialectName,
                         std::unique_ptr<DialectVersion> dialectVersion) const;

  //===--------------------------------------------------------------------===//
  // Types and Attributes encoding
  //===--------------------------------------------------------------------===//

  /// Retrieve the callbacks.
  ArrayRef<std::unique_ptr<AttrTypeBytecodeWriter<Attribute>>>
  getAttributeWriterCallbacks() const;
  ArrayRef<std::unique_ptr<AttrTypeBytecodeWriter<Type>>>
  getTypeWriterCallbacks() const;

  /// Attach a custom bytecode printer callback to the configuration for the
  /// emission of custom type/attributes encodings.
  void attachAttributeCallback(
      std::unique_ptr<AttrTypeBytecodeWriter<Attribute>> callback);
  void
  attachTypeCallback(std::unique_ptr<AttrTypeBytecodeWriter<Type>> callback);

  /// Attach a custom bytecode printer callback to the configuration for the
  /// emission of custom type/attributes encodings.
  template <typename CallableT>
  std::enable_if_t<std::is_convertible_v<
      CallableT,
      std::function<LogicalResult(Attribute, std::optional<StringRef> &,
                                  DialectBytecodeWriter &)>>>
  attachAttributeCallback(CallableT &&emitFn) {
    attachAttributeCallback(AttrTypeBytecodeWriter<Attribute>::fromCallable(
        std::forward<CallableT>(emitFn)));
  }
  template <typename CallableT>
  std::enable_if_t<std::is_convertible_v<
      CallableT, std::function<LogicalResult(Type, std::optional<StringRef> &,
                                             DialectBytecodeWriter &)>>>
  attachTypeCallback(CallableT &&emitFn) {
    attachTypeCallback(AttrTypeBytecodeWriter<Type>::fromCallable(
        std::forward<CallableT>(emitFn)));
  }

  //===--------------------------------------------------------------------===//
  // Resources
  //===--------------------------------------------------------------------===//

  /// Set a boolean flag to skip emission of resources into the bytecode file.
  void setElideResourceDataFlag(bool shouldElideResourceData = true);

  /// Attach the given resource printer to the writer configuration.
  void attachResourcePrinter(std::unique_ptr<AsmResourcePrinter> printer);

  /// Attach an resource printer, in the form of a callable, to the
  /// configuration.
  template <typename CallableT>
  std::enable_if_t<std::is_convertible<
      CallableT, function_ref<void(Operation *, AsmResourceBuilder &)>>::value>
  attachResourcePrinter(StringRef name, CallableT &&printFn) {
    attachResourcePrinter(AsmResourcePrinter::fromCallable(
        name, std::forward<CallableT>(printFn)));
  }

  /// Attach resource printers to the AsmState for the fallback resources
  /// in the given map.
  void attachFallbackResourcePrinter(FallbackAsmResourceMap &map) {
    for (auto &printer : map.getPrinters())
      attachResourcePrinter(std::move(printer));
  }

private:
  /// A pointer to allocated storage for the impl state.
  std::unique_ptr<Impl> impl;
};

//===----------------------------------------------------------------------===//
// Entry Points
//===----------------------------------------------------------------------===//

/// Write the bytecode for the given operation to the provided output stream.
/// For streams where it matters, the given stream should be in "binary" mode.
/// It only ever fails if setDesiredByteCodeVersion can't be honored.
LogicalResult writeBytecodeToFile(Operation *op, raw_ostream &os,
                                  const BytecodeWriterConfig &config = {});

} // namespace mlir

#endif // MLIR_BYTECODE_BYTECODEWRITER_H
