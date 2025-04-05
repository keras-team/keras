//===- BytecodeReaderConfig.h - MLIR Bytecode Reader Config -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header config for reading MLIR bytecode files/streams.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BYTECODE_BYTECODEREADERCONFIG_H
#define MLIR_BYTECODE_BYTECODEREADERCONFIG_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class Attribute;
class DialectBytecodeReader;
class Type;

/// A class to interact with the attributes and types parser when parsing MLIR
/// bytecode.
template <class T>
class AttrTypeBytecodeReader {
public:
  AttrTypeBytecodeReader() = default;
  virtual ~AttrTypeBytecodeReader() = default;

  virtual LogicalResult read(DialectBytecodeReader &reader,
                             StringRef dialectName, T &entry) = 0;

  /// Return an Attribute/Type printer implemented via the given callable, whose
  /// form should match that of the `parse` function above.
  template <typename CallableT,
            std::enable_if_t<
                std::is_convertible_v<
                    CallableT, std::function<LogicalResult(
                                   DialectBytecodeReader &, StringRef, T &)>>,
                bool> = true>
  static std::unique_ptr<AttrTypeBytecodeReader<T>>
  fromCallable(CallableT &&readFn) {
    struct Processor : public AttrTypeBytecodeReader<T> {
      Processor(CallableT &&readFn)
          : AttrTypeBytecodeReader(), readFn(std::move(readFn)) {}
      LogicalResult read(DialectBytecodeReader &reader, StringRef dialectName,
                         T &entry) override {
        return readFn(reader, dialectName, entry);
      }

      std::decay_t<CallableT> readFn;
    };
    return std::make_unique<Processor>(std::forward<CallableT>(readFn));
  }
};

//===----------------------------------------------------------------------===//
// BytecodeReaderConfig
//===----------------------------------------------------------------------===//

/// A class containing bytecode-specific configurations of the `ParserConfig`.
class BytecodeReaderConfig {
public:
  BytecodeReaderConfig() = default;

  /// Returns the callbacks available to the parser.
  ArrayRef<std::unique_ptr<AttrTypeBytecodeReader<Attribute>>>
  getAttributeCallbacks() const {
    return attributeBytecodeParsers;
  }
  ArrayRef<std::unique_ptr<AttrTypeBytecodeReader<Type>>>
  getTypeCallbacks() const {
    return typeBytecodeParsers;
  }

  /// Attach a custom bytecode parser callback to the configuration for parsing
  /// of custom type/attributes encodings.
  void attachAttributeCallback(
      std::unique_ptr<AttrTypeBytecodeReader<Attribute>> parser) {
    attributeBytecodeParsers.emplace_back(std::move(parser));
  }
  void
  attachTypeCallback(std::unique_ptr<AttrTypeBytecodeReader<Type>> parser) {
    typeBytecodeParsers.emplace_back(std::move(parser));
  }

  /// Attach a custom bytecode parser callback to the configuration for parsing
  /// of custom type/attributes encodings.
  template <typename CallableT>
  std::enable_if_t<std::is_convertible_v<
      CallableT, std::function<LogicalResult(DialectBytecodeReader &, StringRef,
                                             Attribute &)>>>
  attachAttributeCallback(CallableT &&parserFn) {
    attachAttributeCallback(AttrTypeBytecodeReader<Attribute>::fromCallable(
        std::forward<CallableT>(parserFn)));
  }
  template <typename CallableT>
  std::enable_if_t<std::is_convertible_v<
      CallableT,
      std::function<LogicalResult(DialectBytecodeReader &, StringRef, Type &)>>>
  attachTypeCallback(CallableT &&parserFn) {
    attachTypeCallback(AttrTypeBytecodeReader<Type>::fromCallable(
        std::forward<CallableT>(parserFn)));
  }

private:
  llvm::SmallVector<std::unique_ptr<AttrTypeBytecodeReader<Attribute>>>
      attributeBytecodeParsers;
  llvm::SmallVector<std::unique_ptr<AttrTypeBytecodeReader<Type>>>
      typeBytecodeParsers;
};

} // namespace mlir

#endif // MLIR_BYTECODE_BYTECODEREADERCONFIG_H
