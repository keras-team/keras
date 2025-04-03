//===- CodeComplete.h - MLIR Asm CodeComplete Context -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ASMPARSER_CODECOMPLETE_H
#define MLIR_ASMPARSER_CODECOMPLETE_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {
class Attribute;
class Type;

/// This class provides an abstract interface into the parser for hooking in
/// code completion events. This class is only really useful for providing
/// language tooling for MLIR, general clients should not need to use this
/// class.
class AsmParserCodeCompleteContext {
public:
  virtual ~AsmParserCodeCompleteContext();

  /// Return the source location used to provide code completion.
  SMLoc getCodeCompleteLoc() const { return codeCompleteLoc; }

  //===--------------------------------------------------------------------===//
  // Completion Hooks
  //===--------------------------------------------------------------------===//

  /// Signal code completion for a dialect name, with an optional prefix.
  virtual void completeDialectName(StringRef prefix) = 0;
  void completeDialectName() { completeDialectName(""); }

  /// Signal code completion for an operation name within the given dialect.
  virtual void completeOperationName(StringRef dialectName) = 0;

  /// Append the given SSA value as a code completion result for SSA value
  /// completions.
  virtual void appendSSAValueCompletion(StringRef name,
                                        std::string typeData) = 0;

  /// Append the given block as a code completion result for block name
  /// completions.
  virtual void appendBlockCompletion(StringRef name) = 0;

  /// Signal a completion for the given expected tokens, which are optional if
  /// `optional` is set.
  virtual void completeExpectedTokens(ArrayRef<StringRef> tokens,
                                      bool optional) = 0;

  /// Signal a completion for an attribute.
  virtual void completeAttribute(const llvm::StringMap<Attribute> &aliases) = 0;
  virtual void completeDialectAttributeOrAlias(
      const llvm::StringMap<Attribute> &aliases) = 0;

  /// Signal a completion for a type.
  virtual void completeType(const llvm::StringMap<Type> &aliases) = 0;
  virtual void
  completeDialectTypeOrAlias(const llvm::StringMap<Type> &aliases) = 0;

protected:
  /// Create a new code completion context with the given code complete
  /// location.
  explicit AsmParserCodeCompleteContext(SMLoc codeCompleteLoc)
      : codeCompleteLoc(codeCompleteLoc) {}

private:
  /// The location used to code complete.
  SMLoc codeCompleteLoc;
};
} // namespace mlir

#endif // MLIR_ASMPARSER_CODECOMPLETE_H
