//===- DimLvlMapParser.h - `DimLvlMap` parser -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_DIMLVLMAPPARSER_H
#define MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_DIMLVLMAPPARSER_H

#include "DimLvlMap.h"
#include "LvlTypeParser.h"

namespace mlir {
namespace sparse_tensor {
namespace ir_detail {

///
/// Parses the Sparse Tensor Encoding Attribute (STEA).
///
/// General syntax is as follows,
///
///   [s0, ...]     // optional forward decl sym-vars
///   {l0, ...}     // optional forward decl lvl-vars
///   (
///     d0 = ...,   // dim-var = dim-exp
///     ...
///   ) -> (
///     l0 = ...,   // lvl-var = lvl-exp
///     ...
///   )
///
/// with simplifications when variables are implicit.
///
class DimLvlMapParser final {
public:
  explicit DimLvlMapParser(AsmParser &parser) : parser(parser) {}

  // Parses the input for a sparse tensor dimension-level map
  // and returns the map on success.
  FailureOr<DimLvlMap> parseDimLvlMap();

private:
  /// Client code should prefer using `parseVarUsage`
  /// and `parseVarBinding` rather than calling this method directly.
  OptionalParseResult parseVar(VarKind vk, bool isOptional,
                               Policy creationPolicy, VarInfo::ID &id,
                               bool &didCreate);

  /// Parses a variable occurence which is a *use* of that variable.
  /// When a valid variable name is currently unused, if
  /// `requireKnown=true`, an error is raised; if `requireKnown=false`,
  /// a new unbound variable will be created.
  FailureOr<VarInfo::ID> parseVarUsage(VarKind vk, bool requireKnown);

  /// Parses a variable occurence which is a *binding* of that variable.
  /// The `requireKnown` parameter is for handling the binding of
  /// forward-declared variables.
  FailureOr<VarInfo::ID> parseVarBinding(VarKind vk, bool requireKnown = false);

  /// Parses an optional variable binding. When the next token is
  /// not a valid variable name, this will bind a new unnamed variable.
  /// The returned `bool` indicates whether a variable name was parsed.
  FailureOr<std::pair<Var, bool>>
  parseOptionalVarBinding(VarKind vk, bool requireKnown = false);

  /// Binds the given variable: both updating the `VarEnv` itself, and
  /// the `{dims,lvls}AndSymbols` lists (which will be passed
  /// to `AsmParser::parseAffineExpr`). This method is already called by the
  /// `parseVarBinding`/`parseOptionalVarBinding` methods, therefore should
  /// not need to be called elsewhere.
  Var bindVar(llvm::SMLoc loc, VarInfo::ID id);

  ParseResult parseSymbolBindingList();
  ParseResult parseLvlVarBindingList();
  ParseResult parseDimSpec();
  ParseResult parseDimSpecList();
  FailureOr<LvlVar> parseLvlVarBinding(bool requireLvlVarBinding);
  ParseResult parseLvlSpec(bool requireLvlVarBinding);
  ParseResult parseLvlSpecList();

  AsmParser &parser;
  LvlTypeParser lvlTypeParser;
  VarEnv env;
  // The parser maintains the `{dims,lvls}AndSymbols` lists to avoid
  // the O(n^2) cost of repeatedly constructing them inside of the
  // `parse{Dim,Lvl}Spec` methods.
  SmallVector<std::pair<StringRef, AffineExpr>, 4> dimsAndSymbols;
  SmallVector<std::pair<StringRef, AffineExpr>, 4> lvlsAndSymbols;
  SmallVector<DimSpec> dimSpecs;
  SmallVector<LvlSpec> lvlSpecs;
};

} // namespace ir_detail
} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_DETAIL_DIMLVLMAPPARSER_H
