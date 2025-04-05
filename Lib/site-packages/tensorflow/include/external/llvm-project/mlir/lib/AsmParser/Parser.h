//===- Parser.h - MLIR Base Parser Class ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_ASMPARSER_PARSER_H
#define MLIR_LIB_ASMPARSER_PARSER_H

#include "ParserState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include <optional>

namespace mlir {
namespace detail {
//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

/// This class implement support for parsing global entities like attributes and
/// types. It is intended to be subclassed by specialized subparsers that
/// include state.
class Parser {
public:
  using Delimiter = OpAsmParser::Delimiter;

  Builder builder;

  Parser(ParserState &state)
      : builder(state.config.getContext()), state(state) {}

  // Helper methods to get stuff from the parser-global state.
  ParserState &getState() const { return state; }
  MLIRContext *getContext() const { return state.config.getContext(); }
  const llvm::SourceMgr &getSourceMgr() { return state.lex.getSourceMgr(); }

  /// Parse a comma-separated list of elements up until the specified end token.
  ParseResult
  parseCommaSeparatedListUntil(Token::Kind rightToken,
                               function_ref<ParseResult()> parseElement,
                               bool allowEmptyList = true);

  /// Parse a list of comma-separated items with an optional delimiter.  If a
  /// delimiter is provided, then an empty list is allowed.  If not, then at
  /// least one element will be parsed.
  ParseResult
  parseCommaSeparatedList(Delimiter delimiter,
                          function_ref<ParseResult()> parseElementFn,
                          StringRef contextMessage = StringRef());

  /// Parse a comma separated list of elements that must have at least one entry
  /// in it.
  ParseResult
  parseCommaSeparatedList(function_ref<ParseResult()> parseElementFn) {
    return parseCommaSeparatedList(Delimiter::None, parseElementFn);
  }

  /// Parse the body of a dialect symbol, which starts and ends with <>'s, and
  /// may be recursive. Return with the 'body' StringRef encompassing the entire
  /// body. `isCodeCompletion` is set to true if the body contained a code
  /// completion location, in which case the body is only populated up to the
  /// completion.
  ParseResult parseDialectSymbolBody(StringRef &body, bool &isCodeCompletion);
  ParseResult parseDialectSymbolBody(StringRef &body) {
    bool isCodeCompletion = false;
    return parseDialectSymbolBody(body, isCodeCompletion);
  }

  // We have two forms of parsing methods - those that return a non-null
  // pointer on success, and those that return a ParseResult to indicate whether
  // they returned a failure.  The second class fills in by-reference arguments
  // as the results of their action.

  //===--------------------------------------------------------------------===//
  // Error Handling
  //===--------------------------------------------------------------------===//

  /// Emit an error and return failure.
  InFlightDiagnostic emitError(const Twine &message = {});
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {});

  /// Emit an error about a "wrong token".  If the current token is at the
  /// start of a source line, this will apply heuristics to back up and report
  /// the error at the end of the previous line, which is where the expected
  /// token is supposed to be.
  InFlightDiagnostic emitWrongTokenError(const Twine &message = {});

  /// Encode the specified source location information into an attribute for
  /// attachment to the IR.
  Location getEncodedSourceLocation(SMLoc loc) {
    return state.lex.getEncodedSourceLocation(loc);
  }

  //===--------------------------------------------------------------------===//
  // Token Parsing
  //===--------------------------------------------------------------------===//

  /// Return the current token the parser is inspecting.
  const Token &getToken() const { return state.curToken; }
  StringRef getTokenSpelling() const { return state.curToken.getSpelling(); }

  /// Return the last parsed token.
  const Token &getLastToken() const { return state.lastToken; }

  /// If the current token has the specified kind, consume it and return true.
  /// If not, return false.
  bool consumeIf(Token::Kind kind) {
    if (state.curToken.isNot(kind))
      return false;
    consumeToken(kind);
    return true;
  }

  /// Advance the current lexer onto the next token.
  void consumeToken() {
    assert(state.curToken.isNot(Token::eof, Token::error) &&
           "shouldn't advance past EOF or errors");
    state.lastToken = state.curToken;
    state.curToken = state.lex.lexToken();
  }

  /// Advance the current lexer onto the next token, asserting what the expected
  /// current token is.  This is preferred to the above method because it leads
  /// to more self-documenting code with better checking.
  void consumeToken(Token::Kind kind) {
    assert(state.curToken.is(kind) && "consumed an unexpected token");
    consumeToken();
  }

  /// Reset the parser to the given lexer position. Resetting the parser/lexer
  /// position does not update 'state.lastToken'. 'state.lastToken' is the
  /// last parsed token, and is used to provide the scope end location for
  /// OperationDefinitions. To ensure the correctness of the end location, the
  /// last consumed token of an OperationDefinition needs to be the last token
  /// belonging to it.
  void resetToken(const char *tokPos) {
    state.lex.resetPointer(tokPos);
    state.curToken = state.lex.lexToken();
  }

  /// Consume the specified token if present and return success.  On failure,
  /// output a diagnostic and return failure.
  ParseResult parseToken(Token::Kind expectedToken, const Twine &message);

  /// Parse an optional integer value from the stream.
  OptionalParseResult parseOptionalInteger(APInt &result);

  /// Parse an optional integer value only in decimal format from the stream.
  OptionalParseResult parseOptionalDecimalInteger(APInt &result);

  /// Parse a floating point value from an integer literal token.
  ParseResult parseFloatFromIntegerLiteral(std::optional<APFloat> &result,
                                           const Token &tok, bool isNegative,
                                           const llvm::fltSemantics &semantics,
                                           size_t typeSizeInBits);

  /// Returns true if the current token corresponds to a keyword.
  bool isCurrentTokenAKeyword() const {
    return getToken().isAny(Token::bare_identifier, Token::inttype) ||
           getToken().isKeyword();
  }

  /// Parse a keyword, if present, into 'keyword'.
  ParseResult parseOptionalKeyword(StringRef *keyword);

  //===--------------------------------------------------------------------===//
  // Resource Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a handle to a dialect resource within the assembly format.
  FailureOr<AsmDialectResourceHandle>
  parseResourceHandle(const OpAsmDialectInterface *dialect, StringRef &name);
  FailureOr<AsmDialectResourceHandle> parseResourceHandle(Dialect *dialect);

  //===--------------------------------------------------------------------===//
  // Type Parsing
  //===--------------------------------------------------------------------===//

  /// Invoke the `getChecked` method of the given Attribute or Type class, using
  /// the provided location to emit errors in the case of failure. Note that
  /// unlike `OpBuilder::getType`, this method does not implicitly insert a
  /// context parameter.
  template <typename T, typename... ParamsT>
  T getChecked(SMLoc loc, ParamsT &&...params) {
    return T::getChecked([&] { return emitError(loc); },
                         std::forward<ParamsT>(params)...);
  }

  ParseResult parseFunctionResultTypes(SmallVectorImpl<Type> &elements);
  ParseResult parseTypeListNoParens(SmallVectorImpl<Type> &elements);
  ParseResult parseTypeListParens(SmallVectorImpl<Type> &elements);

  /// Optionally parse a type.
  OptionalParseResult parseOptionalType(Type &type);

  /// Parse an arbitrary type.
  Type parseType();

  /// Parse a complex type.
  Type parseComplexType();

  /// Parse an extended type.
  Type parseExtendedType();

  /// Parse a function type.
  Type parseFunctionType();

  /// Parse a memref type.
  Type parseMemRefType();

  /// Parse a non function type.
  Type parseNonFunctionType();

  /// Parse a tensor type.
  Type parseTensorType();

  /// Parse a tuple type.
  Type parseTupleType();

  /// Parse a vector type.
  VectorType parseVectorType();
  ParseResult parseVectorDimensionList(SmallVectorImpl<int64_t> &dimensions,
                                       SmallVectorImpl<bool> &scalableDims);
  ParseResult parseDimensionListRanked(SmallVectorImpl<int64_t> &dimensions,
                                       bool allowDynamic = true,
                                       bool withTrailingX = true);
  ParseResult parseIntegerInDimensionList(int64_t &value);
  ParseResult parseXInDimensionList();

  //===--------------------------------------------------------------------===//
  // Attribute Parsing
  //===--------------------------------------------------------------------===//

  /// Parse an arbitrary attribute with an optional type.
  Attribute parseAttribute(Type type = {});

  /// Parse an optional attribute with the provided type.
  OptionalParseResult parseOptionalAttribute(Attribute &attribute,
                                             Type type = {});
  OptionalParseResult parseOptionalAttribute(ArrayAttr &attribute, Type type);
  OptionalParseResult parseOptionalAttribute(StringAttr &attribute, Type type);
  OptionalParseResult parseOptionalAttribute(SymbolRefAttr &result, Type type);

  /// Parse an optional attribute that is demarcated by a specific token.
  template <typename AttributeT>
  OptionalParseResult parseOptionalAttributeWithToken(Token::Kind kind,
                                                      AttributeT &attr,
                                                      Type type = {}) {
    if (getToken().isNot(kind))
      return std::nullopt;

    if (Attribute parsedAttr = parseAttribute(type)) {
      attr = cast<AttributeT>(parsedAttr);
      return success();
    }
    return failure();
  }

  /// Parse an attribute dictionary.
  ParseResult parseAttributeDict(NamedAttrList &attributes);

  /// Parse a distinct attribute.
  Attribute parseDistinctAttr(Type type);

  /// Parse an extended attribute.
  Attribute parseExtendedAttr(Type type);

  /// Parse a float attribute.
  Attribute parseFloatAttr(Type type, bool isNegative);

  /// Parse a decimal or a hexadecimal literal, which can be either an integer
  /// or a float attribute.
  Attribute parseDecOrHexAttr(Type type, bool isNegative);

  /// Parse a dense elements attribute.
  Attribute parseDenseElementsAttr(Type attrType);
  ShapedType parseElementsLiteralType(Type type);

  /// Parse a dense resource elements attribute.
  Attribute parseDenseResourceElementsAttr(Type attrType);

  /// Parse a DenseArrayAttr.
  Attribute parseDenseArrayAttr(Type type);

  /// Parse a sparse elements attribute.
  Attribute parseSparseElementsAttr(Type attrType);

  /// Parse a strided layout attribute.
  Attribute parseStridedLayoutAttr();

  //===--------------------------------------------------------------------===//
  // Location Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a raw location instance.
  ParseResult parseLocationInstance(LocationAttr &loc);

  /// Parse a callsite location instance.
  ParseResult parseCallSiteLocation(LocationAttr &loc);

  /// Parse a fused location instance.
  ParseResult parseFusedLocation(LocationAttr &loc);

  /// Parse a name or FileLineCol location instance.
  ParseResult parseNameOrFileLineColLocation(LocationAttr &loc);

  //===--------------------------------------------------------------------===//
  // Affine Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a reference to either an affine map, expr, or an integer set.
  ParseResult parseAffineMapOrIntegerSetReference(AffineMap &map,
                                                  IntegerSet &set);
  ParseResult parseAffineMapReference(AffineMap &map);
  ParseResult
  parseAffineExprReference(ArrayRef<std::pair<StringRef, AffineExpr>> symbolSet,
                           AffineExpr &expr);
  ParseResult parseIntegerSetReference(IntegerSet &set);

  /// Parse an AffineMap where the dim and symbol identifiers are SSA ids.
  ParseResult
  parseAffineMapOfSSAIds(AffineMap &map,
                         function_ref<ParseResult(bool)> parseElement,
                         Delimiter delimiter);

  /// Parse an AffineExpr where dim and symbol identifiers are SSA ids.
  ParseResult
  parseAffineExprOfSSAIds(AffineExpr &expr,
                          function_ref<ParseResult(bool)> parseElement);

  //===--------------------------------------------------------------------===//
  // Code Completion
  //===--------------------------------------------------------------------===//

  /// The set of various code completion methods. Every completion method
  /// returns `failure` to signal that parsing should abort after any desired
  /// completions have been enqueued. Note that `failure` is does not mean
  /// completion failed, it's just a signal to the parser to stop.

  ParseResult codeCompleteDialectName();
  ParseResult codeCompleteOperationName(StringRef dialectName);
  ParseResult codeCompleteDialectOrElidedOpName(SMLoc loc);
  ParseResult codeCompleteStringDialectOrOperationName(StringRef name);
  ParseResult codeCompleteExpectedTokens(ArrayRef<StringRef> tokens);
  ParseResult codeCompleteOptionalTokens(ArrayRef<StringRef> tokens);

  Attribute codeCompleteAttribute();
  Type codeCompleteType();
  Attribute
  codeCompleteDialectSymbol(const llvm::StringMap<Attribute> &aliases);
  Type codeCompleteDialectSymbol(const llvm::StringMap<Type> &aliases);

protected:
  /// The Parser is subclassed and reinstantiated.  Do not add additional
  /// non-trivial state here, add it to the ParserState class.
  ParserState &state;
};
} // namespace detail
} // namespace mlir

#endif // MLIR_LIB_ASMPARSER_PARSER_H
