//===- OpImplementation.h - Classes for implementing Op types ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This classes used by the implementation details of Op types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_OPIMPLEMENTATION_H
#define MLIR_IR_OPIMPLEMENTATION_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SMLoc.h"
#include <optional>

namespace mlir {
class AsmParsedResourceEntry;
class AsmResourceBuilder;
class Builder;

//===----------------------------------------------------------------------===//
// AsmDialectResourceHandle
//===----------------------------------------------------------------------===//

/// This class represents an opaque handle to a dialect resource entry.
class AsmDialectResourceHandle {
public:
  AsmDialectResourceHandle() = default;
  AsmDialectResourceHandle(void *resource, TypeID resourceID, Dialect *dialect)
      : resource(resource), opaqueID(resourceID), dialect(dialect) {}
  bool operator==(const AsmDialectResourceHandle &other) const {
    return resource == other.resource;
  }

  /// Return an opaque pointer to the referenced resource.
  void *getResource() const { return resource; }

  /// Return the type ID of the resource.
  TypeID getTypeID() const { return opaqueID; }

  /// Return the dialect that owns the resource.
  Dialect *getDialect() const { return dialect; }

private:
  /// The opaque handle to the dialect resource.
  void *resource = nullptr;
  /// The type of the resource referenced.
  TypeID opaqueID;
  /// The dialect owning the given resource.
  Dialect *dialect;
};

/// This class represents a CRTP base class for dialect resource handles. It
/// abstracts away various utilities necessary for defined derived resource
/// handles.
template <typename DerivedT, typename ResourceT, typename DialectT>
class AsmDialectResourceHandleBase : public AsmDialectResourceHandle {
public:
  using Dialect = DialectT;

  /// Construct a handle from a pointer to the resource. The given pointer
  /// should be guaranteed to live beyond the life of this handle.
  AsmDialectResourceHandleBase(ResourceT *resource, DialectT *dialect)
      : AsmDialectResourceHandle(resource, TypeID::get<DerivedT>(), dialect) {}
  AsmDialectResourceHandleBase(AsmDialectResourceHandle handle)
      : AsmDialectResourceHandle(handle) {
    assert(handle.getTypeID() == TypeID::get<DerivedT>());
  }

  /// Return the resource referenced by this handle.
  ResourceT *getResource() {
    return static_cast<ResourceT *>(AsmDialectResourceHandle::getResource());
  }
  const ResourceT *getResource() const {
    return const_cast<AsmDialectResourceHandleBase *>(this)->getResource();
  }

  /// Return the dialect that owns the resource.
  DialectT *getDialect() const {
    return static_cast<DialectT *>(AsmDialectResourceHandle::getDialect());
  }

  /// Support llvm style casting.
  static bool classof(const AsmDialectResourceHandle *handle) {
    return handle->getTypeID() == TypeID::get<DerivedT>();
  }
};

inline llvm::hash_code hash_value(const AsmDialectResourceHandle &param) {
  return llvm::hash_value(param.getResource());
}

//===----------------------------------------------------------------------===//
// AsmPrinter
//===----------------------------------------------------------------------===//

/// This base class exposes generic asm printer hooks, usable across the various
/// derived printers.
class AsmPrinter {
public:
  /// This class contains the internal default implementation of the base
  /// printer methods.
  class Impl;

  /// Initialize the printer with the given internal implementation.
  AsmPrinter(Impl &impl) : impl(&impl) {}
  virtual ~AsmPrinter();

  /// Return the raw output stream used by this printer.
  virtual raw_ostream &getStream() const;

  /// Print the given floating point value in a stabilized form that can be
  /// roundtripped through the IR. This is the companion to the 'parseFloat'
  /// hook on the AsmParser.
  virtual void printFloat(const APFloat &value);

  virtual void printType(Type type);
  virtual void printAttribute(Attribute attr);

  /// Trait to check if `AttrType` provides a `print` method.
  template <typename AttrOrType>
  using has_print_method =
      decltype(std::declval<AttrOrType>().print(std::declval<AsmPrinter &>()));
  template <typename AttrOrType>
  using detect_has_print_method =
      llvm::is_detected<has_print_method, AttrOrType>;

  /// Print the provided attribute in the context of an operation custom
  /// printer/parser: this will invoke directly the print method on the
  /// attribute class and skip the `#dialect.mnemonic` prefix in most cases.
  template <typename AttrOrType,
            std::enable_if_t<detect_has_print_method<AttrOrType>::value>
                *sfinae = nullptr>
  void printStrippedAttrOrType(AttrOrType attrOrType) {
    if (succeeded(printAlias(attrOrType)))
      return;

    raw_ostream &os = getStream();
    uint64_t posPrior = os.tell();
    attrOrType.print(*this);
    if (posPrior != os.tell())
      return;

    // Fallback to printing with prefix if the above failed to write anything
    // to the output stream.
    *this << attrOrType;
  }

  /// Print the provided array of attributes or types in the context of an
  /// operation custom printer/parser: this will invoke directly the print
  /// method on the attribute class and skip the `#dialect.mnemonic` prefix in
  /// most cases.
  template <typename AttrOrType,
            std::enable_if_t<detect_has_print_method<AttrOrType>::value>
                *sfinae = nullptr>
  void printStrippedAttrOrType(ArrayRef<AttrOrType> attrOrTypes) {
    llvm::interleaveComma(
        attrOrTypes, getStream(),
        [this](AttrOrType attrOrType) { printStrippedAttrOrType(attrOrType); });
  }

  /// SFINAE for printing the provided attribute in the context of an operation
  /// custom printer in the case where the attribute does not define a print
  /// method.
  template <typename AttrOrType,
            std::enable_if_t<!detect_has_print_method<AttrOrType>::value>
                *sfinae = nullptr>
  void printStrippedAttrOrType(AttrOrType attrOrType) {
    *this << attrOrType;
  }

  /// Print the given attribute without its type. The corresponding parser must
  /// provide a valid type for the attribute.
  virtual void printAttributeWithoutType(Attribute attr);

  /// Print the alias for the given attribute, return failure if no alias could
  /// be printed.
  virtual LogicalResult printAlias(Attribute attr);

  /// Print the alias for the given type, return failure if no alias could
  /// be printed.
  virtual LogicalResult printAlias(Type type);

  /// Print the given string as a keyword, or a quoted and escaped string if it
  /// has any special or non-printable characters in it.
  virtual void printKeywordOrString(StringRef keyword);

  /// Print the given string as a quoted string, escaping any special or
  /// non-printable characters in it.
  virtual void printString(StringRef string);

  /// Print the given string as a symbol reference, i.e. a form representable by
  /// a SymbolRefAttr. A symbol reference is represented as a string prefixed
  /// with '@'. The reference is surrounded with ""'s and escaped if it has any
  /// special or non-printable characters in it.
  virtual void printSymbolName(StringRef symbolRef);

  /// Print a handle to the given dialect resource.
  virtual void printResourceHandle(const AsmDialectResourceHandle &resource);

  /// Print an optional arrow followed by a type list.
  template <typename TypeRange>
  void printOptionalArrowTypeList(TypeRange &&types) {
    if (types.begin() != types.end())
      printArrowTypeList(types);
  }
  template <typename TypeRange>
  void printArrowTypeList(TypeRange &&types) {
    auto &os = getStream() << " -> ";

    bool wrapped = !llvm::hasSingleElement(types) ||
                   llvm::isa<FunctionType>((*types.begin()));
    if (wrapped)
      os << '(';
    llvm::interleaveComma(types, *this);
    if (wrapped)
      os << ')';
  }

  /// Print the two given type ranges in a functional form.
  template <typename InputRangeT, typename ResultRangeT>
  void printFunctionalType(InputRangeT &&inputs, ResultRangeT &&results) {
    auto &os = getStream();
    os << '(';
    llvm::interleaveComma(inputs, *this);
    os << ')';
    printArrowTypeList(results);
  }

  void printDimensionList(ArrayRef<int64_t> shape);

  /// Class used to automatically end a cyclic region on destruction.
  class CyclicPrintReset {
  public:
    explicit CyclicPrintReset(AsmPrinter *printer) : printer(printer) {}

    ~CyclicPrintReset() {
      if (printer)
        printer->popCyclicPrinting();
    }

    CyclicPrintReset(const CyclicPrintReset &) = delete;

    CyclicPrintReset &operator=(const CyclicPrintReset &) = delete;

    CyclicPrintReset(CyclicPrintReset &&rhs)
        : printer(std::exchange(rhs.printer, nullptr)) {}

    CyclicPrintReset &operator=(CyclicPrintReset &&rhs) {
      printer = std::exchange(rhs.printer, nullptr);
      return *this;
    }

  private:
    AsmPrinter *printer;
  };

  /// Attempts to start a cyclic printing region for `attrOrType`.
  /// A cyclic printing region starts with this call and ends with the
  /// destruction of the returned `CyclicPrintReset`. During this time,
  /// calling `tryStartCyclicPrint` with the same attribute in any printer
  /// will lead to returning failure.
  ///
  /// This makes it possible to break infinite recursions when trying to print
  /// cyclic attributes or types by printing only immutable parameters if nested
  /// within itself.
  template <class AttrOrTypeT>
  FailureOr<CyclicPrintReset> tryStartCyclicPrint(AttrOrTypeT attrOrType) {
    static_assert(
        std::is_base_of_v<AttributeTrait::IsMutable<AttrOrTypeT>,
                          AttrOrTypeT> ||
            std::is_base_of_v<TypeTrait::IsMutable<AttrOrTypeT>, AttrOrTypeT>,
        "Only mutable attributes or types can be cyclic");
    if (failed(pushCyclicPrinting(attrOrType.getAsOpaquePointer())))
      return failure();
    return CyclicPrintReset(this);
  }

protected:
  /// Initialize the printer with no internal implementation. In this case, all
  /// virtual methods of this class must be overriden.
  AsmPrinter() = default;

  /// Pushes a new attribute or type in the form of a type erased pointer
  /// into an internal set.
  /// Returns success if the type or attribute was inserted in the set or
  /// failure if it was already contained.
  virtual LogicalResult pushCyclicPrinting(const void *opaquePointer);

  /// Removes the element that was last inserted with a successful call to
  /// `pushCyclicPrinting`. There must be exactly one `popCyclicPrinting` call
  /// in reverse order of all successful `pushCyclicPrinting`.
  virtual void popCyclicPrinting();

private:
  AsmPrinter(const AsmPrinter &) = delete;
  void operator=(const AsmPrinter &) = delete;

  /// The internal implementation of the printer.
  Impl *impl{nullptr};
};

template <typename AsmPrinterT>
inline std::enable_if_t<std::is_base_of<AsmPrinter, AsmPrinterT>::value,
                        AsmPrinterT &>
operator<<(AsmPrinterT &p, Type type) {
  p.printType(type);
  return p;
}

template <typename AsmPrinterT>
inline std::enable_if_t<std::is_base_of<AsmPrinter, AsmPrinterT>::value,
                        AsmPrinterT &>
operator<<(AsmPrinterT &p, Attribute attr) {
  p.printAttribute(attr);
  return p;
}

template <typename AsmPrinterT>
inline std::enable_if_t<std::is_base_of<AsmPrinter, AsmPrinterT>::value,
                        AsmPrinterT &>
operator<<(AsmPrinterT &p, const APFloat &value) {
  p.printFloat(value);
  return p;
}
template <typename AsmPrinterT>
inline std::enable_if_t<std::is_base_of<AsmPrinter, AsmPrinterT>::value,
                        AsmPrinterT &>
operator<<(AsmPrinterT &p, float value) {
  return p << APFloat(value);
}
template <typename AsmPrinterT>
inline std::enable_if_t<std::is_base_of<AsmPrinter, AsmPrinterT>::value,
                        AsmPrinterT &>
operator<<(AsmPrinterT &p, double value) {
  return p << APFloat(value);
}

// Support printing anything that isn't convertible to one of the other
// streamable types, even if it isn't exactly one of them. For example, we want
// to print FunctionType with the Type version above, not have it match this.
template <typename AsmPrinterT, typename T,
          std::enable_if_t<!std::is_convertible<T &, Value &>::value &&
                               !std::is_convertible<T &, Type &>::value &&
                               !std::is_convertible<T &, Attribute &>::value &&
                               !std::is_convertible<T &, ValueRange>::value &&
                               !std::is_convertible<T &, APFloat &>::value &&
                               !llvm::is_one_of<T, bool, float, double>::value,
                           T> * = nullptr>
inline std::enable_if_t<std::is_base_of<AsmPrinter, AsmPrinterT>::value,
                        AsmPrinterT &>
operator<<(AsmPrinterT &p, const T &other) {
  p.getStream() << other;
  return p;
}

template <typename AsmPrinterT>
inline std::enable_if_t<std::is_base_of<AsmPrinter, AsmPrinterT>::value,
                        AsmPrinterT &>
operator<<(AsmPrinterT &p, bool value) {
  return p << (value ? StringRef("true") : "false");
}

template <typename AsmPrinterT, typename ValueRangeT>
inline std::enable_if_t<std::is_base_of<AsmPrinter, AsmPrinterT>::value,
                        AsmPrinterT &>
operator<<(AsmPrinterT &p, const ValueTypeRange<ValueRangeT> &types) {
  llvm::interleaveComma(types, p);
  return p;
}

template <typename AsmPrinterT>
inline std::enable_if_t<std::is_base_of<AsmPrinter, AsmPrinterT>::value,
                        AsmPrinterT &>
operator<<(AsmPrinterT &p, const TypeRange &types) {
  llvm::interleaveComma(types, p);
  return p;
}

// Prevent matching the TypeRange version above for ValueRange
// printing through base AsmPrinter. This is needed so that the
// ValueRange printing behaviour does not change from printing
// the SSA values to printing the types for the operands when
// using AsmPrinter instead of OpAsmPrinter.
template <typename AsmPrinterT, typename T>
inline std::enable_if_t<std::is_same<AsmPrinter, AsmPrinterT>::value &&
                            std::is_convertible<T &, ValueRange>::value,
                        AsmPrinterT &>
operator<<(AsmPrinterT &p, const T &other) = delete;

template <typename AsmPrinterT, typename ElementT>
inline std::enable_if_t<std::is_base_of<AsmPrinter, AsmPrinterT>::value,
                        AsmPrinterT &>
operator<<(AsmPrinterT &p, ArrayRef<ElementT> types) {
  llvm::interleaveComma(types, p);
  return p;
}

//===----------------------------------------------------------------------===//
// OpAsmPrinter
//===----------------------------------------------------------------------===//

/// This is a pure-virtual base class that exposes the asmprinter hooks
/// necessary to implement a custom print() method.
class OpAsmPrinter : public AsmPrinter {
public:
  using AsmPrinter::AsmPrinter;
  ~OpAsmPrinter() override;

  /// Print a loc(...) specifier if printing debug info is enabled.
  virtual void printOptionalLocationSpecifier(Location loc) = 0;

  /// Print a newline and indent the printer to the start of the current
  /// operation.
  virtual void printNewline() = 0;

  /// Increase indentation.
  virtual void increaseIndent() = 0;

  /// Decrease indentation.
  virtual void decreaseIndent() = 0;

  /// Print a block argument in the usual format of:
  ///   %ssaName : type {attr1=42} loc("here")
  /// where location printing is controlled by the standard internal option.
  /// You may pass omitType=true to not print a type, and pass an empty
  /// attribute list if you don't care for attributes.
  virtual void printRegionArgument(BlockArgument arg,
                                   ArrayRef<NamedAttribute> argAttrs = {},
                                   bool omitType = false) = 0;

  /// Print implementations for various things an operation contains.
  virtual void printOperand(Value value) = 0;
  virtual void printOperand(Value value, raw_ostream &os) = 0;

  /// Print a comma separated list of operands.
  template <typename ContainerType>
  void printOperands(const ContainerType &container) {
    printOperands(container.begin(), container.end());
  }

  /// Print a comma separated list of operands.
  template <typename IteratorType>
  void printOperands(IteratorType it, IteratorType end) {
    llvm::interleaveComma(llvm::make_range(it, end), getStream(),
                          [this](Value value) { printOperand(value); });
  }

  /// Print the given successor.
  virtual void printSuccessor(Block *successor) = 0;

  /// Print the successor and its operands.
  virtual void printSuccessorAndUseList(Block *successor,
                                        ValueRange succOperands) = 0;

  /// If the specified operation has attributes, print out an attribute
  /// dictionary with their values.  elidedAttrs allows the client to ignore
  /// specific well known attributes, commonly used if the attribute value is
  /// printed some other way (like as a fixed operand).
  virtual void printOptionalAttrDict(ArrayRef<NamedAttribute> attrs,
                                     ArrayRef<StringRef> elidedAttrs = {}) = 0;

  /// If the specified operation has attributes, print out an attribute
  /// dictionary prefixed with 'attributes'.
  virtual void
  printOptionalAttrDictWithKeyword(ArrayRef<NamedAttribute> attrs,
                                   ArrayRef<StringRef> elidedAttrs = {}) = 0;

  /// Prints the entire operation with the custom assembly form, if available,
  /// or the generic assembly form, otherwise.
  virtual void printCustomOrGenericOp(Operation *op) = 0;

  /// Print the entire operation with the default generic assembly form.
  /// If `printOpName` is true, then the operation name is printed (the default)
  /// otherwise it is omitted and the print will start with the operand list.
  virtual void printGenericOp(Operation *op, bool printOpName = true) = 0;

  /// Prints a region.
  /// If 'printEntryBlockArgs' is false, the arguments of the
  /// block are not printed. If 'printBlockTerminator' is false, the terminator
  /// operation of the block is not printed. If printEmptyBlock is true, then
  /// the block header is printed even if the block is empty.
  virtual void printRegion(Region &blocks, bool printEntryBlockArgs = true,
                           bool printBlockTerminators = true,
                           bool printEmptyBlock = false) = 0;

  /// Renumber the arguments for the specified region to the same names as the
  /// SSA values in namesToUse.  This may only be used for IsolatedFromAbove
  /// operations.  If any entry in namesToUse is null, the corresponding
  /// argument name is left alone.
  virtual void shadowRegionArgs(Region &region, ValueRange namesToUse) = 0;

  /// Prints an affine map of SSA ids, where SSA id names are used in place
  /// of dims/symbols.
  /// Operand values must come from single-result sources, and be valid
  /// dimensions/symbol identifiers according to mlir::isValidDim/Symbol.
  virtual void printAffineMapOfSSAIds(AffineMapAttr mapAttr,
                                      ValueRange operands) = 0;

  /// Prints an affine expression of SSA ids with SSA id names used instead of
  /// dims and symbols.
  /// Operand values must come from single-result sources, and be valid
  /// dimensions/symbol identifiers according to mlir::isValidDim/Symbol.
  virtual void printAffineExprOfSSAIds(AffineExpr expr, ValueRange dimOperands,
                                       ValueRange symOperands) = 0;

  /// Print the complete type of an operation in functional form.
  void printFunctionalType(Operation *op);
  using AsmPrinter::printFunctionalType;
};

// Make the implementations convenient to use.
inline OpAsmPrinter &operator<<(OpAsmPrinter &p, Value value) {
  p.printOperand(value);
  return p;
}

template <typename T,
          std::enable_if_t<std::is_convertible<T &, ValueRange>::value &&
                               !std::is_convertible<T &, Value &>::value,
                           T> * = nullptr>
inline OpAsmPrinter &operator<<(OpAsmPrinter &p, const T &values) {
  p.printOperands(values);
  return p;
}

inline OpAsmPrinter &operator<<(OpAsmPrinter &p, Block *value) {
  p.printSuccessor(value);
  return p;
}

//===----------------------------------------------------------------------===//
// AsmParser
//===----------------------------------------------------------------------===//

/// This base class exposes generic asm parser hooks, usable across the various
/// derived parsers.
class AsmParser {
public:
  AsmParser() = default;
  virtual ~AsmParser();

  MLIRContext *getContext() const;

  /// Return the location of the original name token.
  virtual SMLoc getNameLoc() const = 0;

  //===--------------------------------------------------------------------===//
  // Utilities
  //===--------------------------------------------------------------------===//

  /// Emit a diagnostic at the specified location and return failure.
  virtual InFlightDiagnostic emitError(SMLoc loc,
                                       const Twine &message = {}) = 0;

  /// Return a builder which provides useful access to MLIRContext, global
  /// objects like types and attributes.
  virtual Builder &getBuilder() const = 0;

  /// Get the location of the next token and store it into the argument.  This
  /// always succeeds.
  virtual SMLoc getCurrentLocation() = 0;
  ParseResult getCurrentLocation(SMLoc *loc) {
    *loc = getCurrentLocation();
    return success();
  }

  /// Re-encode the given source location as an MLIR location and return it.
  /// Note: This method should only be used when a `Location` is necessary, as
  /// the encoding process is not efficient.
  virtual Location getEncodedSourceLoc(SMLoc loc) = 0;

  //===--------------------------------------------------------------------===//
  // Token Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a '->' token.
  virtual ParseResult parseArrow() = 0;

  /// Parse a '->' token if present
  virtual ParseResult parseOptionalArrow() = 0;

  /// Parse a `{` token.
  virtual ParseResult parseLBrace() = 0;

  /// Parse a `{` token if present.
  virtual ParseResult parseOptionalLBrace() = 0;

  /// Parse a `}` token.
  virtual ParseResult parseRBrace() = 0;

  /// Parse a `}` token if present.
  virtual ParseResult parseOptionalRBrace() = 0;

  /// Parse a `:` token.
  virtual ParseResult parseColon() = 0;

  /// Parse a `:` token if present.
  virtual ParseResult parseOptionalColon() = 0;

  /// Parse a `,` token.
  virtual ParseResult parseComma() = 0;

  /// Parse a `,` token if present.
  virtual ParseResult parseOptionalComma() = 0;

  /// Parse a `=` token.
  virtual ParseResult parseEqual() = 0;

  /// Parse a `=` token if present.
  virtual ParseResult parseOptionalEqual() = 0;

  /// Parse a '<' token.
  virtual ParseResult parseLess() = 0;

  /// Parse a '<' token if present.
  virtual ParseResult parseOptionalLess() = 0;

  /// Parse a '>' token.
  virtual ParseResult parseGreater() = 0;

  /// Parse a '>' token if present.
  virtual ParseResult parseOptionalGreater() = 0;

  /// Parse a '?' token.
  virtual ParseResult parseQuestion() = 0;

  /// Parse a '?' token if present.
  virtual ParseResult parseOptionalQuestion() = 0;

  /// Parse a '+' token.
  virtual ParseResult parsePlus() = 0;

  /// Parse a '+' token if present.
  virtual ParseResult parseOptionalPlus() = 0;

  /// Parse a '-' token.
  virtual ParseResult parseMinus() = 0;

  /// Parse a '-' token if present.
  virtual ParseResult parseOptionalMinus() = 0;

  /// Parse a '*' token.
  virtual ParseResult parseStar() = 0;

  /// Parse a '*' token if present.
  virtual ParseResult parseOptionalStar() = 0;

  /// Parse a '|' token.
  virtual ParseResult parseVerticalBar() = 0;

  /// Parse a '|' token if present.
  virtual ParseResult parseOptionalVerticalBar() = 0;

  /// Parse a quoted string token.
  ParseResult parseString(std::string *string) {
    auto loc = getCurrentLocation();
    if (parseOptionalString(string))
      return emitError(loc, "expected string");
    return success();
  }

  /// Parse a quoted string token if present.
  virtual ParseResult parseOptionalString(std::string *string) = 0;

  /// Parses a Base64 encoded string of bytes.
  virtual ParseResult parseBase64Bytes(std::vector<char> *bytes) = 0;

  /// Parse a `(` token.
  virtual ParseResult parseLParen() = 0;

  /// Parse a `(` token if present.
  virtual ParseResult parseOptionalLParen() = 0;

  /// Parse a `)` token.
  virtual ParseResult parseRParen() = 0;

  /// Parse a `)` token if present.
  virtual ParseResult parseOptionalRParen() = 0;

  /// Parse a `[` token.
  virtual ParseResult parseLSquare() = 0;

  /// Parse a `[` token if present.
  virtual ParseResult parseOptionalLSquare() = 0;

  /// Parse a `]` token.
  virtual ParseResult parseRSquare() = 0;

  /// Parse a `]` token if present.
  virtual ParseResult parseOptionalRSquare() = 0;

  /// Parse a `...` token.
  virtual ParseResult parseEllipsis() = 0;

  /// Parse a `...` token if present;
  virtual ParseResult parseOptionalEllipsis() = 0;

  /// Parse a floating point value from the stream.
  virtual ParseResult parseFloat(double &result) = 0;

  /// Parse a floating point value into APFloat from the stream.
  virtual ParseResult parseFloat(const llvm::fltSemantics &semantics,
                                 APFloat &result) = 0;

  /// Parse an integer value from the stream.
  template <typename IntT>
  ParseResult parseInteger(IntT &result) {
    auto loc = getCurrentLocation();
    OptionalParseResult parseResult = parseOptionalInteger(result);
    if (!parseResult.has_value())
      return emitError(loc, "expected integer value");
    return *parseResult;
  }

  /// Parse a decimal integer value from the stream.
  template <typename IntT>
  ParseResult parseDecimalInteger(IntT &result) {
    auto loc = getCurrentLocation();
    OptionalParseResult parseResult = parseOptionalDecimalInteger(result);
    if (!parseResult.has_value())
      return emitError(loc, "expected decimal integer value");
    return *parseResult;
  }

  /// Parse an optional integer value from the stream.
  virtual OptionalParseResult parseOptionalInteger(APInt &result) = 0;
  virtual OptionalParseResult parseOptionalDecimalInteger(APInt &result) = 0;

 private:
  template <typename IntT, typename ParseFn>
  OptionalParseResult parseOptionalIntegerAndCheck(IntT &result,
                                                   ParseFn &&parseFn) {
    auto loc = getCurrentLocation();
    APInt uintResult;
    OptionalParseResult parseResult = parseFn(uintResult);
    if (!parseResult.has_value() || failed(*parseResult))
      return parseResult;

    // Try to convert to the provided integer type.  sextOrTrunc is correct even
    // for unsigned types because parseOptionalInteger ensures the sign bit is
    // zero for non-negated integers.
    result =
        (IntT)uintResult.sextOrTrunc(sizeof(IntT) * CHAR_BIT).getLimitedValue();
    if (APInt(uintResult.getBitWidth(), result) != uintResult)
      return emitError(loc, "integer value too large");
    return success();
  }

 public:
  template <typename IntT>
  OptionalParseResult parseOptionalInteger(IntT &result) {
    return parseOptionalIntegerAndCheck(
        result, [&](APInt &result) { return parseOptionalInteger(result); });
  }

  template <typename IntT>
  OptionalParseResult parseOptionalDecimalInteger(IntT &result) {
    return parseOptionalIntegerAndCheck(result, [&](APInt &result) {
      return parseOptionalDecimalInteger(result);
    });
  }

  /// These are the supported delimiters around operand lists and region
  /// argument lists, used by parseOperandList.
  enum class Delimiter {
    /// Zero or more operands with no delimiters.
    None,
    /// Parens surrounding zero or more operands.
    Paren,
    /// Square brackets surrounding zero or more operands.
    Square,
    /// <> brackets surrounding zero or more operands.
    LessGreater,
    /// {} brackets surrounding zero or more operands.
    Braces,
    /// Parens supporting zero or more operands, or nothing.
    OptionalParen,
    /// Square brackets supporting zero or more ops, or nothing.
    OptionalSquare,
    /// <> brackets supporting zero or more ops, or nothing.
    OptionalLessGreater,
    /// {} brackets surrounding zero or more operands, or nothing.
    OptionalBraces,
  };

  /// Parse a list of comma-separated items with an optional delimiter.  If a
  /// delimiter is provided, then an empty list is allowed.  If not, then at
  /// least one element will be parsed.
  ///
  /// contextMessage is an optional message appended to "expected '('" sorts of
  /// diagnostics when parsing the delimeters.
  virtual ParseResult
  parseCommaSeparatedList(Delimiter delimiter,
                          function_ref<ParseResult()> parseElementFn,
                          StringRef contextMessage = StringRef()) = 0;

  /// Parse a comma separated list of elements that must have at least one entry
  /// in it.
  ParseResult
  parseCommaSeparatedList(function_ref<ParseResult()> parseElementFn) {
    return parseCommaSeparatedList(Delimiter::None, parseElementFn);
  }

  //===--------------------------------------------------------------------===//
  // Keyword Parsing
  //===--------------------------------------------------------------------===//

  /// This class represents a StringSwitch like class that is useful for parsing
  /// expected keywords. On construction, unless a non-empty keyword is
  /// provided, it invokes `parseKeyword` and processes each of the provided
  /// cases statements until a match is hit. The provided `ResultT` must be
  /// assignable from `failure()`.
  template <typename ResultT = ParseResult>
  class KeywordSwitch {
  public:
    KeywordSwitch(AsmParser &parser, StringRef *keyword = nullptr)
        : parser(parser), loc(parser.getCurrentLocation()) {
      if (keyword && !keyword->empty())
        this->keyword = *keyword;
      else if (failed(parser.parseKeywordOrCompletion(&this->keyword)))
        result = failure();
    }
    /// Case that uses the provided value when true.
    KeywordSwitch &Case(StringLiteral str, ResultT value) {
      return Case(str, [&](StringRef, SMLoc) { return std::move(value); });
    }
    KeywordSwitch &Default(ResultT value) {
      return Default([&](StringRef, SMLoc) { return std::move(value); });
    }
    /// Case that invokes the provided functor when true. The parameters passed
    /// to the functor are the keyword, and the location of the keyword (in case
    /// any errors need to be emitted).
    template <typename FnT>
    std::enable_if_t<!std::is_convertible<FnT, ResultT>::value, KeywordSwitch &>
    Case(StringLiteral str, FnT &&fn) {
      if (result)
        return *this;

      // If the word was empty, record this as a completion.
      if (keyword.empty())
        parser.codeCompleteExpectedTokens(str);
      else if (keyword == str)
        result.emplace(std::move(fn(keyword, loc)));
      return *this;
    }
    template <typename FnT>
    std::enable_if_t<!std::is_convertible<FnT, ResultT>::value, KeywordSwitch &>
    Default(FnT &&fn) {
      if (!result)
        result.emplace(fn(keyword, loc));
      return *this;
    }

    /// Returns true if this switch has a value yet.
    bool hasValue() const { return result.has_value(); }

    /// Return the result of the switch.
    [[nodiscard]] operator ResultT() {
      if (!result)
        return parser.emitError(loc, "unexpected keyword: ") << keyword;
      return std::move(*result);
    }

  private:
    /// The parser used to construct this switch.
    AsmParser &parser;

    /// The location of the keyword, used to emit errors as necessary.
    SMLoc loc;

    /// The parsed keyword itself.
    StringRef keyword;

    /// The result of the switch statement or std::nullopt if currently unknown.
    std::optional<ResultT> result;
  };

  /// Parse a given keyword.
  ParseResult parseKeyword(StringRef keyword) {
    return parseKeyword(keyword, "");
  }
  virtual ParseResult parseKeyword(StringRef keyword, const Twine &msg) = 0;

  /// Parse a keyword into 'keyword'.
  ParseResult parseKeyword(StringRef *keyword) {
    auto loc = getCurrentLocation();
    if (parseOptionalKeyword(keyword))
      return emitError(loc, "expected valid keyword");
    return success();
  }

  /// Parse the given keyword if present.
  virtual ParseResult parseOptionalKeyword(StringRef keyword) = 0;

  /// Parse a keyword, if present, into 'keyword'.
  virtual ParseResult parseOptionalKeyword(StringRef *keyword) = 0;

  /// Parse a keyword, if present, and if one of the 'allowedValues',
  /// into 'keyword'
  virtual ParseResult
  parseOptionalKeyword(StringRef *keyword,
                       ArrayRef<StringRef> allowedValues) = 0;

  /// Parse a keyword or a quoted string.
  ParseResult parseKeywordOrString(std::string *result) {
    if (failed(parseOptionalKeywordOrString(result)))
      return emitError(getCurrentLocation())
             << "expected valid keyword or string";
    return success();
  }

  /// Parse an optional keyword or string.
  virtual ParseResult parseOptionalKeywordOrString(std::string *result) = 0;

  //===--------------------------------------------------------------------===//
  // Attribute/Type Parsing
  //===--------------------------------------------------------------------===//

  /// Invoke the `getChecked` method of the given Attribute or Type class, using
  /// the provided location to emit errors in the case of failure. Note that
  /// unlike `OpBuilder::getType`, this method does not implicitly insert a
  /// context parameter.
  template <typename T, typename... ParamsT>
  auto getChecked(SMLoc loc, ParamsT &&...params) {
    return T::getChecked([&] { return emitError(loc); },
                         std::forward<ParamsT>(params)...);
  }
  /// A variant of `getChecked` that uses the result of `getNameLoc` to emit
  /// errors.
  template <typename T, typename... ParamsT>
  auto getChecked(ParamsT &&...params) {
    return T::getChecked([&] { return emitError(getNameLoc()); },
                         std::forward<ParamsT>(params)...);
  }

  //===--------------------------------------------------------------------===//
  // Attribute Parsing
  //===--------------------------------------------------------------------===//

  /// Parse an arbitrary attribute of a given type and return it in result.
  virtual ParseResult parseAttribute(Attribute &result, Type type = {}) = 0;

  /// Parse a custom attribute with the provided callback, unless the next
  /// token is `#`, in which case the generic parser is invoked.
  virtual ParseResult parseCustomAttributeWithFallback(
      Attribute &result, Type type,
      function_ref<ParseResult(Attribute &result, Type type)>
          parseAttribute) = 0;

  /// Parse an attribute of a specific kind and type.
  template <typename AttrType>
  ParseResult parseAttribute(AttrType &result, Type type = {}) {
    SMLoc loc = getCurrentLocation();

    // Parse any kind of attribute.
    Attribute attr;
    if (parseAttribute(attr, type))
      return failure();

    // Check for the right kind of attribute.
    if (!(result = llvm::dyn_cast<AttrType>(attr)))
      return emitError(loc, "invalid kind of attribute specified");

    return success();
  }

  /// Parse an arbitrary attribute and return it in result.  This also adds the
  /// attribute to the specified attribute list with the specified name.
  ParseResult parseAttribute(Attribute &result, StringRef attrName,
                             NamedAttrList &attrs) {
    return parseAttribute(result, Type(), attrName, attrs);
  }

  /// Parse an attribute of a specific kind and type.
  template <typename AttrType>
  ParseResult parseAttribute(AttrType &result, StringRef attrName,
                             NamedAttrList &attrs) {
    return parseAttribute(result, Type(), attrName, attrs);
  }

  /// Parse an arbitrary attribute of a given type and populate it in `result`.
  /// This also adds the attribute to the specified attribute list with the
  /// specified name.
  template <typename AttrType>
  ParseResult parseAttribute(AttrType &result, Type type, StringRef attrName,
                             NamedAttrList &attrs) {
    SMLoc loc = getCurrentLocation();

    // Parse any kind of attribute.
    Attribute attr;
    if (parseAttribute(attr, type))
      return failure();

    // Check for the right kind of attribute.
    result = llvm::dyn_cast<AttrType>(attr);
    if (!result)
      return emitError(loc, "invalid kind of attribute specified");

    attrs.append(attrName, result);
    return success();
  }

  /// Trait to check if `AttrType` provides a `parse` method.
  template <typename AttrType>
  using has_parse_method = decltype(AttrType::parse(std::declval<AsmParser &>(),
                                                    std::declval<Type>()));
  template <typename AttrType>
  using detect_has_parse_method = llvm::is_detected<has_parse_method, AttrType>;

  /// Parse a custom attribute of a given type unless the next token is `#`, in
  /// which case the generic parser is invoked. The parsed attribute is
  /// populated in `result` and also added to the specified attribute list with
  /// the specified name.
  template <typename AttrType>
  std::enable_if_t<detect_has_parse_method<AttrType>::value, ParseResult>
  parseCustomAttributeWithFallback(AttrType &result, Type type,
                                   StringRef attrName, NamedAttrList &attrs) {
    SMLoc loc = getCurrentLocation();

    // Parse any kind of attribute.
    Attribute attr;
    if (parseCustomAttributeWithFallback(
            attr, type, [&](Attribute &result, Type type) -> ParseResult {
              result = AttrType::parse(*this, type);
              if (!result)
                return failure();
              return success();
            }))
      return failure();

    // Check for the right kind of attribute.
    result = llvm::dyn_cast<AttrType>(attr);
    if (!result)
      return emitError(loc, "invalid kind of attribute specified");

    attrs.append(attrName, result);
    return success();
  }

  /// SFINAE parsing method for Attribute that don't implement a parse method.
  template <typename AttrType>
  std::enable_if_t<!detect_has_parse_method<AttrType>::value, ParseResult>
  parseCustomAttributeWithFallback(AttrType &result, Type type,
                                   StringRef attrName, NamedAttrList &attrs) {
    return parseAttribute(result, type, attrName, attrs);
  }

  /// Parse a custom attribute of a given type unless the next token is `#`, in
  /// which case the generic parser is invoked. The parsed attribute is
  /// populated in `result`.
  template <typename AttrType>
  std::enable_if_t<detect_has_parse_method<AttrType>::value, ParseResult>
  parseCustomAttributeWithFallback(AttrType &result, Type type = {}) {
    SMLoc loc = getCurrentLocation();

    // Parse any kind of attribute.
    Attribute attr;
    if (parseCustomAttributeWithFallback(
            attr, type, [&](Attribute &result, Type type) -> ParseResult {
              result = AttrType::parse(*this, type);
              return success(!!result);
            }))
      return failure();

    // Check for the right kind of attribute.
    result = llvm::dyn_cast<AttrType>(attr);
    if (!result)
      return emitError(loc, "invalid kind of attribute specified");
    return success();
  }

  /// SFINAE parsing method for Attribute that don't implement a parse method.
  template <typename AttrType>
  std::enable_if_t<!detect_has_parse_method<AttrType>::value, ParseResult>
  parseCustomAttributeWithFallback(AttrType &result, Type type = {}) {
    return parseAttribute(result, type);
  }

  /// Parse an arbitrary optional attribute of a given type and return it in
  /// result.
  virtual OptionalParseResult parseOptionalAttribute(Attribute &result,
                                                     Type type = {}) = 0;

  /// Parse an optional array attribute and return it in result.
  virtual OptionalParseResult parseOptionalAttribute(ArrayAttr &result,
                                                     Type type = {}) = 0;

  /// Parse an optional string attribute and return it in result.
  virtual OptionalParseResult parseOptionalAttribute(StringAttr &result,
                                                     Type type = {}) = 0;

  /// Parse an optional symbol ref attribute and return it in result.
  virtual OptionalParseResult parseOptionalAttribute(SymbolRefAttr &result,
                                                     Type type = {}) = 0;

  /// Parse an optional attribute of a specific type and add it to the list with
  /// the specified name.
  template <typename AttrType>
  OptionalParseResult parseOptionalAttribute(AttrType &result,
                                             StringRef attrName,
                                             NamedAttrList &attrs) {
    return parseOptionalAttribute(result, Type(), attrName, attrs);
  }

  /// Parse an optional attribute of a specific type and add it to the list with
  /// the specified name.
  template <typename AttrType>
  OptionalParseResult parseOptionalAttribute(AttrType &result, Type type,
                                             StringRef attrName,
                                             NamedAttrList &attrs) {
    OptionalParseResult parseResult = parseOptionalAttribute(result, type);
    if (parseResult.has_value() && succeeded(*parseResult))
      attrs.append(attrName, result);
    return parseResult;
  }

  /// Parse a named dictionary into 'result' if it is present.
  virtual ParseResult parseOptionalAttrDict(NamedAttrList &result) = 0;

  /// Parse a named dictionary into 'result' if the `attributes` keyword is
  /// present.
  virtual ParseResult
  parseOptionalAttrDictWithKeyword(NamedAttrList &result) = 0;

  /// Parse an affine map instance into 'map'.
  virtual ParseResult parseAffineMap(AffineMap &map) = 0;

  /// Parse an affine expr instance into 'expr' using the already computed
  /// mapping from symbols to affine expressions in 'symbolSet'.
  virtual ParseResult
  parseAffineExpr(ArrayRef<std::pair<StringRef, AffineExpr>> symbolSet,
                  AffineExpr &expr) = 0;

  /// Parse an integer set instance into 'set'.
  virtual ParseResult parseIntegerSet(IntegerSet &set) = 0;

  //===--------------------------------------------------------------------===//
  // Identifier Parsing
  //===--------------------------------------------------------------------===//

  /// Parse an @-identifier and store it (without the '@' symbol) in a string
  /// attribute.
  ParseResult parseSymbolName(StringAttr &result) {
    if (failed(parseOptionalSymbolName(result)))
      return emitError(getCurrentLocation())
             << "expected valid '@'-identifier for symbol name";
    return success();
  }

  /// Parse an @-identifier and store it (without the '@' symbol) in a string
  /// attribute named 'attrName'.
  ParseResult parseSymbolName(StringAttr &result, StringRef attrName,
                              NamedAttrList &attrs) {
    if (parseSymbolName(result))
      return failure();
    attrs.append(attrName, result);
    return success();
  }

  /// Parse an optional @-identifier and store it (without the '@' symbol) in a
  /// string attribute.
  virtual ParseResult parseOptionalSymbolName(StringAttr &result) = 0;

  /// Parse an optional @-identifier and store it (without the '@' symbol) in a
  /// string attribute named 'attrName'.
  ParseResult parseOptionalSymbolName(StringAttr &result, StringRef attrName,
                                      NamedAttrList &attrs) {
    if (succeeded(parseOptionalSymbolName(result))) {
      attrs.append(attrName, result);
      return success();
    }
    return failure();
  }

  //===--------------------------------------------------------------------===//
  // Resource Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a handle to a resource within the assembly format.
  template <typename ResourceT>
  FailureOr<ResourceT> parseResourceHandle() {
    SMLoc handleLoc = getCurrentLocation();

    // Try to load the dialect that owns the handle.
    auto *dialect =
        getContext()->getOrLoadDialect<typename ResourceT::Dialect>();
    if (!dialect) {
      return emitError(handleLoc)
             << "dialect '" << ResourceT::Dialect::getDialectNamespace()
             << "' is unknown";
    }

    FailureOr<AsmDialectResourceHandle> handle = parseResourceHandle(dialect);
    if (failed(handle))
      return failure();
    if (auto *result = dyn_cast<ResourceT>(&*handle))
      return std::move(*result);
    return emitError(handleLoc) << "provided resource handle differs from the "
                                   "expected resource type";
  }

  //===--------------------------------------------------------------------===//
  // Type Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a type.
  virtual ParseResult parseType(Type &result) = 0;

  /// Parse a custom type with the provided callback, unless the next
  /// token is `#`, in which case the generic parser is invoked.
  virtual ParseResult parseCustomTypeWithFallback(
      Type &result, function_ref<ParseResult(Type &result)> parseType) = 0;

  /// Parse an optional type.
  virtual OptionalParseResult parseOptionalType(Type &result) = 0;

  /// Parse a type of a specific type.
  template <typename TypeT>
  ParseResult parseType(TypeT &result) {
    SMLoc loc = getCurrentLocation();

    // Parse any kind of type.
    Type type;
    if (parseType(type))
      return failure();

    // Check for the right kind of type.
    result = llvm::dyn_cast<TypeT>(type);
    if (!result)
      return emitError(loc, "invalid kind of type specified");

    return success();
  }

  /// Trait to check if `TypeT` provides a `parse` method.
  template <typename TypeT>
  using type_has_parse_method =
      decltype(TypeT::parse(std::declval<AsmParser &>()));
  template <typename TypeT>
  using detect_type_has_parse_method =
      llvm::is_detected<type_has_parse_method, TypeT>;

  /// Parse a custom Type of a given type unless the next token is `#`, in
  /// which case the generic parser is invoked. The parsed Type is
  /// populated in `result`.
  template <typename TypeT>
  std::enable_if_t<detect_type_has_parse_method<TypeT>::value, ParseResult>
  parseCustomTypeWithFallback(TypeT &result) {
    SMLoc loc = getCurrentLocation();

    // Parse any kind of Type.
    Type type;
    if (parseCustomTypeWithFallback(type, [&](Type &result) -> ParseResult {
          result = TypeT::parse(*this);
          return success(!!result);
        }))
      return failure();

    // Check for the right kind of Type.
    result = llvm::dyn_cast<TypeT>(type);
    if (!result)
      return emitError(loc, "invalid kind of Type specified");
    return success();
  }

  /// SFINAE parsing method for Type that don't implement a parse method.
  template <typename TypeT>
  std::enable_if_t<!detect_type_has_parse_method<TypeT>::value, ParseResult>
  parseCustomTypeWithFallback(TypeT &result) {
    return parseType(result);
  }

  /// Parse a type list.
  ParseResult parseTypeList(SmallVectorImpl<Type> &result);

  /// Parse an arrow followed by a type list.
  virtual ParseResult parseArrowTypeList(SmallVectorImpl<Type> &result) = 0;

  /// Parse an optional arrow followed by a type list.
  virtual ParseResult
  parseOptionalArrowTypeList(SmallVectorImpl<Type> &result) = 0;

  /// Parse a colon followed by a type.
  virtual ParseResult parseColonType(Type &result) = 0;

  /// Parse a colon followed by a type of a specific kind, e.g. a FunctionType.
  template <typename TypeType>
  ParseResult parseColonType(TypeType &result) {
    SMLoc loc = getCurrentLocation();

    // Parse any kind of type.
    Type type;
    if (parseColonType(type))
      return failure();

    // Check for the right kind of type.
    result = llvm::dyn_cast<TypeType>(type);
    if (!result)
      return emitError(loc, "invalid kind of type specified");

    return success();
  }

  /// Parse a colon followed by a type list, which must have at least one type.
  virtual ParseResult parseColonTypeList(SmallVectorImpl<Type> &result) = 0;

  /// Parse an optional colon followed by a type list, which if present must
  /// have at least one type.
  virtual ParseResult
  parseOptionalColonTypeList(SmallVectorImpl<Type> &result) = 0;

  /// Parse a keyword followed by a type.
  ParseResult parseKeywordType(const char *keyword, Type &result) {
    return failure(parseKeyword(keyword) || parseType(result));
  }

  /// Add the specified type to the end of the specified type list and return
  /// success.  This is a helper designed to allow parse methods to be simple
  /// and chain through || operators.
  ParseResult addTypeToList(Type type, SmallVectorImpl<Type> &result) {
    result.push_back(type);
    return success();
  }

  /// Add the specified types to the end of the specified type list and return
  /// success.  This is a helper designed to allow parse methods to be simple
  /// and chain through || operators.
  ParseResult addTypesToList(ArrayRef<Type> types,
                             SmallVectorImpl<Type> &result) {
    result.append(types.begin(), types.end());
    return success();
  }

  /// Parse a dimension list of a tensor or memref type.  This populates the
  /// dimension list, using ShapedType::kDynamic for the `?` dimensions if
  /// `allowDynamic` is set and errors out on `?` otherwise. Parsing the
  /// trailing `x` is configurable.
  ///
  ///   dimension-list ::= eps | dimension (`x` dimension)*
  ///   dimension-list-with-trailing-x ::= (dimension `x`)*
  ///   dimension ::= `?` | decimal-literal
  ///
  /// When `allowDynamic` is not set, this is used to parse:
  ///
  ///   static-dimension-list ::= eps | decimal-literal (`x` decimal-literal)*
  ///   static-dimension-list-with-trailing-x ::= (dimension `x`)*
  virtual ParseResult parseDimensionList(SmallVectorImpl<int64_t> &dimensions,
                                         bool allowDynamic = true,
                                         bool withTrailingX = true) = 0;

  /// Parse an 'x' token in a dimension list, handling the case where the x is
  /// juxtaposed with an element type, as in "xf32", leaving the "f32" as the
  /// next token.
  virtual ParseResult parseXInDimensionList() = 0;

  /// Class used to automatically end a cyclic region on destruction.
  class CyclicParseReset {
  public:
    explicit CyclicParseReset(AsmParser *parser) : parser(parser) {}

    ~CyclicParseReset() {
      if (parser)
        parser->popCyclicParsing();
    }

    CyclicParseReset(const CyclicParseReset &) = delete;
    CyclicParseReset &operator=(const CyclicParseReset &) = delete;
    CyclicParseReset(CyclicParseReset &&rhs)
        : parser(std::exchange(rhs.parser, nullptr)) {}
    CyclicParseReset &operator=(CyclicParseReset &&rhs) {
      parser = std::exchange(rhs.parser, nullptr);
      return *this;
    }

  private:
    AsmParser *parser;
  };

  /// Attempts to start a cyclic parsing region for `attrOrType`.
  /// A cyclic parsing region starts with this call and ends with the
  /// destruction of the returned `CyclicParseReset`. During this time,
  /// calling `tryStartCyclicParse` with the same attribute in any parser
  /// will lead to returning failure.
  ///
  /// This makes it possible to parse cyclic attributes or types by parsing a
  /// short from if nested within itself.
  template <class AttrOrTypeT>
  FailureOr<CyclicParseReset> tryStartCyclicParse(AttrOrTypeT attrOrType) {
    static_assert(
        std::is_base_of_v<AttributeTrait::IsMutable<AttrOrTypeT>,
                          AttrOrTypeT> ||
            std::is_base_of_v<TypeTrait::IsMutable<AttrOrTypeT>, AttrOrTypeT>,
        "Only mutable attributes or types can be cyclic");
    if (failed(pushCyclicParsing(attrOrType.getAsOpaquePointer())))
      return failure();

    return CyclicParseReset(this);
  }

protected:
  /// Parse a handle to a resource within the assembly format for the given
  /// dialect.
  virtual FailureOr<AsmDialectResourceHandle>
  parseResourceHandle(Dialect *dialect) = 0;

  /// Pushes a new attribute or type in the form of a type erased pointer
  /// into an internal set.
  /// Returns success if the type or attribute was inserted in the set or
  /// failure if it was already contained.
  virtual LogicalResult pushCyclicParsing(const void *opaquePointer) = 0;

  /// Removes the element that was last inserted with a successful call to
  /// `pushCyclicParsing`. There must be exactly one `popCyclicParsing` call
  /// in reverse order of all successful `pushCyclicParsing`.
  virtual void popCyclicParsing() = 0;

  //===--------------------------------------------------------------------===//
  // Code Completion
  //===--------------------------------------------------------------------===//

  /// Parse a keyword, or an empty string if the current location signals a code
  /// completion.
  virtual ParseResult parseKeywordOrCompletion(StringRef *keyword) = 0;

  /// Signal the code completion of a set of expected tokens.
  virtual void codeCompleteExpectedTokens(ArrayRef<StringRef> tokens) = 0;

private:
  AsmParser(const AsmParser &) = delete;
  void operator=(const AsmParser &) = delete;
};

//===----------------------------------------------------------------------===//
// OpAsmParser
//===----------------------------------------------------------------------===//

/// The OpAsmParser has methods for interacting with the asm parser: parsing
/// things from it, emitting errors etc.  It has an intentionally high-level API
/// that is designed to reduce/constrain syntax innovation in individual
/// operations.
///
/// For example, consider an op like this:
///
///    %x = load %p[%1, %2] : memref<...>
///
/// The "%x = load" tokens are already parsed and therefore invisible to the
/// custom op parser.  This can be supported by calling `parseOperandList` to
/// parse the %p, then calling `parseOperandList` with a `SquareDelimiter` to
/// parse the indices, then calling `parseColonTypeList` to parse the result
/// type.
///
class OpAsmParser : public AsmParser {
public:
  using AsmParser::AsmParser;
  ~OpAsmParser() override;

  /// Parse a loc(...) specifier if present, filling in result if so.
  /// Location for BlockArgument and Operation may be deferred with an alias, in
  /// which case an OpaqueLoc is set and will be resolved when parsing
  /// completes.
  virtual ParseResult
  parseOptionalLocationSpecifier(std::optional<Location> &result) = 0;

  /// Return the name of the specified result in the specified syntax, as well
  /// as the sub-element in the name.  It returns an empty string and ~0U for
  /// invalid result numbers.  For example, in this operation:
  ///
  ///  %x, %y:2, %z = foo.op
  ///
  ///    getResultName(0) == {"x", 0 }
  ///    getResultName(1) == {"y", 0 }
  ///    getResultName(2) == {"y", 1 }
  ///    getResultName(3) == {"z", 0 }
  ///    getResultName(4) == {"", ~0U }
  virtual std::pair<StringRef, unsigned>
  getResultName(unsigned resultNo) const = 0;

  /// Return the number of declared SSA results.  This returns 4 for the foo.op
  /// example in the comment for `getResultName`.
  virtual size_t getNumResults() const = 0;

  // These methods emit an error and return failure or success. This allows
  // these to be chained together into a linear sequence of || expressions in
  // many cases.

  /// Parse an operation in its generic form.
  /// The parsed operation is parsed in the current context and inserted in the
  /// provided block and insertion point. The results produced by this operation
  /// aren't mapped to any named value in the parser. Returns nullptr on
  /// failure.
  virtual Operation *parseGenericOperation(Block *insertBlock,
                                           Block::iterator insertPt) = 0;

  /// Parse the name of an operation, in the custom form. On success, return a
  /// an object of type 'OperationName'. Otherwise, failure is returned.
  virtual FailureOr<OperationName> parseCustomOperationName() = 0;

  //===--------------------------------------------------------------------===//
  // Operand Parsing
  //===--------------------------------------------------------------------===//

  /// This is the representation of an operand reference.
  struct UnresolvedOperand {
    SMLoc location;  // Location of the token.
    StringRef name;  // Value name, e.g. %42 or %abc
    unsigned number; // Number, e.g. 12 for an operand like %xyz#12
  };

  /// Parse different components, viz., use-info of operand(s), successor(s),
  /// region(s), attribute(s) and function-type, of the generic form of an
  /// operation instance and populate the input operation-state 'result' with
  /// those components. If any of the components is explicitly provided, then
  /// skip parsing that component.
  virtual ParseResult parseGenericOperationAfterOpName(
      OperationState &result,
      std::optional<ArrayRef<UnresolvedOperand>> parsedOperandType =
          std::nullopt,
      std::optional<ArrayRef<Block *>> parsedSuccessors = std::nullopt,
      std::optional<MutableArrayRef<std::unique_ptr<Region>>> parsedRegions =
          std::nullopt,
      std::optional<ArrayRef<NamedAttribute>> parsedAttributes = std::nullopt,
      std::optional<Attribute> parsedPropertiesAttribute = std::nullopt,
      std::optional<FunctionType> parsedFnType = std::nullopt) = 0;

  /// Parse a single SSA value operand name along with a result number if
  /// `allowResultNumber` is true.
  virtual ParseResult parseOperand(UnresolvedOperand &result,
                                   bool allowResultNumber = true) = 0;

  /// Parse a single operand if present.
  virtual OptionalParseResult
  parseOptionalOperand(UnresolvedOperand &result,
                       bool allowResultNumber = true) = 0;

  /// Parse zero or more SSA comma-separated operand references with a specified
  /// surrounding delimiter, and an optional required operand count.
  virtual ParseResult
  parseOperandList(SmallVectorImpl<UnresolvedOperand> &result,
                   Delimiter delimiter = Delimiter::None,
                   bool allowResultNumber = true,
                   int requiredOperandCount = -1) = 0;

  /// Parse a specified number of comma separated operands.
  ParseResult parseOperandList(SmallVectorImpl<UnresolvedOperand> &result,
                               int requiredOperandCount,
                               Delimiter delimiter = Delimiter::None) {
    return parseOperandList(result, delimiter,
                            /*allowResultNumber=*/true, requiredOperandCount);
  }

  /// Parse zero or more trailing SSA comma-separated trailing operand
  /// references with a specified surrounding delimiter, and an optional
  /// required operand count. A leading comma is expected before the
  /// operands.
  ParseResult
  parseTrailingOperandList(SmallVectorImpl<UnresolvedOperand> &result,
                           Delimiter delimiter = Delimiter::None) {
    if (failed(parseOptionalComma()))
      return success(); // The comma is optional.
    return parseOperandList(result, delimiter);
  }

  /// Resolve an operand to an SSA value, emitting an error on failure.
  virtual ParseResult resolveOperand(const UnresolvedOperand &operand,
                                     Type type,
                                     SmallVectorImpl<Value> &result) = 0;

  /// Resolve a list of operands to SSA values, emitting an error on failure, or
  /// appending the results to the list on success. This method should be used
  /// when all operands have the same type.
  template <typename Operands = ArrayRef<UnresolvedOperand>>
  ParseResult resolveOperands(Operands &&operands, Type type,
                              SmallVectorImpl<Value> &result) {
    for (const UnresolvedOperand &operand : operands)
      if (resolveOperand(operand, type, result))
        return failure();
    return success();
  }
  template <typename Operands = ArrayRef<UnresolvedOperand>>
  ParseResult resolveOperands(Operands &&operands, Type type, SMLoc loc,
                              SmallVectorImpl<Value> &result) {
    return resolveOperands(std::forward<Operands>(operands), type, result);
  }

  /// Resolve a list of operands and a list of operand types to SSA values,
  /// emitting an error and returning failure, or appending the results
  /// to the list on success.
  template <typename Operands = ArrayRef<UnresolvedOperand>,
            typename Types = ArrayRef<Type>>
  std::enable_if_t<!std::is_convertible<Types, Type>::value, ParseResult>
  resolveOperands(Operands &&operands, Types &&types, SMLoc loc,
                  SmallVectorImpl<Value> &result) {
    size_t operandSize = llvm::range_size(operands);
    size_t typeSize = llvm::range_size(types);
    if (operandSize != typeSize)
      return emitError(loc)
             << operandSize << " operands present, but expected " << typeSize;

    for (auto [operand, type] : llvm::zip_equal(operands, types))
      if (resolveOperand(operand, type, result))
        return failure();
    return success();
  }

  /// Parses an affine map attribute where dims and symbols are SSA operands.
  /// Operand values must come from single-result sources, and be valid
  /// dimensions/symbol identifiers according to mlir::isValidDim/Symbol.
  virtual ParseResult
  parseAffineMapOfSSAIds(SmallVectorImpl<UnresolvedOperand> &operands,
                         Attribute &map, StringRef attrName,
                         NamedAttrList &attrs,
                         Delimiter delimiter = Delimiter::Square) = 0;

  /// Parses an affine expression where dims and symbols are SSA operands.
  /// Operand values must come from single-result sources, and be valid
  /// dimensions/symbol identifiers according to mlir::isValidDim/Symbol.
  virtual ParseResult
  parseAffineExprOfSSAIds(SmallVectorImpl<UnresolvedOperand> &dimOperands,
                          SmallVectorImpl<UnresolvedOperand> &symbOperands,
                          AffineExpr &expr) = 0;

  //===--------------------------------------------------------------------===//
  // Argument Parsing
  //===--------------------------------------------------------------------===//

  struct Argument {
    UnresolvedOperand ssaName;         // SourceLoc, SSA name, result #.
    Type type;                         // Type.
    DictionaryAttr attrs;              // Attributes if present.
    std::optional<Location> sourceLoc; // Source location specifier if present.
  };

  /// Parse a single argument with the following syntax:
  ///
  ///   `%ssaName : !type { optionalAttrDict} loc(optionalSourceLoc)`
  ///
  /// If `allowType` is false or `allowAttrs` are false then the respective
  /// parts of the grammar are not parsed.
  virtual ParseResult parseArgument(Argument &result, bool allowType = false,
                                    bool allowAttrs = false) = 0;

  /// Parse a single argument if present.
  virtual OptionalParseResult
  parseOptionalArgument(Argument &result, bool allowType = false,
                        bool allowAttrs = false) = 0;

  /// Parse zero or more arguments with a specified surrounding delimiter.
  virtual ParseResult parseArgumentList(SmallVectorImpl<Argument> &result,
                                        Delimiter delimiter = Delimiter::None,
                                        bool allowType = false,
                                        bool allowAttrs = false) = 0;

  //===--------------------------------------------------------------------===//
  // Region Parsing
  //===--------------------------------------------------------------------===//

  /// Parses a region. Any parsed blocks are appended to 'region' and must be
  /// moved to the op regions after the op is created. The first block of the
  /// region takes 'arguments'.
  ///
  /// If 'enableNameShadowing' is set to true, the argument names are allowed to
  /// shadow the names of other existing SSA values defined above the region
  /// scope. 'enableNameShadowing' can only be set to true for regions attached
  /// to operations that are 'IsolatedFromAbove'.
  virtual ParseResult parseRegion(Region &region,
                                  ArrayRef<Argument> arguments = {},
                                  bool enableNameShadowing = false) = 0;

  /// Parses a region if present.
  virtual OptionalParseResult
  parseOptionalRegion(Region &region, ArrayRef<Argument> arguments = {},
                      bool enableNameShadowing = false) = 0;

  /// Parses a region if present. If the region is present, a new region is
  /// allocated and placed in `region`. If no region is present or on failure,
  /// `region` remains untouched.
  virtual OptionalParseResult
  parseOptionalRegion(std::unique_ptr<Region> &region,
                      ArrayRef<Argument> arguments = {},
                      bool enableNameShadowing = false) = 0;

  //===--------------------------------------------------------------------===//
  // Successor Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a single operation successor.
  virtual ParseResult parseSuccessor(Block *&dest) = 0;

  /// Parse an optional operation successor.
  virtual OptionalParseResult parseOptionalSuccessor(Block *&dest) = 0;

  /// Parse a single operation successor and its operand list.
  virtual ParseResult
  parseSuccessorAndUseList(Block *&dest, SmallVectorImpl<Value> &operands) = 0;

  //===--------------------------------------------------------------------===//
  // Type Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a list of assignments of the form
  ///   (%x1 = %y1, %x2 = %y2, ...)
  ParseResult parseAssignmentList(SmallVectorImpl<Argument> &lhs,
                                  SmallVectorImpl<UnresolvedOperand> &rhs) {
    OptionalParseResult result = parseOptionalAssignmentList(lhs, rhs);
    if (!result.has_value())
      return emitError(getCurrentLocation(), "expected '('");
    return result.value();
  }

  virtual OptionalParseResult
  parseOptionalAssignmentList(SmallVectorImpl<Argument> &lhs,
                              SmallVectorImpl<UnresolvedOperand> &rhs) = 0;
};

//===--------------------------------------------------------------------===//
// Dialect OpAsm interface.
//===--------------------------------------------------------------------===//

/// A functor used to set the name of the start of a result group of an
/// operation. See 'getAsmResultNames' below for more details.
using OpAsmSetValueNameFn = function_ref<void(Value, StringRef)>;

/// A functor used to set the name of blocks in regions directly nested under
/// an operation.
using OpAsmSetBlockNameFn = function_ref<void(Block *, StringRef)>;

class OpAsmDialectInterface
    : public DialectInterface::Base<OpAsmDialectInterface> {
public:
  OpAsmDialectInterface(Dialect *dialect) : Base(dialect) {}

  //===------------------------------------------------------------------===//
  // Aliases
  //===------------------------------------------------------------------===//

  /// Holds the result of `getAlias` hook call.
  enum class AliasResult {
    /// The object (type or attribute) is not supported by the hook
    /// and an alias was not provided.
    NoAlias,
    /// An alias was provided, but it might be overriden by other hook.
    OverridableAlias,
    /// An alias was provided and it should be used
    /// (no other hooks will be checked).
    FinalAlias
  };

  /// Hooks for getting an alias identifier alias for a given symbol, that is
  /// not necessarily a part of this dialect. The identifier is used in place of
  /// the symbol when printing textual IR. These aliases must not contain `.` or
  /// end with a numeric digit([0-9]+).
  virtual AliasResult getAlias(Attribute attr, raw_ostream &os) const {
    return AliasResult::NoAlias;
  }
  virtual AliasResult getAlias(Type type, raw_ostream &os) const {
    return AliasResult::NoAlias;
  }

  //===--------------------------------------------------------------------===//
  // Resources
  //===--------------------------------------------------------------------===//

  /// Declare a resource with the given key, returning a handle to use for any
  /// references of this resource key within the IR during parsing. The result
  /// of `getResourceKey` on the returned handle is permitted to be different
  /// than `key`.
  virtual FailureOr<AsmDialectResourceHandle>
  declareResource(StringRef key) const {
    return failure();
  }

  /// Return a key to use for the given resource. This key should uniquely
  /// identify this resource within the dialect.
  virtual std::string
  getResourceKey(const AsmDialectResourceHandle &handle) const {
    llvm_unreachable(
        "Dialect must implement `getResourceKey` when defining resources");
  }

  /// Hook for parsing resource entries. Returns failure if the entry was not
  /// valid, or could otherwise not be processed correctly. Any necessary errors
  /// can be emitted via the provided entry.
  virtual LogicalResult parseResource(AsmParsedResourceEntry &entry) const;

  /// Hook for building resources to use during printing. The given `op` may be
  /// inspected to help determine what information to include.
  /// `referencedResources` contains all of the resources detected when printing
  /// 'op'.
  virtual void
  buildResources(Operation *op,
                 const SetVector<AsmDialectResourceHandle> &referencedResources,
                 AsmResourceBuilder &builder) const {}
};

//===--------------------------------------------------------------------===//
// Custom printers and parsers.
//===--------------------------------------------------------------------===//

// Handles custom<DimensionList>(...) in TableGen.
void printDimensionList(OpAsmPrinter &printer, Operation *op,
                        ArrayRef<int64_t> dimensions);
ParseResult parseDimensionList(OpAsmParser &parser,
                               DenseI64ArrayAttr &dimensions);

} // namespace mlir

//===--------------------------------------------------------------------===//
// Operation OpAsm interface.
//===--------------------------------------------------------------------===//

/// The OpAsmOpInterface, see OpAsmInterface.td for more details.
#include "mlir/IR/OpAsmInterface.h.inc"

namespace llvm {
template <>
struct DenseMapInfo<mlir::AsmDialectResourceHandle> {
  static inline mlir::AsmDialectResourceHandle getEmptyKey() {
    return {DenseMapInfo<void *>::getEmptyKey(),
            DenseMapInfo<mlir::TypeID>::getEmptyKey(), nullptr};
  }
  static inline mlir::AsmDialectResourceHandle getTombstoneKey() {
    return {DenseMapInfo<void *>::getTombstoneKey(),
            DenseMapInfo<mlir::TypeID>::getTombstoneKey(), nullptr};
  }
  static unsigned getHashValue(const mlir::AsmDialectResourceHandle &handle) {
    return DenseMapInfo<void *>::getHashValue(handle.getResource());
  }
  static bool isEqual(const mlir::AsmDialectResourceHandle &lhs,
                      const mlir::AsmDialectResourceHandle &rhs) {
    return lhs.getResource() == rhs.getResource();
  }
};
} // namespace llvm

#endif
