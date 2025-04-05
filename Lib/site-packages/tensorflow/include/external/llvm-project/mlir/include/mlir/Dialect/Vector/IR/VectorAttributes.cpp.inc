/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::vector::CombiningKindAttr,
::mlir::vector::IteratorTypeAttr,
::mlir::vector::PrintPunctuationAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::vector::CombiningKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vector::CombiningKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vector::IteratorTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vector::IteratorTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vector::PrintPunctuationAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vector::PrintPunctuationAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::vector::CombiningKindAttr>([&](auto t) {
      printer << ::mlir::vector::CombiningKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vector::IteratorTypeAttr>([&](auto t) {
      printer << ::mlir::vector::IteratorTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vector::PrintPunctuationAttr>([&](auto t) {
      printer << ::mlir::vector::PrintPunctuationAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace vector {
namespace detail {
struct CombiningKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::vector::CombiningKind>;
  CombiningKindAttrStorage(::mlir::vector::CombiningKind value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static CombiningKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<CombiningKindAttrStorage>()) CombiningKindAttrStorage(std::move(value));
  }

  ::mlir::vector::CombiningKind value;
};
} // namespace detail
CombiningKindAttr CombiningKindAttr::get(::mlir::MLIRContext *context, ::mlir::vector::CombiningKind value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute CombiningKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::vector::CombiningKind> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::vector::CombiningKind> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::vector::symbolizeCombiningKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::vector::CombiningKind" << " to be one of: " << "add" << ", " << "mul" << ", " << "minui" << ", " << "minsi" << ", " << "minnumf" << ", " << "maxui" << ", " << "maxsi" << ", " << "maxnumf" << ", " << "and" << ", " << "or" << ", " << "xor" << ", " << "maximumf" << ", " << "minimumf")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Vector_CombiningKindAttr parameter 'value' which is to be a `::mlir::vector::CombiningKind`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return CombiningKindAttr::get(odsParser.getContext(),
      ::mlir::vector::CombiningKind((*_result_value)));
}

void CombiningKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyCombiningKind(getValue());
  odsPrinter << ">";
}

::mlir::vector::CombiningKind CombiningKindAttr::getValue() const {
  return getImpl()->value;
}

} // namespace vector
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::CombiningKindAttr)
namespace mlir {
namespace vector {
namespace detail {
struct IteratorTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::vector::IteratorType>;
  IteratorTypeAttrStorage(::mlir::vector::IteratorType value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static IteratorTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<IteratorTypeAttrStorage>()) IteratorTypeAttrStorage(std::move(value));
  }

  ::mlir::vector::IteratorType value;
};
} // namespace detail
IteratorTypeAttr IteratorTypeAttr::get(::mlir::MLIRContext *context, ::mlir::vector::IteratorType value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute IteratorTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::vector::IteratorType> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::vector::IteratorType> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::vector::symbolizeIteratorType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::vector::IteratorType" << " to be one of: " << "parallel" << ", " << "reduction")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Vector_IteratorTypeEnum parameter 'value' which is to be a `::mlir::vector::IteratorType`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return IteratorTypeAttr::get(odsParser.getContext(),
      ::mlir::vector::IteratorType((*_result_value)));
}

void IteratorTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyIteratorType(getValue());
  odsPrinter << ">";
}

::mlir::vector::IteratorType IteratorTypeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace vector
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::IteratorTypeAttr)
namespace mlir {
namespace vector {
namespace detail {
struct PrintPunctuationAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::vector::PrintPunctuation>;
  PrintPunctuationAttrStorage(::mlir::vector::PrintPunctuation value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static PrintPunctuationAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<PrintPunctuationAttrStorage>()) PrintPunctuationAttrStorage(std::move(value));
  }

  ::mlir::vector::PrintPunctuation value;
};
} // namespace detail
PrintPunctuationAttr PrintPunctuationAttr::get(::mlir::MLIRContext *context, ::mlir::vector::PrintPunctuation value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute PrintPunctuationAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::vector::PrintPunctuation> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::vector::PrintPunctuation> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::vector::symbolizePrintPunctuation(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::vector::PrintPunctuation" << " to be one of: " << "no_punctuation" << ", " << "newline" << ", " << "comma" << ", " << "open" << ", " << "close")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Vector_PrintPunctuation parameter 'value' which is to be a `::mlir::vector::PrintPunctuation`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return PrintPunctuationAttr::get(odsParser.getContext(),
      ::mlir::vector::PrintPunctuation((*_result_value)));
}

void PrintPunctuationAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyPrintPunctuation(getValue());
  odsPrinter << ">";
}

::mlir::vector::PrintPunctuation PrintPunctuationAttr::getValue() const {
  return getImpl()->value;
}

} // namespace vector
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::PrintPunctuationAttr)
namespace mlir {
namespace vector {

/// Parse an attribute registered to this dialect.
::mlir::Attribute VectorDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                      ::mlir::Type type) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef attrTag;
  {
    ::mlir::Attribute attr;
    auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
    if (parseResult.has_value())
      return attr;
  }
  
  parser.emitError(typeLoc) << "unknown attribute `"
      << attrTag << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print an attribute registered to this dialect.
void VectorDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace vector
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

