/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::arith::FastMathFlagsAttr,
::mlir::arith::IntegerOverflowFlagsAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::arith::FastMathFlagsAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::arith::FastMathFlagsAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::arith::IntegerOverflowFlagsAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::arith::IntegerOverflowFlagsAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::arith::FastMathFlagsAttr>([&](auto t) {
      printer << ::mlir::arith::FastMathFlagsAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::arith::IntegerOverflowFlagsAttr>([&](auto t) {
      printer << ::mlir::arith::IntegerOverflowFlagsAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace arith {
namespace detail {
struct FastMathFlagsAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::arith::FastMathFlags>;
  FastMathFlagsAttrStorage(::mlir::arith::FastMathFlags value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static FastMathFlagsAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<FastMathFlagsAttrStorage>()) FastMathFlagsAttrStorage(std::move(value));
  }

  ::mlir::arith::FastMathFlags value;
};
} // namespace detail
FastMathFlagsAttr FastMathFlagsAttr::get(::mlir::MLIRContext *context, ::mlir::arith::FastMathFlags value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute FastMathFlagsAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::arith::FastMathFlags> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::arith::FastMathFlags> {
      ::mlir::arith::FastMathFlags flags = {};
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      do {
        if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
          return ::mlir::failure();
        auto maybeEnum = ::mlir::arith::symbolizeFastMathFlags(enumKeyword);
        if (!maybeEnum) {
            return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::arith::FastMathFlags" << " to be one of: " << "none" << ", " << "reassoc" << ", " << "nnan" << ", " << "ninf" << ", " << "nsz" << ", " << "arcp" << ", " << "contract" << ", " << "afn" << ", " << "fast")};
        }
        flags = flags | *maybeEnum;
      } while(::mlir::succeeded(odsParser.parseOptionalComma()));
      return flags;
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Arith_FastMathAttr parameter 'value' which is to be a `::mlir::arith::FastMathFlags`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return FastMathFlagsAttr::get(odsParser.getContext(),
      ::mlir::arith::FastMathFlags((*_result_value)));
}

void FastMathFlagsAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyFastMathFlags(getValue());
  odsPrinter << ">";
}

::mlir::arith::FastMathFlags FastMathFlagsAttr::getValue() const {
  return getImpl()->value;
}

} // namespace arith
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::arith::FastMathFlagsAttr)
namespace mlir {
namespace arith {
namespace detail {
struct IntegerOverflowFlagsAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::arith::IntegerOverflowFlags>;
  IntegerOverflowFlagsAttrStorage(::mlir::arith::IntegerOverflowFlags value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static IntegerOverflowFlagsAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<IntegerOverflowFlagsAttrStorage>()) IntegerOverflowFlagsAttrStorage(std::move(value));
  }

  ::mlir::arith::IntegerOverflowFlags value;
};
} // namespace detail
IntegerOverflowFlagsAttr IntegerOverflowFlagsAttr::get(::mlir::MLIRContext *context, ::mlir::arith::IntegerOverflowFlags value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute IntegerOverflowFlagsAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::arith::IntegerOverflowFlags> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::arith::IntegerOverflowFlags> {
      ::mlir::arith::IntegerOverflowFlags flags = {};
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      do {
        if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
          return ::mlir::failure();
        auto maybeEnum = ::mlir::arith::symbolizeIntegerOverflowFlags(enumKeyword);
        if (!maybeEnum) {
            return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::arith::IntegerOverflowFlags" << " to be one of: " << "none" << ", " << "nsw" << ", " << "nuw")};
        }
        flags = flags | *maybeEnum;
      } while(::mlir::succeeded(odsParser.parseOptionalComma()));
      return flags;
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Arith_IntegerOverflowAttr parameter 'value' which is to be a `::mlir::arith::IntegerOverflowFlags`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return IntegerOverflowFlagsAttr::get(odsParser.getContext(),
      ::mlir::arith::IntegerOverflowFlags((*_result_value)));
}

void IntegerOverflowFlagsAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyIntegerOverflowFlags(getValue());
  odsPrinter << ">";
}

::mlir::arith::IntegerOverflowFlags IntegerOverflowFlagsAttr::getValue() const {
  return getImpl()->value;
}

} // namespace arith
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::arith::IntegerOverflowFlagsAttr)
namespace mlir {
namespace arith {

/// Parse an attribute registered to this dialect.
::mlir::Attribute ArithDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void ArithDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace arith
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

