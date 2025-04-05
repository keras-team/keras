/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::chlo::ComparisonDirectionAttr,
::mlir::chlo::ComparisonTypeAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::chlo::ComparisonDirectionAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::chlo::ComparisonDirectionAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::chlo::ComparisonTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::chlo::ComparisonTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::chlo::ComparisonDirectionAttr>([&](auto t) {
      printer << ::mlir::chlo::ComparisonDirectionAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::chlo::ComparisonTypeAttr>([&](auto t) {
      printer << ::mlir::chlo::ComparisonTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace chlo {
namespace detail {
struct ComparisonDirectionAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::chlo::ComparisonDirection>;
  ComparisonDirectionAttrStorage(::mlir::chlo::ComparisonDirection value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ComparisonDirectionAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ComparisonDirectionAttrStorage>()) ComparisonDirectionAttrStorage(std::move(value));
  }

  ::mlir::chlo::ComparisonDirection value;
};
} // namespace detail
ComparisonDirectionAttr ComparisonDirectionAttr::get(::mlir::MLIRContext *context, ::mlir::chlo::ComparisonDirection value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ComparisonDirectionAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::chlo::ComparisonDirection> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::chlo::ComparisonDirection> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::chlo::symbolizeComparisonDirection(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::chlo::ComparisonDirection" << " to be one of: " << "EQ" << ", " << "NE" << ", " << "GE" << ", " << "GT" << ", " << "LE" << ", " << "LT")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse CHLO_ComparisonDirectionAttr parameter 'value' which is to be a `::mlir::chlo::ComparisonDirection`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ComparisonDirectionAttr::get(odsParser.getContext(),
      ::mlir::chlo::ComparisonDirection((*_result_value)));
}

void ComparisonDirectionAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyComparisonDirection(getValue());
}

::mlir::chlo::ComparisonDirection ComparisonDirectionAttr::getValue() const {
  return getImpl()->value;
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::ComparisonDirectionAttr)
namespace mlir {
namespace chlo {
namespace detail {
struct ComparisonTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::chlo::ComparisonType>;
  ComparisonTypeAttrStorage(::mlir::chlo::ComparisonType value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ComparisonTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ComparisonTypeAttrStorage>()) ComparisonTypeAttrStorage(std::move(value));
  }

  ::mlir::chlo::ComparisonType value;
};
} // namespace detail
ComparisonTypeAttr ComparisonTypeAttr::get(::mlir::MLIRContext *context, ::mlir::chlo::ComparisonType value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ComparisonTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::chlo::ComparisonType> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::chlo::ComparisonType> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::chlo::symbolizeComparisonType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::chlo::ComparisonType" << " to be one of: " << "NOTYPE" << ", " << "FLOAT" << ", " << "TOTALORDER" << ", " << "SIGNED" << ", " << "UNSIGNED")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse CHLO_ComparisonTypeAttr parameter 'value' which is to be a `::mlir::chlo::ComparisonType`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ComparisonTypeAttr::get(odsParser.getContext(),
      ::mlir::chlo::ComparisonType((*_result_value)));
}

void ComparisonTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyComparisonType(getValue());
}

::mlir::chlo::ComparisonType ComparisonTypeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace chlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::chlo::ComparisonTypeAttr)

#endif  // GET_ATTRDEF_CLASSES

