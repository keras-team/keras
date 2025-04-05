/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::index::IndexCmpPredicateAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::index::IndexCmpPredicateAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::index::IndexCmpPredicateAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::index::IndexCmpPredicateAttr>([&](auto t) {
      printer << ::mlir::index::IndexCmpPredicateAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace index {
namespace detail {
struct IndexCmpPredicateAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::index::IndexCmpPredicate>;
  IndexCmpPredicateAttrStorage(::mlir::index::IndexCmpPredicate value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static IndexCmpPredicateAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<IndexCmpPredicateAttrStorage>()) IndexCmpPredicateAttrStorage(std::move(value));
  }

  ::mlir::index::IndexCmpPredicate value;
};
} // namespace detail
IndexCmpPredicateAttr IndexCmpPredicateAttr::get(::mlir::MLIRContext *context, ::mlir::index::IndexCmpPredicate value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute IndexCmpPredicateAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::index::IndexCmpPredicate> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::index::IndexCmpPredicate> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::index::symbolizeIndexCmpPredicate(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::index::IndexCmpPredicate" << " to be one of: " << "eq" << ", " << "ne" << ", " << "slt" << ", " << "sle" << ", " << "sgt" << ", " << "sge" << ", " << "ult" << ", " << "ule" << ", " << "ugt" << ", " << "uge")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse IndexCmpPredicateAttr parameter 'value' which is to be a `::mlir::index::IndexCmpPredicate`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return IndexCmpPredicateAttr::get(odsParser.getContext(),
      ::mlir::index::IndexCmpPredicate((*_result_value)));
}

void IndexCmpPredicateAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyIndexCmpPredicate(getValue());
}

::mlir::index::IndexCmpPredicate IndexCmpPredicateAttr::getValue() const {
  return getImpl()->value;
}

} // namespace index
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::index::IndexCmpPredicateAttr)
namespace mlir {
namespace index {

/// Parse an attribute registered to this dialect.
::mlir::Attribute IndexDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void IndexDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace index
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

