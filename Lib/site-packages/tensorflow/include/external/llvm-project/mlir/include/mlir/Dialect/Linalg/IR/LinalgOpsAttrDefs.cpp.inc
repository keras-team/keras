/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::linalg::UnaryFnAttr,
::mlir::linalg::BinaryFnAttr,
::mlir::linalg::TernaryFnAttr,
::mlir::linalg::TypeFnAttr,
::mlir::linalg::IteratorTypeAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::linalg::UnaryFnAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::linalg::UnaryFnAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::linalg::BinaryFnAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::linalg::BinaryFnAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::linalg::TernaryFnAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::linalg::TernaryFnAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::linalg::TypeFnAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::linalg::TypeFnAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::linalg::IteratorTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::linalg::IteratorTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::linalg::UnaryFnAttr>([&](auto t) {
      printer << ::mlir::linalg::UnaryFnAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::linalg::BinaryFnAttr>([&](auto t) {
      printer << ::mlir::linalg::BinaryFnAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::linalg::TernaryFnAttr>([&](auto t) {
      printer << ::mlir::linalg::TernaryFnAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::linalg::TypeFnAttr>([&](auto t) {
      printer << ::mlir::linalg::TypeFnAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::linalg::IteratorTypeAttr>([&](auto t) {
      printer << ::mlir::linalg::IteratorTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace linalg {
namespace detail {
struct UnaryFnAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::linalg::UnaryFn>;
  UnaryFnAttrStorage(::mlir::linalg::UnaryFn value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static UnaryFnAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<UnaryFnAttrStorage>()) UnaryFnAttrStorage(std::move(value));
  }

  ::mlir::linalg::UnaryFn value;
};
} // namespace detail
UnaryFnAttr UnaryFnAttr::get(::mlir::MLIRContext *context, ::mlir::linalg::UnaryFn value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute UnaryFnAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::linalg::UnaryFn> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::linalg::UnaryFn> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::linalg::symbolizeUnaryFn(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::linalg::UnaryFn" << " to be one of: " << "exp" << ", " << "log" << ", " << "abs" << ", " << "ceil" << ", " << "floor" << ", " << "negf" << ", " << "reciprocal" << ", " << "round" << ", " << "sqrt" << ", " << "rsqrt" << ", " << "square" << ", " << "tanh" << ", " << "erf")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse UnaryFnAttr parameter 'value' which is to be a `::mlir::linalg::UnaryFn`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return UnaryFnAttr::get(odsParser.getContext(),
      ::mlir::linalg::UnaryFn((*_result_value)));
}

void UnaryFnAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyUnaryFn(getValue());
  odsPrinter << ">";
}

::mlir::linalg::UnaryFn UnaryFnAttr::getValue() const {
  return getImpl()->value;
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::UnaryFnAttr)
namespace mlir {
namespace linalg {
namespace detail {
struct BinaryFnAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::linalg::BinaryFn>;
  BinaryFnAttrStorage(::mlir::linalg::BinaryFn value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static BinaryFnAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<BinaryFnAttrStorage>()) BinaryFnAttrStorage(std::move(value));
  }

  ::mlir::linalg::BinaryFn value;
};
} // namespace detail
BinaryFnAttr BinaryFnAttr::get(::mlir::MLIRContext *context, ::mlir::linalg::BinaryFn value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute BinaryFnAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::linalg::BinaryFn> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::linalg::BinaryFn> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::linalg::symbolizeBinaryFn(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::linalg::BinaryFn" << " to be one of: " << "add" << ", " << "sub" << ", " << "mul" << ", " << "div" << ", " << "div_unsigned" << ", " << "max_signed" << ", " << "min_signed" << ", " << "max_unsigned" << ", " << "min_unsigned" << ", " << "powf")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse BinaryFnAttr parameter 'value' which is to be a `::mlir::linalg::BinaryFn`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return BinaryFnAttr::get(odsParser.getContext(),
      ::mlir::linalg::BinaryFn((*_result_value)));
}

void BinaryFnAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyBinaryFn(getValue());
  odsPrinter << ">";
}

::mlir::linalg::BinaryFn BinaryFnAttr::getValue() const {
  return getImpl()->value;
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::BinaryFnAttr)
namespace mlir {
namespace linalg {
namespace detail {
struct TernaryFnAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::linalg::TernaryFn>;
  TernaryFnAttrStorage(::mlir::linalg::TernaryFn value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TernaryFnAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<TernaryFnAttrStorage>()) TernaryFnAttrStorage(std::move(value));
  }

  ::mlir::linalg::TernaryFn value;
};
} // namespace detail
TernaryFnAttr TernaryFnAttr::get(::mlir::MLIRContext *context, ::mlir::linalg::TernaryFn value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute TernaryFnAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::linalg::TernaryFn> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::linalg::TernaryFn> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::linalg::symbolizeTernaryFn(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::linalg::TernaryFn" << " to be one of: " << "select")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TernaryFnAttr parameter 'value' which is to be a `::mlir::linalg::TernaryFn`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return TernaryFnAttr::get(odsParser.getContext(),
      ::mlir::linalg::TernaryFn((*_result_value)));
}

void TernaryFnAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyTernaryFn(getValue());
  odsPrinter << ">";
}

::mlir::linalg::TernaryFn TernaryFnAttr::getValue() const {
  return getImpl()->value;
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::TernaryFnAttr)
namespace mlir {
namespace linalg {
namespace detail {
struct TypeFnAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::linalg::TypeFn>;
  TypeFnAttrStorage(::mlir::linalg::TypeFn value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TypeFnAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<TypeFnAttrStorage>()) TypeFnAttrStorage(std::move(value));
  }

  ::mlir::linalg::TypeFn value;
};
} // namespace detail
TypeFnAttr TypeFnAttr::get(::mlir::MLIRContext *context, ::mlir::linalg::TypeFn value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute TypeFnAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::linalg::TypeFn> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::linalg::TypeFn> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::linalg::symbolizeTypeFn(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::linalg::TypeFn" << " to be one of: " << "cast_signed" << ", " << "cast_unsigned")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TypeFnAttr parameter 'value' which is to be a `::mlir::linalg::TypeFn`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return TypeFnAttr::get(odsParser.getContext(),
      ::mlir::linalg::TypeFn((*_result_value)));
}

void TypeFnAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyTypeFn(getValue());
  odsPrinter << ">";
}

::mlir::linalg::TypeFn TypeFnAttr::getValue() const {
  return getImpl()->value;
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::TypeFnAttr)
namespace mlir {
namespace linalg {
namespace detail {
struct IteratorTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::utils::IteratorType>;
  IteratorTypeAttrStorage(::mlir::utils::IteratorType value) : value(std::move(value)) {}

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

  ::mlir::utils::IteratorType value;
};
} // namespace detail
IteratorTypeAttr IteratorTypeAttr::get(::mlir::MLIRContext *context, ::mlir::utils::IteratorType value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute IteratorTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::utils::IteratorType> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::utils::IteratorType> {
      auto loc = odsParser.getCurrentLocation();
      ::llvm::StringRef enumKeyword;
      if (::mlir::failed(odsParser.parseKeyword(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::utils::symbolizeIteratorType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::utils::IteratorType" << " to be one of: " << "parallel" << ", " << "reduction")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse IteratorTypeEnum parameter 'value' which is to be a `::mlir::utils::IteratorType`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return IteratorTypeAttr::get(odsParser.getContext(),
      ::mlir::utils::IteratorType((*_result_value)));
}

void IteratorTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyIteratorType(getValue());
  odsPrinter << ">";
}

::mlir::utils::IteratorType IteratorTypeAttr::getValue() const {
  return getImpl()->value;
}

} // namespace linalg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::IteratorTypeAttr)
namespace mlir {
namespace linalg {

/// Parse an attribute registered to this dialect.
::mlir::Attribute LinalgDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void LinalgDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace linalg
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

