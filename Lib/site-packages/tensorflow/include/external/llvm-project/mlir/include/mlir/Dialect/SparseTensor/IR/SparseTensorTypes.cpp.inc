/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::sparse_tensor::StorageSpecifierType,
::mlir::sparse_tensor::IterSpaceType,
::mlir::sparse_tensor::IteratorType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::sparse_tensor::StorageSpecifierType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::sparse_tensor::StorageSpecifierType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::sparse_tensor::IterSpaceType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::sparse_tensor::IterSpaceType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::sparse_tensor::IteratorType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::sparse_tensor::IteratorType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::mlir::sparse_tensor::StorageSpecifierType>([&](auto t) {
      printer << ::mlir::sparse_tensor::StorageSpecifierType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::sparse_tensor::IterSpaceType>([&](auto t) {
      printer << ::mlir::sparse_tensor::IterSpaceType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::sparse_tensor::IteratorType>([&](auto t) {
      printer << ::mlir::sparse_tensor::IteratorType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace sparse_tensor {
namespace detail {
struct StorageSpecifierTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::mlir::sparse_tensor::SparseTensorEncodingAttr>;
  StorageSpecifierTypeStorage(::mlir::sparse_tensor::SparseTensorEncodingAttr encoding) : encoding(std::move(encoding)) {}

  KeyTy getAsKey() const {
    return KeyTy(encoding);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (encoding == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static StorageSpecifierTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto encoding = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<StorageSpecifierTypeStorage>()) StorageSpecifierTypeStorage(std::move(encoding));
  }

  ::mlir::sparse_tensor::SparseTensorEncodingAttr encoding;
};
} // namespace detail
StorageSpecifierType StorageSpecifierType::get(SparseTensorEncodingAttr encoding) {
  return get(encoding.getContext(), encoding);
}

StorageSpecifierType StorageSpecifierType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, SparseTensorEncodingAttr encoding) {
  return get(encoding.getContext(), encoding);
}

StorageSpecifierType StorageSpecifierType::get(Type type) {
  return get(getSparseTensorEncoding(type));
}

StorageSpecifierType StorageSpecifierType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type type) {
  return get(getSparseTensorEncoding(type));
}

StorageSpecifierType StorageSpecifierType::get(Value tensor) {
  return get(tensor.getType());
}

StorageSpecifierType StorageSpecifierType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Value tensor) {
  return get(tensor.getType());
}

::llvm::LogicalResult StorageSpecifierType::verifyInvariantsImpl(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::sparse_tensor::SparseTensorEncodingAttr encoding) {
  if (!((::llvm::isa<::mlir::sparse_tensor::SparseTensorEncodingAttr>(encoding)))) {
    emitError() << "failed to verify 'encoding': ";
    return ::mlir::failure();
  }

  return ::mlir::success();
}

::llvm::LogicalResult StorageSpecifierType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::sparse_tensor::SparseTensorEncodingAttr encoding) {
  if (::mlir::failed(verifyInvariantsImpl(emitError, encoding)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type StorageSpecifierType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::sparse_tensor::SparseTensorEncodingAttr> _result_encoding;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'encoding'
  odsParser.getContext()->getOrLoadDialect<::mlir::sparse_tensor::SparseTensorDialect>();
  _result_encoding = ::mlir::FieldParser<::mlir::sparse_tensor::SparseTensorEncodingAttr>::parse(odsParser);
  if (::mlir::failed(_result_encoding)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse SparseTensor_StorageSpecifier parameter 'encoding' which is to be a `::mlir::sparse_tensor::SparseTensorEncodingAttr`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_encoding));
  return odsParser.getChecked<StorageSpecifierType>(odsLoc, odsParser.getContext(),
      ::mlir::sparse_tensor::SparseTensorEncodingAttr(::llvm::cast<::mlir::sparse_tensor::SparseTensorEncodingAttr>((*_result_encoding))));
}

void StorageSpecifierType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << getEncoding();
  odsPrinter << ">";
}

::mlir::sparse_tensor::SparseTensorEncodingAttr StorageSpecifierType::getEncoding() const {
  return getImpl()->encoding;
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::StorageSpecifierType)
namespace mlir {
namespace sparse_tensor {
namespace detail {
struct IterSpaceTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::mlir::sparse_tensor::SparseTensorEncodingAttr, Level, Level>;
  IterSpaceTypeStorage(::mlir::sparse_tensor::SparseTensorEncodingAttr encoding, Level loLvl, Level hiLvl) : encoding(std::move(encoding)), loLvl(std::move(loLvl)), hiLvl(std::move(hiLvl)) {}

  KeyTy getAsKey() const {
    return KeyTy(encoding, loLvl, hiLvl);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (encoding == std::get<0>(tblgenKey)) && (loLvl == std::get<1>(tblgenKey)) && (hiLvl == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static IterSpaceTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto encoding = std::move(std::get<0>(tblgenKey));
    auto loLvl = std::move(std::get<1>(tblgenKey));
    auto hiLvl = std::move(std::get<2>(tblgenKey));
    return new (allocator.allocate<IterSpaceTypeStorage>()) IterSpaceTypeStorage(std::move(encoding), std::move(loLvl), std::move(hiLvl));
  }

  ::mlir::sparse_tensor::SparseTensorEncodingAttr encoding;
  Level loLvl;
  Level hiLvl;
};
} // namespace detail
IterSpaceType IterSpaceType::get(::mlir::MLIRContext *context, ::mlir::sparse_tensor::SparseTensorEncodingAttr encoding, Level loLvl, Level hiLvl) {
  return Base::get(context, std::move(encoding), std::move(loLvl), std::move(hiLvl));
}

IterSpaceType IterSpaceType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::mlir::sparse_tensor::SparseTensorEncodingAttr encoding, Level loLvl, Level hiLvl) {
  return Base::getChecked(emitError, context, encoding, loLvl, hiLvl);
}

::llvm::LogicalResult IterSpaceType::verifyInvariantsImpl(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::sparse_tensor::SparseTensorEncodingAttr encoding, Level loLvl, Level hiLvl) {
  if (!((::llvm::isa<::mlir::sparse_tensor::SparseTensorEncodingAttr>(encoding)))) {
    emitError() << "failed to verify 'encoding': ";
    return ::mlir::failure();
  }

  return ::mlir::success();
}

::llvm::LogicalResult IterSpaceType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::sparse_tensor::SparseTensorEncodingAttr encoding, Level loLvl, Level hiLvl) {
  if (::mlir::failed(verifyInvariantsImpl(emitError, encoding, loLvl, hiLvl)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type IterSpaceType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::sparse_tensor::SparseTensorEncodingAttr> _result_encoding;
  ::mlir::FailureOr<Level> _result_loLvl;
  ::mlir::FailureOr<Level> _result_hiLvl;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'encoding'
  odsParser.getContext()->getOrLoadDialect<::mlir::sparse_tensor::SparseTensorDialect>();
  _result_encoding = ::mlir::FieldParser<::mlir::sparse_tensor::SparseTensorEncodingAttr>::parse(odsParser);
  if (::mlir::failed(_result_encoding)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse SparseTensor_IterSpace parameter 'encoding' which is to be a `::mlir::sparse_tensor::SparseTensorEncodingAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal 'lvls'
  if (odsParser.parseKeyword("lvls")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parseLevelRange(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_loLvl),
      ::mlir::detail::unwrapForCustomParse(_result_hiLvl));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_loLvl)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'loLvl'");
      return {};
    }
    if (::mlir::failed(_result_hiLvl)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'hiLvl'");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_encoding));
  assert(::mlir::succeeded(_result_loLvl));
  assert(::mlir::succeeded(_result_hiLvl));
  return odsParser.getChecked<IterSpaceType>(odsLoc, odsParser.getContext(),
      ::mlir::sparse_tensor::SparseTensorEncodingAttr(::llvm::cast<::mlir::sparse_tensor::SparseTensorEncodingAttr>((*_result_encoding))),
      Level((*_result_loLvl)),
      Level((*_result_hiLvl)));
}

void IterSpaceType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getEncoding());
  odsPrinter << ",";
  odsPrinter << ' ' << "lvls";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  printLevelRange(odsPrinter,
    getLoLvl(),
    getHiLvl());
  odsPrinter << ">";
}

::mlir::sparse_tensor::SparseTensorEncodingAttr IterSpaceType::getEncoding() const {
  return getImpl()->encoding;
}

Level IterSpaceType::getLoLvl() const {
  return getImpl()->loLvl;
}

Level IterSpaceType::getHiLvl() const {
  return getImpl()->hiLvl;
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::IterSpaceType)
namespace mlir {
namespace sparse_tensor {
namespace detail {
struct IteratorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::mlir::sparse_tensor::SparseTensorEncodingAttr, Level, Level>;
  IteratorTypeStorage(::mlir::sparse_tensor::SparseTensorEncodingAttr encoding, Level loLvl, Level hiLvl) : encoding(std::move(encoding)), loLvl(std::move(loLvl)), hiLvl(std::move(hiLvl)) {}

  KeyTy getAsKey() const {
    return KeyTy(encoding, loLvl, hiLvl);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (encoding == std::get<0>(tblgenKey)) && (loLvl == std::get<1>(tblgenKey)) && (hiLvl == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static IteratorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto encoding = std::move(std::get<0>(tblgenKey));
    auto loLvl = std::move(std::get<1>(tblgenKey));
    auto hiLvl = std::move(std::get<2>(tblgenKey));
    return new (allocator.allocate<IteratorTypeStorage>()) IteratorTypeStorage(std::move(encoding), std::move(loLvl), std::move(hiLvl));
  }

  ::mlir::sparse_tensor::SparseTensorEncodingAttr encoding;
  Level loLvl;
  Level hiLvl;
};
} // namespace detail
IteratorType IteratorType::get(::mlir::MLIRContext *context, ::mlir::sparse_tensor::SparseTensorEncodingAttr encoding, Level loLvl, Level hiLvl) {
  return Base::get(context, std::move(encoding), std::move(loLvl), std::move(hiLvl));
}

IteratorType IteratorType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::mlir::sparse_tensor::SparseTensorEncodingAttr encoding, Level loLvl, Level hiLvl) {
  return Base::getChecked(emitError, context, encoding, loLvl, hiLvl);
}

::llvm::LogicalResult IteratorType::verifyInvariantsImpl(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::sparse_tensor::SparseTensorEncodingAttr encoding, Level loLvl, Level hiLvl) {
  if (!((::llvm::isa<::mlir::sparse_tensor::SparseTensorEncodingAttr>(encoding)))) {
    emitError() << "failed to verify 'encoding': ";
    return ::mlir::failure();
  }

  return ::mlir::success();
}

::llvm::LogicalResult IteratorType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::sparse_tensor::SparseTensorEncodingAttr encoding, Level loLvl, Level hiLvl) {
  if (::mlir::failed(verifyInvariantsImpl(emitError, encoding, loLvl, hiLvl)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type IteratorType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::sparse_tensor::SparseTensorEncodingAttr> _result_encoding;
  ::mlir::FailureOr<Level> _result_loLvl;
  ::mlir::FailureOr<Level> _result_hiLvl;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'encoding'
  odsParser.getContext()->getOrLoadDialect<::mlir::sparse_tensor::SparseTensorDialect>();
  _result_encoding = ::mlir::FieldParser<::mlir::sparse_tensor::SparseTensorEncodingAttr>::parse(odsParser);
  if (::mlir::failed(_result_encoding)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse SparseTensor_Iterator parameter 'encoding' which is to be a `::mlir::sparse_tensor::SparseTensorEncodingAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal 'lvls'
  if (odsParser.parseKeyword("lvls")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parseLevelRange(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_loLvl),
      ::mlir::detail::unwrapForCustomParse(_result_hiLvl));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_loLvl)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'loLvl'");
      return {};
    }
    if (::mlir::failed(_result_hiLvl)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'hiLvl'");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_encoding));
  assert(::mlir::succeeded(_result_loLvl));
  assert(::mlir::succeeded(_result_hiLvl));
  return odsParser.getChecked<IteratorType>(odsLoc, odsParser.getContext(),
      ::mlir::sparse_tensor::SparseTensorEncodingAttr(::llvm::cast<::mlir::sparse_tensor::SparseTensorEncodingAttr>((*_result_encoding))),
      Level((*_result_loLvl)),
      Level((*_result_hiLvl)));
}

void IteratorType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getEncoding());
  odsPrinter << ",";
  odsPrinter << ' ' << "lvls";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  printLevelRange(odsPrinter,
    getLoLvl(),
    getHiLvl());
  odsPrinter << ">";
}

::mlir::sparse_tensor::SparseTensorEncodingAttr IteratorType::getEncoding() const {
  return getImpl()->encoding;
}

Level IteratorType::getLoLvl() const {
  return getImpl()->loLvl;
}

Level IteratorType::getHiLvl() const {
  return getImpl()->hiLvl;
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::IteratorType)
namespace mlir {
namespace sparse_tensor {

/// Parse a type registered to this dialect.
::mlir::Type SparseTensorDialect::parseType(::mlir::DialectAsmParser &parser) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  ::mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  
  parser.emitError(typeLoc) << "unknown  type `"
      << mnemonic << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print a type registered to this dialect.
void SparseTensorDialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}
} // namespace sparse_tensor
} // namespace mlir

#endif  // GET_TYPEDEF_CLASSES

