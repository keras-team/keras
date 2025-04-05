/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::LLVM::LLVMArrayType,
::mlir::LLVM::LLVMFunctionType,
::mlir::LLVM::LLVMPointerType,
::mlir::LLVM::LLVMFixedVectorType,
::mlir::LLVM::LLVMScalableVectorType,
::mlir::LLVM::LLVMTargetExtType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::LLVM::LLVMArrayType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::LLVM::LLVMArrayType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::LLVM::LLVMFunctionType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::LLVM::LLVMFunctionType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::LLVM::LLVMPointerType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::LLVM::LLVMPointerType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::LLVM::LLVMFixedVectorType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::LLVM::LLVMFixedVectorType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::LLVM::LLVMScalableVectorType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::LLVM::LLVMScalableVectorType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::LLVM::LLVMTargetExtType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::LLVM::LLVMTargetExtType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::mlir::LLVM::LLVMArrayType>([&](auto t) {
      printer << ::mlir::LLVM::LLVMArrayType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::LLVM::LLVMFunctionType>([&](auto t) {
      printer << ::mlir::LLVM::LLVMFunctionType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::LLVM::LLVMPointerType>([&](auto t) {
      printer << ::mlir::LLVM::LLVMPointerType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::LLVM::LLVMFixedVectorType>([&](auto t) {
      printer << ::mlir::LLVM::LLVMFixedVectorType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::LLVM::LLVMScalableVectorType>([&](auto t) {
      printer << ::mlir::LLVM::LLVMScalableVectorType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::LLVM::LLVMTargetExtType>([&](auto t) {
      printer << ::mlir::LLVM::LLVMTargetExtType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace LLVM {
namespace detail {
struct LLVMArrayTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type, uint64_t>;
  LLVMArrayTypeStorage(Type elementType, uint64_t numElements) : elementType(std::move(elementType)), numElements(std::move(numElements)) {}

  KeyTy getAsKey() const {
    return KeyTy(elementType, numElements);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey)) && (numElements == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static LLVMArrayTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto elementType = std::move(std::get<0>(tblgenKey));
    auto numElements = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<LLVMArrayTypeStorage>()) LLVMArrayTypeStorage(std::move(elementType), std::move(numElements));
  }

  Type elementType;
  uint64_t numElements;
};
} // namespace detail
LLVMArrayType LLVMArrayType::get(::mlir::MLIRContext *context, Type elementType, uint64_t numElements) {
  return Base::get(context, std::move(elementType), std::move(numElements));
}

LLVMArrayType LLVMArrayType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Type elementType, uint64_t numElements) {
  return Base::getChecked(emitError, context, elementType, numElements);
}

::llvm::LogicalResult LLVMArrayType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType, uint64_t numElements) {
  if (::mlir::failed(verify(emitError, elementType, numElements)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type LLVMArrayType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Type> _result_elementType;
  ::mlir::FailureOr<uint64_t> _result_numElements;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'numElements'
  _result_numElements = ::mlir::FieldParser<uint64_t>::parse(odsParser);
  if (::mlir::failed(_result_numElements)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse LLVMArrayType parameter 'numElements' which is to be a `uint64_t`");
    return {};
  }
  // Parse literal 'x'
  if (odsParser.parseKeyword("x")) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parsePrettyLLVMType(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_elementType));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_elementType)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'elementType'");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_elementType));
  assert(::mlir::succeeded(_result_numElements));
  return odsParser.getChecked<LLVMArrayType>(odsLoc, odsParser.getContext(),
      Type((*_result_elementType)),
      uint64_t((*_result_numElements)));
}

void LLVMArrayType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getNumElements());
  odsPrinter << ' ' << "x";
  odsPrinter << ' ';
  printPrettyLLVMType(odsPrinter,
    getElementType());
  odsPrinter << ">";
}

Type LLVMArrayType::getElementType() const {
  return getImpl()->elementType;
}

uint64_t LLVMArrayType::getNumElements() const {
  return getImpl()->numElements;
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LLVMArrayType)
namespace mlir {
namespace LLVM {
namespace detail {
struct LLVMFunctionTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type, ::llvm::ArrayRef<Type>, bool>;
  LLVMFunctionTypeStorage(Type returnType, ::llvm::ArrayRef<Type> params, bool varArg) : returnType(std::move(returnType)), params(std::move(params)), varArg(std::move(varArg)) {}

  KeyTy getAsKey() const {
    return KeyTy(returnType, params, varArg);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (returnType == std::get<0>(tblgenKey)) && (params == std::get<1>(tblgenKey)) && (varArg == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static LLVMFunctionTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto returnType = std::move(std::get<0>(tblgenKey));
    auto params = std::move(std::get<1>(tblgenKey));
    auto varArg = std::move(std::get<2>(tblgenKey));
    params = allocator.copyInto(params);
    return new (allocator.allocate<LLVMFunctionTypeStorage>()) LLVMFunctionTypeStorage(std::move(returnType), std::move(params), std::move(varArg));
  }

  Type returnType;
  ::llvm::ArrayRef<Type> params;
  bool varArg;
};
} // namespace detail
LLVMFunctionType LLVMFunctionType::get(::mlir::MLIRContext *context, Type returnType, ::llvm::ArrayRef<Type> params, bool varArg) {
  return Base::get(context, std::move(returnType), std::move(params), std::move(varArg));
}

LLVMFunctionType LLVMFunctionType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Type returnType, ::llvm::ArrayRef<Type> params, bool varArg) {
  return Base::getChecked(emitError, context, returnType, params, varArg);
}

::llvm::LogicalResult LLVMFunctionType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type returnType, ::llvm::ArrayRef<Type> params, bool varArg) {
  if (::mlir::failed(verify(emitError, returnType, params, varArg)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type LLVMFunctionType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Type> _result_returnType;
  ::mlir::FailureOr<::llvm::SmallVector<Type>> _result_params;
  ::mlir::FailureOr<bool> _result_varArg;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parsePrettyLLVMType(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_returnType));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_returnType)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'returnType'");
      return {};
    }
  }
  // Parse literal '('
  if (odsParser.parseLParen()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parseFunctionTypes(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_params),
      ::mlir::detail::unwrapForCustomParse(_result_varArg));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_params)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'params'");
      return {};
    }
    if (::mlir::failed(_result_varArg)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'varArg'");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_returnType));
  assert(::mlir::succeeded(_result_params));
  assert(::mlir::succeeded(_result_varArg));
  return odsParser.getChecked<LLVMFunctionType>(odsLoc, odsParser.getContext(),
      Type((*_result_returnType)),
      ::llvm::ArrayRef<Type>((*_result_params)),
      bool((*_result_varArg)));
}

void LLVMFunctionType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  printPrettyLLVMType(odsPrinter,
    getReturnType());
  odsPrinter << " ";
  odsPrinter << "(";
  printFunctionTypes(odsPrinter,
    getParams(),
    getVarArg());
  odsPrinter << ">";
}

Type LLVMFunctionType::getReturnType() const {
  return getImpl()->returnType;
}

::llvm::ArrayRef<Type> LLVMFunctionType::getParams() const {
  return getImpl()->params;
}

bool LLVMFunctionType::getVarArg() const {
  return getImpl()->varArg;
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LLVMFunctionType)
namespace mlir {
namespace LLVM {
namespace detail {
struct LLVMPointerTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<unsigned>;
  LLVMPointerTypeStorage(unsigned addressSpace) : addressSpace(std::move(addressSpace)) {}

  KeyTy getAsKey() const {
    return KeyTy(addressSpace);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (addressSpace == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static LLVMPointerTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto addressSpace = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<LLVMPointerTypeStorage>()) LLVMPointerTypeStorage(std::move(addressSpace));
  }

  unsigned addressSpace;
};
} // namespace detail
LLVMPointerType LLVMPointerType::get(::mlir::MLIRContext *context, unsigned addressSpace) {
  return Base::get(context, addressSpace);
}

::mlir::Type LLVMPointerType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<unsigned> _result_addressSpace;
  // Parse literal '<'
  if (odsParser.parseOptionalLess()) {
  } else {

    // Parse variable 'addressSpace'
    _result_addressSpace = ::mlir::FieldParser<unsigned>::parse(odsParser);
    if (::mlir::failed(_result_addressSpace)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse LLVMPointerType parameter 'addressSpace' which is to be a `unsigned`");
      return {};
    }
    // Parse literal '>'
    if (odsParser.parseGreater()) return {};
  }
  return LLVMPointerType::get(odsParser.getContext(),
      unsigned((_result_addressSpace.value_or(0))));
}

void LLVMPointerType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  if (!(getAddressSpace() == 0)) {
    odsPrinter << "<";
    if (!(getAddressSpace() == 0)) {
      odsPrinter.printStrippedAttrOrType(getAddressSpace());
    }
    odsPrinter << ">";
  } else {
  }
}

unsigned LLVMPointerType::getAddressSpace() const {
  return getImpl()->addressSpace;
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LLVMPointerType)
namespace mlir {
namespace LLVM {
namespace detail {
struct LLVMFixedVectorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type, unsigned>;
  LLVMFixedVectorTypeStorage(Type elementType, unsigned numElements) : elementType(std::move(elementType)), numElements(std::move(numElements)) {}

  KeyTy getAsKey() const {
    return KeyTy(elementType, numElements);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey)) && (numElements == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static LLVMFixedVectorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto elementType = std::move(std::get<0>(tblgenKey));
    auto numElements = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<LLVMFixedVectorTypeStorage>()) LLVMFixedVectorTypeStorage(std::move(elementType), std::move(numElements));
  }

  Type elementType;
  unsigned numElements;
};
} // namespace detail
LLVMFixedVectorType LLVMFixedVectorType::get(::mlir::MLIRContext *context, Type elementType, unsigned numElements) {
  return Base::get(context, std::move(elementType), std::move(numElements));
}

LLVMFixedVectorType LLVMFixedVectorType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Type elementType, unsigned numElements) {
  return Base::getChecked(emitError, context, elementType, numElements);
}

::llvm::LogicalResult LLVMFixedVectorType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType, unsigned numElements) {
  if (::mlir::failed(verify(emitError, elementType, numElements)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type LLVMFixedVectorType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Type> _result_elementType;
  ::mlir::FailureOr<unsigned> _result_numElements;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'numElements'
  _result_numElements = ::mlir::FieldParser<unsigned>::parse(odsParser);
  if (::mlir::failed(_result_numElements)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse LLVMFixedVectorType parameter 'numElements' which is to be a `unsigned`");
    return {};
  }
  // Parse literal 'x'
  if (odsParser.parseKeyword("x")) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parsePrettyLLVMType(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_elementType));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_elementType)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'elementType'");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_elementType));
  assert(::mlir::succeeded(_result_numElements));
  return odsParser.getChecked<LLVMFixedVectorType>(odsLoc, odsParser.getContext(),
      Type((*_result_elementType)),
      unsigned((*_result_numElements)));
}

void LLVMFixedVectorType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getNumElements());
  odsPrinter << ' ' << "x";
  odsPrinter << ' ';
  printPrettyLLVMType(odsPrinter,
    getElementType());
  odsPrinter << ">";
}

Type LLVMFixedVectorType::getElementType() const {
  return getImpl()->elementType;
}

unsigned LLVMFixedVectorType::getNumElements() const {
  return getImpl()->numElements;
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LLVMFixedVectorType)
namespace mlir {
namespace LLVM {
namespace detail {
struct LLVMScalableVectorTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type, unsigned>;
  LLVMScalableVectorTypeStorage(Type elementType, unsigned minNumElements) : elementType(std::move(elementType)), minNumElements(std::move(minNumElements)) {}

  KeyTy getAsKey() const {
    return KeyTy(elementType, minNumElements);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey)) && (minNumElements == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static LLVMScalableVectorTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto elementType = std::move(std::get<0>(tblgenKey));
    auto minNumElements = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<LLVMScalableVectorTypeStorage>()) LLVMScalableVectorTypeStorage(std::move(elementType), std::move(minNumElements));
  }

  Type elementType;
  unsigned minNumElements;
};
} // namespace detail
LLVMScalableVectorType LLVMScalableVectorType::get(::mlir::MLIRContext *context, Type elementType, unsigned minNumElements) {
  return Base::get(context, std::move(elementType), std::move(minNumElements));
}

LLVMScalableVectorType LLVMScalableVectorType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Type elementType, unsigned minNumElements) {
  return Base::getChecked(emitError, context, elementType, minNumElements);
}

::llvm::LogicalResult LLVMScalableVectorType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType, unsigned minNumElements) {
  if (::mlir::failed(verify(emitError, elementType, minNumElements)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type LLVMScalableVectorType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Type> _result_elementType;
  ::mlir::FailureOr<unsigned> _result_minNumElements;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse literal '?'
  if (odsParser.parseQuestion()) return {};
  // Parse literal 'x'
  if (odsParser.parseKeyword("x")) return {};

  // Parse variable 'minNumElements'
  _result_minNumElements = ::mlir::FieldParser<unsigned>::parse(odsParser);
  if (::mlir::failed(_result_minNumElements)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse LLVMScalableVectorType parameter 'minNumElements' which is to be a `unsigned`");
    return {};
  }
  // Parse literal 'x'
  if (odsParser.parseKeyword("x")) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parsePrettyLLVMType(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_elementType));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_elementType)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'elementType'");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_elementType));
  assert(::mlir::succeeded(_result_minNumElements));
  return odsParser.getChecked<LLVMScalableVectorType>(odsLoc, odsParser.getContext(),
      Type((*_result_elementType)),
      unsigned((*_result_minNumElements)));
}

void LLVMScalableVectorType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << "?";
  odsPrinter << ' ' << "x";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getMinNumElements());
  odsPrinter << ' ' << "x";
  odsPrinter << " ";
  odsPrinter << ' ';
  printPrettyLLVMType(odsPrinter,
    getElementType());
  odsPrinter << ">";
}

Type LLVMScalableVectorType::getElementType() const {
  return getImpl()->elementType;
}

unsigned LLVMScalableVectorType::getMinNumElements() const {
  return getImpl()->minNumElements;
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LLVMScalableVectorType)
namespace mlir {
namespace LLVM {
namespace detail {
struct LLVMTargetExtTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::StringRef, ::llvm::ArrayRef<Type>, ::llvm::ArrayRef<unsigned int>>;
  LLVMTargetExtTypeStorage(::llvm::StringRef extTypeName, ::llvm::ArrayRef<Type> typeParams, ::llvm::ArrayRef<unsigned int> intParams) : extTypeName(std::move(extTypeName)), typeParams(std::move(typeParams)), intParams(std::move(intParams)) {}

  KeyTy getAsKey() const {
    return KeyTy(extTypeName, typeParams, intParams);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (extTypeName == std::get<0>(tblgenKey)) && (::llvm::ArrayRef<Type>(typeParams) == ::llvm::ArrayRef<Type>(std::get<1>(tblgenKey))) && (::llvm::ArrayRef<unsigned int>(intParams) == ::llvm::ArrayRef<unsigned int>(std::get<2>(tblgenKey)));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static LLVMTargetExtTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto extTypeName = std::move(std::get<0>(tblgenKey));
    auto typeParams = std::move(std::get<1>(tblgenKey));
    auto intParams = std::move(std::get<2>(tblgenKey));
    extTypeName = allocator.copyInto(extTypeName);
    typeParams = allocator.copyInto(typeParams);
    intParams = allocator.copyInto(intParams);
    return new (allocator.allocate<LLVMTargetExtTypeStorage>()) LLVMTargetExtTypeStorage(std::move(extTypeName), std::move(typeParams), std::move(intParams));
  }

  ::llvm::StringRef extTypeName;
  ::llvm::ArrayRef<Type> typeParams;
  ::llvm::ArrayRef<unsigned int> intParams;
};
} // namespace detail
LLVMTargetExtType LLVMTargetExtType::get(::mlir::MLIRContext *context, ::llvm::StringRef extTypeName, ::llvm::ArrayRef<Type> typeParams, ::llvm::ArrayRef<unsigned int> intParams) {
  return Base::get(context, std::move(extTypeName), std::move(typeParams), std::move(intParams));
}

::mlir::Type LLVMTargetExtType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<std::string> _result_extTypeName;
  ::mlir::FailureOr<::llvm::SmallVector<Type>> _result_typeParams;
  ::mlir::FailureOr<::llvm::SmallVector<unsigned int>> _result_intParams;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'extTypeName'
  _result_extTypeName = ::mlir::FieldParser<std::string>::parse(odsParser);
  if (::mlir::failed(_result_extTypeName)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse LLVMTargetExtType parameter 'extTypeName' which is to be a `::llvm::StringRef`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseOptionalComma()) {
  } else {
    {
      auto odsCustomLoc = odsParser.getCurrentLocation();
      (void)odsCustomLoc;
      auto odsCustomResult = parseExtTypeParams(odsParser,
        ::mlir::detail::unwrapForCustomParse(_result_typeParams),
        ::mlir::detail::unwrapForCustomParse(_result_intParams));
      if (::mlir::failed(odsCustomResult)) return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_extTypeName));
  return LLVMTargetExtType::get(odsParser.getContext(),
      ::llvm::StringRef((*_result_extTypeName)),
      ::llvm::ArrayRef<Type>((_result_typeParams.value_or(::llvm::SmallVector<Type>()))),
      ::llvm::ArrayRef<unsigned int>((_result_intParams.value_or(::llvm::SmallVector<unsigned int>()))));
}

void LLVMTargetExtType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printString(getExtTypeName());;
  if (!(::llvm::ArrayRef<Type>(getTypeParams()) == ::llvm::ArrayRef<Type>(::llvm::SmallVector<Type>())) || !(::llvm::ArrayRef<unsigned int>(getIntParams()) == ::llvm::ArrayRef<unsigned int>(::llvm::SmallVector<unsigned int>()))) {
    odsPrinter << ",";
    odsPrinter << ' ';
    printExtTypeParams(odsPrinter,
      getTypeParams(),
      getIntParams());
  } else {
  }
  odsPrinter << ">";
}

::llvm::StringRef LLVMTargetExtType::getExtTypeName() const {
  return getImpl()->extTypeName;
}

::llvm::ArrayRef<Type> LLVMTargetExtType::getTypeParams() const {
  return getImpl()->typeParams;
}

::llvm::ArrayRef<unsigned int> LLVMTargetExtType::getIntParams() const {
  return getImpl()->intParams;
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LLVMTargetExtType)

#endif  // GET_TYPEDEF_CLASSES

