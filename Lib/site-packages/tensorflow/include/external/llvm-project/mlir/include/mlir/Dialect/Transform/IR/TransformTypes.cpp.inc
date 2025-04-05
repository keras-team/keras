/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::transform::AffineMapParamType,
::mlir::transform::AnyOpType,
::mlir::transform::AnyValueType,
::mlir::transform::OperationType,
::mlir::transform::AnyParamType,
::mlir::transform::ParamType,
::mlir::transform::TypeParamType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::transform::AffineMapParamType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::transform::AffineMapParamType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::transform::AnyOpType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::transform::AnyOpType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::transform::AnyValueType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::transform::AnyValueType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::transform::OperationType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::transform::OperationType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::transform::AnyParamType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::transform::AnyParamType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::transform::ParamType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::transform::ParamType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::transform::TypeParamType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::transform::TypeParamType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::mlir::transform::AffineMapParamType>([&](auto t) {
      printer << ::mlir::transform::AffineMapParamType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::transform::AnyOpType>([&](auto t) {
      printer << ::mlir::transform::AnyOpType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::transform::AnyValueType>([&](auto t) {
      printer << ::mlir::transform::AnyValueType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::transform::OperationType>([&](auto t) {
      printer << ::mlir::transform::OperationType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::transform::AnyParamType>([&](auto t) {
      printer << ::mlir::transform::AnyParamType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::transform::ParamType>([&](auto t) {
      printer << ::mlir::transform::ParamType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::transform::TypeParamType>([&](auto t) {
      printer << ::mlir::transform::TypeParamType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace transform {
::mlir::Type AffineMapParamType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  return AffineMapParamType::get(odsParser.getContext());
}

void AffineMapParamType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::AffineMapParamType)
namespace mlir {
namespace transform {
::mlir::Type AnyOpType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  return AnyOpType::get(odsParser.getContext());
}

void AnyOpType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::AnyOpType)
namespace mlir {
namespace transform {
::mlir::Type AnyValueType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  return AnyValueType::get(odsParser.getContext());
}

void AnyValueType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::AnyValueType)
namespace mlir {
namespace transform {
namespace detail {
struct OperationTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::StringRef>;
  OperationTypeStorage(::llvm::StringRef operation_name) : operation_name(std::move(operation_name)) {}

  KeyTy getAsKey() const {
    return KeyTy(operation_name);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (operation_name == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static OperationTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto operation_name = std::move(std::get<0>(tblgenKey));
    operation_name = allocator.copyInto(operation_name);
    return new (allocator.allocate<OperationTypeStorage>()) OperationTypeStorage(std::move(operation_name));
  }

  ::llvm::StringRef operation_name;
};
} // namespace detail
OperationType OperationType::get(::mlir::MLIRContext *context, ::llvm::StringRef operation_name) {
  return Base::get(context, std::move(operation_name));
}

::mlir::Type OperationType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<std::string> _result_operation_name;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'operation_name'
  _result_operation_name = ::mlir::FieldParser<std::string>::parse(odsParser);
  if (::mlir::failed(_result_operation_name)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Transform_OperationType parameter 'operation_name' which is to be a `::llvm::StringRef`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_operation_name));
  return OperationType::get(odsParser.getContext(),
      ::llvm::StringRef((*_result_operation_name)));
}

void OperationType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printString(getOperationName());;
  odsPrinter << ">";
}

::llvm::StringRef OperationType::getOperationName() const {
  return getImpl()->operation_name;
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::OperationType)
namespace mlir {
namespace transform {
::mlir::Type AnyParamType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  return AnyParamType::get(odsParser.getContext());
}

void AnyParamType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::AnyParamType)
namespace mlir {
namespace transform {
namespace detail {
struct ParamTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::mlir::Type>;
  ParamTypeStorage(::mlir::Type type) : type(std::move(type)) {}

  KeyTy getAsKey() const {
    return KeyTy(type);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (type == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ParamTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto type = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ParamTypeStorage>()) ParamTypeStorage(std::move(type));
  }

  ::mlir::Type type;
};
} // namespace detail
ParamType ParamType::get(::mlir::MLIRContext *context, ::mlir::Type type) {
  return Base::get(context, std::move(type));
}

ParamType ParamType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::mlir::Type type) {
  return Base::getChecked(emitError, context, type);
}

::llvm::LogicalResult ParamType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::Type type) {
  if (::mlir::failed(verify(emitError, type)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type ParamType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::Type> _result_type;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'type'
  _result_type = ::mlir::FieldParser<::mlir::Type>::parse(odsParser);
  if (::mlir::failed(_result_type)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Transform_ParamType parameter 'type' which is to be a `::mlir::Type`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_type));
  return odsParser.getChecked<ParamType>(odsLoc, odsParser.getContext(),
      ::mlir::Type((*_result_type)));
}

void ParamType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getType());
  odsPrinter << ">";
}

::mlir::Type ParamType::getType() const {
  return getImpl()->type;
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::ParamType)
namespace mlir {
namespace transform {
::mlir::Type TypeParamType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  return TypeParamType::get(odsParser.getContext());
}

void TypeParamType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::TypeParamType)

#endif  // GET_TYPEDEF_CLASSES

