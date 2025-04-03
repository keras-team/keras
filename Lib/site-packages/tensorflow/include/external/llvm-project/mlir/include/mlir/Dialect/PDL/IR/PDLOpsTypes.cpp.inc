/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::pdl::AttributeType,
::mlir::pdl::OperationType,
::mlir::pdl::RangeType,
::mlir::pdl::TypeType,
::mlir::pdl::ValueType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::pdl::AttributeType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pdl::AttributeType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pdl::OperationType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pdl::OperationType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pdl::RangeType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pdl::RangeType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pdl::TypeType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pdl::TypeType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::pdl::ValueType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::pdl::ValueType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::mlir::pdl::AttributeType>([&](auto t) {
      printer << ::mlir::pdl::AttributeType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::pdl::OperationType>([&](auto t) {
      printer << ::mlir::pdl::OperationType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::pdl::RangeType>([&](auto t) {
      printer << ::mlir::pdl::RangeType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::pdl::TypeType>([&](auto t) {
      printer << ::mlir::pdl::TypeType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::pdl::ValueType>([&](auto t) {
      printer << ::mlir::pdl::ValueType::getMnemonic();
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace pdl {
} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::AttributeType)
namespace mlir {
namespace pdl {
} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::OperationType)
namespace mlir {
namespace pdl {
namespace detail {
struct RangeTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type>;
  RangeTypeStorage(Type elementType) : elementType(std::move(elementType)) {}

  KeyTy getAsKey() const {
    return KeyTy(elementType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static RangeTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto elementType = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<RangeTypeStorage>()) RangeTypeStorage(std::move(elementType));
  }

  Type elementType;
};
} // namespace detail
RangeType RangeType::get(Type elementType) {
  return Base::get(elementType.getContext(), elementType);
}

RangeType RangeType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType) {
  return Base::getChecked(emitError, elementType.getContext(), elementType);
}

::llvm::LogicalResult RangeType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType) {
  if (::mlir::failed(verify(emitError, elementType)))
    return ::mlir::failure();
  return ::mlir::success();
}

Type RangeType::getElementType() const {
  return getImpl()->elementType;
}

} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::RangeType)
namespace mlir {
namespace pdl {
} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::TypeType)
namespace mlir {
namespace pdl {
} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::ValueType)
namespace mlir {
namespace pdl {

/// Parse a type registered to this dialect.
::mlir::Type PDLDialect::parseType(::mlir::DialectAsmParser &parser) const {
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
void PDLDialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}
} // namespace pdl
} // namespace mlir

#endif  // GET_TYPEDEF_CLASSES

