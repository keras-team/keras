/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::complex::NumberAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::complex::NumberAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::complex::NumberAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::complex::NumberAttr>([&](auto t) {
      printer << ::mlir::complex::NumberAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace complex {
namespace detail {
struct NumberAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::APFloat, ::llvm::APFloat, ::mlir::Type>;
  NumberAttrStorage(::llvm::APFloat real, ::llvm::APFloat imag, ::mlir::Type type) : real(std::move(real)), imag(std::move(imag)), type(std::move(type)) {}

  KeyTy getAsKey() const {
    return KeyTy(real, imag, type);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (real.bitwiseIsEqual(std::get<0>(tblgenKey))) && (imag.bitwiseIsEqual(std::get<1>(tblgenKey))) && (type == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static NumberAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto real = std::move(std::get<0>(tblgenKey));
    auto imag = std::move(std::get<1>(tblgenKey));
    auto type = std::move(std::get<2>(tblgenKey));
    return new (allocator.allocate<NumberAttrStorage>()) NumberAttrStorage(std::move(real), std::move(imag), std::move(type));
  }

  ::llvm::APFloat real;
  ::llvm::APFloat imag;
  ::mlir::Type type;
};
} // namespace detail
NumberAttr NumberAttr::get(mlir::ComplexType type, double real, double imag) {
  auto elementType = ::llvm::cast<FloatType>(type.getElementType());
  APFloat realFloat(real);
  bool unused;
  realFloat.convert(elementType.getFloatSemantics(),
                    APFloat::rmNearestTiesToEven, &unused);
  APFloat imagFloat(imag);
  imagFloat.convert(elementType.getFloatSemantics(),
                    APFloat::rmNearestTiesToEven, &unused);
  return Base::get(type.getContext(), realFloat, imagFloat, type);
}

NumberAttr NumberAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, mlir::ComplexType type, double real, double imag) {
  auto elementType = ::llvm::cast<FloatType>(type.getElementType());
  APFloat realFloat(real);
  bool unused;
  realFloat.convert(elementType.getFloatSemantics(),
                    APFloat::rmNearestTiesToEven, &unused);
  APFloat imagFloat(imag);
  imagFloat.convert(elementType.getFloatSemantics(),
                    APFloat::rmNearestTiesToEven, &unused);
  return Base::getChecked(emitError, type.getContext(), realFloat, imagFloat, type);
}

::llvm::LogicalResult NumberAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::APFloat real, ::llvm::APFloat imag, ::mlir::Type type) {
  if (::mlir::failed(verify(emitError, real, imag, type)))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::APFloat NumberAttr::getReal() const {
  return getImpl()->real;
}

::llvm::APFloat NumberAttr::getImag() const {
  return getImpl()->imag;
}

::mlir::Type NumberAttr::getType() const {
  return getImpl()->type;
}

} // namespace complex
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::complex::NumberAttr)
namespace mlir {
namespace complex {

/// Parse an attribute registered to this dialect.
::mlir::Attribute ComplexDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void ComplexDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace complex
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

