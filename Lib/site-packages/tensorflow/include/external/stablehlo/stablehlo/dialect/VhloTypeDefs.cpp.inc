/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::vhlo::BooleanV1Type,
::mlir::vhlo::ComplexV1Type,
::mlir::vhlo::FloatBF16V1Type,
::mlir::vhlo::FloatF16V1Type,
::mlir::vhlo::FloatF32V1Type,
::mlir::vhlo::FloatF64V1Type,
::mlir::vhlo::FloatF8E3M4V1Type,
::mlir::vhlo::FloatF8E4M3V1Type,
::mlir::vhlo::FloatF8E4M3FNV1Type,
::mlir::vhlo::FloatF8E5M2V1Type,
::mlir::vhlo::FloatF8E4M3FNUZV1Type,
::mlir::vhlo::FloatF8E4M3B11FNUZV1Type,
::mlir::vhlo::FloatF8E5M2FNUZV1Type,
::mlir::vhlo::FloatTF32V1Type,
::mlir::vhlo::FunctionV1Type,
::mlir::vhlo::IndexV1Type,
::mlir::vhlo::IntegerSI2V1Type,
::mlir::vhlo::IntegerSI4V1Type,
::mlir::vhlo::IntegerSI8V1Type,
::mlir::vhlo::IntegerSI16V1Type,
::mlir::vhlo::IntegerSI32V1Type,
::mlir::vhlo::IntegerSI64V1Type,
::mlir::vhlo::IntegerUI2V1Type,
::mlir::vhlo::IntegerUI4V1Type,
::mlir::vhlo::IntegerUI8V1Type,
::mlir::vhlo::IntegerUI16V1Type,
::mlir::vhlo::IntegerUI32V1Type,
::mlir::vhlo::IntegerUI64V1Type,
::mlir::vhlo::NoneV1Type,
::mlir::vhlo::RankedTensorV1Type,
::mlir::vhlo::TokenV1Type,
::mlir::vhlo::TupleV1Type,
::mlir::vhlo::UniformQuantizedV1Type,
::mlir::vhlo::UniformQuantizedPerAxisV1Type,
::mlir::vhlo::UnrankedTensorV1Type,
::mlir::vhlo::WitnessV1Type

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::vhlo::BooleanV1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::BooleanV1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::ComplexV1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::ComplexV1Type::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::FloatBF16V1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::FloatBF16V1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::FloatF16V1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::FloatF16V1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::FloatF32V1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::FloatF32V1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::FloatF64V1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::FloatF64V1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::FloatF8E3M4V1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::FloatF8E3M4V1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::FloatF8E4M3V1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::FloatF8E4M3V1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::FloatF8E4M3FNV1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::FloatF8E4M3FNV1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::FloatF8E5M2V1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::FloatF8E5M2V1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::FloatF8E4M3FNUZV1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::FloatF8E4M3FNUZV1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::FloatF8E4M3B11FNUZV1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::FloatF8E4M3B11FNUZV1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::FloatF8E5M2FNUZV1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::FloatF8E5M2FNUZV1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::FloatTF32V1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::FloatTF32V1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::FunctionV1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::FunctionV1Type::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::IndexV1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::IndexV1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::IntegerSI2V1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::IntegerSI2V1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::IntegerSI4V1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::IntegerSI4V1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::IntegerSI8V1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::IntegerSI8V1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::IntegerSI16V1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::IntegerSI16V1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::IntegerSI32V1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::IntegerSI32V1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::IntegerSI64V1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::IntegerSI64V1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::IntegerUI2V1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::IntegerUI2V1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::IntegerUI4V1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::IntegerUI4V1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::IntegerUI8V1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::IntegerUI8V1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::IntegerUI16V1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::IntegerUI16V1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::IntegerUI32V1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::IntegerUI32V1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::IntegerUI64V1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::IntegerUI64V1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::NoneV1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::NoneV1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::RankedTensorV1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::RankedTensorV1Type::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::TokenV1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::TokenV1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::TupleV1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::TupleV1Type::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::UniformQuantizedV1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::UniformQuantizedV1Type::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::UniformQuantizedPerAxisV1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::UniformQuantizedPerAxisV1Type::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::UnrankedTensorV1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::UnrankedTensorV1Type::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::vhlo::WitnessV1Type::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::vhlo::WitnessV1Type::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::mlir::vhlo::BooleanV1Type>([&](auto t) {
      printer << ::mlir::vhlo::BooleanV1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::ComplexV1Type>([&](auto t) {
      printer << ::mlir::vhlo::ComplexV1Type::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::FloatBF16V1Type>([&](auto t) {
      printer << ::mlir::vhlo::FloatBF16V1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::FloatF16V1Type>([&](auto t) {
      printer << ::mlir::vhlo::FloatF16V1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::FloatF32V1Type>([&](auto t) {
      printer << ::mlir::vhlo::FloatF32V1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::FloatF64V1Type>([&](auto t) {
      printer << ::mlir::vhlo::FloatF64V1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::FloatF8E3M4V1Type>([&](auto t) {
      printer << ::mlir::vhlo::FloatF8E3M4V1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::FloatF8E4M3V1Type>([&](auto t) {
      printer << ::mlir::vhlo::FloatF8E4M3V1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::FloatF8E4M3FNV1Type>([&](auto t) {
      printer << ::mlir::vhlo::FloatF8E4M3FNV1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::FloatF8E5M2V1Type>([&](auto t) {
      printer << ::mlir::vhlo::FloatF8E5M2V1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::FloatF8E4M3FNUZV1Type>([&](auto t) {
      printer << ::mlir::vhlo::FloatF8E4M3FNUZV1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::FloatF8E4M3B11FNUZV1Type>([&](auto t) {
      printer << ::mlir::vhlo::FloatF8E4M3B11FNUZV1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::FloatF8E5M2FNUZV1Type>([&](auto t) {
      printer << ::mlir::vhlo::FloatF8E5M2FNUZV1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::FloatTF32V1Type>([&](auto t) {
      printer << ::mlir::vhlo::FloatTF32V1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::FunctionV1Type>([&](auto t) {
      printer << ::mlir::vhlo::FunctionV1Type::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::IndexV1Type>([&](auto t) {
      printer << ::mlir::vhlo::IndexV1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::IntegerSI2V1Type>([&](auto t) {
      printer << ::mlir::vhlo::IntegerSI2V1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::IntegerSI4V1Type>([&](auto t) {
      printer << ::mlir::vhlo::IntegerSI4V1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::IntegerSI8V1Type>([&](auto t) {
      printer << ::mlir::vhlo::IntegerSI8V1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::IntegerSI16V1Type>([&](auto t) {
      printer << ::mlir::vhlo::IntegerSI16V1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::IntegerSI32V1Type>([&](auto t) {
      printer << ::mlir::vhlo::IntegerSI32V1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::IntegerSI64V1Type>([&](auto t) {
      printer << ::mlir::vhlo::IntegerSI64V1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::IntegerUI2V1Type>([&](auto t) {
      printer << ::mlir::vhlo::IntegerUI2V1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::IntegerUI4V1Type>([&](auto t) {
      printer << ::mlir::vhlo::IntegerUI4V1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::IntegerUI8V1Type>([&](auto t) {
      printer << ::mlir::vhlo::IntegerUI8V1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::IntegerUI16V1Type>([&](auto t) {
      printer << ::mlir::vhlo::IntegerUI16V1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::IntegerUI32V1Type>([&](auto t) {
      printer << ::mlir::vhlo::IntegerUI32V1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::IntegerUI64V1Type>([&](auto t) {
      printer << ::mlir::vhlo::IntegerUI64V1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::NoneV1Type>([&](auto t) {
      printer << ::mlir::vhlo::NoneV1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::RankedTensorV1Type>([&](auto t) {
      printer << ::mlir::vhlo::RankedTensorV1Type::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::TokenV1Type>([&](auto t) {
      printer << ::mlir::vhlo::TokenV1Type::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::TupleV1Type>([&](auto t) {
      printer << ::mlir::vhlo::TupleV1Type::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::UniformQuantizedV1Type>([&](auto t) {
      printer << ::mlir::vhlo::UniformQuantizedV1Type::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::UniformQuantizedPerAxisV1Type>([&](auto t) {
      printer << ::mlir::vhlo::UniformQuantizedPerAxisV1Type::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::UnrankedTensorV1Type>([&](auto t) {
      printer << ::mlir::vhlo::UnrankedTensorV1Type::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::vhlo::WitnessV1Type>([&](auto t) {
      printer << ::mlir::vhlo::WitnessV1Type::getMnemonic();
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::BooleanV1Type)
namespace mlir {
namespace vhlo {
namespace detail {
struct ComplexV1TypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Type>;
  ComplexV1TypeStorage(Type elementType) : elementType(std::move(elementType)) {}

  KeyTy getAsKey() const {
    return KeyTy(elementType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ComplexV1TypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto elementType = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ComplexV1TypeStorage>()) ComplexV1TypeStorage(std::move(elementType));
  }

  Type elementType;
};
} // namespace detail
LogicalResult ComplexV1Type::verify(
    llvm::function_ref<mlir::InFlightDiagnostic ()> errFn, mlir::Type elementType) {
  if (!isFromVhlo(elementType)) return errFn() << "expected VHLO type";
  return success();
}
ComplexV1Type ComplexV1Type::get(::mlir::MLIRContext *context, Type elementType) {
  return Base::get(context, std::move(elementType));
}

ComplexV1Type ComplexV1Type::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Type elementType) {
  return Base::getChecked(emitError, context, elementType);
}

::llvm::LogicalResult ComplexV1Type::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType) {
  if (::mlir::failed(verify(emitError, elementType)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type ComplexV1Type::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Type> _result_elementType;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'elementType'
  _result_elementType = ::mlir::FieldParser<Type>::parse(odsParser);
  if (::mlir::failed(_result_elementType)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_ComplexV1 parameter 'elementType' which is to be a `Type`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_elementType));
  return odsParser.getChecked<ComplexV1Type>(odsLoc, odsParser.getContext(),
      Type((*_result_elementType)));
}

void ComplexV1Type::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getElementType());
  odsPrinter << ">";
}

Type ComplexV1Type::getElementType() const {
  return getImpl()->elementType;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::ComplexV1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatBF16V1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatF16V1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatF32V1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatF64V1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatF8E3M4V1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatF8E4M3V1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatF8E4M3FNV1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatF8E5M2V1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatF8E4M3FNUZV1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatF8E4M3B11FNUZV1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatF8E5M2FNUZV1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::FloatTF32V1Type)
namespace mlir {
namespace vhlo {
namespace detail {
struct FunctionV1TypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<::mlir::Type>, ::llvm::ArrayRef<mlir::Type>>;
  FunctionV1TypeStorage(::llvm::ArrayRef<::mlir::Type> inputs, ::llvm::ArrayRef<mlir::Type> outputs) : inputs(std::move(inputs)), outputs(std::move(outputs)) {}

  KeyTy getAsKey() const {
    return KeyTy(inputs, outputs);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (inputs == std::get<0>(tblgenKey)) && (outputs == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static FunctionV1TypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto inputs = std::move(std::get<0>(tblgenKey));
    auto outputs = std::move(std::get<1>(tblgenKey));
    inputs = allocator.copyInto(inputs);
    outputs = allocator.copyInto(outputs);
    return new (allocator.allocate<FunctionV1TypeStorage>()) FunctionV1TypeStorage(std::move(inputs), std::move(outputs));
  }

  ::llvm::ArrayRef<::mlir::Type> inputs;
  ::llvm::ArrayRef<mlir::Type> outputs;
};
} // namespace detail
LogicalResult FunctionV1Type::verify(
    llvm::function_ref<mlir::InFlightDiagnostic ()> errFn,
    ArrayRef<mlir::Type> inputs, ArrayRef<mlir::Type> outputs) {
  if (!allFromVhlo(inputs) || !allFromVhlo(outputs))
    return errFn() << "expected VHLO types";
  return success();
}
FunctionV1Type FunctionV1Type::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<::mlir::Type> inputs, ::llvm::ArrayRef<mlir::Type> outputs) {
  return Base::get(context, std::move(inputs), std::move(outputs));
}

FunctionV1Type FunctionV1Type::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<::mlir::Type> inputs, ::llvm::ArrayRef<mlir::Type> outputs) {
  return Base::getChecked(emitError, context, inputs, outputs);
}

::llvm::LogicalResult FunctionV1Type::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<::mlir::Type> inputs, ::llvm::ArrayRef<mlir::Type> outputs) {
  if (::mlir::failed(verify(emitError, inputs, outputs)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type FunctionV1Type::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<::mlir::Type>> _result_inputs;
  ::mlir::FailureOr<::llvm::SmallVector<mlir::Type>> _result_outputs;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse literal '('
  if (odsParser.parseLParen()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parseTypeArray(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_inputs));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_inputs)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'inputs'");
      return {};
    }
  }
  // Parse literal ')'
  if (odsParser.parseRParen()) return {};
  // Parse literal '->'
  if (odsParser.parseArrow()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parseTypeArray(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_outputs));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_outputs)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'outputs'");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_inputs));
  assert(::mlir::succeeded(_result_outputs));
  return odsParser.getChecked<FunctionV1Type>(odsLoc, odsParser.getContext(),
      ::llvm::ArrayRef<::mlir::Type>((*_result_inputs)),
      ::llvm::ArrayRef<mlir::Type>((*_result_outputs)));
}

void FunctionV1Type::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << "(";
  printTypeArray(odsPrinter,
    getInputs());
  odsPrinter << ")";
  odsPrinter << ' ' << "->";
  odsPrinter << ' ';
  printTypeArray(odsPrinter,
    getOutputs());
  odsPrinter << ">";
}

::llvm::ArrayRef<::mlir::Type> FunctionV1Type::getInputs() const {
  return getImpl()->inputs;
}

::llvm::ArrayRef<mlir::Type> FunctionV1Type::getOutputs() const {
  return getImpl()->outputs;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::FunctionV1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::IndexV1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerSI2V1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerSI4V1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerSI8V1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerSI16V1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerSI32V1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerSI64V1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerUI2V1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerUI4V1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerUI8V1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerUI16V1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerUI32V1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::IntegerUI64V1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::NoneV1Type)
namespace mlir {
namespace vhlo {
namespace detail {
struct RankedTensorV1TypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, ::mlir::Type, ::mlir::Attribute>;
  RankedTensorV1TypeStorage(::llvm::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::mlir::Attribute encoding) : shape(std::move(shape)), elementType(std::move(elementType)), encoding(std::move(encoding)) {}

  KeyTy getAsKey() const {
    return KeyTy(shape, elementType, encoding);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (elementType == std::get<1>(tblgenKey)) && (encoding == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static RankedTensorV1TypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto shape = std::move(std::get<0>(tblgenKey));
    auto elementType = std::move(std::get<1>(tblgenKey));
    auto encoding = std::move(std::get<2>(tblgenKey));
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<RankedTensorV1TypeStorage>()) RankedTensorV1TypeStorage(std::move(shape), std::move(elementType), std::move(encoding));
  }

  ::llvm::ArrayRef<int64_t> shape;
  ::mlir::Type elementType;
  ::mlir::Attribute encoding;
};
} // namespace detail
LogicalResult RankedTensorV1Type::verify(
    llvm::function_ref<mlir::InFlightDiagnostic ()> errFn,
    ArrayRef<int64_t> shape, ::mlir::Type elementType, ::mlir::Attribute encoding) {
  if (!isFromVhlo(elementType) || (encoding && !isFromVhlo(encoding)))
    return errFn() << "expected VHLO type or attribute";
  return success();
}
RankedTensorV1Type RankedTensorV1Type::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::mlir::Attribute encoding) {
  return Base::get(context, std::move(shape), std::move(elementType), std::move(encoding));
}

RankedTensorV1Type RankedTensorV1Type::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::mlir::Attribute encoding) {
  return Base::getChecked(emitError, context, shape, elementType, encoding);
}

::llvm::LogicalResult RankedTensorV1Type::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, ::mlir::Type elementType, ::mlir::Attribute encoding) {
  if (::mlir::failed(verify(emitError, shape, elementType, encoding)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type RankedTensorV1Type::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<int64_t>> _result_shape;
  ::mlir::FailureOr<::mlir::Type> _result_elementType;
  ::mlir::FailureOr<::mlir::Attribute> _result_encoding;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parseShape(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_shape));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_shape)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'shape'");
      return {};
    }
  }

  // Parse variable 'elementType'
  _result_elementType = ::mlir::FieldParser<::mlir::Type>::parse(odsParser);
  if (::mlir::failed(_result_elementType)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_RankedTensorV1 parameter 'elementType' which is to be a `::mlir::Type`");
    return {};
  }
  {
    auto odsCustomLoc = odsParser.getCurrentLocation();
    (void)odsCustomLoc;
    auto odsCustomResult = parseEncoding(odsParser,
      ::mlir::detail::unwrapForCustomParse(_result_encoding));
    if (::mlir::failed(odsCustomResult)) return {};
    if (::mlir::failed(_result_encoding)) {
      odsParser.emitError(odsCustomLoc, "custom parser failed to parse parameter 'encoding'");
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_shape));
  assert(::mlir::succeeded(_result_elementType));
  assert(::mlir::succeeded(_result_encoding));
  return odsParser.getChecked<RankedTensorV1Type>(odsLoc, odsParser.getContext(),
      ::llvm::ArrayRef<int64_t>((*_result_shape)),
      ::mlir::Type((*_result_elementType)),
      ::mlir::Attribute((*_result_encoding)));
}

void RankedTensorV1Type::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  printShape(odsPrinter,
    getShape());
  odsPrinter.printStrippedAttrOrType(getElementType());
  printEncoding(odsPrinter,
    getEncoding());
  odsPrinter << ">";
}

::llvm::ArrayRef<int64_t> RankedTensorV1Type::getShape() const {
  return getImpl()->shape;
}

::mlir::Type RankedTensorV1Type::getElementType() const {
  return getImpl()->elementType;
}

::mlir::Attribute RankedTensorV1Type::getEncoding() const {
  return getImpl()->encoding;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::RankedTensorV1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::TokenV1Type)
namespace mlir {
namespace vhlo {
namespace detail {
struct TupleV1TypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<::mlir::Type>>;
  TupleV1TypeStorage(::llvm::ArrayRef<::mlir::Type> types) : types(std::move(types)) {}

  KeyTy getAsKey() const {
    return KeyTy(types);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (types == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TupleV1TypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto types = std::move(std::get<0>(tblgenKey));
    types = allocator.copyInto(types);
    return new (allocator.allocate<TupleV1TypeStorage>()) TupleV1TypeStorage(std::move(types));
  }

  ::llvm::ArrayRef<::mlir::Type> types;
};
} // namespace detail
LogicalResult TupleV1Type::verify(
    llvm::function_ref<mlir::InFlightDiagnostic ()> errFn, ArrayRef<mlir::Type> types) {
  if (!allFromVhlo(types)) return errFn() << "expected VHLO types";
  return success();
}
TupleV1Type TupleV1Type::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<::mlir::Type> types) {
  return Base::get(context, std::move(types));
}

TupleV1Type TupleV1Type::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<::mlir::Type> types) {
  return Base::getChecked(emitError, context, types);
}

::llvm::LogicalResult TupleV1Type::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<::mlir::Type> types) {
  if (::mlir::failed(verify(emitError, types)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type TupleV1Type::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<::mlir::Type>> _result_types;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'types'
  _result_types = ::mlir::FieldParser<::llvm::SmallVector<::mlir::Type>>::parse(odsParser);
  if (::mlir::failed(_result_types)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_TupleV1 parameter 'types' which is to be a `::llvm::ArrayRef<::mlir::Type>`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_types));
  return odsParser.getChecked<TupleV1Type>(odsLoc, odsParser.getContext(),
      ::llvm::ArrayRef<::mlir::Type>((*_result_types)));
}

void TupleV1Type::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getTypes());
  odsPrinter << ">";
}

::llvm::ArrayRef<::mlir::Type> TupleV1Type::getTypes() const {
  return getImpl()->types;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::TupleV1Type)
namespace mlir {
namespace vhlo {
namespace detail {
struct UniformQuantizedV1TypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<unsigned, ::mlir::Type, ::mlir::Type, ::llvm::APFloat, int64_t, int64_t, int64_t>;
  UniformQuantizedV1TypeStorage(unsigned flags, ::mlir::Type storageType, ::mlir::Type expressedType, ::llvm::APFloat scale, int64_t zeroPoint, int64_t storageTypeMin, int64_t storageTypeMax) : flags(std::move(flags)), storageType(std::move(storageType)), expressedType(std::move(expressedType)), scale(std::move(scale)), zeroPoint(std::move(zeroPoint)), storageTypeMin(std::move(storageTypeMin)), storageTypeMax(std::move(storageTypeMax)) {}

  KeyTy getAsKey() const {
    return KeyTy(flags, storageType, expressedType, scale, zeroPoint, storageTypeMin, storageTypeMax);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (flags == std::get<0>(tblgenKey)) && (storageType == std::get<1>(tblgenKey)) && (expressedType == std::get<2>(tblgenKey)) && (scale.bitwiseIsEqual(std::get<3>(tblgenKey))) && (zeroPoint == std::get<4>(tblgenKey)) && (storageTypeMin == std::get<5>(tblgenKey)) && (storageTypeMax == std::get<6>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey), std::get<5>(tblgenKey), std::get<6>(tblgenKey));
  }

  static UniformQuantizedV1TypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto flags = std::move(std::get<0>(tblgenKey));
    auto storageType = std::move(std::get<1>(tblgenKey));
    auto expressedType = std::move(std::get<2>(tblgenKey));
    auto scale = std::move(std::get<3>(tblgenKey));
    auto zeroPoint = std::move(std::get<4>(tblgenKey));
    auto storageTypeMin = std::move(std::get<5>(tblgenKey));
    auto storageTypeMax = std::move(std::get<6>(tblgenKey));
    return new (allocator.allocate<UniformQuantizedV1TypeStorage>()) UniformQuantizedV1TypeStorage(std::move(flags), std::move(storageType), std::move(expressedType), std::move(scale), std::move(zeroPoint), std::move(storageTypeMin), std::move(storageTypeMax));
  }

  unsigned flags;
  ::mlir::Type storageType;
  ::mlir::Type expressedType;
  ::llvm::APFloat scale;
  int64_t zeroPoint;
  int64_t storageTypeMin;
  int64_t storageTypeMax;
};
} // namespace detail
LogicalResult UniformQuantizedV1Type::verify(
    llvm::function_ref<mlir::InFlightDiagnostic ()> errFn,
    unsigned int, mlir::Type storageType, mlir::Type expressedType,
    llvm::APFloat, int64_t, int64_t, int64_t) {
  if (!isFromVhlo(storageType) || !isFromVhlo(expressedType))
    return errFn() << "expected VHLO type";
  return success();
}
UniformQuantizedV1Type UniformQuantizedV1Type::get(::mlir::MLIRContext *context, unsigned flags, ::mlir::Type storageType, ::mlir::Type expressedType, ::llvm::APFloat scale, int64_t zeroPoint, int64_t storageTypeMin, int64_t storageTypeMax) {
  return Base::get(context, std::move(flags), std::move(storageType), std::move(expressedType), std::move(scale), std::move(zeroPoint), std::move(storageTypeMin), std::move(storageTypeMax));
}

UniformQuantizedV1Type UniformQuantizedV1Type::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned flags, ::mlir::Type storageType, ::mlir::Type expressedType, ::llvm::APFloat scale, int64_t zeroPoint, int64_t storageTypeMin, int64_t storageTypeMax) {
  return Base::getChecked(emitError, context, flags, storageType, expressedType, scale, zeroPoint, storageTypeMin, storageTypeMax);
}

::llvm::LogicalResult UniformQuantizedV1Type::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned flags, ::mlir::Type storageType, ::mlir::Type expressedType, ::llvm::APFloat scale, int64_t zeroPoint, int64_t storageTypeMin, int64_t storageTypeMax) {
  if (::mlir::failed(verify(emitError, flags, storageType, expressedType, scale, zeroPoint, storageTypeMin, storageTypeMax)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type UniformQuantizedV1Type::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<unsigned> _result_flags;
  ::mlir::FailureOr<::mlir::Type> _result_storageType;
  ::mlir::FailureOr<::mlir::Type> _result_expressedType;
  ::mlir::FailureOr<::llvm::APFloat> _result_scale;
  ::mlir::FailureOr<int64_t> _result_zeroPoint;
  ::mlir::FailureOr<int64_t> _result_storageTypeMin;
  ::mlir::FailureOr<int64_t> _result_storageTypeMax;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'storageType'
  _result_storageType = ::mlir::FieldParser<::mlir::Type>::parse(odsParser);
  if (::mlir::failed(_result_storageType)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_UniformQuantizedV1 parameter 'storageType' which is to be a `::mlir::Type`");
    return {};
  }
  // Parse literal ':'
  if (odsParser.parseColon()) return {};

  // Parse variable 'expressedType'
  _result_expressedType = ::mlir::FieldParser<::mlir::Type>::parse(odsParser);
  if (::mlir::failed(_result_expressedType)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_UniformQuantizedV1 parameter 'expressedType' which is to be a `::mlir::Type`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'scale'
  _result_scale = 
      [&]() -> FailureOr<llvm::APFloat> {
        double value;
        if (failed(odsParser.parseFloat(value))) {
          return failure();
        }
        return APFloat(value);
      }()
    ;
  if (::mlir::failed(_result_scale)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_UniformQuantizedV1 parameter 'scale' which is to be a `::llvm::APFloat`");
    return {};
  }
  // Parse literal ':'
  if (odsParser.parseColon()) return {};

  // Parse variable 'zeroPoint'
  _result_zeroPoint = ::mlir::FieldParser<int64_t>::parse(odsParser);
  if (::mlir::failed(_result_zeroPoint)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_UniformQuantizedV1 parameter 'zeroPoint' which is to be a `int64_t`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'storageTypeMin'
  _result_storageTypeMin = ::mlir::FieldParser<int64_t>::parse(odsParser);
  if (::mlir::failed(_result_storageTypeMin)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_UniformQuantizedV1 parameter 'storageTypeMin' which is to be a `int64_t`");
    return {};
  }
  // Parse literal ':'
  if (odsParser.parseColon()) return {};

  // Parse variable 'storageTypeMax'
  _result_storageTypeMax = ::mlir::FieldParser<int64_t>::parse(odsParser);
  if (::mlir::failed(_result_storageTypeMax)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_UniformQuantizedV1 parameter 'storageTypeMax' which is to be a `int64_t`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'flags'
  _result_flags = ::mlir::FieldParser<unsigned>::parse(odsParser);
  if (::mlir::failed(_result_flags)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_UniformQuantizedV1 parameter 'flags' which is to be a `unsigned`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_flags));
  assert(::mlir::succeeded(_result_storageType));
  assert(::mlir::succeeded(_result_expressedType));
  assert(::mlir::succeeded(_result_scale));
  assert(::mlir::succeeded(_result_zeroPoint));
  assert(::mlir::succeeded(_result_storageTypeMin));
  assert(::mlir::succeeded(_result_storageTypeMax));
  return odsParser.getChecked<UniformQuantizedV1Type>(odsLoc, odsParser.getContext(),
      unsigned((*_result_flags)),
      ::mlir::Type((*_result_storageType)),
      ::mlir::Type((*_result_expressedType)),
      ::llvm::APFloat((*_result_scale)),
      int64_t((*_result_zeroPoint)),
      int64_t((*_result_storageTypeMin)),
      int64_t((*_result_storageTypeMax)));
}

void UniformQuantizedV1Type::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getStorageType());
  odsPrinter << ":";
  odsPrinter.printStrippedAttrOrType(getExpressedType());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printFloat(getScale());;
  odsPrinter << ":";
  odsPrinter.printStrippedAttrOrType(getZeroPoint());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getStorageTypeMin());
  odsPrinter << ":";
  odsPrinter.printStrippedAttrOrType(getStorageTypeMax());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getFlags());
  odsPrinter << ">";
}

unsigned UniformQuantizedV1Type::getFlags() const {
  return getImpl()->flags;
}

::mlir::Type UniformQuantizedV1Type::getStorageType() const {
  return getImpl()->storageType;
}

::mlir::Type UniformQuantizedV1Type::getExpressedType() const {
  return getImpl()->expressedType;
}

::llvm::APFloat UniformQuantizedV1Type::getScale() const {
  return getImpl()->scale;
}

int64_t UniformQuantizedV1Type::getZeroPoint() const {
  return getImpl()->zeroPoint;
}

int64_t UniformQuantizedV1Type::getStorageTypeMin() const {
  return getImpl()->storageTypeMin;
}

int64_t UniformQuantizedV1Type::getStorageTypeMax() const {
  return getImpl()->storageTypeMax;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::UniformQuantizedV1Type)
namespace mlir {
namespace vhlo {
namespace detail {
struct UniformQuantizedPerAxisV1TypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<unsigned, ::mlir::Type, ::mlir::Type, int32_t, ::llvm::ArrayRef<::llvm::APFloat>, ::llvm::ArrayRef<int64_t>, int64_t, int64_t>;
  UniformQuantizedPerAxisV1TypeStorage(unsigned flags, ::mlir::Type storageType, ::mlir::Type expressedType, int32_t quantizedDimension, ::llvm::ArrayRef<::llvm::APFloat> scales, ::llvm::ArrayRef<int64_t> zeroPoints, int64_t storageTypeMin, int64_t storageTypeMax) : flags(std::move(flags)), storageType(std::move(storageType)), expressedType(std::move(expressedType)), quantizedDimension(std::move(quantizedDimension)), scales(std::move(scales)), zeroPoints(std::move(zeroPoints)), storageTypeMin(std::move(storageTypeMin)), storageTypeMax(std::move(storageTypeMax)) {}

  KeyTy getAsKey() const {
    return KeyTy(flags, storageType, expressedType, quantizedDimension, scales, zeroPoints, storageTypeMin, storageTypeMax);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (flags == std::get<0>(tblgenKey)) && (storageType == std::get<1>(tblgenKey)) && (expressedType == std::get<2>(tblgenKey)) && (quantizedDimension == std::get<3>(tblgenKey)) && (scales == std::get<4>(tblgenKey)) && (zeroPoints == std::get<5>(tblgenKey)) && (storageTypeMin == std::get<6>(tblgenKey)) && (storageTypeMax == std::get<7>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey), std::get<5>(tblgenKey), std::get<6>(tblgenKey), std::get<7>(tblgenKey));
  }

  static UniformQuantizedPerAxisV1TypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto flags = std::move(std::get<0>(tblgenKey));
    auto storageType = std::move(std::get<1>(tblgenKey));
    auto expressedType = std::move(std::get<2>(tblgenKey));
    auto quantizedDimension = std::move(std::get<3>(tblgenKey));
    auto scales = std::move(std::get<4>(tblgenKey));
    auto zeroPoints = std::move(std::get<5>(tblgenKey));
    auto storageTypeMin = std::move(std::get<6>(tblgenKey));
    auto storageTypeMax = std::move(std::get<7>(tblgenKey));
    scales = allocator.copyInto(scales);
    zeroPoints = allocator.copyInto(zeroPoints);
    return new (allocator.allocate<UniformQuantizedPerAxisV1TypeStorage>()) UniformQuantizedPerAxisV1TypeStorage(std::move(flags), std::move(storageType), std::move(expressedType), std::move(quantizedDimension), std::move(scales), std::move(zeroPoints), std::move(storageTypeMin), std::move(storageTypeMax));
  }

  unsigned flags;
  ::mlir::Type storageType;
  ::mlir::Type expressedType;
  int32_t quantizedDimension;
  ::llvm::ArrayRef<::llvm::APFloat> scales;
  ::llvm::ArrayRef<int64_t> zeroPoints;
  int64_t storageTypeMin;
  int64_t storageTypeMax;
};
} // namespace detail
LogicalResult UniformQuantizedPerAxisV1Type::verify(
    llvm::function_ref<mlir::InFlightDiagnostic ()> errFn,
    unsigned int, mlir::Type storageType, mlir::Type expressedType,
    int32_t, ::llvm::ArrayRef<::llvm::APFloat>, ::llvm::ArrayRef<int64_t>, int64_t, int64_t) {
  if (!isFromVhlo(storageType) || !isFromVhlo(expressedType))
    return errFn() << "expected VHLO type";
  return success();
}
UniformQuantizedPerAxisV1Type UniformQuantizedPerAxisV1Type::get(::mlir::MLIRContext *context, unsigned flags, ::mlir::Type storageType, ::mlir::Type expressedType, int32_t quantizedDimension, ::llvm::ArrayRef<::llvm::APFloat> scales, ::llvm::ArrayRef<int64_t> zeroPoints, int64_t storageTypeMin, int64_t storageTypeMax) {
  return Base::get(context, std::move(flags), std::move(storageType), std::move(expressedType), std::move(quantizedDimension), std::move(scales), std::move(zeroPoints), std::move(storageTypeMin), std::move(storageTypeMax));
}

UniformQuantizedPerAxisV1Type UniformQuantizedPerAxisV1Type::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned flags, ::mlir::Type storageType, ::mlir::Type expressedType, int32_t quantizedDimension, ::llvm::ArrayRef<::llvm::APFloat> scales, ::llvm::ArrayRef<int64_t> zeroPoints, int64_t storageTypeMin, int64_t storageTypeMax) {
  return Base::getChecked(emitError, context, flags, storageType, expressedType, quantizedDimension, scales, zeroPoints, storageTypeMin, storageTypeMax);
}

::llvm::LogicalResult UniformQuantizedPerAxisV1Type::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned flags, ::mlir::Type storageType, ::mlir::Type expressedType, int32_t quantizedDimension, ::llvm::ArrayRef<::llvm::APFloat> scales, ::llvm::ArrayRef<int64_t> zeroPoints, int64_t storageTypeMin, int64_t storageTypeMax) {
  if (::mlir::failed(verify(emitError, flags, storageType, expressedType, quantizedDimension, scales, zeroPoints, storageTypeMin, storageTypeMax)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type UniformQuantizedPerAxisV1Type::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<unsigned> _result_flags;
  ::mlir::FailureOr<::mlir::Type> _result_storageType;
  ::mlir::FailureOr<::mlir::Type> _result_expressedType;
  ::mlir::FailureOr<int32_t> _result_quantizedDimension;
  ::mlir::FailureOr<::llvm::SmallVector<::llvm::APFloat>> _result_scales;
  ::mlir::FailureOr<::llvm::SmallVector<int64_t>> _result_zeroPoints;
  ::mlir::FailureOr<int64_t> _result_storageTypeMin;
  ::mlir::FailureOr<int64_t> _result_storageTypeMax;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'storageType'
  _result_storageType = ::mlir::FieldParser<::mlir::Type>::parse(odsParser);
  if (::mlir::failed(_result_storageType)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_UniformQuantizedPerAxisV1 parameter 'storageType' which is to be a `::mlir::Type`");
    return {};
  }
  // Parse literal ':'
  if (odsParser.parseColon()) return {};

  // Parse variable 'expressedType'
  _result_expressedType = ::mlir::FieldParser<::mlir::Type>::parse(odsParser);
  if (::mlir::failed(_result_expressedType)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_UniformQuantizedPerAxisV1 parameter 'expressedType' which is to be a `::mlir::Type`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'quantizedDimension'
  _result_quantizedDimension = ::mlir::FieldParser<int32_t>::parse(odsParser);
  if (::mlir::failed(_result_quantizedDimension)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_UniformQuantizedPerAxisV1 parameter 'quantizedDimension' which is to be a `int32_t`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'scales'
  _result_scales = 
      [&]() -> FailureOr<llvm::SmallVector<::llvm::APFloat>> {
        ::llvm::SmallVector<double> scales;

        auto parseResult = odsParser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
          return odsParser.parseFloat(scales.emplace_back());
        });
        if(failed(parseResult)) return failure();
        return llvm::map_to_vector(
          scales, [](double scale) { return APFloat(scale); });
      }()
    ;
  if (::mlir::failed(_result_scales)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_UniformQuantizedPerAxisV1 parameter 'scales' which is to be a `::llvm::ArrayRef<::llvm::APFloat>`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal '['
  if (odsParser.parseLSquare()) return {};

  // Parse variable 'zeroPoints'
  _result_zeroPoints = ::mlir::FieldParser<::llvm::SmallVector<int64_t>>::parse(odsParser);
  if (::mlir::failed(_result_zeroPoints)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_UniformQuantizedPerAxisV1 parameter 'zeroPoints' which is to be a `::llvm::ArrayRef<int64_t>`");
    return {};
  }
  // Parse literal ']'
  if (odsParser.parseRSquare()) return {};
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'storageTypeMin'
  _result_storageTypeMin = ::mlir::FieldParser<int64_t>::parse(odsParser);
  if (::mlir::failed(_result_storageTypeMin)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_UniformQuantizedPerAxisV1 parameter 'storageTypeMin' which is to be a `int64_t`");
    return {};
  }
  // Parse literal ':'
  if (odsParser.parseColon()) return {};

  // Parse variable 'storageTypeMax'
  _result_storageTypeMax = ::mlir::FieldParser<int64_t>::parse(odsParser);
  if (::mlir::failed(_result_storageTypeMax)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_UniformQuantizedPerAxisV1 parameter 'storageTypeMax' which is to be a `int64_t`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'flags'
  _result_flags = ::mlir::FieldParser<unsigned>::parse(odsParser);
  if (::mlir::failed(_result_flags)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_UniformQuantizedPerAxisV1 parameter 'flags' which is to be a `unsigned`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_flags));
  assert(::mlir::succeeded(_result_storageType));
  assert(::mlir::succeeded(_result_expressedType));
  assert(::mlir::succeeded(_result_quantizedDimension));
  assert(::mlir::succeeded(_result_scales));
  assert(::mlir::succeeded(_result_zeroPoints));
  assert(::mlir::succeeded(_result_storageTypeMin));
  assert(::mlir::succeeded(_result_storageTypeMax));
  return odsParser.getChecked<UniformQuantizedPerAxisV1Type>(odsLoc, odsParser.getContext(),
      unsigned((*_result_flags)),
      ::mlir::Type((*_result_storageType)),
      ::mlir::Type((*_result_expressedType)),
      int32_t((*_result_quantizedDimension)),
      ::llvm::ArrayRef<::llvm::APFloat>((*_result_scales)),
      ::llvm::ArrayRef<int64_t>((*_result_zeroPoints)),
      int64_t((*_result_storageTypeMin)),
      int64_t((*_result_storageTypeMax)));
}

void UniformQuantizedPerAxisV1Type::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getStorageType());
  odsPrinter << ":";
  odsPrinter.printStrippedAttrOrType(getExpressedType());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getQuantizedDimension());
  odsPrinter << ",";
  odsPrinter << ' ';

      odsPrinter << '[';
      llvm::interleaveComma(getScales(), odsPrinter, [&](APFloat scale) {
        odsPrinter << scale;
      });
      odsPrinter << ']';
    ;
  odsPrinter << ",";
  odsPrinter << ' ' << "[";
  odsPrinter.printStrippedAttrOrType(getZeroPoints());
  odsPrinter << "]";
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getStorageTypeMin());
  odsPrinter << ":";
  odsPrinter.printStrippedAttrOrType(getStorageTypeMax());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getFlags());
  odsPrinter << ">";
}

unsigned UniformQuantizedPerAxisV1Type::getFlags() const {
  return getImpl()->flags;
}

::mlir::Type UniformQuantizedPerAxisV1Type::getStorageType() const {
  return getImpl()->storageType;
}

::mlir::Type UniformQuantizedPerAxisV1Type::getExpressedType() const {
  return getImpl()->expressedType;
}

int32_t UniformQuantizedPerAxisV1Type::getQuantizedDimension() const {
  return getImpl()->quantizedDimension;
}

::llvm::ArrayRef<::llvm::APFloat> UniformQuantizedPerAxisV1Type::getScales() const {
  return getImpl()->scales;
}

::llvm::ArrayRef<int64_t> UniformQuantizedPerAxisV1Type::getZeroPoints() const {
  return getImpl()->zeroPoints;
}

int64_t UniformQuantizedPerAxisV1Type::getStorageTypeMin() const {
  return getImpl()->storageTypeMin;
}

int64_t UniformQuantizedPerAxisV1Type::getStorageTypeMax() const {
  return getImpl()->storageTypeMax;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::UniformQuantizedPerAxisV1Type)
namespace mlir {
namespace vhlo {
namespace detail {
struct UnrankedTensorV1TypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::mlir::Type>;
  UnrankedTensorV1TypeStorage(::mlir::Type elementType) : elementType(std::move(elementType)) {}

  KeyTy getAsKey() const {
    return KeyTy(elementType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (elementType == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static UnrankedTensorV1TypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto elementType = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<UnrankedTensorV1TypeStorage>()) UnrankedTensorV1TypeStorage(std::move(elementType));
  }

  ::mlir::Type elementType;
};
} // namespace detail
LogicalResult UnrankedTensorV1Type::verify(
    llvm::function_ref<mlir::InFlightDiagnostic ()> errFn, mlir::Type elementType) {
  if (!isFromVhlo(elementType)) return errFn() << "expected VHLO type";
  return success();
}
UnrankedTensorV1Type UnrankedTensorV1Type::get(::mlir::MLIRContext *context, ::mlir::Type elementType) {
  return Base::get(context, std::move(elementType));
}

UnrankedTensorV1Type UnrankedTensorV1Type::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::mlir::Type elementType) {
  return Base::getChecked(emitError, context, elementType);
}

::llvm::LogicalResult UnrankedTensorV1Type::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::Type elementType) {
  if (::mlir::failed(verify(emitError, elementType)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Type UnrankedTensorV1Type::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::Type> _result_elementType;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'elementType'
  _result_elementType = ::mlir::FieldParser<::mlir::Type>::parse(odsParser);
  if (::mlir::failed(_result_elementType)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse VHLO_UnrankedTensorV1 parameter 'elementType' which is to be a `::mlir::Type`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_elementType));
  return odsParser.getChecked<UnrankedTensorV1Type>(odsLoc, odsParser.getContext(),
      ::mlir::Type((*_result_elementType)));
}

void UnrankedTensorV1Type::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getElementType());
  odsPrinter << ">";
}

::mlir::Type UnrankedTensorV1Type::getElementType() const {
  return getImpl()->elementType;
}

} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::UnrankedTensorV1Type)
namespace mlir {
namespace vhlo {
} // namespace vhlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::vhlo::WitnessV1Type)

#endif  // GET_TYPEDEF_CLASSES

