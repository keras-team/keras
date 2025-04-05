/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::shape::ShapeType,
::mlir::shape::SizeType,
::mlir::shape::ValueShapeType,
::mlir::shape::WitnessType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::shape::ShapeType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::shape::ShapeType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::shape::SizeType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::shape::SizeType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::shape::ValueShapeType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::shape::ValueShapeType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::shape::WitnessType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::shape::WitnessType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::mlir::shape::ShapeType>([&](auto t) {
      printer << ::mlir::shape::ShapeType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::shape::SizeType>([&](auto t) {
      printer << ::mlir::shape::SizeType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::shape::ValueShapeType>([&](auto t) {
      printer << ::mlir::shape::ValueShapeType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::shape::WitnessType>([&](auto t) {
      printer << ::mlir::shape::WitnessType::getMnemonic();
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace shape {
} // namespace shape
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::shape::ShapeType)
namespace mlir {
namespace shape {
} // namespace shape
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::shape::SizeType)
namespace mlir {
namespace shape {
} // namespace shape
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::shape::ValueShapeType)
namespace mlir {
namespace shape {
} // namespace shape
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::shape::WitnessType)
namespace mlir {
namespace shape {

/// Parse a type registered to this dialect.
::mlir::Type ShapeDialect::parseType(::mlir::DialectAsmParser &parser) const {
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
void ShapeDialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}
} // namespace shape
} // namespace mlir

#endif  // GET_TYPEDEF_CLASSES

