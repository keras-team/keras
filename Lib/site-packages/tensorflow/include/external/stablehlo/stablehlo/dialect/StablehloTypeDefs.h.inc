/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace stablehlo {
class TokenType;
class TokenType : public ::mlir::Type::TypeBase<TokenType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "stablehlo.token";
  static constexpr ::llvm::StringLiteral dialectName = "stablehlo";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"token"};
  }

};
} // namespace stablehlo
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::stablehlo::TokenType)

#endif  // GET_TYPEDEF_CLASSES

