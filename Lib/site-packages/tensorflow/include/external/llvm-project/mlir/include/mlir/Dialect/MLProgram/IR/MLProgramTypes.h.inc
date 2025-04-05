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
namespace ml_program {
class TokenType;
class TokenType : public ::mlir::Type::TypeBase<TokenType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ml_program.token";
  static constexpr ::llvm::StringLiteral dialectName = "ml_program";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"token"};
  }

};
} // namespace ml_program
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ml_program::TokenType)

#endif  // GET_TYPEDEF_CLASSES

