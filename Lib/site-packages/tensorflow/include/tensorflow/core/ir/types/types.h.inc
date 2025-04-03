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
namespace tf_type {
class ControlType;
class OpaqueTensorType;
class ControlType : public ::mlir::Type::TypeBase<ControlType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tf_type.control";
  static constexpr ::llvm::StringLiteral dialectName = "tf_type";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"control"};
  }

};
class OpaqueTensorType : public ::mlir::Type::TypeBase<OpaqueTensorType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tf_type.tensor";
  static constexpr ::llvm::StringLiteral dialectName = "tf_type";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensor"};
  }

};
} // namespace tf_type
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_type::ControlType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_type::OpaqueTensorType)

#endif  // GET_TYPEDEF_CLASSES

