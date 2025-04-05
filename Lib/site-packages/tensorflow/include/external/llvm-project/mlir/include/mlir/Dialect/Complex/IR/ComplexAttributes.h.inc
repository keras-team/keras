/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace complex {
class NumberAttr;
namespace detail {
struct NumberAttrStorage;
} // namespace detail
class NumberAttr : public ::mlir::Attribute::AttrBase<NumberAttr, ::mlir::Attribute, detail::NumberAttrStorage, ::mlir::TypedAttr::Trait> {
public:
  using Base::Base;
  std::complex<APFloat> getValue() {
    return std::complex<APFloat>(getReal(), getImag());
  }
  static constexpr ::llvm::StringLiteral name = "complex.number";
  static constexpr ::llvm::StringLiteral dialectName = "complex";
  using Base::getChecked;
  static NumberAttr get(mlir::ComplexType type, double real, double imag);
  static NumberAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, mlir::ComplexType type, double real, double imag);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::APFloat real, ::llvm::APFloat imag, ::mlir::Type type);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::APFloat real, ::llvm::APFloat imag, ::mlir::Type type);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"number"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::APFloat getReal() const;
  ::llvm::APFloat getImag() const;
  ::mlir::Type getType() const;
};
} // namespace complex
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::complex::NumberAttr)

#endif  // GET_ATTRDEF_CLASSES

