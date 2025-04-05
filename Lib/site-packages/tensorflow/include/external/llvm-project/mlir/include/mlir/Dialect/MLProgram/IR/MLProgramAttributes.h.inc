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
namespace ml_program {
class ExternAttr;
namespace detail {
struct ExternAttrStorage;
} // namespace detail
class ExternAttr : public ::mlir::Attribute::AttrBase<ExternAttr, ::mlir::Attribute, detail::ExternAttrStorage, ::mlir::TypedAttr::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ml_program.extern";
  static constexpr ::llvm::StringLiteral dialectName = "ml_program";
  static ExternAttr get(::mlir::MLIRContext *context, ::mlir::Type type);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"extern"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::Type getType() const;
};
} // namespace ml_program
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ml_program::ExternAttr)

#endif  // GET_ATTRDEF_CLASSES

