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
namespace ub {
class PoisonAttr;
class PoisonAttr : public ::mlir::Attribute::AttrBase<PoisonAttr, ::mlir::Attribute, ::mlir::AttributeStorage, ::mlir::ub::PoisonAttrInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ub.poison";
  static constexpr ::llvm::StringLiteral dialectName = "ub";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"poison"};
  }

};
} // namespace ub
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ub::PoisonAttr)

#endif  // GET_ATTRDEF_CLASSES

