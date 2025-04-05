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
namespace tfg {
class RegionAttr;
namespace detail {
struct RegionAttrStorage;
} // namespace detail
class RegionAttr : public ::mlir::Attribute::AttrBase<RegionAttr, ::mlir::Attribute, detail::RegionAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tfg.region_attrs";
  static constexpr ::llvm::StringLiteral dialectName = "tfg";
  static RegionAttr get(::mlir::MLIRContext *context, DictionaryAttr attrs, ArrayAttr arg_attrs, ArrayAttr res_attrs);
  static RegionAttr get(DictionaryAttr attrs, ArrayAttr arg_attrs, ArrayAttr res_attrs);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"region_attrs"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  DictionaryAttr getAttrs() const;
  ArrayAttr getArgAttrs() const;
  ArrayAttr getResAttrs() const;
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::RegionAttr)

#endif  // GET_ATTRDEF_CLASSES

