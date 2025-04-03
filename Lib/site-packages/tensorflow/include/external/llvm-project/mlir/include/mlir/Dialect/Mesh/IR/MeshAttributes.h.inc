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
namespace mesh {
class ReductionKindAttr;
class MeshAxesArrayAttr;
namespace detail {
struct ReductionKindAttrStorage;
} // namespace detail
class ReductionKindAttr : public ::mlir::Attribute::AttrBase<ReductionKindAttr, ::mlir::Attribute, detail::ReductionKindAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "mesh.partial";
  static constexpr ::llvm::StringLiteral dialectName = "mesh";
  static ReductionKindAttr get(::mlir::MLIRContext *context, ::mlir::mesh::ReductionKind value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"partial"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::mesh::ReductionKind getValue() const;
};
namespace detail {
struct MeshAxesArrayAttrStorage;
} // namespace detail
class MeshAxesArrayAttr : public ::mlir::Attribute::AttrBase<MeshAxesArrayAttr, ::mlir::Attribute, detail::MeshAxesArrayAttrStorage> {
public:
  using Base::Base;
  size_t size() const { return getAxes().size(); }
  auto begin() const { return getAxes().begin(); }
  auto end() const { return getAxes().end(); }
  static constexpr ::llvm::StringLiteral name = "mesh.axisarray";
  static constexpr ::llvm::StringLiteral dialectName = "mesh";
  static MeshAxesArrayAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<MeshAxesAttr> axes);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"axisarray"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<MeshAxesAttr> getAxes() const;
};
} // namespace mesh
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mesh::ReductionKindAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mesh::MeshAxesArrayAttr)

#endif  // GET_ATTRDEF_CLASSES

