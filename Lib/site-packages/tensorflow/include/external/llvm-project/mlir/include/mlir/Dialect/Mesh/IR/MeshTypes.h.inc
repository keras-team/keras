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
namespace mesh {
class ShardingType;
class ShardingType : public ::mlir::Type::TypeBase<ShardingType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "mesh.sharding";
  static constexpr ::llvm::StringLiteral dialectName = "mesh";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"sharding"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
};
} // namespace mesh
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mesh::ShardingType)

#endif  // GET_TYPEDEF_CLASSES

