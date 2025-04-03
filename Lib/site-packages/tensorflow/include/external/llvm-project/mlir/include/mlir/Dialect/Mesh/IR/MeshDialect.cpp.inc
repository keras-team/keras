/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: MeshOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::mesh::MeshDialect)
namespace mlir {
namespace mesh {

MeshDialect::MeshDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<MeshDialect>())
    
     {
  getContext()->loadDialect<arith::ArithDialect>();
  initialize();
}

MeshDialect::~MeshDialect() = default;

} // namespace mesh
} // namespace mlir
