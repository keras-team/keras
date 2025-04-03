/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: SCFOps.td                                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::scf::SCFDialect)
namespace mlir {
namespace scf {

SCFDialect::SCFDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<SCFDialect>())
    
     {
  getContext()->loadDialect<arith::ArithDialect>();
  initialize();
}

SCFDialect::~SCFDialect() = default;

} // namespace scf
} // namespace mlir
