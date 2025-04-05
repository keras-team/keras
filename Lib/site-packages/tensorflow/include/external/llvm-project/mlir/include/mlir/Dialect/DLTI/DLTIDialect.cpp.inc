/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: DLTIBase.td                                                          *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::DLTIDialect)
namespace mlir {

DLTIDialect::DLTIDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<DLTIDialect>())
    
     {
  
  initialize();
}

DLTIDialect::~DLTIDialect() = default;

} // namespace mlir
