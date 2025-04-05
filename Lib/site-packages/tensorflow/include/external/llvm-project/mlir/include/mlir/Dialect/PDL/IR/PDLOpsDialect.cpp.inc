/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: PDLOps.td                                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::PDLDialect)
namespace mlir {
namespace pdl {

PDLDialect::PDLDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<PDLDialect>())
    
     {
  
  initialize();
}

PDLDialect::~PDLDialect() = default;

} // namespace pdl
} // namespace mlir
