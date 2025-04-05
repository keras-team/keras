/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: dialect.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tf_type::TFTypeDialect)
namespace mlir {
namespace tf_type {

TFTypeDialect::TFTypeDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<TFTypeDialect>())
    
     {
  
  initialize();
}

TFTypeDialect::~TFTypeDialect() = default;

} // namespace tf_type
} // namespace mlir
