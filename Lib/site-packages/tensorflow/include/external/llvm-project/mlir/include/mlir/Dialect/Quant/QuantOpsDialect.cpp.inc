/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: QuantOps.td                                                          *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::quant::QuantizationDialect)
namespace mlir {
namespace quant {

QuantizationDialect::QuantizationDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<QuantizationDialect>())
    
     {
  
  initialize();
}

QuantizationDialect::~QuantizationDialect() = default;

} // namespace quant
} // namespace mlir
