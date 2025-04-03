/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: BuiltinDialect.td                                                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::BuiltinDialect)
namespace mlir {

BuiltinDialect::BuiltinDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<BuiltinDialect>())
    
     {
  
  initialize();
}

BuiltinDialect::~BuiltinDialect() = default;

} // namespace mlir
