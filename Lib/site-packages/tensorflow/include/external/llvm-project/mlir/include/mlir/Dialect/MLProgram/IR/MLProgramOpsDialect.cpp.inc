/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: MLProgramOps.td                                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ml_program::MLProgramDialect)
namespace mlir {
namespace ml_program {

MLProgramDialect::MLProgramDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<MLProgramDialect>())
    
     {
  
  initialize();
}

MLProgramDialect::~MLProgramDialect() = default;

} // namespace ml_program
} // namespace mlir
