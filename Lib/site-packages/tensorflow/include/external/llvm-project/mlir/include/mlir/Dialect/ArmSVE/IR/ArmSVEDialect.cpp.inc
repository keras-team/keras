/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ArmSVE.td                                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_sve::ArmSVEDialect)
namespace mlir {
namespace arm_sve {

ArmSVEDialect::ArmSVEDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<ArmSVEDialect>())
    
     {
  getContext()->loadDialect<vector::VectorDialect>();
  initialize();
}

ArmSVEDialect::~ArmSVEDialect() = default;

} // namespace arm_sve
} // namespace mlir
