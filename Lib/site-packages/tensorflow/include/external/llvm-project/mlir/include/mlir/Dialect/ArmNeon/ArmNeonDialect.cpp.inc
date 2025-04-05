/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ArmNeon.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_neon::ArmNeonDialect)
namespace mlir {
namespace arm_neon {

ArmNeonDialect::ArmNeonDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<ArmNeonDialect>())
    
     {
  
  initialize();
}

ArmNeonDialect::~ArmNeonDialect() = default;

} // namespace arm_neon
} // namespace mlir
