/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: MathBase.td                                                          *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::math::MathDialect)
namespace mlir {
namespace math {

MathDialect::MathDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<MathDialect>())
    
     {
  getContext()->loadDialect<::mlir::arith::ArithDialect>();
  initialize();
}

MathDialect::~MathDialect() = default;

} // namespace math
} // namespace mlir
