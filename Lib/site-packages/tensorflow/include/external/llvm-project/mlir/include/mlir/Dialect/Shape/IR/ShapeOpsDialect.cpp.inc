/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ShapeOps.td                                                          *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::shape::ShapeDialect)
namespace mlir {
namespace shape {

ShapeDialect::ShapeDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<ShapeDialect>())
    
     {
  getContext()->loadDialect<arith::ArithDialect>();
  getContext()->loadDialect<tensor::TensorDialect>();
  initialize();
}

ShapeDialect::~ShapeDialect() = default;

} // namespace shape
} // namespace mlir
