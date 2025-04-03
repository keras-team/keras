/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TensorOps.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::TensorDialect)
namespace mlir {
namespace tensor {

TensorDialect::TensorDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<TensorDialect>())
    
     {
  getContext()->loadDialect<affine::AffineDialect>();
  getContext()->loadDialect<arith::ArithDialect>();
  getContext()->loadDialect<complex::ComplexDialect>();
  initialize();
}

TensorDialect::~TensorDialect() = default;

} // namespace tensor
} // namespace mlir
