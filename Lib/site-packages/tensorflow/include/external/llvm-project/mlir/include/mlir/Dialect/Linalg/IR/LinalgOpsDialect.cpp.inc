/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: LinalgOps.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::linalg::LinalgDialect)
namespace mlir {
namespace linalg {

LinalgDialect::LinalgDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<LinalgDialect>())
    
     {
  getContext()->loadDialect<arith::ArithDialect>();
  getContext()->loadDialect<affine::AffineDialect>();
  getContext()->loadDialect<math::MathDialect>();
  getContext()->loadDialect<memref::MemRefDialect>();
  getContext()->loadDialect<tensor::TensorDialect>();
  initialize();
}

LinalgDialect::~LinalgDialect() = default;

} // namespace linalg
} // namespace mlir
