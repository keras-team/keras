/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: BufferizationBase.td                                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::bufferization::BufferizationDialect)
namespace mlir {
namespace bufferization {

BufferizationDialect::BufferizationDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<BufferizationDialect>())
    
     {
  getContext()->loadDialect<affine::AffineDialect>();
  getContext()->loadDialect<memref::MemRefDialect>();
  getContext()->loadDialect<tensor::TensorDialect>();
  getContext()->loadDialect<arith::ArithDialect>();
  initialize();
}

BufferizationDialect::~BufferizationDialect() = default;

} // namespace bufferization
} // namespace mlir
