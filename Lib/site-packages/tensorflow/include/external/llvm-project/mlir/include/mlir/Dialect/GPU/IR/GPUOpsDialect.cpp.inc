/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: GPUOps.td                                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::GPUDialect)
namespace mlir {
namespace gpu {

GPUDialect::GPUDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<GPUDialect>())
    , knownBlockSizeAttrName(context), knownGridSizeAttrName(context)
     {
  getContext()->loadDialect<arith::ArithDialect>();
  initialize();
}

GPUDialect::~GPUDialect() = default;

} // namespace gpu
} // namespace mlir
