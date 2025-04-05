/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: AffineOps.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineDialect)
namespace mlir {
namespace affine {

AffineDialect::AffineDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<AffineDialect>())
    
     {
  getContext()->loadDialect<arith::ArithDialect>();
  getContext()->loadDialect<ub::UBDialect>();
  initialize();
}

AffineDialect::~AffineDialect() = default;

} // namespace affine
} // namespace mlir
