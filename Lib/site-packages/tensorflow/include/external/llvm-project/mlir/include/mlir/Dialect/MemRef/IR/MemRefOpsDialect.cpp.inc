/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: MemRefBase.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::MemRefDialect)
namespace mlir {
namespace memref {

MemRefDialect::MemRefDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<MemRefDialect>())
    
     {
  getContext()->loadDialect<arith::ArithDialect>();
  initialize();
}

MemRefDialect::~MemRefDialect() = default;

} // namespace memref
} // namespace mlir
