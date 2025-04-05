/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: IndexOps.td                                                          *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::index::IndexDialect)
namespace mlir {
namespace index {

IndexDialect::IndexDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<IndexDialect>())
    
     {
  
  initialize();
}

IndexDialect::~IndexDialect() = default;

} // namespace index
} // namespace mlir
