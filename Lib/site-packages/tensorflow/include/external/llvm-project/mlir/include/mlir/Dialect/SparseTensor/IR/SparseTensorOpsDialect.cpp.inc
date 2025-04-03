/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: SparseTensorOps.td                                                   *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::SparseTensorDialect)
namespace mlir {
namespace sparse_tensor {

SparseTensorDialect::SparseTensorDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<SparseTensorDialect>())
    
     {
  
  initialize();
}

SparseTensorDialect::~SparseTensorDialect() = default;

} // namespace sparse_tensor
} // namespace mlir
