/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: OpenMPOps.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::OpenMPDialect)
namespace mlir {
namespace omp {

OpenMPDialect::OpenMPDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<OpenMPDialect>())
    
     {
  getContext()->loadDialect<::mlir::LLVM::LLVMDialect, ::mlir::func::FuncDialect>();
  initialize();
}

OpenMPDialect::~OpenMPDialect() = default;

} // namespace omp
} // namespace mlir
