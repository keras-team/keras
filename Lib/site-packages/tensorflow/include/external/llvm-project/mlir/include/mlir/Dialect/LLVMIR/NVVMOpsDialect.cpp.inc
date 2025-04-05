/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: NVVMOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::NVVMDialect)
namespace mlir {
namespace NVVM {

NVVMDialect::NVVMDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<NVVMDialect>())
    
     {
  getContext()->loadDialect<LLVM::LLVMDialect>();
  initialize();
}

NVVMDialect::~NVVMDialect() = default;

} // namespace NVVM
} // namespace mlir
