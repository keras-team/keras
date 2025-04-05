/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: LLVMOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LLVMDialect)
namespace mlir {
namespace LLVM {

LLVMDialect::LLVMDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<LLVMDialect>())
    , workgroupAttributionAttrName(context)
     {
  
  initialize();
}

LLVMDialect::~LLVMDialect() = default;

} // namespace LLVM
} // namespace mlir
