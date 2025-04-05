/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: FuncOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::func::FuncDialect)
namespace mlir {
namespace func {

FuncDialect::FuncDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<FuncDialect>())
    
     {
  
  initialize();
}

FuncDialect::~FuncDialect() = default;

} // namespace func
} // namespace mlir
