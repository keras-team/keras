/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ControlFlowOps.td                                                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::cf::ControlFlowDialect)
namespace mlir {
namespace cf {

ControlFlowDialect::ControlFlowDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<ControlFlowDialect>())
    
     {
  getContext()->loadDialect<arith::ArithDialect>();
  initialize();
}

ControlFlowDialect::~ControlFlowDialect() = default;

} // namespace cf
} // namespace mlir
