/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ControlFlowOps.td                                                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace cf {

class ControlFlowDialect : public ::mlir::Dialect {
  explicit ControlFlowDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~ControlFlowDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("cf");
  }
};
} // namespace cf
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::cf::ControlFlowDialect)
