/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: X86Vector.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace x86vector {

class X86VectorDialect : public ::mlir::Dialect {
  explicit X86VectorDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~X86VectorDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("x86vector");
  }
};
} // namespace x86vector
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::x86vector::X86VectorDialect)
