/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: PDLInterpOps.td                                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace pdl_interp {

class PDLInterpDialect : public ::mlir::Dialect {
  explicit PDLInterpDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~PDLInterpDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("pdl_interp");
  }

    /// Returns the name of the function containing the matcher code. This
    /// function is called by the interpreter when matching an operation.
    static StringRef getMatcherFunctionName() { return "matcher"; }

    /// Returns the name of the module containing the rewrite functions. These
    /// functions are invoked by distinct patterns within the matcher function
    /// to rewrite the IR after a successful match.
    static StringRef getRewriterModuleName() { return "rewriters"; }
  };
} // namespace pdl_interp
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::PDLInterpDialect)
