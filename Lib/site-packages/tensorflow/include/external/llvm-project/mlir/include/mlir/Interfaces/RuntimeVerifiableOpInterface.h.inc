/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class RuntimeVerifiableOpInterface;
namespace detail {
struct RuntimeVerifiableOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    void (*generateRuntimeVerification)(const Concept *impl, ::mlir::Operation *, ::mlir::OpBuilder &, ::mlir::Location);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::RuntimeVerifiableOpInterface;
    Model() : Concept{generateRuntimeVerification} {}

    static inline void generateRuntimeVerification(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::RuntimeVerifiableOpInterface;
    FallbackModel() : Concept{generateRuntimeVerification} {}

    static inline void generateRuntimeVerification(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct RuntimeVerifiableOpInterfaceTrait;

} // namespace detail
class RuntimeVerifiableOpInterface : public ::mlir::OpInterface<RuntimeVerifiableOpInterface, detail::RuntimeVerifiableOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<RuntimeVerifiableOpInterface, detail::RuntimeVerifiableOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::RuntimeVerifiableOpInterfaceTrait<ConcreteOp> {};
  /// Generate IR to verify this op at runtime, aborting runtime execution if
  /// verification fails.
  void generateRuntimeVerification(::mlir::OpBuilder & builder, ::mlir::Location loc);

    /// Generate the error message that will be printed to the user when 
    /// verification fails.
    static std::string generateErrorMessage(Operation *op, const std::string &msg);
};
namespace detail {
  template <typename ConcreteOp>
  struct RuntimeVerifiableOpInterfaceTrait : public ::mlir::OpInterface<RuntimeVerifiableOpInterface, detail::RuntimeVerifiableOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
void detail::RuntimeVerifiableOpInterfaceInterfaceTraits::Model<ConcreteOp>::generateRuntimeVerification(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).generateRuntimeVerification(builder, loc);
}
template<typename ConcreteOp>
void detail::RuntimeVerifiableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::generateRuntimeVerification(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc) {
  return static_cast<const ConcreteOp *>(impl)->generateRuntimeVerification(tablegen_opaque_val, builder, loc);
}
} // namespace mlir
