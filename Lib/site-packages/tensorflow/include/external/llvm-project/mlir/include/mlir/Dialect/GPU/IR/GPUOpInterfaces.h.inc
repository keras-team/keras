/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace gpu {
class AsyncOpInterface;
namespace detail {
struct AsyncOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    OperandRange (*getAsyncDependencies)(const Concept *impl, ::mlir::Operation *);
    void (*addAsyncDependency)(const Concept *impl, ::mlir::Operation *, Value);
    Value (*getAsyncToken)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::gpu::AsyncOpInterface;
    Model() : Concept{getAsyncDependencies, addAsyncDependency, getAsyncToken} {}

    static inline OperandRange getAsyncDependencies(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void addAsyncDependency(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Value token);
    static inline Value getAsyncToken(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::gpu::AsyncOpInterface;
    FallbackModel() : Concept{getAsyncDependencies, addAsyncDependency, getAsyncToken} {}

    static inline OperandRange getAsyncDependencies(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void addAsyncDependency(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Value token);
    static inline Value getAsyncToken(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    OperandRange getAsyncDependencies(::mlir::Operation *tablegen_opaque_val) const;
    void addAsyncDependency(::mlir::Operation *tablegen_opaque_val, Value token) const;
  };
};
template <typename ConcreteOp>
struct AsyncOpInterfaceTrait;

} // namespace detail
class AsyncOpInterface : public ::mlir::OpInterface<AsyncOpInterface, detail::AsyncOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AsyncOpInterface, detail::AsyncOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::AsyncOpInterfaceTrait<ConcreteOp> {};
  /// Query the operands that represent async dependency tokens.
  OperandRange getAsyncDependencies();
  /// Adds a new token to the list of async dependencies if it is not already there.
  void addAsyncDependency(Value token);
  /// Query the result that represents the async token to depend on.
  Value getAsyncToken();
};
namespace detail {
  template <typename ConcreteOp>
  struct AsyncOpInterfaceTrait : public ::mlir::OpInterface<AsyncOpInterface, detail::AsyncOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Query the operands that represent async dependency tokens.
    OperandRange getAsyncDependencies() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getAsyncDependencies();
    }
    /// Adds a new token to the list of async dependencies if it is not already there.
    void addAsyncDependency(Value token) {
      if (!::llvm::is_contained(this->getAsyncDependencies(), token))
          ::mlir::gpu::addAsyncDependency(this->getOperation(), token);
    }
  };
}// namespace detail
} // namespace gpu
} // namespace mlir
namespace mlir {
namespace gpu {
template<typename ConcreteOp>
OperandRange detail::AsyncOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAsyncDependencies(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAsyncDependencies();
}
template<typename ConcreteOp>
void detail::AsyncOpInterfaceInterfaceTraits::Model<ConcreteOp>::addAsyncDependency(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Value token) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).addAsyncDependency(token);
}
template<typename ConcreteOp>
Value detail::AsyncOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAsyncToken(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAsyncToken();
}
template<typename ConcreteOp>
OperandRange detail::AsyncOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAsyncDependencies(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAsyncDependencies(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::AsyncOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::addAsyncDependency(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Value token) {
  return static_cast<const ConcreteOp *>(impl)->addAsyncDependency(tablegen_opaque_val, token);
}
template<typename ConcreteOp>
Value detail::AsyncOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAsyncToken(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAsyncToken(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
OperandRange detail::AsyncOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAsyncDependencies(::mlir::Operation *tablegen_opaque_val) const {
ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getAsyncDependencies();
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::AsyncOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::addAsyncDependency(::mlir::Operation *tablegen_opaque_val, Value token) const {
if (!::llvm::is_contained(this->getAsyncDependencies(), token))
          ::mlir::gpu::addAsyncDependency(this->getOperation(), token);
}
} // namespace gpu
} // namespace mlir
