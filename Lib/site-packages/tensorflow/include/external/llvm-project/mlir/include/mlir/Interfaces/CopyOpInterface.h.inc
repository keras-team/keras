/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class CopyOpInterface;
namespace detail {
struct CopyOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getSource)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getTarget)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::CopyOpInterface;
    Model() : Concept{getSource, getTarget} {}

    static inline ::mlir::Value getSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getTarget(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::CopyOpInterface;
    FallbackModel() : Concept{getSource, getTarget} {}

    static inline ::mlir::Value getSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getTarget(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct CopyOpInterfaceTrait;

} // namespace detail
class CopyOpInterface : public ::mlir::OpInterface<CopyOpInterface, detail::CopyOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<CopyOpInterface, detail::CopyOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::CopyOpInterfaceTrait<ConcreteOp> {};
  /// Returns the source value for this copy operation
  ::mlir::Value getSource();
  /// Returns the target value for this copy operation
  ::mlir::Value getTarget();
};
namespace detail {
  template <typename ConcreteOp>
  struct CopyOpInterfaceTrait : public ::mlir::OpInterface<CopyOpInterface, detail::CopyOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::mlir::Value detail::CopyOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSource();
}
template<typename ConcreteOp>
::mlir::Value detail::CopyOpInterfaceInterfaceTraits::Model<ConcreteOp>::getTarget(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTarget();
}
template<typename ConcreteOp>
::mlir::Value detail::CopyOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSource(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::CopyOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getTarget(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getTarget(tablegen_opaque_val);
}
} // namespace mlir
