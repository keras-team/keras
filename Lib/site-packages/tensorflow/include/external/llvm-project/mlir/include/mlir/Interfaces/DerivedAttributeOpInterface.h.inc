/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class DerivedAttributeOpInterface;
namespace detail {
struct DerivedAttributeOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    bool (*isDerivedAttribute)(::mlir::StringRef);
    ::mlir::DictionaryAttr (*materializeDerivedAttributes)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::DerivedAttributeOpInterface;
    Model() : Concept{isDerivedAttribute, materializeDerivedAttributes} {}

    static inline bool isDerivedAttribute(::mlir::StringRef name);
    static inline ::mlir::DictionaryAttr materializeDerivedAttributes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::DerivedAttributeOpInterface;
    FallbackModel() : Concept{isDerivedAttribute, materializeDerivedAttributes} {}

    static inline bool isDerivedAttribute(::mlir::StringRef name);
    static inline ::mlir::DictionaryAttr materializeDerivedAttributes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct DerivedAttributeOpInterfaceTrait;

} // namespace detail
class DerivedAttributeOpInterface : public ::mlir::OpInterface<DerivedAttributeOpInterface, detail::DerivedAttributeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<DerivedAttributeOpInterface, detail::DerivedAttributeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::DerivedAttributeOpInterfaceTrait<ConcreteOp> {};
  /// Returns whether name corresponds to a derived attribute.
  bool isDerivedAttribute(::mlir::StringRef name);
  /// Materializes the derived attributes. Returns null attribute where
  /// unable to materialize a derived attribute as attribute.
  ::mlir::DictionaryAttr materializeDerivedAttributes();
};
namespace detail {
  template <typename ConcreteOp>
  struct DerivedAttributeOpInterfaceTrait : public ::mlir::OpInterface<DerivedAttributeOpInterface, detail::DerivedAttributeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
bool detail::DerivedAttributeOpInterfaceInterfaceTraits::Model<ConcreteOp>::isDerivedAttribute(::mlir::StringRef name) {
  return ConcreteOp::isDerivedAttribute(name);
}
template<typename ConcreteOp>
::mlir::DictionaryAttr detail::DerivedAttributeOpInterfaceInterfaceTraits::Model<ConcreteOp>::materializeDerivedAttributes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).materializeDerivedAttributes();
}
template<typename ConcreteOp>
bool detail::DerivedAttributeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isDerivedAttribute(::mlir::StringRef name) {
  return ConcreteOp::isDerivedAttribute(name);
}
template<typename ConcreteOp>
::mlir::DictionaryAttr detail::DerivedAttributeOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::materializeDerivedAttributes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->materializeDerivedAttributes(tablegen_opaque_val);
}
} // namespace mlir
