/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace hlo {
class BoundedAttrInterface;
namespace detail {
struct BoundedAttrInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::llvm::ArrayRef<int64_t> (*getBounds)(const Concept *impl, ::mlir::Attribute );
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::hlo::BoundedAttrInterface;
    Model() : Concept{getBounds} {}

    static inline ::llvm::ArrayRef<int64_t> getBounds(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::hlo::BoundedAttrInterface;
    FallbackModel() : Concept{getBounds} {}

    static inline ::llvm::ArrayRef<int64_t> getBounds(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
  };
};
template <typename ConcreteAttr>
struct BoundedAttrInterfaceTrait;

} // namespace detail
class BoundedAttrInterface : public ::mlir::AttributeInterface<BoundedAttrInterface, detail::BoundedAttrInterfaceInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<BoundedAttrInterface, detail::BoundedAttrInterfaceInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::BoundedAttrInterfaceTrait<ConcreteAttr> {};
  /// Get the attribute's bounds
  ::llvm::ArrayRef<int64_t> getBounds() const;
};
namespace detail {
  template <typename ConcreteAttr>
  struct BoundedAttrInterfaceTrait : public ::mlir::AttributeInterface<BoundedAttrInterface, detail::BoundedAttrInterfaceInterfaceTraits>::Trait<ConcreteAttr> {
  };
}// namespace detail
} // namespace hlo
} // namespace mlir
namespace mlir {
namespace hlo {
template<typename ConcreteAttr>
::llvm::ArrayRef<int64_t> detail::BoundedAttrInterfaceInterfaceTraits::Model<ConcreteAttr>::getBounds(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getBounds();
}
template<typename ConcreteAttr>
::llvm::ArrayRef<int64_t> detail::BoundedAttrInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getBounds(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getBounds(tablegen_opaque_val);
}
} // namespace hlo
} // namespace mlir
