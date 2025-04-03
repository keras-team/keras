/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class DeviceMappingAttrInterface;
namespace detail {
struct DeviceMappingAttrInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    int64_t (*getMappingId)(const Concept *impl, ::mlir::Attribute );
    bool (*isLinearMapping)(const Concept *impl, ::mlir::Attribute );
    int64_t (*getRelativeIndex)(const Concept *impl, ::mlir::Attribute );
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::DeviceMappingAttrInterface;
    Model() : Concept{getMappingId, isLinearMapping, getRelativeIndex} {}

    static inline int64_t getMappingId(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline bool isLinearMapping(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline int64_t getRelativeIndex(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::DeviceMappingAttrInterface;
    FallbackModel() : Concept{getMappingId, isLinearMapping, getRelativeIndex} {}

    static inline int64_t getMappingId(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline bool isLinearMapping(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline int64_t getRelativeIndex(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
  };
};
template <typename ConcreteAttr>
struct DeviceMappingAttrInterfaceTrait;

} // namespace detail
class DeviceMappingAttrInterface : public ::mlir::AttributeInterface<DeviceMappingAttrInterface, detail::DeviceMappingAttrInterfaceInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<DeviceMappingAttrInterface, detail::DeviceMappingAttrInterfaceInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::DeviceMappingAttrInterfaceTrait<ConcreteAttr> {};
  /// Return mapping as an integer from the attribute.
  int64_t getMappingId() const;
  /// Return true if the attribute specifies a linear mapping
  bool isLinearMapping() const;
  /// Return the [0..n) relative index of the attribute depending on its group.
  /// This can be used to index into a contiguous array.
  int64_t getRelativeIndex() const;
};
namespace detail {
  template <typename ConcreteAttr>
  struct DeviceMappingAttrInterfaceTrait : public ::mlir::AttributeInterface<DeviceMappingAttrInterface, detail::DeviceMappingAttrInterfaceInterfaceTraits>::Trait<ConcreteAttr> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteAttr>
int64_t detail::DeviceMappingAttrInterfaceInterfaceTraits::Model<ConcreteAttr>::getMappingId(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getMappingId();
}
template<typename ConcreteAttr>
bool detail::DeviceMappingAttrInterfaceInterfaceTraits::Model<ConcreteAttr>::isLinearMapping(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).isLinearMapping();
}
template<typename ConcreteAttr>
int64_t detail::DeviceMappingAttrInterfaceInterfaceTraits::Model<ConcreteAttr>::getRelativeIndex(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getRelativeIndex();
}
template<typename ConcreteAttr>
int64_t detail::DeviceMappingAttrInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getMappingId(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getMappingId(tablegen_opaque_val);
}
template<typename ConcreteAttr>
bool detail::DeviceMappingAttrInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::isLinearMapping(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->isLinearMapping(tablegen_opaque_val);
}
template<typename ConcreteAttr>
int64_t detail::DeviceMappingAttrInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getRelativeIndex(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getRelativeIndex(tablegen_opaque_val);
}
} // namespace mlir
