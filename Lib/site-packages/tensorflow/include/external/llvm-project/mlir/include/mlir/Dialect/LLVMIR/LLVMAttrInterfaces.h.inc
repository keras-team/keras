/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace LLVM {
class DIRecursiveTypeAttrInterface;
namespace detail {
struct DIRecursiveTypeAttrInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    bool (*getIsRecSelf)(const Concept *impl, ::mlir::Attribute );
    DistinctAttr (*getRecId)(const Concept *impl, ::mlir::Attribute );
    DIRecursiveTypeAttrInterface (*withRecId)(const Concept *impl, ::mlir::Attribute , DistinctAttr);
    DIRecursiveTypeAttrInterface (*getRecSelf)(DistinctAttr);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::LLVM::DIRecursiveTypeAttrInterface;
    Model() : Concept{getIsRecSelf, getRecId, withRecId, getRecSelf} {}

    static inline bool getIsRecSelf(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline DistinctAttr getRecId(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline DIRecursiveTypeAttrInterface withRecId(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, DistinctAttr recId);
    static inline DIRecursiveTypeAttrInterface getRecSelf(DistinctAttr recId);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::LLVM::DIRecursiveTypeAttrInterface;
    FallbackModel() : Concept{getIsRecSelf, getRecId, withRecId, getRecSelf} {}

    static inline bool getIsRecSelf(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline DistinctAttr getRecId(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline DIRecursiveTypeAttrInterface withRecId(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, DistinctAttr recId);
    static inline DIRecursiveTypeAttrInterface getRecSelf(DistinctAttr recId);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
  };
};
template <typename ConcreteAttr>
struct DIRecursiveTypeAttrInterfaceTrait;

} // namespace detail
class DIRecursiveTypeAttrInterface : public ::mlir::AttributeInterface<DIRecursiveTypeAttrInterface, detail::DIRecursiveTypeAttrInterfaceInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<DIRecursiveTypeAttrInterface, detail::DIRecursiveTypeAttrInterfaceInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::DIRecursiveTypeAttrInterfaceTrait<ConcreteAttr> {};
  /// Get whether this attr describes a recursive self reference.
  bool getIsRecSelf() const;
  /// Get the recursive ID used for matching "rec-decl" with "rec-self".
  /// If this attr instance is not recursive, return a null attribute.
  DistinctAttr getRecId() const;
  /// Get a copy of this type attr but with the recursive ID set to `recId`.
  DIRecursiveTypeAttrInterface withRecId(DistinctAttr recId) const;
  /// Build a rec-self instance using the provided `recId`.
  DIRecursiveTypeAttrInterface getRecSelf(DistinctAttr recId) const;
};
namespace detail {
  template <typename ConcreteAttr>
  struct DIRecursiveTypeAttrInterfaceTrait : public ::mlir::AttributeInterface<DIRecursiveTypeAttrInterface, detail::DIRecursiveTypeAttrInterfaceInterfaceTraits>::Trait<ConcreteAttr> {
  };
}// namespace detail
} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {
template<typename ConcreteAttr>
bool detail::DIRecursiveTypeAttrInterfaceInterfaceTraits::Model<ConcreteAttr>::getIsRecSelf(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getIsRecSelf();
}
template<typename ConcreteAttr>
DistinctAttr detail::DIRecursiveTypeAttrInterfaceInterfaceTraits::Model<ConcreteAttr>::getRecId(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getRecId();
}
template<typename ConcreteAttr>
DIRecursiveTypeAttrInterface detail::DIRecursiveTypeAttrInterfaceInterfaceTraits::Model<ConcreteAttr>::withRecId(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, DistinctAttr recId) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).withRecId(recId);
}
template<typename ConcreteAttr>
DIRecursiveTypeAttrInterface detail::DIRecursiveTypeAttrInterfaceInterfaceTraits::Model<ConcreteAttr>::getRecSelf(DistinctAttr recId) {
  return ConcreteAttr::getRecSelf(recId);
}
template<typename ConcreteAttr>
bool detail::DIRecursiveTypeAttrInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getIsRecSelf(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getIsRecSelf(tablegen_opaque_val);
}
template<typename ConcreteAttr>
DistinctAttr detail::DIRecursiveTypeAttrInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getRecId(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getRecId(tablegen_opaque_val);
}
template<typename ConcreteAttr>
DIRecursiveTypeAttrInterface detail::DIRecursiveTypeAttrInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::withRecId(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, DistinctAttr recId) {
  return static_cast<const ConcreteAttr *>(impl)->withRecId(tablegen_opaque_val, recId);
}
template<typename ConcreteAttr>
DIRecursiveTypeAttrInterface detail::DIRecursiveTypeAttrInterfaceInterfaceTraits::FallbackModel<ConcreteAttr>::getRecSelf(DistinctAttr recId) {
  return ConcreteAttr::getRecSelf(recId);
}
} // namespace LLVM
} // namespace mlir
