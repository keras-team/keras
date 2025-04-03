/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class DestructurableTypeInterface;
namespace detail {
struct DestructurableTypeInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::std::optional<::llvm::DenseMap<::mlir::Attribute, ::mlir::Type>> (*getSubelementIndexMap)(const Concept *impl, ::mlir::Type );
    ::mlir::Type (*getTypeAtIndex)(const Concept *impl, ::mlir::Type , ::mlir::Attribute);
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::DestructurableTypeInterface;
    Model() : Concept{getSubelementIndexMap, getTypeAtIndex} {}

    static inline ::std::optional<::llvm::DenseMap<::mlir::Attribute, ::mlir::Type>> getSubelementIndexMap(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline ::mlir::Type getTypeAtIndex(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Attribute index);
  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::DestructurableTypeInterface;
    FallbackModel() : Concept{getSubelementIndexMap, getTypeAtIndex} {}

    static inline ::std::optional<::llvm::DenseMap<::mlir::Attribute, ::mlir::Type>> getSubelementIndexMap(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline ::mlir::Type getTypeAtIndex(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Attribute index);
  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
  };
};
template <typename ConcreteType>
struct DestructurableTypeInterfaceTrait;

} // namespace detail
class DestructurableTypeInterface : public ::mlir::TypeInterface<DestructurableTypeInterface, detail::DestructurableTypeInterfaceInterfaceTraits> {
public:
  using ::mlir::TypeInterface<DestructurableTypeInterface, detail::DestructurableTypeInterfaceInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::DestructurableTypeInterfaceTrait<ConcreteType> {};
  /// Destructures the type into subelements into a map of index attributes to
  /// types of subelements. Returns nothing if the type cannot be destructured.
  ::std::optional<::llvm::DenseMap<::mlir::Attribute, ::mlir::Type>> getSubelementIndexMap() const;
  /// Indicates which type is held at the provided index, returning a null
  /// Type if no type could be computed. While this can return information
  /// even when the type cannot be completely destructured, it must be coherent
  /// with the types returned by `getSubelementIndexMap` when they exist.
  ::mlir::Type getTypeAtIndex(::mlir::Attribute index) const;
};
namespace detail {
  template <typename ConcreteType>
  struct DestructurableTypeInterfaceTrait : public ::mlir::TypeInterface<DestructurableTypeInterface, detail::DestructurableTypeInterfaceInterfaceTraits>::Trait<ConcreteType> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteType>
::std::optional<::llvm::DenseMap<::mlir::Attribute, ::mlir::Type>> detail::DestructurableTypeInterfaceInterfaceTraits::Model<ConcreteType>::getSubelementIndexMap(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getSubelementIndexMap();
}
template<typename ConcreteType>
::mlir::Type detail::DestructurableTypeInterfaceInterfaceTraits::Model<ConcreteType>::getTypeAtIndex(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Attribute index) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getTypeAtIndex(index);
}
template<typename ConcreteType>
::std::optional<::llvm::DenseMap<::mlir::Attribute, ::mlir::Type>> detail::DestructurableTypeInterfaceInterfaceTraits::FallbackModel<ConcreteType>::getSubelementIndexMap(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getSubelementIndexMap(tablegen_opaque_val);
}
template<typename ConcreteType>
::mlir::Type detail::DestructurableTypeInterfaceInterfaceTraits::FallbackModel<ConcreteType>::getTypeAtIndex(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Attribute index) {
  return static_cast<const ConcreteType *>(impl)->getTypeAtIndex(tablegen_opaque_val, index);
}
} // namespace mlir
