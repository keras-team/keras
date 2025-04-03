/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace vhlo {
class VersionedTypeInterface;
namespace detail {
struct VersionedTypeInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    mlir::vhlo::Version (*getMinVersion)(const Concept *impl, ::mlir::Type );
    mlir::vhlo::Version (*getMaxVersion)(const Concept *impl, ::mlir::Type );
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::vhlo::VersionedTypeInterface;
    Model() : Concept{getMinVersion, getMaxVersion} {}

    static inline mlir::vhlo::Version getMinVersion(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline mlir::vhlo::Version getMaxVersion(const Concept *impl, ::mlir::Type tablegen_opaque_val);
  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::vhlo::VersionedTypeInterface;
    FallbackModel() : Concept{getMinVersion, getMaxVersion} {}

    static inline mlir::vhlo::Version getMinVersion(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline mlir::vhlo::Version getMaxVersion(const Concept *impl, ::mlir::Type tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
  };
};
template <typename ConcreteType>
struct VersionedTypeInterfaceTrait;

} // namespace detail
class VersionedTypeInterface : public ::mlir::TypeInterface<VersionedTypeInterface, detail::VersionedTypeInterfaceInterfaceTraits> {
public:
  using ::mlir::TypeInterface<VersionedTypeInterface, detail::VersionedTypeInterfaceInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::VersionedTypeInterfaceTrait<ConcreteType> {};
  /// Returns the minimum version of the VHLO dialect an attribute is supported in.
  mlir::vhlo::Version getMinVersion() const;
  /// Returns the maximum version (inclusive) of the VHLO dialect an attribute is supported in.
  mlir::vhlo::Version getMaxVersion() const;
};
namespace detail {
  template <typename ConcreteType>
  struct VersionedTypeInterfaceTrait : public ::mlir::TypeInterface<VersionedTypeInterface, detail::VersionedTypeInterfaceInterfaceTraits>::Trait<ConcreteType> {
  };
}// namespace detail
} // namespace vhlo
} // namespace mlir
namespace mlir {
namespace vhlo {
template<typename ConcreteType>
mlir::vhlo::Version detail::VersionedTypeInterfaceInterfaceTraits::Model<ConcreteType>::getMinVersion(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getMinVersion();
}
template<typename ConcreteType>
mlir::vhlo::Version detail::VersionedTypeInterfaceInterfaceTraits::Model<ConcreteType>::getMaxVersion(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getMaxVersion();
}
template<typename ConcreteType>
mlir::vhlo::Version detail::VersionedTypeInterfaceInterfaceTraits::FallbackModel<ConcreteType>::getMinVersion(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getMinVersion(tablegen_opaque_val);
}
template<typename ConcreteType>
mlir::vhlo::Version detail::VersionedTypeInterfaceInterfaceTraits::FallbackModel<ConcreteType>::getMaxVersion(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getMaxVersion(tablegen_opaque_val);
}
} // namespace vhlo
} // namespace mlir
