/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace omp {
class PointerLikeType;
namespace detail {
struct PointerLikeTypeInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Type (*getElementType)(const Concept *impl, ::mlir::Type );
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::omp::PointerLikeType;
    Model() : Concept{getElementType} {}

    static inline ::mlir::Type getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::omp::PointerLikeType;
    FallbackModel() : Concept{getElementType} {}

    static inline ::mlir::Type getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
  };
};
template <typename ConcreteType>
struct PointerLikeTypeTrait;

} // namespace detail
class PointerLikeType : public ::mlir::TypeInterface<PointerLikeType, detail::PointerLikeTypeInterfaceTraits> {
public:
  using ::mlir::TypeInterface<PointerLikeType, detail::PointerLikeTypeInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::PointerLikeTypeTrait<ConcreteType> {};
  /// Returns the pointee type or null if the pointer has no pointee type
  ::mlir::Type getElementType() const;
};
namespace detail {
  template <typename ConcreteType>
  struct PointerLikeTypeTrait : public ::mlir::TypeInterface<PointerLikeType, detail::PointerLikeTypeInterfaceTraits>::Trait<ConcreteType> {
  };
}// namespace detail
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
template<typename ConcreteType>
::mlir::Type detail::PointerLikeTypeInterfaceTraits::Model<ConcreteType>::getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getElementType();
}
template<typename ConcreteType>
::mlir::Type detail::PointerLikeTypeInterfaceTraits::FallbackModel<ConcreteType>::getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getElementType(tablegen_opaque_val);
}
} // namespace omp
} // namespace mlir
