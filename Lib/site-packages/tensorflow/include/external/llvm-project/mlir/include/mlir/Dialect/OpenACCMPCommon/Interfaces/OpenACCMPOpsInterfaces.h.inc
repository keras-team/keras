/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace accomp {
class RecipeInterface;
namespace detail {
struct RecipeInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Block *(*getAllocaBlock)(const Concept *impl, ::mlir::Operation *, ::mlir::Region &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::accomp::RecipeInterface;
    Model() : Concept{getAllocaBlock} {}

    static inline ::mlir::Block *getAllocaBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Region & region);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::accomp::RecipeInterface;
    FallbackModel() : Concept{getAllocaBlock} {}

    static inline ::mlir::Block *getAllocaBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Region & region);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::Block *getAllocaBlock(::mlir::Operation *tablegen_opaque_val, ::mlir::Region &region) const;
  };
};
template <typename ConcreteOp>
struct RecipeInterfaceTrait;

} // namespace detail
class RecipeInterface : public ::mlir::OpInterface<RecipeInterface, detail::RecipeInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<RecipeInterface, detail::RecipeInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::RecipeInterfaceTrait<ConcreteOp> {};
  /// For the given region of the operation return the block
  /// inside the region, where an alloca-like operation should be inserted.
  /// The default implementation returns the entry block of the region.
  ::mlir::Block *getAllocaBlock(::mlir::Region & region);
};
namespace detail {
  template <typename ConcreteOp>
  struct RecipeInterfaceTrait : public ::mlir::OpInterface<RecipeInterface, detail::RecipeInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// For the given region of the operation return the block
    /// inside the region, where an alloca-like operation should be inserted.
    /// The default implementation returns the entry block of the region.
    ::mlir::Block *getAllocaBlock(::mlir::Region & region) {
      return &region.front();
    }
  };
}// namespace detail
} // namespace accomp
} // namespace mlir
namespace mlir {
namespace accomp {
template<typename ConcreteOp>
::mlir::Block *detail::RecipeInterfaceInterfaceTraits::Model<ConcreteOp>::getAllocaBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Region & region) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAllocaBlock(region);
}
template<typename ConcreteOp>
::mlir::Block *detail::RecipeInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAllocaBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Region & region) {
  return static_cast<const ConcreteOp *>(impl)->getAllocaBlock(tablegen_opaque_val, region);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::Block *detail::RecipeInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAllocaBlock(::mlir::Operation *tablegen_opaque_val, ::mlir::Region &region) const {
return &region.front();
}
} // namespace accomp
} // namespace mlir
