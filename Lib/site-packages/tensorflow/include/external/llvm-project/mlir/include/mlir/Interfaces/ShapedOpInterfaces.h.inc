/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class ShapedDimOpInterface;
namespace detail {
struct ShapedDimOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getShapedValue)(const Concept *impl, ::mlir::Operation *);
    ::mlir::OpFoldResult (*getDimension)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ShapedDimOpInterface;
    Model() : Concept{getShapedValue, getDimension} {}

    static inline ::mlir::Value getShapedValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpFoldResult getDimension(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ShapedDimOpInterface;
    FallbackModel() : Concept{getShapedValue, getDimension} {}

    static inline ::mlir::Value getShapedValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpFoldResult getDimension(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct ShapedDimOpInterfaceTrait;

} // namespace detail
class ShapedDimOpInterface : public ::mlir::OpInterface<ShapedDimOpInterface, detail::ShapedDimOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ShapedDimOpInterface, detail::ShapedDimOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ShapedDimOpInterfaceTrait<ConcreteOp> {};
  /// Return the shaped value operand. This is the value that the dimension
  /// is taken from.
  ::mlir::Value getShapedValue();
  /// Return the dimension operand. This can be a constant or an SSA value.
  ::mlir::OpFoldResult getDimension();
};
namespace detail {
  template <typename ConcreteOp>
  struct ShapedDimOpInterfaceTrait : public ::mlir::OpInterface<ShapedDimOpInterface, detail::ShapedDimOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      return verifyShapedDimOpInterface(op);
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::mlir::Value detail::ShapedDimOpInterfaceInterfaceTraits::Model<ConcreteOp>::getShapedValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShapedValue();
}
template<typename ConcreteOp>
::mlir::OpFoldResult detail::ShapedDimOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDimension(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDimension();
}
template<typename ConcreteOp>
::mlir::Value detail::ShapedDimOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getShapedValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getShapedValue(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::OpFoldResult detail::ShapedDimOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDimension(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getDimension(tablegen_opaque_val);
}
} // namespace mlir
