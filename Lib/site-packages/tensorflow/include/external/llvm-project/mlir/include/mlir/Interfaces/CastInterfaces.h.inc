/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class CastOpInterface;
namespace detail {
struct CastOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    bool (*areCastCompatible)(::mlir::TypeRange, ::mlir::TypeRange);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::CastOpInterface;
    Model() : Concept{areCastCompatible} {}

    static inline bool areCastCompatible(::mlir::TypeRange inputs, ::mlir::TypeRange outputs);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::CastOpInterface;
    FallbackModel() : Concept{areCastCompatible} {}

    static inline bool areCastCompatible(::mlir::TypeRange inputs, ::mlir::TypeRange outputs);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct CastOpInterfaceTrait;

} // namespace detail
class CastOpInterface : public ::mlir::OpInterface<CastOpInterface, detail::CastOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<CastOpInterface, detail::CastOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::CastOpInterfaceTrait<ConcreteOp> {};
  /// Returns true if the given set of input and result types are compatible
  /// to cast using this cast operation.
  bool areCastCompatible(::mlir::TypeRange inputs, ::mlir::TypeRange outputs);
};
namespace detail {
  template <typename ConcreteOp>
  struct CastOpInterfaceTrait : public ::mlir::OpInterface<CastOpInterface, detail::CastOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      return impl::verifyCastInterfaceOp(op);
    }

    /// Attempt to fold the given cast operation.
    static LogicalResult foldTrait(Operation *op, ArrayRef<Attribute> operands,
                                   SmallVectorImpl<OpFoldResult> &results) {
      return impl::foldCastInterfaceOp(op, operands, results);
    }
  
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
bool detail::CastOpInterfaceInterfaceTraits::Model<ConcreteOp>::areCastCompatible(::mlir::TypeRange inputs, ::mlir::TypeRange outputs) {
  return ConcreteOp::areCastCompatible(inputs, outputs);
}
template<typename ConcreteOp>
bool detail::CastOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::areCastCompatible(::mlir::TypeRange inputs, ::mlir::TypeRange outputs) {
  return ConcreteOp::areCastCompatible(inputs, outputs);
}
} // namespace mlir
