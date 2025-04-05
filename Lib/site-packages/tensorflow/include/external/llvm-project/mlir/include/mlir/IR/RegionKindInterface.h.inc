/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class RegionKindInterface;
namespace detail {
struct RegionKindInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::RegionKind (*getRegionKind)(unsigned);
    bool (*hasSSADominance)(unsigned);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::RegionKindInterface;
    Model() : Concept{getRegionKind, hasSSADominance} {}

    static inline ::mlir::RegionKind getRegionKind(unsigned index);
    static inline bool hasSSADominance(unsigned index);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::RegionKindInterface;
    FallbackModel() : Concept{getRegionKind, hasSSADominance} {}

    static inline ::mlir::RegionKind getRegionKind(unsigned index);
    static inline bool hasSSADominance(unsigned index);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct RegionKindInterfaceTrait;

} // namespace detail
class RegionKindInterface : public ::mlir::OpInterface<RegionKindInterface, detail::RegionKindInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<RegionKindInterface, detail::RegionKindInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::RegionKindInterfaceTrait<ConcreteOp> {};
  /// Return the kind of the region with the given index inside this operation.
  ::mlir::RegionKind getRegionKind(unsigned index);
  /// Return true if the kind of the given region requires the SSA-Dominance property
  bool hasSSADominance(unsigned index);
};
namespace detail {
  template <typename ConcreteOp>
  struct RegionKindInterfaceTrait : public ::mlir::OpInterface<RegionKindInterface, detail::RegionKindInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::mlir::RegionKind detail::RegionKindInterfaceInterfaceTraits::Model<ConcreteOp>::getRegionKind(unsigned index) {
  return ConcreteOp::getRegionKind(index);
}
template<typename ConcreteOp>
bool detail::RegionKindInterfaceInterfaceTraits::Model<ConcreteOp>::hasSSADominance(unsigned index) {
  return getRegionKind(index) == ::mlir::RegionKind::SSACFG;
}
template<typename ConcreteOp>
::mlir::RegionKind detail::RegionKindInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getRegionKind(unsigned index) {
  return ConcreteOp::getRegionKind(index);
}
template<typename ConcreteOp>
bool detail::RegionKindInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::hasSSADominance(unsigned index) {
  return ConcreteOp::hasSSADominance(index);
}
} // namespace mlir
