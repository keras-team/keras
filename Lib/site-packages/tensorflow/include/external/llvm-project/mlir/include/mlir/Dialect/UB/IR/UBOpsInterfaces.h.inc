/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace ub {
class PoisonAttrInterface;
namespace detail {
struct PoisonAttrInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::ub::PoisonAttrInterface;
    Model() : Concept{} {}

  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::ub::PoisonAttrInterface;
    FallbackModel() : Concept{} {}

  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
  };
};
template <typename ConcreteAttr>
struct PoisonAttrInterfaceTrait;

} // namespace detail
class PoisonAttrInterface : public ::mlir::AttributeInterface<PoisonAttrInterface, detail::PoisonAttrInterfaceInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<PoisonAttrInterface, detail::PoisonAttrInterfaceInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::PoisonAttrInterfaceTrait<ConcreteAttr> {};
};
namespace detail {
  template <typename ConcreteAttr>
  struct PoisonAttrInterfaceTrait : public ::mlir::AttributeInterface<PoisonAttrInterface, detail::PoisonAttrInterfaceInterfaceTraits>::Trait<ConcreteAttr> {
  };
}// namespace detail
} // namespace ub
} // namespace mlir
namespace mlir {
namespace ub {
} // namespace ub
} // namespace mlir
