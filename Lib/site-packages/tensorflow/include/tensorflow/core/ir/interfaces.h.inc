/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace tfg {
class TensorFlowRegistryInterface;
namespace detail {
struct TensorFlowRegistryInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    bool (*isStateful)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::tfg::TensorFlowRegistryInterface;
    Model() : Concept{isStateful} {}

    static inline bool isStateful(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::tfg::TensorFlowRegistryInterface;
    FallbackModel() : Concept{isStateful} {}

    static inline bool isStateful(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct TensorFlowRegistryInterfaceTrait;

} // namespace detail
class TensorFlowRegistryInterface : public ::mlir::OpInterface<TensorFlowRegistryInterface, detail::TensorFlowRegistryInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<TensorFlowRegistryInterface, detail::TensorFlowRegistryInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::TensorFlowRegistryInterfaceTrait<ConcreteOp> {};
  /// Returns true if the current op is stateful, according to TensorFlow.
  bool isStateful();
};
namespace detail {
  template <typename ConcreteOp>
  struct TensorFlowRegistryInterfaceTrait : public ::mlir::OpInterface<TensorFlowRegistryInterface, detail::TensorFlowRegistryInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class ControlArgumentInterface;
namespace detail {
struct ControlArgumentInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    mlir::BlockArgument (*getDataValueOf)(BlockArgument);
    mlir::BlockArgument (*getControlTokenOf)(BlockArgument);
    mlir::BlockArgument (*getDataValue)(Region &, unsigned);
    mlir::BlockArgument (*getControlToken)(Region &, unsigned);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::tfg::ControlArgumentInterface;
    Model() : Concept{getDataValueOf, getControlTokenOf, getDataValue, getControlToken} {}

    static inline mlir::BlockArgument getDataValueOf(BlockArgument ctl);
    static inline mlir::BlockArgument getControlTokenOf(BlockArgument data);
    static inline mlir::BlockArgument getDataValue(Region & region, unsigned idx);
    static inline mlir::BlockArgument getControlToken(Region & region, unsigned idx);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::tfg::ControlArgumentInterface;
    FallbackModel() : Concept{getDataValueOf, getControlTokenOf, getDataValue, getControlToken} {}

    static inline mlir::BlockArgument getDataValueOf(BlockArgument ctl);
    static inline mlir::BlockArgument getControlTokenOf(BlockArgument data);
    static inline mlir::BlockArgument getDataValue(Region & region, unsigned idx);
    static inline mlir::BlockArgument getControlToken(Region & region, unsigned idx);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    static mlir::BlockArgument getControlToken(Region &region, unsigned idx);
  };
};
template <typename ConcreteOp>
struct ControlArgumentInterfaceTrait;

} // namespace detail
class ControlArgumentInterface : public ::mlir::OpInterface<ControlArgumentInterface, detail::ControlArgumentInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ControlArgumentInterface, detail::ControlArgumentInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ControlArgumentInterfaceTrait<ConcreteOp> {};
  /// Given a block argument that is a control token, return the associated
  /// data argument.
  mlir::BlockArgument getDataValueOf(BlockArgument ctl);
  /// Given a block argument that is a data value, return the associated
  /// control token.
  mlir::BlockArgument getControlTokenOf(BlockArgument data);
  /// Get the data value corresponding to a given argument index.
  mlir::BlockArgument getDataValue(Region & region, unsigned idx);
  /// Get the control token corresponding to a given argument index.
  mlir::BlockArgument getControlToken(Region & region, unsigned idx);

    /// Verify that a region has the same number of data and control arguments.
    static LogicalResult verifyRegion(Operation *op, Region &region);
};
namespace detail {
  template <typename ConcreteOp>
  struct ControlArgumentInterfaceTrait : public ::mlir::OpInterface<ControlArgumentInterface, detail::ControlArgumentInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Get the control token corresponding to a given argument index.
    static mlir::BlockArgument getControlToken(Region & region, unsigned idx) {
      mlir::BlockArgument data = ConcreteOp::getDataValue(region, idx);
        return ConcreteOp::getControlTokenOf(data);
    }
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      for (Region &region : op->getRegions())
      if (failed(ControlArgumentInterface::verifyRegion(op, region)))
        return failure();
    return success();
    }
  };
}// namespace detail
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class PreservedAttributesInterface;
namespace detail {
struct PreservedAttributesInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    mlir::tfg::RegionAttr (*getPreservedAttrs)(const Concept *impl, ::mlir::Operation *, unsigned);
    void (*setPreservedAttrs)(const Concept *impl, ::mlir::Operation *, unsigned, RegionAttr);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::tfg::PreservedAttributesInterface;
    Model() : Concept{getPreservedAttrs, setPreservedAttrs} {}

    static inline mlir::tfg::RegionAttr getPreservedAttrs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline void setPreservedAttrs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx, RegionAttr attrs);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::tfg::PreservedAttributesInterface;
    FallbackModel() : Concept{getPreservedAttrs, setPreservedAttrs} {}

    static inline mlir::tfg::RegionAttr getPreservedAttrs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx);
    static inline void setPreservedAttrs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx, RegionAttr attrs);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct PreservedAttributesInterfaceTrait;

} // namespace detail
class PreservedAttributesInterface : public ::mlir::OpInterface<PreservedAttributesInterface, detail::PreservedAttributesInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<PreservedAttributesInterface, detail::PreservedAttributesInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::PreservedAttributesInterfaceTrait<ConcreteOp> {};
  /// Get the (potentially null) RegionAttr corresponding to the region
  /// at the given index.
  mlir::tfg::RegionAttr getPreservedAttrs(unsigned idx);
  /// Set the non-null RegionAttr corresponding to the region at the
  /// given index.
  void setPreservedAttrs(unsigned idx, RegionAttr attrs);
};
namespace detail {
  template <typename ConcreteOp>
  struct PreservedAttributesInterfaceTrait : public ::mlir::OpInterface<PreservedAttributesInterface, detail::PreservedAttributesInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
template<typename ConcreteOp>
bool detail::TensorFlowRegistryInterfaceInterfaceTraits::Model<ConcreteOp>::isStateful(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isStateful();
}
template<typename ConcreteOp>
bool detail::TensorFlowRegistryInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isStateful(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isStateful(tablegen_opaque_val);
}
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
template<typename ConcreteOp>
mlir::BlockArgument detail::ControlArgumentInterfaceInterfaceTraits::Model<ConcreteOp>::getDataValueOf(BlockArgument ctl) {
  return ConcreteOp::getDataValueOf(ctl);
}
template<typename ConcreteOp>
mlir::BlockArgument detail::ControlArgumentInterfaceInterfaceTraits::Model<ConcreteOp>::getControlTokenOf(BlockArgument data) {
  return ConcreteOp::getControlTokenOf(data);
}
template<typename ConcreteOp>
mlir::BlockArgument detail::ControlArgumentInterfaceInterfaceTraits::Model<ConcreteOp>::getDataValue(Region & region, unsigned idx) {
  return ConcreteOp::getDataValue(region, idx);
}
template<typename ConcreteOp>
mlir::BlockArgument detail::ControlArgumentInterfaceInterfaceTraits::Model<ConcreteOp>::getControlToken(Region & region, unsigned idx) {
  return ConcreteOp::getControlToken(region, idx);
}
template<typename ConcreteOp>
mlir::BlockArgument detail::ControlArgumentInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDataValueOf(BlockArgument ctl) {
  return ConcreteOp::getDataValueOf(ctl);
}
template<typename ConcreteOp>
mlir::BlockArgument detail::ControlArgumentInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getControlTokenOf(BlockArgument data) {
  return ConcreteOp::getControlTokenOf(data);
}
template<typename ConcreteOp>
mlir::BlockArgument detail::ControlArgumentInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDataValue(Region & region, unsigned idx) {
  return ConcreteOp::getDataValue(region, idx);
}
template<typename ConcreteOp>
mlir::BlockArgument detail::ControlArgumentInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getControlToken(Region & region, unsigned idx) {
  return ConcreteOp::getControlToken(region, idx);
}
template<typename ConcreteModel, typename ConcreteOp>
mlir::BlockArgument detail::ControlArgumentInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getControlToken(Region &region, unsigned idx) {
mlir::BlockArgument data = ConcreteOp::getDataValue(region, idx);
        return ConcreteOp::getControlTokenOf(data);
}
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
template<typename ConcreteOp>
mlir::tfg::RegionAttr detail::PreservedAttributesInterfaceInterfaceTraits::Model<ConcreteOp>::getPreservedAttrs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getPreservedAttrs(idx);
}
template<typename ConcreteOp>
void detail::PreservedAttributesInterfaceInterfaceTraits::Model<ConcreteOp>::setPreservedAttrs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx, RegionAttr attrs) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setPreservedAttrs(idx, attrs);
}
template<typename ConcreteOp>
mlir::tfg::RegionAttr detail::PreservedAttributesInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getPreservedAttrs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx) {
  return static_cast<const ConcreteOp *>(impl)->getPreservedAttrs(tablegen_opaque_val, idx);
}
template<typename ConcreteOp>
void detail::PreservedAttributesInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setPreservedAttrs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned idx, RegionAttr attrs) {
  return static_cast<const ConcreteOp *>(impl)->setPreservedAttrs(tablegen_opaque_val, idx, attrs);
}
} // namespace tfg
} // namespace mlir
