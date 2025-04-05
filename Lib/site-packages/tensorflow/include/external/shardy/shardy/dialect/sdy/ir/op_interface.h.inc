/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace sdy {
class ShardableDataFlowOpInterface;
namespace detail {
struct ShardableDataFlowOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    mlir::ArrayRef<mlir::sdy::TensorShardingAttr> (*getBlockArgumentEdgeOwnerShardings)(const Concept *impl, ::mlir::Operation *);
    void (*setBlockArgumentEdgeOwnerSharding)(const Concept *impl, ::mlir::Operation *, unsigned, mlir::sdy::TensorShardingAttr);
    void (*setBlockArgumentEdgeOwnerShardings)(const Concept *impl, ::mlir::Operation *, mlir::ArrayRef<mlir::sdy::TensorShardingAttr>);
    mlir::ArrayRef<mlir::BlockArgument> (*getBlockArgumentEdgeOwners)(const Concept *impl, ::mlir::Operation *);
    mlir::ResultRange (*getOpResultEdgeOwners)(const Concept *impl, ::mlir::Operation *);
    mlir::SmallVector<mlir::Value> (*getEdgeSources)(const Concept *impl, ::mlir::Operation *, mlir::Value);
    mlir::Value (*getEdgeOwnerFromTarget)(const Concept *impl, ::mlir::Operation *, mlir::Value);
    mlir::Value (*getEdgeOwnerFromSource)(const Concept *impl, ::mlir::Operation *, mlir::OpOperand&);
    mlir::SmallVector<mlir::Value> (*getNonEdgeOwnerTargets)(const Concept *impl, ::mlir::Operation *, mlir::Value);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::sdy::ShardableDataFlowOpInterface;
    Model() : Concept{getBlockArgumentEdgeOwnerShardings, setBlockArgumentEdgeOwnerSharding, setBlockArgumentEdgeOwnerShardings, getBlockArgumentEdgeOwners, getOpResultEdgeOwners, getEdgeSources, getEdgeOwnerFromTarget, getEdgeOwnerFromSource, getNonEdgeOwnerTargets} {}

    static inline mlir::ArrayRef<mlir::sdy::TensorShardingAttr> getBlockArgumentEdgeOwnerShardings(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setBlockArgumentEdgeOwnerSharding(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index, mlir::sdy::TensorShardingAttr sharding);
    static inline void setBlockArgumentEdgeOwnerShardings(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::ArrayRef<mlir::sdy::TensorShardingAttr> shardings);
    static inline mlir::ArrayRef<mlir::BlockArgument> getBlockArgumentEdgeOwners(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline mlir::ResultRange getOpResultEdgeOwners(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline mlir::SmallVector<mlir::Value> getEdgeSources(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::Value target);
    static inline mlir::Value getEdgeOwnerFromTarget(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::Value target);
    static inline mlir::Value getEdgeOwnerFromSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::OpOperand& source);
    static inline mlir::SmallVector<mlir::Value> getNonEdgeOwnerTargets(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::Value owner);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::sdy::ShardableDataFlowOpInterface;
    FallbackModel() : Concept{getBlockArgumentEdgeOwnerShardings, setBlockArgumentEdgeOwnerSharding, setBlockArgumentEdgeOwnerShardings, getBlockArgumentEdgeOwners, getOpResultEdgeOwners, getEdgeSources, getEdgeOwnerFromTarget, getEdgeOwnerFromSource, getNonEdgeOwnerTargets} {}

    static inline mlir::ArrayRef<mlir::sdy::TensorShardingAttr> getBlockArgumentEdgeOwnerShardings(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setBlockArgumentEdgeOwnerSharding(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index, mlir::sdy::TensorShardingAttr sharding);
    static inline void setBlockArgumentEdgeOwnerShardings(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::ArrayRef<mlir::sdy::TensorShardingAttr> shardings);
    static inline mlir::ArrayRef<mlir::BlockArgument> getBlockArgumentEdgeOwners(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline mlir::ResultRange getOpResultEdgeOwners(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline mlir::SmallVector<mlir::Value> getEdgeSources(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::Value target);
    static inline mlir::Value getEdgeOwnerFromTarget(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::Value target);
    static inline mlir::Value getEdgeOwnerFromSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::OpOperand& source);
    static inline mlir::SmallVector<mlir::Value> getNonEdgeOwnerTargets(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::Value owner);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    mlir::SmallVector<mlir::Value> getNonEdgeOwnerTargets(::mlir::Operation *tablegen_opaque_val, mlir::Value owner) const;
  };
};
template <typename ConcreteOp>
struct ShardableDataFlowOpInterfaceTrait;

} // namespace detail
class ShardableDataFlowOpInterface : public ::mlir::OpInterface<ShardableDataFlowOpInterface, detail::ShardableDataFlowOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ShardableDataFlowOpInterface, detail::ShardableDataFlowOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ShardableDataFlowOpInterfaceTrait<ConcreteOp> {};
  /// Returns the shardings of all block argument data flow edge owners.
  mlir::ArrayRef<mlir::sdy::TensorShardingAttr> getBlockArgumentEdgeOwnerShardings();
  /// Sets the shardings of the block argument edge owner with the given index.
  void setBlockArgumentEdgeOwnerSharding(unsigned index, mlir::sdy::TensorShardingAttr sharding);
  /// Sets shardings of all block argument edge owners.
  void setBlockArgumentEdgeOwnerShardings(mlir::ArrayRef<mlir::sdy::TensorShardingAttr> shardings);
  /// Gets all block argument edge owners.
  mlir::ArrayRef<mlir::BlockArgument> getBlockArgumentEdgeOwners();
  /// Gets all op result edge owners.
  mlir::ResultRange getOpResultEdgeOwners();
  /// Gets the data flow edge sources given a target value.
  mlir::SmallVector<mlir::Value> getEdgeSources(mlir::Value target);
  /// Gets the owner target of a data flow edge given a target that may or may not be the owner.
  mlir::Value getEdgeOwnerFromTarget(mlir::Value target);
  /// Gets the owner target of a data flow edge given a source.
  mlir::Value getEdgeOwnerFromSource(mlir::OpOperand& source);
  /// Gets the non-owner targets of a data flow edge given the edge owner.
  mlir::SmallVector<mlir::Value> getNonEdgeOwnerTargets(mlir::Value owner);

    // Gets the sharding of the block argument edge owner with the given index.
    mlir::sdy::TensorShardingAttr getBlockArgumentEdgeOwnerSharding(unsigned index){
      mlir::ArrayRef<mlir::sdy::TensorShardingAttr> argSharding = getBlockArgumentEdgeOwnerShardings();
      if(!argSharding.empty()) {
        return argSharding[index];
      }
      return nullptr;
    }
};
namespace detail {
  template <typename ConcreteOp>
  struct ShardableDataFlowOpInterfaceTrait : public ::mlir::OpInterface<ShardableDataFlowOpInterface, detail::ShardableDataFlowOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Gets the non-owner targets of a data flow edge given the edge owner.
    mlir::SmallVector<mlir::Value> getNonEdgeOwnerTargets(mlir::Value owner) {
      return {};
    }
  };
}// namespace detail
} // namespace sdy
} // namespace mlir
namespace mlir {
namespace sdy {
class ShardingRuleOpInterface;
namespace detail {
struct ShardingRuleOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    mlir::sdy::OpShardingRuleAttr (*getShardingRule)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::sdy::ShardingRuleOpInterface;
    Model() : Concept{getShardingRule} {}

    static inline mlir::sdy::OpShardingRuleAttr getShardingRule(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::sdy::ShardingRuleOpInterface;
    FallbackModel() : Concept{getShardingRule} {}

    static inline mlir::sdy::OpShardingRuleAttr getShardingRule(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct ShardingRuleOpInterfaceTrait;

} // namespace detail
class ShardingRuleOpInterface : public ::mlir::OpInterface<ShardingRuleOpInterface, detail::ShardingRuleOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ShardingRuleOpInterface, detail::ShardingRuleOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ShardingRuleOpInterfaceTrait<ConcreteOp> {};
  /// Returns the sharding rule of the op.
  mlir::sdy::OpShardingRuleAttr getShardingRule();
};
namespace detail {
  template <typename ConcreteOp>
  struct ShardingRuleOpInterfaceTrait : public ::mlir::OpInterface<ShardingRuleOpInterface, detail::ShardingRuleOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
}// namespace detail
} // namespace sdy
} // namespace mlir
namespace mlir {
namespace sdy {
template<typename ConcreteOp>
mlir::ArrayRef<mlir::sdy::TensorShardingAttr> detail::ShardableDataFlowOpInterfaceInterfaceTraits::Model<ConcreteOp>::getBlockArgumentEdgeOwnerShardings(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getBlockArgumentEdgeOwnerShardings();
}
template<typename ConcreteOp>
void detail::ShardableDataFlowOpInterfaceInterfaceTraits::Model<ConcreteOp>::setBlockArgumentEdgeOwnerSharding(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index, mlir::sdy::TensorShardingAttr sharding) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setBlockArgumentEdgeOwnerSharding(index, sharding);
}
template<typename ConcreteOp>
void detail::ShardableDataFlowOpInterfaceInterfaceTraits::Model<ConcreteOp>::setBlockArgumentEdgeOwnerShardings(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::ArrayRef<mlir::sdy::TensorShardingAttr> shardings) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setBlockArgumentEdgeOwnerShardings(shardings);
}
template<typename ConcreteOp>
mlir::ArrayRef<mlir::BlockArgument> detail::ShardableDataFlowOpInterfaceInterfaceTraits::Model<ConcreteOp>::getBlockArgumentEdgeOwners(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getBlockArgumentEdgeOwners();
}
template<typename ConcreteOp>
mlir::ResultRange detail::ShardableDataFlowOpInterfaceInterfaceTraits::Model<ConcreteOp>::getOpResultEdgeOwners(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOpResultEdgeOwners();
}
template<typename ConcreteOp>
mlir::SmallVector<mlir::Value> detail::ShardableDataFlowOpInterfaceInterfaceTraits::Model<ConcreteOp>::getEdgeSources(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::Value target) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getEdgeSources(target);
}
template<typename ConcreteOp>
mlir::Value detail::ShardableDataFlowOpInterfaceInterfaceTraits::Model<ConcreteOp>::getEdgeOwnerFromTarget(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::Value target) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getEdgeOwnerFromTarget(target);
}
template<typename ConcreteOp>
mlir::Value detail::ShardableDataFlowOpInterfaceInterfaceTraits::Model<ConcreteOp>::getEdgeOwnerFromSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::OpOperand& source) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getEdgeOwnerFromSource(source);
}
template<typename ConcreteOp>
mlir::SmallVector<mlir::Value> detail::ShardableDataFlowOpInterfaceInterfaceTraits::Model<ConcreteOp>::getNonEdgeOwnerTargets(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::Value owner) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNonEdgeOwnerTargets(owner);
}
template<typename ConcreteOp>
mlir::ArrayRef<mlir::sdy::TensorShardingAttr> detail::ShardableDataFlowOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getBlockArgumentEdgeOwnerShardings(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getBlockArgumentEdgeOwnerShardings(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::ShardableDataFlowOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setBlockArgumentEdgeOwnerSharding(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned index, mlir::sdy::TensorShardingAttr sharding) {
  return static_cast<const ConcreteOp *>(impl)->setBlockArgumentEdgeOwnerSharding(tablegen_opaque_val, index, sharding);
}
template<typename ConcreteOp>
void detail::ShardableDataFlowOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setBlockArgumentEdgeOwnerShardings(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::ArrayRef<mlir::sdy::TensorShardingAttr> shardings) {
  return static_cast<const ConcreteOp *>(impl)->setBlockArgumentEdgeOwnerShardings(tablegen_opaque_val, shardings);
}
template<typename ConcreteOp>
mlir::ArrayRef<mlir::BlockArgument> detail::ShardableDataFlowOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getBlockArgumentEdgeOwners(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getBlockArgumentEdgeOwners(tablegen_opaque_val);
}
template<typename ConcreteOp>
mlir::ResultRange detail::ShardableDataFlowOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getOpResultEdgeOwners(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getOpResultEdgeOwners(tablegen_opaque_val);
}
template<typename ConcreteOp>
mlir::SmallVector<mlir::Value> detail::ShardableDataFlowOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getEdgeSources(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::Value target) {
  return static_cast<const ConcreteOp *>(impl)->getEdgeSources(tablegen_opaque_val, target);
}
template<typename ConcreteOp>
mlir::Value detail::ShardableDataFlowOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getEdgeOwnerFromTarget(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::Value target) {
  return static_cast<const ConcreteOp *>(impl)->getEdgeOwnerFromTarget(tablegen_opaque_val, target);
}
template<typename ConcreteOp>
mlir::Value detail::ShardableDataFlowOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getEdgeOwnerFromSource(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::OpOperand& source) {
  return static_cast<const ConcreteOp *>(impl)->getEdgeOwnerFromSource(tablegen_opaque_val, source);
}
template<typename ConcreteOp>
mlir::SmallVector<mlir::Value> detail::ShardableDataFlowOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getNonEdgeOwnerTargets(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, mlir::Value owner) {
  return static_cast<const ConcreteOp *>(impl)->getNonEdgeOwnerTargets(tablegen_opaque_val, owner);
}
template<typename ConcreteModel, typename ConcreteOp>
mlir::SmallVector<mlir::Value> detail::ShardableDataFlowOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getNonEdgeOwnerTargets(::mlir::Operation *tablegen_opaque_val, mlir::Value owner) const {
return {};
}
} // namespace sdy
} // namespace mlir
namespace mlir {
namespace sdy {
template<typename ConcreteOp>
mlir::sdy::OpShardingRuleAttr detail::ShardingRuleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getShardingRule(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShardingRule();
}
template<typename ConcreteOp>
mlir::sdy::OpShardingRuleAttr detail::ShardingRuleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getShardingRule(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getShardingRule(tablegen_opaque_val);
}
} // namespace sdy
} // namespace mlir
