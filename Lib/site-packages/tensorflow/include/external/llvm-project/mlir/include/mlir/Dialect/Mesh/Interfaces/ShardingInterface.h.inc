/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace mesh {
class ShardingInterface;
namespace detail {
struct ShardingInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    SmallVector<mlir::utils::IteratorType> (*getLoopIteratorTypes)(const Concept *impl, ::mlir::Operation *);
    SmallVector<ReductionKind> (*getReductionLoopIteratorKinds)(const Concept *impl, ::mlir::Operation *);
    SmallVector<AffineMap> (*getIndexingMaps)(const Concept *impl, ::mlir::Operation *);
    FailureOr<ShardingOption> (*getShardingOption)(const Concept *impl, ::mlir::Operation *, ArrayRef<MeshSharding>, ArrayRef<MeshSharding>);
    FailureOr<std::vector<MeshSharding>> (*getShardingAnnotations)(const Concept *impl, ::mlir::Operation *, const ShardingOption &);
    LogicalResult (*addShardingAnnotations)(const Concept *impl, ::mlir::Operation *, OpBuilder &, const ShardingOption &);
    LogicalResult (*spmdize)(const Concept *impl, ::mlir::Operation *, ArrayRef<Value>, ArrayRef<MeshSharding>, ArrayRef<MeshSharding>, IRMapping&, SymbolTableCollection &, OpBuilder &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::mesh::ShardingInterface;
    Model() : Concept{getLoopIteratorTypes, getReductionLoopIteratorKinds, getIndexingMaps, getShardingOption, getShardingAnnotations, addShardingAnnotations, spmdize} {}

    static inline SmallVector<mlir::utils::IteratorType> getLoopIteratorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<ReductionKind> getReductionLoopIteratorKinds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<AffineMap> getIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline FailureOr<ShardingOption> getShardingOption(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings);
    static inline FailureOr<std::vector<MeshSharding>> getShardingAnnotations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ShardingOption & shardingOption);
    static inline LogicalResult addShardingAnnotations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, const ShardingOption & shardingOption);
    static inline LogicalResult spmdize(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<Value> spmdizedOperands, ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings, IRMapping& spmdizationMap, SymbolTableCollection & symbolTableCollection, OpBuilder & builder);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::mesh::ShardingInterface;
    FallbackModel() : Concept{getLoopIteratorTypes, getReductionLoopIteratorKinds, getIndexingMaps, getShardingOption, getShardingAnnotations, addShardingAnnotations, spmdize} {}

    static inline SmallVector<mlir::utils::IteratorType> getLoopIteratorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<ReductionKind> getReductionLoopIteratorKinds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<AffineMap> getIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline FailureOr<ShardingOption> getShardingOption(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings);
    static inline FailureOr<std::vector<MeshSharding>> getShardingAnnotations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ShardingOption & shardingOption);
    static inline LogicalResult addShardingAnnotations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, const ShardingOption & shardingOption);
    static inline LogicalResult spmdize(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<Value> spmdizedOperands, ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings, IRMapping& spmdizationMap, SymbolTableCollection & symbolTableCollection, OpBuilder & builder);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    SmallVector<mlir::utils::IteratorType> getLoopIteratorTypes(::mlir::Operation *tablegen_opaque_val) const;
    SmallVector<ReductionKind> getReductionLoopIteratorKinds(::mlir::Operation *tablegen_opaque_val) const;
    SmallVector<AffineMap> getIndexingMaps(::mlir::Operation *tablegen_opaque_val) const;
    FailureOr<ShardingOption> getShardingOption(::mlir::Operation *tablegen_opaque_val, ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings) const;
    FailureOr<std::vector<MeshSharding>> getShardingAnnotations(::mlir::Operation *tablegen_opaque_val, const ShardingOption &shardingOption) const;
    LogicalResult addShardingAnnotations(::mlir::Operation *tablegen_opaque_val, OpBuilder &b, const ShardingOption &shardingOption) const;
    LogicalResult spmdize(::mlir::Operation *tablegen_opaque_val, ArrayRef<Value> spmdizedOperands, ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings, IRMapping&spmdizationMap, SymbolTableCollection &symbolTableCollection, OpBuilder &builder) const;
  };
};
template <typename ConcreteOp>
struct ShardingInterfaceTrait;

} // namespace detail
class ShardingInterface : public ::mlir::OpInterface<ShardingInterface, detail::ShardingInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ShardingInterface, detail::ShardingInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ShardingInterfaceTrait<ConcreteOp> {};
  /// Returns a list of iterator types that describe the number of loops.
  /// The iterator types determine how the operation traverses its input and
  /// output tensors.
  /// 
  /// Example 1: A gemm op has 3 loops, M, N and K. Their loop iterator
  /// types are parallel, parallel, reduction. This indicates that M and
  /// N are traversed in parallel, while the K dimension is used for
  /// reduction.
  SmallVector<mlir::utils::IteratorType> getLoopIteratorTypes();
  /// Return the kind of all reduction loop iterators.
  /// The order is the same as the same as the result from
  /// `getLoopIteratorTypes`.
  /// 
  /// Example 1:
  /// iterator types =  (parallel, reduction, parallel, reduction)
  ///                                    ||                   ||
  /// reduction kinds = (                sum,                 max)
  /// 
  /// Example 2:
  /// A softmax op's loop iterator types are parallel and
  /// reduction.
  /// The reduction iterator will be of kind `generic`, since it is non of
  /// the available presets.
  SmallVector<ReductionKind> getReductionLoopIteratorKinds();
  /// Return the indexing maps attribute within the current operation.
  /// Indexing maps determine how indices in the iteration space map to
  /// tensor indices. They are specified using `affine_map` in MLIR, which
  /// provides an affine transformation of indices.
  SmallVector<AffineMap> getIndexingMaps();
  /// Given that certain operands or results of the operation may have
  /// sharding annotations, this method leverages this information to
  /// deduce how the operation should be sharded.
  /// The passed sharding may be incomplete, this gives freedom for the
  /// op to select the most appropriate shardings for all the operands
  /// and results and the op itself.
  FailureOr<ShardingOption> getShardingOption(ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings);
  /// Based on a given ShardingOption, get the operand and result
  /// operations for the operands and results sharding annotations.
  /// This is what shardings the operands and results need to have in order
  /// to shard the op according to shardingOption.
  FailureOr<std::vector<MeshSharding>> getShardingAnnotations(const ShardingOption & shardingOption);
  /// Based on a given ShardingOption, this method adds `mesh.shard`
  /// operations for the operands and results that previously lacked
  /// sharding annotations.
  LogicalResult addShardingAnnotations(OpBuilder & b, const ShardingOption & shardingOption);
  /// Convert self to SPMD form.
  /// This method is used during the spmdization pass of a program fully
  /// annotated with shardings.
  /// 
  /// The spmdization algorithm would read the surrounding sharding
  /// annotations from the IR for each argument/result and prepare
  /// `operandShardings` and `resultShardings`.
  /// Values that are not ranked tensors do not have sharding annotations.
  /// In this case their corresponding MeshSharding is null.
  /// 
  /// For convenience it will also prepare `spmdizedOperands`, although
  /// they can be retrieved from the `spmdizationMap`.
  /// 
  /// The `spmdizationMap` contains a mapping from unsharded to
  /// sharded/spmdized values that are constructed during the spmdization
  /// pass. The interface implementation must populate `spmdizationMap`
  /// with the mapping for this op's results.
  /// 
  /// `builder` is set to insert new operations in the appropriate point.
  /// The implementation should not return the builder to the original
  /// insertion point.
  /// It should leave it as is after all insertions are done.
  /// 
  /// The default implementation does full replication.
  /// This assumes that all sharding annotations are for full replication.
  LogicalResult spmdize(ArrayRef<Value> spmdizedOperands, ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings, IRMapping& spmdizationMap, SymbolTableCollection & symbolTableCollection, OpBuilder & builder);

      LogicalResult verifyShardingInterfaceImpl();

      void printLoopTypesAndIndexingMaps(raw_ostream &os);
};
namespace detail {
  template <typename ConcreteOp>
  struct ShardingInterfaceTrait : public ::mlir::OpInterface<ShardingInterface, detail::ShardingInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns a list of iterator types that describe the number of loops.
    /// The iterator types determine how the operation traverses its input and
    /// output tensors.
    /// 
    /// Example 1: A gemm op has 3 loops, M, N and K. Their loop iterator
    /// types are parallel, parallel, reduction. This indicates that M and
    /// N are traversed in parallel, while the K dimension is used for
    /// reduction.
    SmallVector<mlir::utils::IteratorType> getLoopIteratorTypes() {
      return {};
    }
    /// Return the kind of all reduction loop iterators.
    /// The order is the same as the same as the result from
    /// `getLoopIteratorTypes`.
    /// 
    /// Example 1:
    /// iterator types =  (parallel, reduction, parallel, reduction)
    ///                                    ||                   ||
    /// reduction kinds = (                sum,                 max)
    /// 
    /// Example 2:
    /// A softmax op's loop iterator types are parallel and
    /// reduction.
    /// The reduction iterator will be of kind `generic`, since it is non of
    /// the available presets.
    SmallVector<ReductionKind> getReductionLoopIteratorKinds() {
      return {};
    }
    /// Return the indexing maps attribute within the current operation.
    /// Indexing maps determine how indices in the iteration space map to
    /// tensor indices. They are specified using `affine_map` in MLIR, which
    /// provides an affine transformation of indices.
    SmallVector<AffineMap> getIndexingMaps() {
      return {};
    }
    /// Given that certain operands or results of the operation may have
    /// sharding annotations, this method leverages this information to
    /// deduce how the operation should be sharded.
    /// The passed sharding may be incomplete, this gives freedom for the
    /// op to select the most appropriate shardings for all the operands
    /// and results and the op itself.
    FailureOr<ShardingOption> getShardingOption(ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings) {
      return detail::defaultGetShardingOption(
            (*static_cast<ConcreteOp *>(this)).getOperation(), operandShardings, resultShardings);
    }
    /// Based on a given ShardingOption, get the operand and result
    /// operations for the operands and results sharding annotations.
    /// This is what shardings the operands and results need to have in order
    /// to shard the op according to shardingOption.
    FailureOr<std::vector<MeshSharding>> getShardingAnnotations(const ShardingOption & shardingOption) {
      return detail::defaultGetShardingAnnotations(
            (*static_cast<ConcreteOp *>(this)).getOperation(), shardingOption);
    }
    /// Based on a given ShardingOption, this method adds `mesh.shard`
    /// operations for the operands and results that previously lacked
    /// sharding annotations.
    LogicalResult addShardingAnnotations(OpBuilder & b, const ShardingOption & shardingOption) {
      return detail::defaultAddShardingAnnotations(
            (*static_cast<ConcreteOp *>(this)).getOperation(), b, shardingOption);
    }
    /// Convert self to SPMD form.
    /// This method is used during the spmdization pass of a program fully
    /// annotated with shardings.
    /// 
    /// The spmdization algorithm would read the surrounding sharding
    /// annotations from the IR for each argument/result and prepare
    /// `operandShardings` and `resultShardings`.
    /// Values that are not ranked tensors do not have sharding annotations.
    /// In this case their corresponding MeshSharding is null.
    /// 
    /// For convenience it will also prepare `spmdizedOperands`, although
    /// they can be retrieved from the `spmdizationMap`.
    /// 
    /// The `spmdizationMap` contains a mapping from unsharded to
    /// sharded/spmdized values that are constructed during the spmdization
    /// pass. The interface implementation must populate `spmdizationMap`
    /// with the mapping for this op's results.
    /// 
    /// `builder` is set to insert new operations in the appropriate point.
    /// The implementation should not return the builder to the original
    /// insertion point.
    /// It should leave it as is after all insertions are done.
    /// 
    /// The default implementation does full replication.
    /// This assumes that all sharding annotations are for full replication.
    LogicalResult spmdize(ArrayRef<Value> spmdizedOperands, ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings, IRMapping& spmdizationMap, SymbolTableCollection & symbolTableCollection, OpBuilder & builder) {
      spmdizeFullyReplicatedOperation(
            *(*static_cast<ConcreteOp *>(this)).getOperation(), spmdizedOperands, operandShardings,
              resultShardings, spmdizationMap, symbolTableCollection, builder);
          return success();
    }
  };
}// namespace detail
} // namespace mesh
} // namespace mlir
namespace mlir {
namespace mesh {
template<typename ConcreteOp>
SmallVector<mlir::utils::IteratorType> detail::ShardingInterfaceInterfaceTraits::Model<ConcreteOp>::getLoopIteratorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLoopIteratorTypes();
}
template<typename ConcreteOp>
SmallVector<ReductionKind> detail::ShardingInterfaceInterfaceTraits::Model<ConcreteOp>::getReductionLoopIteratorKinds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getReductionLoopIteratorKinds();
}
template<typename ConcreteOp>
SmallVector<AffineMap> detail::ShardingInterfaceInterfaceTraits::Model<ConcreteOp>::getIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps();
}
template<typename ConcreteOp>
FailureOr<ShardingOption> detail::ShardingInterfaceInterfaceTraits::Model<ConcreteOp>::getShardingOption(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShardingOption(operandShardings, resultShardings);
}
template<typename ConcreteOp>
FailureOr<std::vector<MeshSharding>> detail::ShardingInterfaceInterfaceTraits::Model<ConcreteOp>::getShardingAnnotations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ShardingOption & shardingOption) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShardingAnnotations(shardingOption);
}
template<typename ConcreteOp>
LogicalResult detail::ShardingInterfaceInterfaceTraits::Model<ConcreteOp>::addShardingAnnotations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, const ShardingOption & shardingOption) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).addShardingAnnotations(b, shardingOption);
}
template<typename ConcreteOp>
LogicalResult detail::ShardingInterfaceInterfaceTraits::Model<ConcreteOp>::spmdize(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<Value> spmdizedOperands, ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings, IRMapping& spmdizationMap, SymbolTableCollection & symbolTableCollection, OpBuilder & builder) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).spmdize(spmdizedOperands, operandShardings, resultShardings, spmdizationMap, symbolTableCollection, builder);
}
template<typename ConcreteOp>
SmallVector<mlir::utils::IteratorType> detail::ShardingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getLoopIteratorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getLoopIteratorTypes(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<ReductionKind> detail::ShardingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getReductionLoopIteratorKinds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getReductionLoopIteratorKinds(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<AffineMap> detail::ShardingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getIndexingMaps(tablegen_opaque_val);
}
template<typename ConcreteOp>
FailureOr<ShardingOption> detail::ShardingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getShardingOption(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings) {
  return static_cast<const ConcreteOp *>(impl)->getShardingOption(tablegen_opaque_val, operandShardings, resultShardings);
}
template<typename ConcreteOp>
FailureOr<std::vector<MeshSharding>> detail::ShardingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getShardingAnnotations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, const ShardingOption & shardingOption) {
  return static_cast<const ConcreteOp *>(impl)->getShardingAnnotations(tablegen_opaque_val, shardingOption);
}
template<typename ConcreteOp>
LogicalResult detail::ShardingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::addShardingAnnotations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, const ShardingOption & shardingOption) {
  return static_cast<const ConcreteOp *>(impl)->addShardingAnnotations(tablegen_opaque_val, b, shardingOption);
}
template<typename ConcreteOp>
LogicalResult detail::ShardingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::spmdize(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<Value> spmdizedOperands, ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings, IRMapping& spmdizationMap, SymbolTableCollection & symbolTableCollection, OpBuilder & builder) {
  return static_cast<const ConcreteOp *>(impl)->spmdize(tablegen_opaque_val, spmdizedOperands, operandShardings, resultShardings, spmdizationMap, symbolTableCollection, builder);
}
template<typename ConcreteModel, typename ConcreteOp>
SmallVector<mlir::utils::IteratorType> detail::ShardingInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getLoopIteratorTypes(::mlir::Operation *tablegen_opaque_val) const {
return {};
}
template<typename ConcreteModel, typename ConcreteOp>
SmallVector<ReductionKind> detail::ShardingInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getReductionLoopIteratorKinds(::mlir::Operation *tablegen_opaque_val) const {
return {};
}
template<typename ConcreteModel, typename ConcreteOp>
SmallVector<AffineMap> detail::ShardingInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getIndexingMaps(::mlir::Operation *tablegen_opaque_val) const {
return {};
}
template<typename ConcreteModel, typename ConcreteOp>
FailureOr<ShardingOption> detail::ShardingInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getShardingOption(::mlir::Operation *tablegen_opaque_val, ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings) const {
return detail::defaultGetShardingOption(
            (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation(), operandShardings, resultShardings);
}
template<typename ConcreteModel, typename ConcreteOp>
FailureOr<std::vector<MeshSharding>> detail::ShardingInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getShardingAnnotations(::mlir::Operation *tablegen_opaque_val, const ShardingOption &shardingOption) const {
return detail::defaultGetShardingAnnotations(
            (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation(), shardingOption);
}
template<typename ConcreteModel, typename ConcreteOp>
LogicalResult detail::ShardingInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::addShardingAnnotations(::mlir::Operation *tablegen_opaque_val, OpBuilder &b, const ShardingOption &shardingOption) const {
return detail::defaultAddShardingAnnotations(
            (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation(), b, shardingOption);
}
template<typename ConcreteModel, typename ConcreteOp>
LogicalResult detail::ShardingInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::spmdize(::mlir::Operation *tablegen_opaque_val, ArrayRef<Value> spmdizedOperands, ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings, IRMapping&spmdizationMap, SymbolTableCollection &symbolTableCollection, OpBuilder &builder) const {
spmdizeFullyReplicatedOperation(
            *(llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation(), spmdizedOperands, operandShardings,
              resultShardings, spmdizationMap, symbolTableCollection, builder);
          return success();
}
} // namespace mesh
} // namespace mlir
