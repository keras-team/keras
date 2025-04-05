/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class SubsetOpInterface;
namespace detail {
struct SubsetOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    bool (*operatesOnEquivalentSubset)(const Concept *impl, ::mlir::Operation *, ::mlir::SubsetOpInterface, ::llvm::function_ref<bool(Value, Value)>);
    bool (*operatesOnDisjointSubset)(const Concept *impl, ::mlir::Operation *, ::mlir::SubsetOpInterface, ::llvm::function_ref<bool(Value, Value)>);
    ::mlir::FailureOr<::mlir::HyperrectangularSlice> (*getAccessedHyperrectangularSlice)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::SubsetOpInterface;
    Model() : Concept{operatesOnEquivalentSubset, operatesOnDisjointSubset, getAccessedHyperrectangularSlice} {}

    static inline bool operatesOnEquivalentSubset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn);
    static inline bool operatesOnDisjointSubset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn);
    static inline ::mlir::FailureOr<::mlir::HyperrectangularSlice> getAccessedHyperrectangularSlice(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::SubsetOpInterface;
    FallbackModel() : Concept{operatesOnEquivalentSubset, operatesOnDisjointSubset, getAccessedHyperrectangularSlice} {}

    static inline bool operatesOnEquivalentSubset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn);
    static inline bool operatesOnDisjointSubset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn);
    static inline ::mlir::FailureOr<::mlir::HyperrectangularSlice> getAccessedHyperrectangularSlice(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    bool operatesOnEquivalentSubset(::mlir::Operation *tablegen_opaque_val, ::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) const;
    bool operatesOnDisjointSubset(::mlir::Operation *tablegen_opaque_val, ::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) const;
    ::mlir::FailureOr<::mlir::HyperrectangularSlice> getAccessedHyperrectangularSlice(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct SubsetOpInterfaceTrait;

} // namespace detail
class SubsetOpInterface : public ::mlir::OpInterface<SubsetOpInterface, detail::SubsetOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<SubsetOpInterface, detail::SubsetOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::SubsetOpInterfaceTrait<ConcreteOp> {};
  /// Return "true" if this op and the given candidate subset op operate on
  /// equivalent subsets. Return "false" if the two subsets are disjoint
  /// or cannot be proven to be equivalent.
  /// 
  /// This interface method does not have to be implemented if
  /// `getAccessedHyperrectangularSlice` is implemented.
  bool operatesOnEquivalentSubset(::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn);
  /// Return "true" if this op and the given candidate subset op operate on
  /// disjoint subsets. Return "false" if the two subsets are equivalent,
  /// overlapping or cannot be proven to be disjoint.
  /// 
  /// This interface method does not have to be implemented if
  /// `getAccessedHyperrectangularSlice` is implemented.
  bool operatesOnDisjointSubset(::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn);
  /// If this op operates on a hyperrectangular subset, return a
  /// description of the subset in terms of offsets, sizes and strides.
  /// Otherwise, return "failure".
  /// 
  /// This interface method is a convenience method for the most common case
  /// of hyperrectangular subset ops. It is optional. If it is implemented,
  /// `operatesOnEquivalentSubset` and `operatesOnDisjointSubset` do not
  /// have to be implemented.
  ::mlir::FailureOr<::mlir::HyperrectangularSlice> getAccessedHyperrectangularSlice();

    /// Return the container that this operation is operating on. In case of an
    /// extraction op, the container is the source tensor. In case of an
    /// insertion op, the container is the destination tensor.
    Value getTensorContainer() {
      return ::mlir::detail::getTensorContainer(getOperation());
    }
};
namespace detail {
  template <typename ConcreteOp>
  struct SubsetOpInterfaceTrait : public ::mlir::OpInterface<SubsetOpInterface, detail::SubsetOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Return "true" if this op and the given candidate subset op operate on
    /// equivalent subsets. Return "false" if the two subsets are disjoint
    /// or cannot be proven to be equivalent.
    /// 
    /// This interface method does not have to be implemented if
    /// `getAccessedHyperrectangularSlice` is implemented.
    bool operatesOnEquivalentSubset(::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) {
      return ::mlir::detail::defaultOperatesOnEquivalentSubset(
              (*static_cast<ConcreteOp *>(this)), candidate, equivalenceFn);
    }
    /// Return "true" if this op and the given candidate subset op operate on
    /// disjoint subsets. Return "false" if the two subsets are equivalent,
    /// overlapping or cannot be proven to be disjoint.
    /// 
    /// This interface method does not have to be implemented if
    /// `getAccessedHyperrectangularSlice` is implemented.
    bool operatesOnDisjointSubset(::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) {
      return ::mlir::detail::defaultOperatesOnDisjointSubset(
              (*static_cast<ConcreteOp *>(this)), candidate, equivalenceFn);
    }
    /// If this op operates on a hyperrectangular subset, return a
    /// description of the subset in terms of offsets, sizes and strides.
    /// Otherwise, return "failure".
    /// 
    /// This interface method is a convenience method for the most common case
    /// of hyperrectangular subset ops. It is optional. If it is implemented,
    /// `operatesOnEquivalentSubset` and `operatesOnDisjointSubset` do not
    /// have to be implemented.
    ::mlir::FailureOr<::mlir::HyperrectangularSlice> getAccessedHyperrectangularSlice() {
      return ::mlir::failure();
    }
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      return ::mlir::detail::verifySubsetOpInterface(
        ::mlir::cast<::mlir::SubsetOpInterface>(op));
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class SubsetExtractionOpInterface;
namespace detail {
struct SubsetExtractionOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::OpOperand &(*getSourceOperand)(const Concept *impl, ::mlir::Operation *);
    /// The base classes of this interface.
    const ::mlir::SubsetOpInterface::Concept *implSubsetOpInterface = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implSubsetOpInterface = interfaceMap.lookup<::mlir::SubsetOpInterface>();
      assert(implSubsetOpInterface && "`::mlir::SubsetExtractionOpInterface` expected its base interface `::mlir::SubsetOpInterface` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::SubsetExtractionOpInterface;
    Model() : Concept{getSourceOperand} {}

    static inline ::mlir::OpOperand &getSourceOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::SubsetExtractionOpInterface;
    FallbackModel() : Concept{getSourceOperand} {}

    static inline ::mlir::OpOperand &getSourceOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct SubsetExtractionOpInterfaceTrait;

} // namespace detail
class SubsetExtractionOpInterface : public ::mlir::OpInterface<SubsetExtractionOpInterface, detail::SubsetExtractionOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<SubsetExtractionOpInterface, detail::SubsetExtractionOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::SubsetExtractionOpInterfaceTrait<ConcreteOp> {};
  /// Return the source tensor operand.
  ::mlir::OpOperand &getSourceOperand();

    /// Return the single result of this op.
    ::mlir::Value getResult() {
      return getOperation()->getResult(0);
    }
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::SubsetOpInterface
  //===----------------------------------------------------------------===//

  operator ::mlir::SubsetOpInterface () const {
    if (!*this) return nullptr;
    return ::mlir::SubsetOpInterface(*this, getImpl()->implSubsetOpInterface);
  }

  /// Return "true" if this op and the given candidate subset op operate on
  /// equivalent subsets. Return "false" if the two subsets are disjoint
  /// or cannot be proven to be equivalent.
  /// 
  /// This interface method does not have to be implemented if
  /// `getAccessedHyperrectangularSlice` is implemented.
  bool operatesOnEquivalentSubset(::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn);
  /// Return "true" if this op and the given candidate subset op operate on
  /// disjoint subsets. Return "false" if the two subsets are equivalent,
  /// overlapping or cannot be proven to be disjoint.
  /// 
  /// This interface method does not have to be implemented if
  /// `getAccessedHyperrectangularSlice` is implemented.
  bool operatesOnDisjointSubset(::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn);
  /// If this op operates on a hyperrectangular subset, return a
  /// description of the subset in terms of offsets, sizes and strides.
  /// Otherwise, return "failure".
  /// 
  /// This interface method is a convenience method for the most common case
  /// of hyperrectangular subset ops. It is optional. If it is implemented,
  /// `operatesOnEquivalentSubset` and `operatesOnDisjointSubset` do not
  /// have to be implemented.
  ::mlir::FailureOr<::mlir::HyperrectangularSlice> getAccessedHyperrectangularSlice();

    /// Return the container that this operation is operating on. In case of an
    /// extraction op, the container is the source tensor. In case of an
    /// insertion op, the container is the destination tensor.
    Value getTensorContainer() {
      return ::mlir::detail::getTensorContainer(getOperation());
    }
};
namespace detail {
  template <typename ConcreteOp>
  struct SubsetExtractionOpInterfaceTrait : public ::mlir::OpInterface<SubsetExtractionOpInterface, detail::SubsetExtractionOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      return ::mlir::detail::verifySubsetExtractionOpInterface(
        ::mlir::cast<::mlir::SubsetExtractionOpInterface>(op));
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class SubsetInsertionOpInterface;
namespace detail {
struct SubsetInsertionOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::OpOperand &(*getSourceOperand)(const Concept *impl, ::mlir::Operation *);
    ::mlir::OpOperand &(*getDestinationOperand)(const Concept *impl, ::mlir::Operation *);
    ::mlir::OpResult (*getUpdatedDestination)(const Concept *impl, ::mlir::Operation *);
    bool (*isEquivalentSubset)(const Concept *impl, ::mlir::Operation *, ::mlir::Value, ::llvm::function_ref<bool(Value, Value)>);
    ::mlir::Value (*buildSubsetExtraction)(const Concept *impl, ::mlir::Operation *, ::mlir::OpBuilder &, Location);
    ::llvm::SmallVector<::mlir::Value> (*getValuesNeededToBuildSubsetExtraction)(const Concept *impl, ::mlir::Operation *);
    /// The base classes of this interface.
    const ::mlir::SubsetOpInterface::Concept *implSubsetOpInterface = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implSubsetOpInterface = interfaceMap.lookup<::mlir::SubsetOpInterface>();
      assert(implSubsetOpInterface && "`::mlir::SubsetInsertionOpInterface` expected its base interface `::mlir::SubsetOpInterface` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::SubsetInsertionOpInterface;
    Model() : Concept{getSourceOperand, getDestinationOperand, getUpdatedDestination, isEquivalentSubset, buildSubsetExtraction, getValuesNeededToBuildSubsetExtraction} {}

    static inline ::mlir::OpOperand &getSourceOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpOperand &getDestinationOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpResult getUpdatedDestination(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isEquivalentSubset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn);
    static inline ::mlir::Value buildSubsetExtraction(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder & builder, Location loc);
    static inline ::llvm::SmallVector<::mlir::Value> getValuesNeededToBuildSubsetExtraction(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::SubsetInsertionOpInterface;
    FallbackModel() : Concept{getSourceOperand, getDestinationOperand, getUpdatedDestination, isEquivalentSubset, buildSubsetExtraction, getValuesNeededToBuildSubsetExtraction} {}

    static inline ::mlir::OpOperand &getSourceOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpOperand &getDestinationOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpResult getUpdatedDestination(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isEquivalentSubset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn);
    static inline ::mlir::Value buildSubsetExtraction(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder & builder, Location loc);
    static inline ::llvm::SmallVector<::mlir::Value> getValuesNeededToBuildSubsetExtraction(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    ::mlir::OpOperand &getDestinationOperand(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::OpResult getUpdatedDestination(::mlir::Operation *tablegen_opaque_val) const;
    bool isEquivalentSubset(::mlir::Operation *tablegen_opaque_val, ::mlir::Value candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) const;
  };
};
template <typename ConcreteOp>
struct SubsetInsertionOpInterfaceTrait;

} // namespace detail
class SubsetInsertionOpInterface : public ::mlir::OpInterface<SubsetInsertionOpInterface, detail::SubsetInsertionOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<SubsetInsertionOpInterface, detail::SubsetInsertionOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::SubsetInsertionOpInterfaceTrait<ConcreteOp> {};
  /// Return the source operand. The source operand can have any type.
  ::mlir::OpOperand &getSourceOperand();
  /// Return the destination operand. The destination operand must be a
  /// tensor.
  /// 
  /// This function does not have to be implemented for destination style
  /// ops that have exactly one "init" operand.
  ::mlir::OpOperand &getDestinationOperand();
  /// Return the updated destination result.
  /// 
  /// This function does not have to be implemented for destination style
  /// ops.
  ::mlir::OpResult getUpdatedDestination();
  /// Return "true" if this operation inserts into a subset that is
  /// equivalent to the subset defined by `candidate`.
  /// 
  /// Two subsets are "equivalent" and "same" if they can bufferize to the
  /// same buffer views/aliases. If they are "equivalent", the tensor IR
  /// may be expressed in terms of different SSA values (but they could
  /// bufferize to MemRef SSA values that can CSE without breaking
  /// correctness). `equivalenceFn` should return "true" if the two given
  /// values are equivalent.
  /// 
  /// Example:
  /// ```
  /// // The subset of the SubsetInsertionOpInterface op %1 is equivalent to
  /// // the subset defined by %2 (but not "same"):
  /// %0 = arith.select %c, %t, %t : tensor<?xf32>
  /// %1 = tensor.insert_slice %x into %0[0][5][1]
  ///     : tensor<5xf32> into tensor<?xf32>
  /// %2 = tensor.extract_slice %t[0][5][1] : tensor<?xf32> to tensor<5xf32>
  /// 
  /// // The subset of the SubsetInsertionOpInterface op %1 is equivalent to
  /// // and "same" as the subset defined by %2.
  /// %1 = tensor.insert_slice %x into %t[0][5][1]
  ///     : tensor<5xf32> into tensor<?xf32>
  /// %2 = tensor.extract_slice %t[0][5][1] : tensor<?xf32> to tensor<5xf32>
  /// ```
  /// 
  /// Note: By default, this function calls
  /// `SubsetOpInterface::operatesOnEquivalentSubset`.
  bool isEquivalentSubset(::mlir::Value candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn);
  /// Return the subset of the destination tensor that this operation
  /// inserts into.
  /// 
  /// Example:
  /// ```
  /// // SubsetOpInterface op:
  /// %0 = tensor.insert_slice %t0 into %t1[%pos][5][1]
  ///     : tensor<5xf32> into tensor<?xf32>
  /// // Subset (built by this function):
  /// %1 = tensor.extract_slice %t1[%pos][5][1]
  ///     : tensor<?xf32> to tensor<5xf32>
  /// ```
  /// 
  /// Note: Implementations do not necessarily have to build new IR. They
  /// may return existing SSA values.
  ::mlir::Value buildSubsetExtraction(::mlir::OpBuilder & builder, Location loc);
  /// Return all SSA values that are needed (i.e., must be in scope) at the
  /// insertion of the builder when calling `buildSubsetExtraction`. Users
  /// of `buildSubsetExtraction` can use this helper method to find a
  /// suitable insertion point.
  /// 
  /// Example: The SSA values needed to build the subset in the example of
  /// `buildSubsetExtraction` are %t1 and %pos.
  ::llvm::SmallVector<::mlir::Value> getValuesNeededToBuildSubsetExtraction();

    /// Return "true" if this operation inserts into the same subset as defined
    /// by `candidate`.
    ///
    /// Note: This function is useful outside of bufferization, where no tensor
    /// equivalence information is available.
    bool isSameSubset(OpResult candidate) {
      auto subsetOp = cast<::mlir::SubsetInsertionOpInterface>(
          getOperation());
      return subsetOp.isEquivalentSubset(
          candidate, [](Value v1, Value v2) { return v1 == v2; });
    }
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::SubsetOpInterface
  //===----------------------------------------------------------------===//

  operator ::mlir::SubsetOpInterface () const {
    if (!*this) return nullptr;
    return ::mlir::SubsetOpInterface(*this, getImpl()->implSubsetOpInterface);
  }

  /// Return "true" if this op and the given candidate subset op operate on
  /// equivalent subsets. Return "false" if the two subsets are disjoint
  /// or cannot be proven to be equivalent.
  /// 
  /// This interface method does not have to be implemented if
  /// `getAccessedHyperrectangularSlice` is implemented.
  bool operatesOnEquivalentSubset(::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn);
  /// Return "true" if this op and the given candidate subset op operate on
  /// disjoint subsets. Return "false" if the two subsets are equivalent,
  /// overlapping or cannot be proven to be disjoint.
  /// 
  /// This interface method does not have to be implemented if
  /// `getAccessedHyperrectangularSlice` is implemented.
  bool operatesOnDisjointSubset(::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn);
  /// If this op operates on a hyperrectangular subset, return a
  /// description of the subset in terms of offsets, sizes and strides.
  /// Otherwise, return "failure".
  /// 
  /// This interface method is a convenience method for the most common case
  /// of hyperrectangular subset ops. It is optional. If it is implemented,
  /// `operatesOnEquivalentSubset` and `operatesOnDisjointSubset` do not
  /// have to be implemented.
  ::mlir::FailureOr<::mlir::HyperrectangularSlice> getAccessedHyperrectangularSlice();

    /// Return the container that this operation is operating on. In case of an
    /// extraction op, the container is the source tensor. In case of an
    /// insertion op, the container is the destination tensor.
    Value getTensorContainer() {
      return ::mlir::detail::getTensorContainer(getOperation());
    }
};
namespace detail {
  template <typename ConcreteOp>
  struct SubsetInsertionOpInterfaceTrait : public ::mlir::OpInterface<SubsetInsertionOpInterface, detail::SubsetInsertionOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Return the destination operand. The destination operand must be a
    /// tensor.
    /// 
    /// This function does not have to be implemented for destination style
    /// ops that have exactly one "init" operand.
    ::mlir::OpOperand &getDestinationOperand() {
      return ::mlir::detail::defaultGetDestinationOperand(
              (*static_cast<ConcreteOp *>(this)).getOperation());
    }
    /// Return the updated destination result.
    /// 
    /// This function does not have to be implemented for destination style
    /// ops.
    ::mlir::OpResult getUpdatedDestination() {
      return ::mlir::detail::defaultGetUpdatedDestination(
              (*static_cast<ConcreteOp *>(this)).getOperation());
    }
    /// Return "true" if this operation inserts into a subset that is
    /// equivalent to the subset defined by `candidate`.
    /// 
    /// Two subsets are "equivalent" and "same" if they can bufferize to the
    /// same buffer views/aliases. If they are "equivalent", the tensor IR
    /// may be expressed in terms of different SSA values (but they could
    /// bufferize to MemRef SSA values that can CSE without breaking
    /// correctness). `equivalenceFn` should return "true" if the two given
    /// values are equivalent.
    /// 
    /// Example:
    /// ```
    /// // The subset of the SubsetInsertionOpInterface op %1 is equivalent to
    /// // the subset defined by %2 (but not "same"):
    /// %0 = arith.select %c, %t, %t : tensor<?xf32>
    /// %1 = tensor.insert_slice %x into %0[0][5][1]
    ///     : tensor<5xf32> into tensor<?xf32>
    /// %2 = tensor.extract_slice %t[0][5][1] : tensor<?xf32> to tensor<5xf32>
    /// 
    /// // The subset of the SubsetInsertionOpInterface op %1 is equivalent to
    /// // and "same" as the subset defined by %2.
    /// %1 = tensor.insert_slice %x into %t[0][5][1]
    ///     : tensor<5xf32> into tensor<?xf32>
    /// %2 = tensor.extract_slice %t[0][5][1] : tensor<?xf32> to tensor<5xf32>
    /// ```
    /// 
    /// Note: By default, this function calls
    /// `SubsetOpInterface::operatesOnEquivalentSubset`.
    bool isEquivalentSubset(::mlir::Value candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) {
      return ::mlir::detail::defaultIsEquivalentSubset(
              (*static_cast<ConcreteOp *>(this)).getOperation(), candidate, equivalenceFn);
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
bool detail::SubsetOpInterfaceInterfaceTraits::Model<ConcreteOp>::operatesOnEquivalentSubset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).operatesOnEquivalentSubset(candidate, equivalenceFn);
}
template<typename ConcreteOp>
bool detail::SubsetOpInterfaceInterfaceTraits::Model<ConcreteOp>::operatesOnDisjointSubset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).operatesOnDisjointSubset(candidate, equivalenceFn);
}
template<typename ConcreteOp>
::mlir::FailureOr<::mlir::HyperrectangularSlice> detail::SubsetOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAccessedHyperrectangularSlice(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAccessedHyperrectangularSlice();
}
template<typename ConcreteOp>
bool detail::SubsetOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::operatesOnEquivalentSubset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) {
  return static_cast<const ConcreteOp *>(impl)->operatesOnEquivalentSubset(tablegen_opaque_val, candidate, equivalenceFn);
}
template<typename ConcreteOp>
bool detail::SubsetOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::operatesOnDisjointSubset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) {
  return static_cast<const ConcreteOp *>(impl)->operatesOnDisjointSubset(tablegen_opaque_val, candidate, equivalenceFn);
}
template<typename ConcreteOp>
::mlir::FailureOr<::mlir::HyperrectangularSlice> detail::SubsetOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAccessedHyperrectangularSlice(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAccessedHyperrectangularSlice(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::SubsetOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::operatesOnEquivalentSubset(::mlir::Operation *tablegen_opaque_val, ::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) const {
return ::mlir::detail::defaultOperatesOnEquivalentSubset(
              (llvm::cast<ConcreteOp>(tablegen_opaque_val)), candidate, equivalenceFn);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::SubsetOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::operatesOnDisjointSubset(::mlir::Operation *tablegen_opaque_val, ::mlir::SubsetOpInterface candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) const {
return ::mlir::detail::defaultOperatesOnDisjointSubset(
              (llvm::cast<ConcreteOp>(tablegen_opaque_val)), candidate, equivalenceFn);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::FailureOr<::mlir::HyperrectangularSlice> detail::SubsetOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAccessedHyperrectangularSlice(::mlir::Operation *tablegen_opaque_val) const {
return ::mlir::failure();
}
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::mlir::OpOperand &detail::SubsetExtractionOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSourceOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSourceOperand();
}
template<typename ConcreteOp>
::mlir::OpOperand &detail::SubsetExtractionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSourceOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSourceOperand(tablegen_opaque_val);
}
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
::mlir::OpOperand &detail::SubsetInsertionOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSourceOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSourceOperand();
}
template<typename ConcreteOp>
::mlir::OpOperand &detail::SubsetInsertionOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDestinationOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDestinationOperand();
}
template<typename ConcreteOp>
::mlir::OpResult detail::SubsetInsertionOpInterfaceInterfaceTraits::Model<ConcreteOp>::getUpdatedDestination(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getUpdatedDestination();
}
template<typename ConcreteOp>
bool detail::SubsetInsertionOpInterfaceInterfaceTraits::Model<ConcreteOp>::isEquivalentSubset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isEquivalentSubset(candidate, equivalenceFn);
}
template<typename ConcreteOp>
::mlir::Value detail::SubsetInsertionOpInterfaceInterfaceTraits::Model<ConcreteOp>::buildSubsetExtraction(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder & builder, Location loc) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).buildSubsetExtraction(builder, loc);
}
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::Value> detail::SubsetInsertionOpInterfaceInterfaceTraits::Model<ConcreteOp>::getValuesNeededToBuildSubsetExtraction(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getValuesNeededToBuildSubsetExtraction();
}
template<typename ConcreteOp>
::mlir::OpOperand &detail::SubsetInsertionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSourceOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSourceOperand(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::OpOperand &detail::SubsetInsertionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDestinationOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getDestinationOperand(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::OpResult detail::SubsetInsertionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getUpdatedDestination(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getUpdatedDestination(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::SubsetInsertionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isEquivalentSubset(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) {
  return static_cast<const ConcreteOp *>(impl)->isEquivalentSubset(tablegen_opaque_val, candidate, equivalenceFn);
}
template<typename ConcreteOp>
::mlir::Value detail::SubsetInsertionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::buildSubsetExtraction(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpBuilder & builder, Location loc) {
  return static_cast<const ConcreteOp *>(impl)->buildSubsetExtraction(tablegen_opaque_val, builder, loc);
}
template<typename ConcreteOp>
::llvm::SmallVector<::mlir::Value> detail::SubsetInsertionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getValuesNeededToBuildSubsetExtraction(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getValuesNeededToBuildSubsetExtraction(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::OpOperand &detail::SubsetInsertionOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getDestinationOperand(::mlir::Operation *tablegen_opaque_val) const {
return ::mlir::detail::defaultGetDestinationOperand(
              (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation());
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::OpResult detail::SubsetInsertionOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getUpdatedDestination(::mlir::Operation *tablegen_opaque_val) const {
return ::mlir::detail::defaultGetUpdatedDestination(
              (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation());
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::SubsetInsertionOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isEquivalentSubset(::mlir::Operation *tablegen_opaque_val, ::mlir::Value candidate, ::llvm::function_ref<bool(Value, Value)> equivalenceFn) const {
return ::mlir::detail::defaultIsEquivalentSubset(
              (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation(), candidate, equivalenceFn);
}
} // namespace mlir
