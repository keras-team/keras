/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace linalg {
class ContractionOpInterface;
namespace detail {
struct ContractionOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    Value (*lhs)(const Concept *impl, ::mlir::Operation *);
    Value (*rhs)(const Concept *impl, ::mlir::Operation *);
    bool (*isRowMajorMatmul)(const Concept *impl, ::mlir::Operation *);
    bool (*isColumnMajorMatmul)(const Concept *impl, ::mlir::Operation *);
    bool (*isRowMajorBatchMatmul)(const Concept *impl, ::mlir::Operation *);
    bool (*isVecmat)(const Concept *impl, ::mlir::Operation *);
    bool (*isBatchVecmat)(const Concept *impl, ::mlir::Operation *);
    bool (*isMatvec)(const Concept *impl, ::mlir::Operation *);
    bool (*isBatchMatvec)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::linalg::ContractionOpInterface;
    Model() : Concept{lhs, rhs, isRowMajorMatmul, isColumnMajorMatmul, isRowMajorBatchMatmul, isVecmat, isBatchVecmat, isMatvec, isBatchMatvec} {}

    static inline Value lhs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value rhs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isRowMajorMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isColumnMajorMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isRowMajorBatchMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isVecmat(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isBatchVecmat(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isMatvec(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isBatchMatvec(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::linalg::ContractionOpInterface;
    FallbackModel() : Concept{lhs, rhs, isRowMajorMatmul, isColumnMajorMatmul, isRowMajorBatchMatmul, isVecmat, isBatchVecmat, isMatvec, isBatchMatvec} {}

    static inline Value lhs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value rhs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isRowMajorMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isColumnMajorMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isRowMajorBatchMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isVecmat(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isBatchVecmat(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isMatvec(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isBatchMatvec(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct ContractionOpInterfaceTrait;

} // namespace detail
class ContractionOpInterface : public ::mlir::OpInterface<ContractionOpInterface, detail::ContractionOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ContractionOpInterface, detail::ContractionOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ContractionOpInterfaceTrait<ConcreteOp> {};
  /// Returns the left-hand side operand.
  Value lhs();
  /// Returns the right-hand side operand.
  Value rhs();
  /// Returns whether the given op has indexing maps that correspond to a
  /// row-major matmul operation.
  bool isRowMajorMatmul();
  /// Returns whether the given op has indexing maps that correspond to a
  /// column-major matmul operation.
  bool isColumnMajorMatmul();
  /// Returns whether the given op has indexing maps that correspond to a
  /// row-major batch matmul operation.
  bool isRowMajorBatchMatmul();
  /// Returns whether the given op has indexing maps that correspond to a
  /// vector-matrix multiplication.
  bool isVecmat();
  /// Returns whether the given op has indexing maps that correspond to a
  /// batched vector-matrix multiplication.
  bool isBatchVecmat();
  /// Returns whether the given op has indexing maps that correspond to a
  /// matrix-vector multiplication.
  bool isMatvec();
  /// Returns whether the given op has indexing maps that correspond to a
  /// batched matrix-vector multiplication.
  bool isBatchMatvec();
};
namespace detail {
  template <typename ConcreteOp>
  struct ContractionOpInterfaceTrait : public ::mlir::OpInterface<ContractionOpInterface, detail::ContractionOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::llvm::LogicalResult verifyRegionTrait(::mlir::Operation *op) {
      return detail::verifyContractionInterface(op);
    }
  };
}// namespace detail
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class ConvolutionOpInterface;
namespace detail {
struct ConvolutionOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    Value (*image)(const Concept *impl, ::mlir::Operation *);
    Value (*filter)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::linalg::ConvolutionOpInterface;
    Model() : Concept{image, filter} {}

    static inline Value image(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value filter(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::linalg::ConvolutionOpInterface;
    FallbackModel() : Concept{image, filter} {}

    static inline Value image(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value filter(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    Value image(::mlir::Operation *tablegen_opaque_val) const;
    Value filter(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct ConvolutionOpInterfaceTrait;

} // namespace detail
class ConvolutionOpInterface : public ::mlir::OpInterface<ConvolutionOpInterface, detail::ConvolutionOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ConvolutionOpInterface, detail::ConvolutionOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ConvolutionOpInterfaceTrait<ConcreteOp> {};
  /// Return the image operand.
  Value image();
  /// Return the filter operand.
  Value filter();
};
namespace detail {
  template <typename ConcreteOp>
  struct ConvolutionOpInterfaceTrait : public ::mlir::OpInterface<ConvolutionOpInterface, detail::ConvolutionOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Return the image operand.
    Value image() {
      return (*static_cast<ConcreteOp *>(this)).getOperation()->getOperand(0);
    }
    /// Return the filter operand.
    Value filter() {
      return (*static_cast<ConcreteOp *>(this)).getOperation()->getOperand(1);
    }
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      return detail::verifyConvolutionInterface(op);
    }
  };
}// namespace detail
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class FillOpInterface;
namespace detail {
struct FillOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    Value (*value)(const Concept *impl, ::mlir::Operation *);
    Value (*output)(const Concept *impl, ::mlir::Operation *);
    Value (*result)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::linalg::FillOpInterface;
    Model() : Concept{value, output, result} {}

    static inline Value value(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value output(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value result(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::linalg::FillOpInterface;
    FallbackModel() : Concept{value, output, result} {}

    static inline Value value(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value output(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value result(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    Value value(::mlir::Operation *tablegen_opaque_val) const;
    Value output(::mlir::Operation *tablegen_opaque_val) const;
    Value result(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct FillOpInterfaceTrait;

} // namespace detail
class FillOpInterface : public ::mlir::OpInterface<FillOpInterface, detail::FillOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<FillOpInterface, detail::FillOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::FillOpInterfaceTrait<ConcreteOp> {};
  /// Return the fill value.
  Value value();
  /// Return the output operand.
  Value output();
  /// Return the result.
  Value result();
};
namespace detail {
  template <typename ConcreteOp>
  struct FillOpInterfaceTrait : public ::mlir::OpInterface<FillOpInterface, detail::FillOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Return the fill value.
    Value value() {
      return (*static_cast<ConcreteOp *>(this)).getOperation()->getOperand(0);
    }
    /// Return the output operand.
    Value output() {
      return (*static_cast<ConcreteOp *>(this)).getOperation()->getOperand(1);
    }
    /// Return the result.
    Value result() {
      if ((*static_cast<ConcreteOp *>(this)).getOperation()->getResults().empty())
          return nullptr;
        return (*static_cast<ConcreteOp *>(this)).getOperation()->getResults().front();
    }
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      return detail::verifyFillInterface(op);
    }
  };
}// namespace detail
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class LinalgOp;
namespace detail {
struct LinalgOpInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    unsigned (*getNumParallelLoops)(const Concept *impl, ::mlir::Operation *);
    void (*getParallelDims)(const Concept *impl, ::mlir::Operation *, SmallVectorImpl<unsigned> &);
    unsigned (*getNumReductionLoops)(const Concept *impl, ::mlir::Operation *);
    void (*getReductionDims)(const Concept *impl, ::mlir::Operation *, SmallVectorImpl<unsigned> &);
    unsigned (*getNumLoops)(const Concept *impl, ::mlir::Operation *);
    bool (*hasSingleReductionLoop)(const Concept *impl, ::mlir::Operation *);
    bool (*payloadUsesValueFromOperand)(const Concept *impl, ::mlir::Operation *, OpOperand *);
    bool (*isInitTensor)(const Concept *impl, ::mlir::Operation *, OpOperand *);
    int64_t (*getRank)(const Concept *impl, ::mlir::Operation *, OpOperand*);
    Block::BlockArgListType (*getRegionInputArgs)(const Concept *impl, ::mlir::Operation *);
    Block::BlockArgListType (*getRegionOutputArgs)(const Concept *impl, ::mlir::Operation *);
    ArrayRef<int64_t> (*getShape)(const Concept *impl, ::mlir::Operation *, OpOperand*);
    BlockArgument (*getMatchingBlockArgument)(const Concept *impl, ::mlir::Operation *, OpOperand *);
    OpOperand *(*getMatchingOpOperand)(const Concept *impl, ::mlir::Operation *, BlockArgument);
    AffineMap (*getMatchingIndexingMap)(const Concept *impl, ::mlir::Operation *, OpOperand*);
    AffineMap (*getIndexingMapMatchingResult)(const Concept *impl, ::mlir::Operation *, OpResult);
    OpOperand *(*getMatchingYieldValue)(const Concept *impl, ::mlir::Operation *, OpOperand*);
    Block*(*getBlock)(const Concept *impl, ::mlir::Operation *);
    SmallVector<utils::IteratorType> (*getIteratorTypesArray)(const Concept *impl, ::mlir::Operation *);
    bool (*hasDynamicIndexingMaps)(const Concept *impl, ::mlir::Operation *);
    LogicalResult (*verifyIndexingMapRequiredAttributes)(const Concept *impl, ::mlir::Operation *);
    ArrayAttr (*getIndexingMaps)(const Concept *impl, ::mlir::Operation *);
    SmallVector<AffineMap> (*getIndexingMapsArray)(const Concept *impl, ::mlir::Operation *);
    bool (*hasDynamicShape)(const Concept *impl, ::mlir::Operation *);
    std::string (*getLibraryCallName)(const Concept *impl, ::mlir::Operation *);
    bool (*hasIndexSemantics)(const Concept *impl, ::mlir::Operation *);
    ::llvm::SmallVector<OpOperand *> (*getOpOperandsMatchingBBargs)(const Concept *impl, ::mlir::Operation *);
    LogicalResult (*mapIterationSpaceDimToOperandDim)(const Concept *impl, ::mlir::Operation *, unsigned, ::mlir::Value &, unsigned &);
    void (*mapIterationSpaceDimToAllOperandDims)(const Concept *impl, ::mlir::Operation *, unsigned, mlir::SmallVectorImpl<std::pair<Value, unsigned>>&);
    AffineMap (*getLoopsToShapesMap)(const Concept *impl, ::mlir::Operation *);
    AffineMap (*getShapesToLoopsMap)(const Concept *impl, ::mlir::Operation *);
    bool (*canOpOperandsBeDropped)(const Concept *impl, ::mlir::Operation *, ArrayRef<OpOperand *>);
    SmallVector<int64_t> (*getStaticShape)(const Concept *impl, ::mlir::Operation *);
    SmallVector<int64_t, 4> (*getStaticLoopRanges)(const Concept *impl, ::mlir::Operation *);
    std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)> (*getRegionBuilder)();
    bool (*hasOnlyProjectedPermutations)(const Concept *impl, ::mlir::Operation *);
    /// The base classes of this interface.
    const ::mlir::DestinationStyleOpInterface::Concept *implDestinationStyleOpInterface = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implDestinationStyleOpInterface = interfaceMap.lookup<::mlir::DestinationStyleOpInterface>();
      assert(implDestinationStyleOpInterface && "`::mlir::linalg::LinalgOp` expected its base interface `::mlir::DestinationStyleOpInterface` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::linalg::LinalgOp;
    Model() : Concept{getNumParallelLoops, getParallelDims, getNumReductionLoops, getReductionDims, getNumLoops, hasSingleReductionLoop, payloadUsesValueFromOperand, isInitTensor, getRank, getRegionInputArgs, getRegionOutputArgs, getShape, getMatchingBlockArgument, getMatchingOpOperand, getMatchingIndexingMap, getIndexingMapMatchingResult, getMatchingYieldValue, getBlock, getIteratorTypesArray, hasDynamicIndexingMaps, verifyIndexingMapRequiredAttributes, getIndexingMaps, getIndexingMapsArray, hasDynamicShape, getLibraryCallName, hasIndexSemantics, getOpOperandsMatchingBBargs, mapIterationSpaceDimToOperandDim, mapIterationSpaceDimToAllOperandDims, getLoopsToShapesMap, getShapesToLoopsMap, canOpOperandsBeDropped, getStaticShape, getStaticLoopRanges, getRegionBuilder, hasOnlyProjectedPermutations} {}

    static inline unsigned getNumParallelLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void getParallelDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res);
    static inline unsigned getNumReductionLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void getReductionDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res);
    static inline unsigned getNumLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasSingleReductionLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool payloadUsesValueFromOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline bool isInitTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline int64_t getRank(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline Block::BlockArgListType getRegionInputArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Block::BlockArgListType getRegionOutputArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ArrayRef<int64_t> getShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline BlockArgument getMatchingBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline OpOperand *getMatchingOpOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, BlockArgument blockArgument);
    static inline AffineMap getMatchingIndexingMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline AffineMap getIndexingMapMatchingResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpResult result);
    static inline OpOperand *getMatchingYieldValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline Block*getBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<utils::IteratorType> getIteratorTypesArray(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasDynamicIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline LogicalResult verifyIndexingMapRequiredAttributes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ArrayAttr getIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<AffineMap> getIndexingMapsArray(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasDynamicShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline std::string getLibraryCallName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasIndexSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::SmallVector<OpOperand *> getOpOperandsMatchingBBargs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline LogicalResult mapIterationSpaceDimToOperandDim(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned dimPos, ::mlir::Value & operand, unsigned & operandDimPos);
    static inline void mapIterationSpaceDimToAllOperandDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned dimPos, mlir::SmallVectorImpl<std::pair<Value, unsigned>>& operandDimPairs);
    static inline AffineMap getLoopsToShapesMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline AffineMap getShapesToLoopsMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool canOpOperandsBeDropped(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<OpOperand *> droppedOperands);
    static inline SmallVector<int64_t> getStaticShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<int64_t, 4> getStaticLoopRanges(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)> getRegionBuilder();
    static inline bool hasOnlyProjectedPermutations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::linalg::LinalgOp;
    FallbackModel() : Concept{getNumParallelLoops, getParallelDims, getNumReductionLoops, getReductionDims, getNumLoops, hasSingleReductionLoop, payloadUsesValueFromOperand, isInitTensor, getRank, getRegionInputArgs, getRegionOutputArgs, getShape, getMatchingBlockArgument, getMatchingOpOperand, getMatchingIndexingMap, getIndexingMapMatchingResult, getMatchingYieldValue, getBlock, getIteratorTypesArray, hasDynamicIndexingMaps, verifyIndexingMapRequiredAttributes, getIndexingMaps, getIndexingMapsArray, hasDynamicShape, getLibraryCallName, hasIndexSemantics, getOpOperandsMatchingBBargs, mapIterationSpaceDimToOperandDim, mapIterationSpaceDimToAllOperandDims, getLoopsToShapesMap, getShapesToLoopsMap, canOpOperandsBeDropped, getStaticShape, getStaticLoopRanges, getRegionBuilder, hasOnlyProjectedPermutations} {}

    static inline unsigned getNumParallelLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void getParallelDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res);
    static inline unsigned getNumReductionLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void getReductionDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res);
    static inline unsigned getNumLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasSingleReductionLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool payloadUsesValueFromOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline bool isInitTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline int64_t getRank(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline Block::BlockArgListType getRegionInputArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Block::BlockArgListType getRegionOutputArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ArrayRef<int64_t> getShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline BlockArgument getMatchingBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline OpOperand *getMatchingOpOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, BlockArgument blockArgument);
    static inline AffineMap getMatchingIndexingMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline AffineMap getIndexingMapMatchingResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpResult result);
    static inline OpOperand *getMatchingYieldValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline Block*getBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<utils::IteratorType> getIteratorTypesArray(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasDynamicIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline LogicalResult verifyIndexingMapRequiredAttributes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ArrayAttr getIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<AffineMap> getIndexingMapsArray(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasDynamicShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline std::string getLibraryCallName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasIndexSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::llvm::SmallVector<OpOperand *> getOpOperandsMatchingBBargs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline LogicalResult mapIterationSpaceDimToOperandDim(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned dimPos, ::mlir::Value & operand, unsigned & operandDimPos);
    static inline void mapIterationSpaceDimToAllOperandDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned dimPos, mlir::SmallVectorImpl<std::pair<Value, unsigned>>& operandDimPairs);
    static inline AffineMap getLoopsToShapesMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline AffineMap getShapesToLoopsMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool canOpOperandsBeDropped(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<OpOperand *> droppedOperands);
    static inline SmallVector<int64_t> getStaticShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<int64_t, 4> getStaticLoopRanges(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)> getRegionBuilder();
    static inline bool hasOnlyProjectedPermutations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    unsigned getNumParallelLoops(::mlir::Operation *tablegen_opaque_val) const;
    void getParallelDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> &res) const;
    unsigned getNumReductionLoops(::mlir::Operation *tablegen_opaque_val) const;
    void getReductionDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> &res) const;
    unsigned getNumLoops(::mlir::Operation *tablegen_opaque_val) const;
    bool hasSingleReductionLoop(::mlir::Operation *tablegen_opaque_val) const;
    bool payloadUsesValueFromOperand(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const;
    bool isInitTensor(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const;
    int64_t getRank(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const;
    Block::BlockArgListType getRegionInputArgs(::mlir::Operation *tablegen_opaque_val) const;
    Block::BlockArgListType getRegionOutputArgs(::mlir::Operation *tablegen_opaque_val) const;
    ArrayRef<int64_t> getShape(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const;
    BlockArgument getMatchingBlockArgument(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const;
    OpOperand *getMatchingOpOperand(::mlir::Operation *tablegen_opaque_val, BlockArgument blockArgument) const;
    AffineMap getMatchingIndexingMap(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const;
    AffineMap getIndexingMapMatchingResult(::mlir::Operation *tablegen_opaque_val, OpResult result) const;
    OpOperand *getMatchingYieldValue(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const;
    Block*getBlock(::mlir::Operation *tablegen_opaque_val) const;
    SmallVector<utils::IteratorType> getIteratorTypesArray(::mlir::Operation *tablegen_opaque_val) const;
    bool hasDynamicIndexingMaps(::mlir::Operation *tablegen_opaque_val) const;
    LogicalResult verifyIndexingMapRequiredAttributes(::mlir::Operation *tablegen_opaque_val) const;
    SmallVector<AffineMap> getIndexingMapsArray(::mlir::Operation *tablegen_opaque_val) const;
    bool hasDynamicShape(::mlir::Operation *tablegen_opaque_val) const;
    std::string getLibraryCallName(::mlir::Operation *tablegen_opaque_val) const;
    ::llvm::SmallVector<OpOperand *> getOpOperandsMatchingBBargs(::mlir::Operation *tablegen_opaque_val) const;
    LogicalResult mapIterationSpaceDimToOperandDim(::mlir::Operation *tablegen_opaque_val, unsigned dimPos, ::mlir::Value &operand, unsigned &operandDimPos) const;
    void mapIterationSpaceDimToAllOperandDims(::mlir::Operation *tablegen_opaque_val, unsigned dimPos, mlir::SmallVectorImpl<std::pair<Value, unsigned>>&operandDimPairs) const;
    AffineMap getLoopsToShapesMap(::mlir::Operation *tablegen_opaque_val) const;
    AffineMap getShapesToLoopsMap(::mlir::Operation *tablegen_opaque_val) const;
    bool canOpOperandsBeDropped(::mlir::Operation *tablegen_opaque_val, ArrayRef<OpOperand *> droppedOperands) const;
    SmallVector<int64_t> getStaticShape(::mlir::Operation *tablegen_opaque_val) const;
    SmallVector<int64_t, 4> getStaticLoopRanges(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct LinalgOpTrait;

} // namespace detail
class LinalgOp : public ::mlir::OpInterface<LinalgOp, detail::LinalgOpInterfaceTraits> {
public:
  using ::mlir::OpInterface<LinalgOp, detail::LinalgOpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::LinalgOpTrait<ConcreteOp> {};
  /// Return the number of parallel loops.
  unsigned getNumParallelLoops();
  /// Return the dims that are parallel loops.
  void getParallelDims(SmallVectorImpl<unsigned> & res);
  /// Return the number of reduction loops.
  unsigned getNumReductionLoops();
  /// Return the dims that are reduction loops.
  void getReductionDims(SmallVectorImpl<unsigned> & res);
  /// Return the total number of loops within the current operation.
  unsigned getNumLoops();
  /// Returns true if the current operation has only one loop and it's a
  /// reduction loop.
  bool hasSingleReductionLoop();
  /// Return true if the payload uses the value loaded from `opOperand`. This
  /// is useful to avoid loading from "write-only" memory that may be
  /// uninitialized, as well as properly cloning "read-write" operands.
  bool payloadUsesValueFromOperand(OpOperand * opOperand);
  /// Return true if `opOperand` is an init tensor. This is true when it is
  /// an output tensor operand whose value is used in the payload region.
  bool isInitTensor(OpOperand * opOperand);
  /// Return the `opOperand` rank or zero for scalars or vectors not wrapped within a tensor or a memref.
  int64_t getRank(OpOperand* opOperand);
  /// Return the input block arguments of the region.
  Block::BlockArgListType getRegionInputArgs();
  /// Return the output block arguments of the region.
  Block::BlockArgListType getRegionOutputArgs();
  /// Return the `opOperand` shape or an empty vector for scalars or vectors
  /// not wrapped within a tensor or a memref.
  ArrayRef<int64_t> getShape(OpOperand* opOperand);
  /// Return the block argument for an `opOperand`.
  BlockArgument getMatchingBlockArgument(OpOperand * opOperand);
  /// Return the operand for a `blockArgument`.
  OpOperand *getMatchingOpOperand(BlockArgument blockArgument);
  /// Return the input or output indexing map for `opOperand`.
  AffineMap getMatchingIndexingMap(OpOperand* opOperand);
  /// Return the indexing map for a `result`.
  AffineMap getIndexingMapMatchingResult(OpResult result);
  /// Return the value yielded by the region corresponding to an output
  /// `opOperand`.
  OpOperand *getMatchingYieldValue(OpOperand* opOperand);
  /// Return the single block constituting the body of the operation by
  /// calling the getBody method on the concrete operation.
  Block*getBlock();
  /// Return iterator types in the current operation.
  /// 
  /// Default implementation assumes that the operation has an attribute
  /// `iterator_types`, but it's not always the case. Sometimes iterator types
  /// can be infered from other parameters and in such cases default
  /// getIteratorTypesArray should be overriden.
  SmallVector<utils::IteratorType> getIteratorTypesArray();
  /// Return true if the indexing map is depending on the current op instance.
  /// This means that the indexing map is dynamically synthesized by using the
  /// op instance's concrete attributes, instead of being static for all
  /// instances of the same op kind.
  bool hasDynamicIndexingMaps();
  /// Verify all attributes used by indexing maps are valid.
  LogicalResult verifyIndexingMapRequiredAttributes();
  /// Return the indexing maps attribute within the current operation.
  ArrayAttr getIndexingMaps();
  /// Return the indexing maps within the current operation.
  SmallVector<AffineMap> getIndexingMapsArray();
  /// Return true if any of the operands has a dynamic shape.
  bool hasDynamicShape();
  /// Return the name registered for this op when lowering to an external
  /// library call.
  std::string getLibraryCallName();
  /// Return whether the op accesses the iteration indices.
  bool hasIndexSemantics();
  /// Return op operands that have a corresponding argument in the basic block.
  /// By default, the block should have an argument for each operand, but there
  /// are expection. For example, in `map` output operand isn't used in
  /// the block.
  ::llvm::SmallVector<OpOperand *> getOpOperandsMatchingBBargs();
  /// Given a dimension of the iteration space of a Linalg operation, finds an
  /// operand in the operation that is defined on such dimension. Returns
  /// whether such operand was found or not. If found, also returns the
  /// operand value and the dimension position within the operand.
  LogicalResult mapIterationSpaceDimToOperandDim(unsigned dimPos, ::mlir::Value & operand, unsigned & operandDimPos);
  /// Given a dimension of the iteration space of a Linalg operation, finds
  /// all the operands in the operation that are defined on such dimension.
  /// Returns all the operand values found and their dimension positions in
  /// `operandDimPairs`.
  void mapIterationSpaceDimToAllOperandDims(unsigned dimPos, mlir::SmallVectorImpl<std::pair<Value, unsigned>>& operandDimPairs);
  /// Hook to provide a custom AffineMap used to compute all the operand
  /// subshapes given loop bounds. This is used to answer the question: "given
  /// an iteration space over the codomain, what are the subshapes of the
  /// operands involved in the computation".
  /// The default behavior is to just concatenate all the indexing maps.
  /// A custom AffineMap allows providing a map that can be used to
  /// compute subshapes even in cases where the concatenation of indexing maps
  /// (i.e. the data traversal order) is not a simple permutation of the loop
  /// traversal order. It is then possible to define ops with skewed data
  /// traversal order for which we can still easily compute hyperrectangular
  /// loop bounds and subviews.
  AffineMap getLoopsToShapesMap();
  /// Hook to provide a custom AffineMap used to construct the
  /// hyperrectangular loop iteration space given all the operand subshapes.
  /// This is used to answer the question:
  /// "Given a list of operand ranges, what is the subportion of the iteration
  /// space involved in the computation".
  /// This is the inverse problem of `getLoopsToShapesMap`.
  /// Return the empty AffineMap when such an AffineMap cannot be constructed.
  /// The default behavior is based on a very simple inference procedure that
  /// only works with permutation affine maps.
  /// A more advanced Tensor-Comprehension like inference is possible but has
  /// proven to be ambiguous in unfavorable case.
  /// A safer and more robust alternative is to allow each op to define
  /// its own AffineMap.
  AffineMap getShapesToLoopsMap();
  /// Checks if the given operands can be dropped, and the remaining
  /// operands can still compute the bounds of the op.
  bool canOpOperandsBeDropped(ArrayRef<OpOperand *> droppedOperands);
  /// Like `getShape`, but only returns statically-known information, without
  /// generating any new IR. For each shape dimension, returns >=0 if that
  /// dimension is statically known, or ShapedType::kDynamic otherwise.
  SmallVector<int64_t> getStaticShape();
  /// Returns the statically-known loop ranges. Composes
  /// `getShapesToLoopsMap()` with the result of `getStaticShape`.
  /// Returns ShapedType::kDynamic for non-statically-known loop ranges.
  /// This is expected to be called by a valid Linalg op
  SmallVector<int64_t, 4> getStaticLoopRanges();
  /// Returns the region builder for constructing the body for linalg.generic.
  /// Returns a null function if this named op does not define a region
  /// builder.
  std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)> getRegionBuilder();
  /// Return true if all the indexing maps are projected permutations.
  /// Otherwise return false.
  bool hasOnlyProjectedPermutations();

    /// Return the flat list of all operand dimension sizes in the order they
    /// appear in the operands.
    SmallVector<OpFoldResult> createFlatListOfOperandDims(OpBuilder &, Location);

    /// Return the flat list of all operands' static dimension sizes in the
    /// order they appear in the operands. All operand dimension sizes have to
    /// be statically known.
    SmallVector<int64_t, 4> createFlatListOfOperandStaticDims();

    /// Create the loop ranges to materialize the computation over the current
    /// operands. This is done by applying `getShapesToLoopsMap` to
    /// `createFlatListOfOperandDims`.
    SmallVector<Range, 4> createLoopRanges(OpBuilder &b, Location loc);

    /// Compute the static loop sizes necessary to vectorize the computation.
    /// This is done by applying `getShapesToLoopsMap` to
    /// `createFlatListOfOperandStaticDims`.
    SmallVector<int64_t, 4> computeStaticLoopSizes();

    /// Returns the value that expresses the shape of the output in terms of
    /// shape of the input operands where possible
    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes);

    /// Return the index in the indexingMaps vector that corresponds to this `opOperand`
    int64_t getIndexingMapIndex(OpOperand *opOperand);
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::DestinationStyleOpInterface
  //===----------------------------------------------------------------===//

  operator ::mlir::DestinationStyleOpInterface () const {
    if (!*this) return nullptr;
    return ::mlir::DestinationStyleOpInterface(*this, getImpl()->implDestinationStyleOpInterface);
  }

  /// Return start and end indices of the init operands range.
  ::mlir::MutableOperandRange getDpsInitsMutable();

    ::mlir::OperandRange getDpsInits() {
      return (*this).getDpsInitsMutable();
    }

    /// Return the number of DPS inits.
    int64_t getNumDpsInits() { return (*this).getDpsInits().size(); }

    /// Return the `i`-th DPS init.
    ::mlir::OpOperand *getDpsInitOperand(int64_t i) {
      return &(*this).getDpsInitsMutable()[i];
    }

    /// Set the `i`-th DPS init.
    void setDpsInitOperand(int64_t i, Value value) {
      assert(i >= 0 && i < (*this).getNumDpsInits() && "invalid index");
      (*this)->setOperand((*this).getDpsInits().getBeginOperandIndex() + i, value);
    }

    /// Return the number of DPS inits.
    int64_t getNumDpsInputs() {
      return (*this)->getNumOperands() - (*this).getNumDpsInits();
    }

    /// Return the DPS input operands.
    ::llvm::SmallVector<::mlir::OpOperand *> getDpsInputOperands() {
      ::llvm::SmallVector<::mlir::OpOperand *> result;
      int64_t numOperands = (*this)->getNumOperands();
      ::mlir::OperandRange range = (*this).getDpsInits();
      if (range.empty()) {
        result.reserve(numOperands);
        for (int64_t i = 0; i < numOperands; ++i)
          result.push_back(&(*this)->getOpOperand(i));
        return result;
      }
      int64_t firstInitPos = range.getBeginOperandIndex();
      int64_t numInits = range.size();
      result.reserve(numOperands - numInits);
      for (int64_t i = 0; i < firstInitPos; ++i)
        result.push_back(&(*this)->getOpOperand(i));
      for (int64_t i = firstInitPos + numInits; i < numOperands; ++i)
        result.push_back(&(*this)->getOpOperand(i));
      return result;
    }

    /// Return the DPS input operands.
    ::llvm::SmallVector<::mlir::Value> getDpsInputs() {
      return ::llvm::to_vector(::llvm::map_range(
          (*this).getDpsInputOperands(), [](OpOperand *o) { return o->get(); }));
    }

    /// Return the `i`-th DPS input operand.
    ::mlir::OpOperand *getDpsInputOperand(int64_t i) {
      ::mlir::OperandRange range = (*this).getDpsInits();
      if (range.empty())
        return &(*this)->getOpOperand(i);
      int64_t firstInitPos = range.getBeginOperandIndex();
      int64_t numInits = range.size();
      assert(i >= 0 && i < (*this)->getNumOperands() - numInits
             && "invalid index");
      return &(*this)->getOpOperand(
          i < firstInitPos ? i : i + firstInitPos + numInits);
    }

    /// Return "true" if `opOperand` is an "input".
    bool isDpsInput(::mlir::OpOperand *opOperand) {
      assert(opOperand->getOwner() == (*this) && "invalid operand");
      return !(*this).isDpsInit(opOperand);
    }

    /// Return "true" if `opOperand` is an "init".
    bool isDpsInit(::mlir::OpOperand *opOperand) {
      assert(opOperand->getOwner() == (*this) && "invalid operand");
      ::mlir::OperandRange range = (*this).getDpsInits();
      if (range.empty())
        return false;
      auto operandNumber = opOperand->getOperandNumber();
      return operandNumber >= range.getBeginOperandIndex()
          && operandNumber < range.getBeginOperandIndex() + range.size();
    }

    /// Return "true" if `opOperand` is a scalar value. A sclar is defined as
    /// neither a MemRef nor a tensor value.
    bool isScalar(::mlir::OpOperand *opOperand) {
      assert(opOperand->getOwner() == (*this) && "invalid operand");
      return !::llvm::isa<BaseMemRefType, TensorType>(
          opOperand->get().getType());
    }

    /// Return the OpResult that is tied to the given OpOperand.
    ::mlir::OpResult getTiedOpResult(::mlir::OpOperand *opOperand) {
        assert(opOperand->getOwner() == (*this) && "invalid operand");
        ::mlir::OperandRange range = (*this).getDpsInits();
        assert(!range.empty() && "op has no inits");
        int64_t resultIndex =
            opOperand->getOperandNumber() - range.getBeginOperandIndex();
        assert(resultIndex >= 0 &&
               resultIndex < (*this)->getNumResults());
        return (*this)->getResult(resultIndex);
    }

    /// Return the OpOperand that is tied to the given OpResult.
    ::mlir::OpOperand *getTiedOpOperand(::mlir::OpResult opResult) {
      assert(opResult.getDefiningOp() == (*this) && "invalid opresult");
      return (*this).getDpsInitOperand(opResult.getResultNumber());
    }

    /// Return whether the op has pure buffer semantics. That is the case if the
    /// op has no tensor operands and at least one memref operand.
    bool hasPureBufferSemantics() {
      // No tensors.
      auto isTensor = [](Value v){
        return ::llvm::isa<::mlir::TensorType>(v.getType());
      };
      if (::llvm::any_of((*this)->getOperands(), isTensor))
        return false;
      // At least one memref.
      auto isMemref = [](Value v){
        return ::llvm::isa<::mlir::BaseMemRefType>(v.getType());
      };
      return llvm::any_of((*this)->getOperands(), isMemref);
    }

    /// Return whether the op has pure tensor semantics. That is the case if the
    /// op has no memref operands and at least one tensor operand.
    bool hasPureTensorSemantics() {
      // No memrefs.
      auto isMemref = [](Value v){
        return ::llvm::isa<::mlir::BaseMemRefType>(v.getType());
      };
      if (::llvm::any_of((*this)->getOperands(), isMemref))
        return false;
      // At least one tensor.
      auto isTensor = [](Value v){
        return ::llvm::isa<::mlir::TensorType>(v.getType());
      };
      return llvm::any_of((*this)->getOperands(), isTensor);    }
};
namespace detail {
  template <typename ConcreteOp>
  struct LinalgOpTrait : public ::mlir::OpInterface<LinalgOp, detail::LinalgOpInterfaceTraits>::Trait<ConcreteOp> {
    /// Return the number of parallel loops.
    unsigned getNumParallelLoops() {
      return llvm::count((*static_cast<ConcreteOp *>(this)).getIteratorTypesArray(),
                           utils::IteratorType::parallel);
    }
    /// Return the dims that are parallel loops.
    void getParallelDims(SmallVectorImpl<unsigned> & res) {
      return findPositionsOfType((*static_cast<ConcreteOp *>(this)).getIteratorTypesArray(),
                                   utils::IteratorType::parallel, res);
    }
    /// Return the number of reduction loops.
    unsigned getNumReductionLoops() {
      return llvm::count((*static_cast<ConcreteOp *>(this)).getIteratorTypesArray(),
                           utils::IteratorType::reduction);
    }
    /// Return the dims that are reduction loops.
    void getReductionDims(SmallVectorImpl<unsigned> & res) {
      return findPositionsOfType((*static_cast<ConcreteOp *>(this)).getIteratorTypesArray(),
                                   utils::IteratorType::reduction, res);
    }
    /// Return the total number of loops within the current operation.
    unsigned getNumLoops() {
      return (*static_cast<ConcreteOp *>(this)).getIteratorTypesArray().size();
    }
    /// Returns true if the current operation has only one loop and it's a
    /// reduction loop.
    bool hasSingleReductionLoop() {
      auto iters = (*static_cast<ConcreteOp *>(this)).getIteratorTypesArray();
        return iters.size() == 1 &&
               llvm::count(iters, utils::IteratorType::reduction) == 1;
    }
    /// Return true if the payload uses the value loaded from `opOperand`. This
    /// is useful to avoid loading from "write-only" memory that may be
    /// uninitialized, as well as properly cloning "read-write" operands.
    bool payloadUsesValueFromOperand(OpOperand * opOperand) {
      unsigned bbArgNumber = opOperand->getOperandNumber();
        // Init tensors have uses.
        return !getBlock()->getArgument(bbArgNumber).use_empty();
    }
    /// Return true if `opOperand` is an init tensor. This is true when it is
    /// an output tensor operand whose value is used in the payload region.
    bool isInitTensor(OpOperand * opOperand) {
      if (!(*static_cast<ConcreteOp *>(this)).isDpsInit(opOperand))
          return false;
        return payloadUsesValueFromOperand(opOperand);
    }
    /// Return the `opOperand` rank or zero for scalars or vectors not wrapped within a tensor or a memref.
    int64_t getRank(OpOperand* opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
        Type t = opOperand->get().getType();
        // A VectorType is an elemental type, do not consider its rank for the operand.
        if (isa<VectorType>(t))
          return 0;
        // Tensor and Memref container types have a rank.
        if (auto shapedType = ::llvm::dyn_cast<ShapedType>(t)) {
          // Failsafe.
          assert((isa<MemRefType>(t) || isa<RankedTensorType>(t)) &&
                 "expected a ranked tensor or memref in LinalgInterface::getRank");
          return shapedType.getRank();
        }
        return 0;
    }
    /// Return the input block arguments of the region.
    Block::BlockArgListType getRegionInputArgs() {
      return getBlock()->getArguments().take_front((*static_cast<ConcreteOp *>(this)).getNumDpsInputs());
    }
    /// Return the output block arguments of the region.
    Block::BlockArgListType getRegionOutputArgs() {
      return getBlock()->getArguments().take_back((*static_cast<ConcreteOp *>(this)).getNumDpsInits());
    }
    /// Return the `opOperand` shape or an empty vector for scalars or vectors
    /// not wrapped within a tensor or a memref.
    ArrayRef<int64_t> getShape(OpOperand* opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
        Type t = opOperand->get().getType();
        // A VectorType is an elemental type, do not consider its rank for the operand.
        if (isa<VectorType>(t))
          return {};
        if (auto shapedType = ::llvm::dyn_cast<ShapedType>(t)) {
          // Failsafe.
          assert((isa<MemRefType>(t) || isa<RankedTensorType>(t)) &&
                 "expected a ranked tensor or memref in LinalgInterface::getRank");
          return shapedType.getShape();
        }
        return {};
    }
    /// Return the block argument for an `opOperand`.
    BlockArgument getMatchingBlockArgument(OpOperand * opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
        return getBlock()->getArgument(opOperand->getOperandNumber());
    }
    /// Return the operand for a `blockArgument`.
    OpOperand *getMatchingOpOperand(BlockArgument blockArgument) {
      assert(blockArgument.getOwner() == getBlock());
        return &this->getOperation()->getOpOperand(
            blockArgument.getArgNumber());
    }
    /// Return the input or output indexing map for `opOperand`.
    AffineMap getMatchingIndexingMap(OpOperand* opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
        auto indexingMaps =
          (*static_cast<ConcreteOp *>(this)).getIndexingMaps().template getAsValueRange<AffineMapAttr>();
        return *(indexingMaps.begin() + opOperand->getOperandNumber());
    }
    /// Return the indexing map for a `result`.
    AffineMap getIndexingMapMatchingResult(OpResult result) {
      assert(result.getOwner() == this->getOperation());
        auto indexingMaps =
          (*static_cast<ConcreteOp *>(this)).getIndexingMaps().template getAsValueRange<AffineMapAttr>();
        return *(indexingMaps.begin() + (*static_cast<ConcreteOp *>(this)).getNumDpsInputs() +
                 result.getResultNumber());
    }
    /// Return the value yielded by the region corresponding to an output
    /// `opOperand`.
    OpOperand *getMatchingYieldValue(OpOperand* opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
        int64_t resultIndex =
            opOperand->getOperandNumber() - (*static_cast<ConcreteOp *>(this)).getNumDpsInputs();
        assert(resultIndex >= 0 &&
               resultIndex < this->getOperation()->getNumResults());
        Operation *yieldOp = getBlock()->getTerminator();
        return &yieldOp->getOpOperand(resultIndex);
    }
    /// Return the single block constituting the body of the operation by
    /// calling the getBody method on the concrete operation.
    Block*getBlock() {
      // Assume the concrete operation implements the
        // SingleBlockImplicitTerminator trait.
        return (*static_cast<ConcreteOp *>(this)).getBody();
    }
    /// Return iterator types in the current operation.
    /// 
    /// Default implementation assumes that the operation has an attribute
    /// `iterator_types`, but it's not always the case. Sometimes iterator types
    /// can be infered from other parameters and in such cases default
    /// getIteratorTypesArray should be overriden.
    SmallVector<utils::IteratorType> getIteratorTypesArray() {
      auto range = (*static_cast<ConcreteOp *>(this)).getIteratorTypes()
                         .template getAsValueRange<IteratorTypeAttr,
                                                   utils::IteratorType>();
        return {range.begin(), range.end()};
    }
    /// Return true if the indexing map is depending on the current op instance.
    /// This means that the indexing map is dynamically synthesized by using the
    /// op instance's concrete attributes, instead of being static for all
    /// instances of the same op kind.
    bool hasDynamicIndexingMaps() {
      return false;
    }
    /// Verify all attributes used by indexing maps are valid.
    LogicalResult verifyIndexingMapRequiredAttributes() {
      return success();
    }
    /// Return the indexing maps within the current operation.
    SmallVector<AffineMap> getIndexingMapsArray() {
      auto range = (*static_cast<ConcreteOp *>(this)).getIndexingMaps()
          .template getAsValueRange<AffineMapAttr>();
        return {range.begin(), range.end()};
    }
    /// Return true if any of the operands has a dynamic shape.
    bool hasDynamicShape() {
      return llvm::any_of(getStaticShape(), ShapedType::isDynamic);
    }
    /// Return the name registered for this op when lowering to an external
    /// library call.
    std::string getLibraryCallName() {
      return (*static_cast<ConcreteOp *>(this)).getLibraryCallName();
    }
    /// Return op operands that have a corresponding argument in the basic block.
    /// By default, the block should have an argument for each operand, but there
    /// are expection. For example, in `map` output operand isn't used in
    /// the block.
    ::llvm::SmallVector<OpOperand *> getOpOperandsMatchingBBargs() {
      ::llvm::SmallVector<OpOperand *> result;
        result.reserve((*static_cast<ConcreteOp *>(this))->getNumOperands());
        llvm::transform(
          this->getOperation()->getOpOperands(),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
    }
    /// Given a dimension of the iteration space of a Linalg operation, finds an
    /// operand in the operation that is defined on such dimension. Returns
    /// whether such operand was found or not. If found, also returns the
    /// operand value and the dimension position within the operand.
    LogicalResult mapIterationSpaceDimToOperandDim(unsigned dimPos, ::mlir::Value & operand, unsigned & operandDimPos) {
      // Retrieve the operand and its dimension position from the first
        // operand with a permutation map that is defined on such dimension.
        for (auto [i, idxMap] : llvm::enumerate((*static_cast<ConcreteOp *>(this)).getIndexingMapsArray())) {
          if (idxMap.isProjectedPermutation()) {
            if (auto mayOperandDim = idxMap.getResultPosition(
                getAffineDimExpr(dimPos, idxMap.getContext()))) {
              operand = (*static_cast<ConcreteOp *>(this))->getOperand(i);
              operandDimPos = *mayOperandDim;
              return success();
            }
          }
        }

        return failure();
    }
    /// Given a dimension of the iteration space of a Linalg operation, finds
    /// all the operands in the operation that are defined on such dimension.
    /// Returns all the operand values found and their dimension positions in
    /// `operandDimPairs`.
    void mapIterationSpaceDimToAllOperandDims(unsigned dimPos, mlir::SmallVectorImpl<std::pair<Value, unsigned>>& operandDimPairs) {
      for (auto [i, idxMap] : llvm::enumerate((*static_cast<ConcreteOp *>(this)).getIndexingMapsArray())) {
          if (idxMap.isProjectedPermutation()) {
            if (auto mayOperandDim = idxMap.getResultPosition(
                getAffineDimExpr(dimPos, idxMap.getContext()))) {
              operandDimPairs.push_back({(*static_cast<ConcreteOp *>(this))->getOperand(i), *mayOperandDim});
            }
          }
        }

        return;
    }
    /// Hook to provide a custom AffineMap used to compute all the operand
    /// subshapes given loop bounds. This is used to answer the question: "given
    /// an iteration space over the codomain, what are the subshapes of the
    /// operands involved in the computation".
    /// The default behavior is to just concatenate all the indexing maps.
    /// A custom AffineMap allows providing a map that can be used to
    /// compute subshapes even in cases where the concatenation of indexing maps
    /// (i.e. the data traversal order) is not a simple permutation of the loop
    /// traversal order. It is then possible to define ops with skewed data
    /// traversal order for which we can still easily compute hyperrectangular
    /// loop bounds and subviews.
    AffineMap getLoopsToShapesMap() {
      auto maps =  (*static_cast<ConcreteOp *>(this)).getIndexingMapsArray();
        return concatAffineMaps(maps);
    }
    /// Hook to provide a custom AffineMap used to construct the
    /// hyperrectangular loop iteration space given all the operand subshapes.
    /// This is used to answer the question:
    /// "Given a list of operand ranges, what is the subportion of the iteration
    /// space involved in the computation".
    /// This is the inverse problem of `getLoopsToShapesMap`.
    /// Return the empty AffineMap when such an AffineMap cannot be constructed.
    /// The default behavior is based on a very simple inference procedure that
    /// only works with permutation affine maps.
    /// A more advanced Tensor-Comprehension like inference is possible but has
    /// proven to be ambiguous in unfavorable case.
    /// A safer and more robust alternative is to allow each op to define
    /// its own AffineMap.
    AffineMap getShapesToLoopsMap() {
      return inversePermutation(getLoopsToShapesMap());
    }
    /// Checks if the given operands can be dropped, and the remaining
    /// operands can still compute the bounds of the op.
    bool canOpOperandsBeDropped(ArrayRef<OpOperand *> droppedOperands) {
      return detail::canOpOperandsBeDroppedImpl((*static_cast<ConcreteOp *>(this)), droppedOperands);
    }
    /// Like `getShape`, but only returns statically-known information, without
    /// generating any new IR. For each shape dimension, returns >=0 if that
    /// dimension is statically known, or ShapedType::kDynamic otherwise.
    SmallVector<int64_t> getStaticShape() {
      SmallVector<int64_t> res;
        for (OpOperand &opOperand : this->getOperation()->getOpOperands())
          llvm::append_range(res, getShape(&opOperand));
        return res;
    }
    /// Returns the statically-known loop ranges. Composes
    /// `getShapesToLoopsMap()` with the result of `getStaticShape`.
    /// Returns ShapedType::kDynamic for non-statically-known loop ranges.
    /// This is expected to be called by a valid Linalg op
    SmallVector<int64_t, 4> getStaticLoopRanges() {
      SmallVector<int64_t> viewSizes = getStaticShape();
        AffineMap invertedMap = getShapesToLoopsMap();
        assert(invertedMap && "expected a valid Linalg op to call the method");
        return invertedMap.compose(viewSizes);
    }
    static ::llvm::LogicalResult verifyRegionTrait(::mlir::Operation *op) {
      return detail::verifyStructuredOpInterface(op);
    }
  };
}// namespace detail
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class AggregatedOpInterface;
namespace detail {
struct AggregatedOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    FailureOr<SmallVector<Value>> (*decomposeOperation)(const Concept *impl, ::mlir::Operation *, OpBuilder &);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::linalg::AggregatedOpInterface;
    Model() : Concept{decomposeOperation} {}

    static inline FailureOr<SmallVector<Value>> decomposeOperation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::linalg::AggregatedOpInterface;
    FallbackModel() : Concept{decomposeOperation} {}

    static inline FailureOr<SmallVector<Value>> decomposeOperation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    FailureOr<SmallVector<Value>> decomposeOperation(::mlir::Operation *tablegen_opaque_val, OpBuilder &b) const;
  };
};
template <typename ConcreteOp>
struct AggregatedOpInterfaceTrait;

} // namespace detail
class AggregatedOpInterface : public ::mlir::OpInterface<AggregatedOpInterface, detail::AggregatedOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AggregatedOpInterface, detail::AggregatedOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::AggregatedOpInterfaceTrait<ConcreteOp> {};
  /// Method to decompose the operation into simpler operations.
  /// 
  /// On success, this method returns one `Value` per result in the
  /// original operation.
  /// The order of the returned values must match the order of the
  /// original values.
  /// In other words, the returned vector can be used directly with
  /// `RewriterBase::replaceOp(this, returnedValues)`.
  FailureOr<SmallVector<Value>> decomposeOperation(OpBuilder & b);
};
namespace detail {
  template <typename ConcreteOp>
  struct AggregatedOpInterfaceTrait : public ::mlir::OpInterface<AggregatedOpInterface, detail::AggregatedOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Method to decompose the operation into simpler operations.
    /// 
    /// On success, this method returns one `Value` per result in the
    /// original operation.
    /// The order of the returned values must match the order of the
    /// original values.
    /// In other words, the returned vector can be used directly with
    /// `RewriterBase::replaceOp(this, returnedValues)`.
    FailureOr<SmallVector<Value>> decomposeOperation(OpBuilder & b) {
      return {};
    }
  };
}// namespace detail
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
template<typename ConcreteOp>
Value detail::ContractionOpInterfaceInterfaceTraits::Model<ConcreteOp>::lhs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()->getOperand(0);
}
template<typename ConcreteOp>
Value detail::ContractionOpInterfaceInterfaceTraits::Model<ConcreteOp>::rhs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()->getOperand(1);
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::Model<ConcreteOp>::isRowMajorMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return mlir::isRowMajorMatmul((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps());
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::Model<ConcreteOp>::isColumnMajorMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return mlir::isColumnMajorMatmul((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps());
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::Model<ConcreteOp>::isRowMajorBatchMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return mlir::isRowMajorBatchMatmul((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps());
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::Model<ConcreteOp>::isVecmat(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return mlir::isVecmat((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps());
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::Model<ConcreteOp>::isBatchVecmat(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return mlir::isBatchVecmat((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps());
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::Model<ConcreteOp>::isMatvec(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return mlir::isMatvec((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps());
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::Model<ConcreteOp>::isBatchMatvec(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return mlir::isBatchMatvec((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps());
}
template<typename ConcreteOp>
Value detail::ContractionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::lhs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->lhs(tablegen_opaque_val);
}
template<typename ConcreteOp>
Value detail::ContractionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::rhs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->rhs(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isRowMajorMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isRowMajorMatmul(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isColumnMajorMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isColumnMajorMatmul(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isRowMajorBatchMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isRowMajorBatchMatmul(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isVecmat(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isVecmat(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isBatchVecmat(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isBatchVecmat(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isMatvec(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isMatvec(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isBatchMatvec(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isBatchMatvec(tablegen_opaque_val);
}
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
template<typename ConcreteOp>
Value detail::ConvolutionOpInterfaceInterfaceTraits::Model<ConcreteOp>::image(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).image();
}
template<typename ConcreteOp>
Value detail::ConvolutionOpInterfaceInterfaceTraits::Model<ConcreteOp>::filter(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).filter();
}
template<typename ConcreteOp>
Value detail::ConvolutionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::image(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->image(tablegen_opaque_val);
}
template<typename ConcreteOp>
Value detail::ConvolutionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::filter(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->filter(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
Value detail::ConvolutionOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::image(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()->getOperand(0);
}
template<typename ConcreteModel, typename ConcreteOp>
Value detail::ConvolutionOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::filter(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()->getOperand(1);
}
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
template<typename ConcreteOp>
Value detail::FillOpInterfaceInterfaceTraits::Model<ConcreteOp>::value(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).value();
}
template<typename ConcreteOp>
Value detail::FillOpInterfaceInterfaceTraits::Model<ConcreteOp>::output(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).output();
}
template<typename ConcreteOp>
Value detail::FillOpInterfaceInterfaceTraits::Model<ConcreteOp>::result(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).result();
}
template<typename ConcreteOp>
Value detail::FillOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::value(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->value(tablegen_opaque_val);
}
template<typename ConcreteOp>
Value detail::FillOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::output(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->output(tablegen_opaque_val);
}
template<typename ConcreteOp>
Value detail::FillOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::result(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->result(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
Value detail::FillOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::value(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()->getOperand(0);
}
template<typename ConcreteModel, typename ConcreteOp>
Value detail::FillOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::output(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()->getOperand(1);
}
template<typename ConcreteModel, typename ConcreteOp>
Value detail::FillOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::result(::mlir::Operation *tablegen_opaque_val) const {
if ((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()->getResults().empty())
          return nullptr;
        return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()->getResults().front();
}
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumParallelLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumParallelLoops();
}
template<typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getParallelDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getParallelDims(res);
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumReductionLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumReductionLoops();
}
template<typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getReductionDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getReductionDims(res);
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumLoops();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::hasSingleReductionLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasSingleReductionLoop();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::payloadUsesValueFromOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).payloadUsesValueFromOperand(opOperand);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::isInitTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isInitTensor(opOperand);
}
template<typename ConcreteOp>
int64_t detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getRank(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRank(opOperand);
}
template<typename ConcreteOp>
Block::BlockArgListType detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getRegionInputArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRegionInputArgs();
}
template<typename ConcreteOp>
Block::BlockArgListType detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getRegionOutputArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRegionOutputArgs();
}
template<typename ConcreteOp>
ArrayRef<int64_t> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShape(opOperand);
}
template<typename ConcreteOp>
BlockArgument detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getMatchingBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMatchingBlockArgument(opOperand);
}
template<typename ConcreteOp>
OpOperand *detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getMatchingOpOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, BlockArgument blockArgument) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMatchingOpOperand(blockArgument);
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getMatchingIndexingMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMatchingIndexingMap(opOperand);
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getIndexingMapMatchingResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpResult result) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMapMatchingResult(result);
}
template<typename ConcreteOp>
OpOperand *detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getMatchingYieldValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMatchingYieldValue(opOperand);
}
template<typename ConcreteOp>
Block*detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getBlock();
}
template<typename ConcreteOp>
SmallVector<utils::IteratorType> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getIteratorTypesArray(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIteratorTypesArray();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::hasDynamicIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasDynamicIndexingMaps();
}
template<typename ConcreteOp>
LogicalResult detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::verifyIndexingMapRequiredAttributes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).verifyIndexingMapRequiredAttributes();
}
template<typename ConcreteOp>
ArrayAttr detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps();
}
template<typename ConcreteOp>
SmallVector<AffineMap> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getIndexingMapsArray(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMapsArray();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::hasDynamicShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasDynamicShape();
}
template<typename ConcreteOp>
std::string detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getLibraryCallName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLibraryCallName();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::hasIndexSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasIndexSemantics();
}
template<typename ConcreteOp>
::llvm::SmallVector<OpOperand *> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getOpOperandsMatchingBBargs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOpOperandsMatchingBBargs();
}
template<typename ConcreteOp>
LogicalResult detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::mapIterationSpaceDimToOperandDim(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned dimPos, ::mlir::Value & operand, unsigned & operandDimPos) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).mapIterationSpaceDimToOperandDim(dimPos, operand, operandDimPos);
}
template<typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::mapIterationSpaceDimToAllOperandDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned dimPos, mlir::SmallVectorImpl<std::pair<Value, unsigned>>& operandDimPairs) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).mapIterationSpaceDimToAllOperandDims(dimPos, operandDimPairs);
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getLoopsToShapesMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLoopsToShapesMap();
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getShapesToLoopsMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShapesToLoopsMap();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::canOpOperandsBeDropped(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<OpOperand *> droppedOperands) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).canOpOperandsBeDropped(droppedOperands);
}
template<typename ConcreteOp>
SmallVector<int64_t> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getStaticShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStaticShape();
}
template<typename ConcreteOp>
SmallVector<int64_t, 4> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getStaticLoopRanges(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStaticLoopRanges();
}
template<typename ConcreteOp>
std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getRegionBuilder() {
  return ConcreteOp::getRegionBuilder();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::hasOnlyProjectedPermutations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return llvm::all_of((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMapsArray(),
                            [](AffineMap map) { return map.isProjectedPermutation(); });
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getNumParallelLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getNumParallelLoops(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getParallelDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res) {
  return static_cast<const ConcreteOp *>(impl)->getParallelDims(tablegen_opaque_val, res);
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getNumReductionLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getNumReductionLoops(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getReductionDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res) {
  return static_cast<const ConcreteOp *>(impl)->getReductionDims(tablegen_opaque_val, res);
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getNumLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getNumLoops(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::hasSingleReductionLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasSingleReductionLoop(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::payloadUsesValueFromOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return static_cast<const ConcreteOp *>(impl)->payloadUsesValueFromOperand(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::isInitTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return static_cast<const ConcreteOp *>(impl)->isInitTensor(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
int64_t detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getRank(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return static_cast<const ConcreteOp *>(impl)->getRank(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
Block::BlockArgListType detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getRegionInputArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getRegionInputArgs(tablegen_opaque_val);
}
template<typename ConcreteOp>
Block::BlockArgListType detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getRegionOutputArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getRegionOutputArgs(tablegen_opaque_val);
}
template<typename ConcreteOp>
ArrayRef<int64_t> detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return static_cast<const ConcreteOp *>(impl)->getShape(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
BlockArgument detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getMatchingBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return static_cast<const ConcreteOp *>(impl)->getMatchingBlockArgument(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
OpOperand *detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getMatchingOpOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, BlockArgument blockArgument) {
  return static_cast<const ConcreteOp *>(impl)->getMatchingOpOperand(tablegen_opaque_val, blockArgument);
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getMatchingIndexingMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return static_cast<const ConcreteOp *>(impl)->getMatchingIndexingMap(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getIndexingMapMatchingResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpResult result) {
  return static_cast<const ConcreteOp *>(impl)->getIndexingMapMatchingResult(tablegen_opaque_val, result);
}
template<typename ConcreteOp>
OpOperand *detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getMatchingYieldValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return static_cast<const ConcreteOp *>(impl)->getMatchingYieldValue(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
Block*detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getBlock(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<utils::IteratorType> detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getIteratorTypesArray(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getIteratorTypesArray(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::hasDynamicIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasDynamicIndexingMaps(tablegen_opaque_val);
}
template<typename ConcreteOp>
LogicalResult detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::verifyIndexingMapRequiredAttributes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->verifyIndexingMapRequiredAttributes(tablegen_opaque_val);
}
template<typename ConcreteOp>
ArrayAttr detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getIndexingMaps(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<AffineMap> detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getIndexingMapsArray(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getIndexingMapsArray(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::hasDynamicShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasDynamicShape(tablegen_opaque_val);
}
template<typename ConcreteOp>
std::string detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getLibraryCallName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getLibraryCallName(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::hasIndexSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasIndexSemantics(tablegen_opaque_val);
}
template<typename ConcreteOp>
::llvm::SmallVector<OpOperand *> detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getOpOperandsMatchingBBargs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getOpOperandsMatchingBBargs(tablegen_opaque_val);
}
template<typename ConcreteOp>
LogicalResult detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::mapIterationSpaceDimToOperandDim(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned dimPos, ::mlir::Value & operand, unsigned & operandDimPos) {
  return static_cast<const ConcreteOp *>(impl)->mapIterationSpaceDimToOperandDim(tablegen_opaque_val, dimPos, operand, operandDimPos);
}
template<typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::mapIterationSpaceDimToAllOperandDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, unsigned dimPos, mlir::SmallVectorImpl<std::pair<Value, unsigned>>& operandDimPairs) {
  return static_cast<const ConcreteOp *>(impl)->mapIterationSpaceDimToAllOperandDims(tablegen_opaque_val, dimPos, operandDimPairs);
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getLoopsToShapesMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getLoopsToShapesMap(tablegen_opaque_val);
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getShapesToLoopsMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getShapesToLoopsMap(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::canOpOperandsBeDropped(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<OpOperand *> droppedOperands) {
  return static_cast<const ConcreteOp *>(impl)->canOpOperandsBeDropped(tablegen_opaque_val, droppedOperands);
}
template<typename ConcreteOp>
SmallVector<int64_t> detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getStaticShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getStaticShape(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<int64_t, 4> detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getStaticLoopRanges(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getStaticLoopRanges(tablegen_opaque_val);
}
template<typename ConcreteOp>
std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)> detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getRegionBuilder() {
  return ConcreteOp::getRegionBuilder();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::hasOnlyProjectedPermutations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasOnlyProjectedPermutations(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getNumParallelLoops(::mlir::Operation *tablegen_opaque_val) const {
return llvm::count((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIteratorTypesArray(),
                           utils::IteratorType::parallel);
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getParallelDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> &res) const {
return findPositionsOfType((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIteratorTypesArray(),
                                   utils::IteratorType::parallel, res);
}
template<typename ConcreteModel, typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getNumReductionLoops(::mlir::Operation *tablegen_opaque_val) const {
return llvm::count((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIteratorTypesArray(),
                           utils::IteratorType::reduction);
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getReductionDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> &res) const {
return findPositionsOfType((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIteratorTypesArray(),
                                   utils::IteratorType::reduction, res);
}
template<typename ConcreteModel, typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getNumLoops(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIteratorTypesArray().size();
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::hasSingleReductionLoop(::mlir::Operation *tablegen_opaque_val) const {
auto iters = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIteratorTypesArray();
        return iters.size() == 1 &&
               llvm::count(iters, utils::IteratorType::reduction) == 1;
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::payloadUsesValueFromOperand(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const {
unsigned bbArgNumber = opOperand->getOperandNumber();
        // Init tensors have uses.
        return !getBlock()->getArgument(bbArgNumber).use_empty();
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isInitTensor(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const {
if (!(llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDpsInit(opOperand))
          return false;
        return payloadUsesValueFromOperand(opOperand);
}
template<typename ConcreteModel, typename ConcreteOp>
int64_t detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getRank(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const {
assert(opOperand->getOwner() == this->getOperation());
        Type t = opOperand->get().getType();
        // A VectorType is an elemental type, do not consider its rank for the operand.
        if (isa<VectorType>(t))
          return 0;
        // Tensor and Memref container types have a rank.
        if (auto shapedType = ::llvm::dyn_cast<ShapedType>(t)) {
          // Failsafe.
          assert((isa<MemRefType>(t) || isa<RankedTensorType>(t)) &&
                 "expected a ranked tensor or memref in LinalgInterface::getRank");
          return shapedType.getRank();
        }
        return 0;
}
template<typename ConcreteModel, typename ConcreteOp>
Block::BlockArgListType detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getRegionInputArgs(::mlir::Operation *tablegen_opaque_val) const {
return getBlock()->getArguments().take_front((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumDpsInputs());
}
template<typename ConcreteModel, typename ConcreteOp>
Block::BlockArgListType detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getRegionOutputArgs(::mlir::Operation *tablegen_opaque_val) const {
return getBlock()->getArguments().take_back((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumDpsInits());
}
template<typename ConcreteModel, typename ConcreteOp>
ArrayRef<int64_t> detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getShape(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const {
assert(opOperand->getOwner() == this->getOperation());
        Type t = opOperand->get().getType();
        // A VectorType is an elemental type, do not consider its rank for the operand.
        if (isa<VectorType>(t))
          return {};
        if (auto shapedType = ::llvm::dyn_cast<ShapedType>(t)) {
          // Failsafe.
          assert((isa<MemRefType>(t) || isa<RankedTensorType>(t)) &&
                 "expected a ranked tensor or memref in LinalgInterface::getRank");
          return shapedType.getShape();
        }
        return {};
}
template<typename ConcreteModel, typename ConcreteOp>
BlockArgument detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getMatchingBlockArgument(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const {
assert(opOperand->getOwner() == this->getOperation());
        return getBlock()->getArgument(opOperand->getOperandNumber());
}
template<typename ConcreteModel, typename ConcreteOp>
OpOperand *detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getMatchingOpOperand(::mlir::Operation *tablegen_opaque_val, BlockArgument blockArgument) const {
assert(blockArgument.getOwner() == getBlock());
        return &this->getOperation()->getOpOperand(
            blockArgument.getArgNumber());
}
template<typename ConcreteModel, typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getMatchingIndexingMap(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const {
assert(opOperand->getOwner() == this->getOperation());
        auto indexingMaps =
          (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps().template getAsValueRange<AffineMapAttr>();
        return *(indexingMaps.begin() + opOperand->getOperandNumber());
}
template<typename ConcreteModel, typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getIndexingMapMatchingResult(::mlir::Operation *tablegen_opaque_val, OpResult result) const {
assert(result.getOwner() == this->getOperation());
        auto indexingMaps =
          (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps().template getAsValueRange<AffineMapAttr>();
        return *(indexingMaps.begin() + (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumDpsInputs() +
                 result.getResultNumber());
}
template<typename ConcreteModel, typename ConcreteOp>
OpOperand *detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getMatchingYieldValue(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const {
assert(opOperand->getOwner() == this->getOperation());
        int64_t resultIndex =
            opOperand->getOperandNumber() - (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumDpsInputs();
        assert(resultIndex >= 0 &&
               resultIndex < this->getOperation()->getNumResults());
        Operation *yieldOp = getBlock()->getTerminator();
        return &yieldOp->getOpOperand(resultIndex);
}
template<typename ConcreteModel, typename ConcreteOp>
Block*detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getBlock(::mlir::Operation *tablegen_opaque_val) const {
// Assume the concrete operation implements the
        // SingleBlockImplicitTerminator trait.
        return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getBody();
}
template<typename ConcreteModel, typename ConcreteOp>
SmallVector<utils::IteratorType> detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getIteratorTypesArray(::mlir::Operation *tablegen_opaque_val) const {
auto range = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIteratorTypes()
                         .template getAsValueRange<IteratorTypeAttr,
                                                   utils::IteratorType>();
        return {range.begin(), range.end()};
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::hasDynamicIndexingMaps(::mlir::Operation *tablegen_opaque_val) const {
return false;
}
template<typename ConcreteModel, typename ConcreteOp>
LogicalResult detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::verifyIndexingMapRequiredAttributes(::mlir::Operation *tablegen_opaque_val) const {
return success();
}
template<typename ConcreteModel, typename ConcreteOp>
SmallVector<AffineMap> detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getIndexingMapsArray(::mlir::Operation *tablegen_opaque_val) const {
auto range = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps()
          .template getAsValueRange<AffineMapAttr>();
        return {range.begin(), range.end()};
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::hasDynamicShape(::mlir::Operation *tablegen_opaque_val) const {
return llvm::any_of(getStaticShape(), ShapedType::isDynamic);
}
template<typename ConcreteModel, typename ConcreteOp>
std::string detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getLibraryCallName(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLibraryCallName();
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::SmallVector<OpOperand *> detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getOpOperandsMatchingBBargs(::mlir::Operation *tablegen_opaque_val) const {
::llvm::SmallVector<OpOperand *> result;
        result.reserve((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getNumOperands());
        llvm::transform(
          this->getOperation()->getOpOperands(),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
}
template<typename ConcreteModel, typename ConcreteOp>
LogicalResult detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::mapIterationSpaceDimToOperandDim(::mlir::Operation *tablegen_opaque_val, unsigned dimPos, ::mlir::Value &operand, unsigned &operandDimPos) const {
// Retrieve the operand and its dimension position from the first
        // operand with a permutation map that is defined on such dimension.
        for (auto [i, idxMap] : llvm::enumerate((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMapsArray())) {
          if (idxMap.isProjectedPermutation()) {
            if (auto mayOperandDim = idxMap.getResultPosition(
                getAffineDimExpr(dimPos, idxMap.getContext()))) {
              operand = (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getOperand(i);
              operandDimPos = *mayOperandDim;
              return success();
            }
          }
        }

        return failure();
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::mapIterationSpaceDimToAllOperandDims(::mlir::Operation *tablegen_opaque_val, unsigned dimPos, mlir::SmallVectorImpl<std::pair<Value, unsigned>>&operandDimPairs) const {
for (auto [i, idxMap] : llvm::enumerate((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMapsArray())) {
          if (idxMap.isProjectedPermutation()) {
            if (auto mayOperandDim = idxMap.getResultPosition(
                getAffineDimExpr(dimPos, idxMap.getContext()))) {
              operandDimPairs.push_back({(llvm::cast<ConcreteOp>(tablegen_opaque_val))->getOperand(i), *mayOperandDim});
            }
          }
        }

        return;
}
template<typename ConcreteModel, typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getLoopsToShapesMap(::mlir::Operation *tablegen_opaque_val) const {
auto maps =  (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMapsArray();
        return concatAffineMaps(maps);
}
template<typename ConcreteModel, typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getShapesToLoopsMap(::mlir::Operation *tablegen_opaque_val) const {
return inversePermutation(getLoopsToShapesMap());
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::canOpOperandsBeDropped(::mlir::Operation *tablegen_opaque_val, ArrayRef<OpOperand *> droppedOperands) const {
return detail::canOpOperandsBeDroppedImpl((llvm::cast<ConcreteOp>(tablegen_opaque_val)), droppedOperands);
}
template<typename ConcreteModel, typename ConcreteOp>
SmallVector<int64_t> detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getStaticShape(::mlir::Operation *tablegen_opaque_val) const {
SmallVector<int64_t> res;
        for (OpOperand &opOperand : this->getOperation()->getOpOperands())
          llvm::append_range(res, getShape(&opOperand));
        return res;
}
template<typename ConcreteModel, typename ConcreteOp>
SmallVector<int64_t, 4> detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getStaticLoopRanges(::mlir::Operation *tablegen_opaque_val) const {
SmallVector<int64_t> viewSizes = getStaticShape();
        AffineMap invertedMap = getShapesToLoopsMap();
        assert(invertedMap && "expected a valid Linalg op to call the method");
        return invertedMap.compose(viewSizes);
}
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
template<typename ConcreteOp>
FailureOr<SmallVector<Value>> detail::AggregatedOpInterfaceInterfaceTraits::Model<ConcreteOp>::decomposeOperation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).decomposeOperation(b);
}
template<typename ConcreteOp>
FailureOr<SmallVector<Value>> detail::AggregatedOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::decomposeOperation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b) {
  return static_cast<const ConcreteOp *>(impl)->decomposeOperation(tablegen_opaque_val, b);
}
template<typename ConcreteModel, typename ConcreteOp>
FailureOr<SmallVector<Value>> detail::AggregatedOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::decomposeOperation(::mlir::Operation *tablegen_opaque_val, OpBuilder &b) const {
return {};
}
} // namespace linalg
} // namespace mlir
