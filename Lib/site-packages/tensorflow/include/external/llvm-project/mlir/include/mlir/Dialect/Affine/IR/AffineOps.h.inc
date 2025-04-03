/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: AffineOps.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace affine {
class AffineApplyOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineDelinearizeIndexOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineForOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineIfOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineLoadOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineMaxOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineMinOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineParallelOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffinePrefetchOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineStoreOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineVectorLoadOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineVectorStoreOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineYieldOp;
} // namespace affine
} // namespace mlir
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineApplyOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineApplyOpGenericAdaptorBase {
public:
  struct Properties {
    using mapTy = ::mlir::AffineMapAttr;
    mapTy map;

    auto getMap() {
      auto &propStorage = this->map;
      return ::llvm::cast<::mlir::AffineMapAttr>(propStorage);
    }
    void setMap(const ::mlir::AffineMapAttr &propValue) {
      this->map = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.map == this->map &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  AffineApplyOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("affine.apply", odsAttrs.getContext());
  }

  AffineApplyOpGenericAdaptorBase(AffineApplyOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::AffineMapAttr getMapAttr() {
    auto attr = ::llvm::cast<::mlir::AffineMapAttr>(getProperties().map);
    return attr;
  }

  ::mlir::AffineMap getMap();
};
} // namespace detail
template <typename RangeT>
class AffineApplyOpGenericAdaptor : public detail::AffineApplyOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineApplyOpGenericAdaptorBase;
public:
  AffineApplyOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AffineApplyOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AffineApplyOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  AffineApplyOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : AffineApplyOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = AffineApplyOp, typename = std::enable_if_t<std::is_same_v<LateInst, AffineApplyOp>>>
  AffineApplyOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getMapOperands() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AffineApplyOpAdaptor : public AffineApplyOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineApplyOpGenericAdaptor::AffineApplyOpGenericAdaptor;
  AffineApplyOpAdaptor(AffineApplyOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AffineApplyOp : public ::mlir::Op<AffineApplyOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineApplyOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineApplyOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("map")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getMapAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getMapAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.apply");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::operand_range getMapOperands() {
    return getODSOperands(0);
  }

  ::mlir::MutableOperandRange getMapOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::AffineMapAttr getMapAttr() {
    return ::llvm::cast<::mlir::AffineMapAttr>(getProperties().map);
  }

  ::mlir::AffineMap getMap();
  void setMapAttr(::mlir::AffineMapAttr attr) {
    getProperties().map = attr;
  }

  void setMap(::mlir::AffineMap attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<AffineExpr>  exprList, ValueRange mapOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMapAttr map, ::mlir::ValueRange mapOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::AffineMapAttr map, ::mlir::ValueRange mapOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMapAttr map, ::mlir::ValueRange mapOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMap map, ::mlir::ValueRange mapOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::AffineMap map, ::mlir::ValueRange mapOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMap map, ::mlir::ValueRange mapOperands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// Returns the affine map to be applied by this operation.
  AffineMap getAffineMap() { return getMap(); }

  /// Returns the affine value map computed from this operation.
  AffineValueMap getAffineValueMap();

  /// Returns true if the result of this operation can be used as dimension id
  /// in the region of the closest surrounding op with trait AffineScope.
  bool isValidDim();

  /// Returns true if the result of this operation can be used as dimension id
  /// within 'region', i.e., for all its uses with `region`.
  bool isValidDim(Region *region);

  /// Returns true if the result of this operation is a symbol in the region
  /// of the closest surrounding op that has the trait AffineScope.
  bool isValidSymbol();

  /// Returns true if the result of this operation is a symbol for all its
  /// uses in `region`.
  bool isValidSymbol(Region *region);

  /// Returns all dimension operands.
  ValueRange getDimOperands() {
    return OperandRange{getOperands().begin(),
                        getOperands().begin() + getMap().getNumDims()};
  }

  /// Returns all symbol operands.
  ValueRange getSymbolOperands() {
    return OperandRange{getOperands().begin() + getMap().getNumDims(),
                        getOperands().end()};
  }
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineApplyOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineDelinearizeIndexOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineDelinearizeIndexOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  AffineDelinearizeIndexOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("affine.delinearize_index", odsAttrs.getContext());
  }

  AffineDelinearizeIndexOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};
} // namespace detail
template <typename RangeT>
class AffineDelinearizeIndexOpGenericAdaptor : public detail::AffineDelinearizeIndexOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineDelinearizeIndexOpGenericAdaptorBase;
public:
  AffineDelinearizeIndexOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AffineDelinearizeIndexOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AffineDelinearizeIndexOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  template <typename LateInst = AffineDelinearizeIndexOp, typename = std::enable_if_t<std::is_same_v<LateInst, AffineDelinearizeIndexOp>>>
  AffineDelinearizeIndexOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLinearIndex() {
    return (*getODSOperands(0).begin());
  }

  RangeT getBasis() {
    return getODSOperands(1);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AffineDelinearizeIndexOpAdaptor : public AffineDelinearizeIndexOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineDelinearizeIndexOpGenericAdaptor::AffineDelinearizeIndexOpGenericAdaptor;
  AffineDelinearizeIndexOpAdaptor(AffineDelinearizeIndexOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AffineDelinearizeIndexOp : public ::mlir::Op<AffineDelinearizeIndexOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineDelinearizeIndexOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineDelinearizeIndexOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.delinearize_index");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IndexType> getLinearIndex() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(0).begin());
  }

  ::mlir::Operation::operand_range getBasis() {
    return getODSOperands(1);
  }

  ::mlir::OpOperand &getLinearIndexMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getBasisMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::result_range getMultiIndex() {
    return getODSResults(0);
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value linear_index, ArrayRef<OpFoldResult> basis);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange multi_index, ::mlir::Value linear_index, ::mlir::ValueRange basis);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value linear_index, ::mlir::ValueRange basis);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineDelinearizeIndexOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineForOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineForOpGenericAdaptorBase {
public:
  struct Properties {
    using lowerBoundMapTy = ::mlir::AffineMapAttr;
    lowerBoundMapTy lowerBoundMap;

    auto getLowerBoundMap() {
      auto &propStorage = this->lowerBoundMap;
      return ::llvm::cast<::mlir::AffineMapAttr>(propStorage);
    }
    void setLowerBoundMap(const ::mlir::AffineMapAttr &propValue) {
      this->lowerBoundMap = propValue;
    }
    using stepTy = ::mlir::IntegerAttr;
    stepTy step;

    auto getStep() {
      auto &propStorage = this->step;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setStep(const ::mlir::IntegerAttr &propValue) {
      this->step = propValue;
    }
    using upperBoundMapTy = ::mlir::AffineMapAttr;
    upperBoundMapTy upperBoundMap;

    auto getUpperBoundMap() {
      auto &propStorage = this->upperBoundMap;
      return ::llvm::cast<::mlir::AffineMapAttr>(propStorage);
    }
    void setUpperBoundMap(const ::mlir::AffineMapAttr &propValue) {
      this->upperBoundMap = propValue;
    }
    using operandSegmentSizesTy = std::array<int32_t, 3>;
    operandSegmentSizesTy operandSegmentSizes;
    ::llvm::ArrayRef<int32_t> getOperandSegmentSizes() const {
      auto &propStorage = this->operandSegmentSizes;
      return propStorage;
    }
    void setOperandSegmentSizes(::llvm::ArrayRef<int32_t> propValue) {
      auto &propStorage = this->operandSegmentSizes;
      ::llvm::copy(propValue, propStorage.begin());
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.lowerBoundMap == this->lowerBoundMap &&
        rhs.step == this->step &&
        rhs.upperBoundMap == this->upperBoundMap &&
        rhs.operandSegmentSizes == this->operandSegmentSizes &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  AffineForOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("affine.for", odsAttrs.getContext());
  }

  AffineForOpGenericAdaptorBase(AffineForOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::AffineMapAttr getLowerBoundMapAttr() {
    auto attr = ::llvm::cast<::mlir::AffineMapAttr>(getProperties().lowerBoundMap);
    return attr;
  }

  ::mlir::AffineMap getLowerBoundMap();
  ::mlir::AffineMapAttr getUpperBoundMapAttr() {
    auto attr = ::llvm::cast<::mlir::AffineMapAttr>(getProperties().upperBoundMap);
    return attr;
  }

  ::mlir::AffineMap getUpperBoundMap();
  ::mlir::IntegerAttr getStepAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().step);
    return attr;
  }

  ::llvm::APInt getStep();
  ::mlir::Region &getRegion() {
    return *odsRegions[0];
  }

  ::mlir::RegionRange getRegions() {
    return odsRegions;
  }

};
} // namespace detail
template <typename RangeT>
class AffineForOpGenericAdaptor : public detail::AffineForOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineForOpGenericAdaptorBase;
public:
  AffineForOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AffineForOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AffineForOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  AffineForOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs) : AffineForOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = AffineForOp, typename = std::enable_if_t<std::is_same_v<LateInst, AffineForOp>>>
  AffineForOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getLowerBoundOperands() {
    return getODSOperands(0);
  }

  RangeT getUpperBoundOperands() {
    return getODSOperands(1);
  }

  RangeT getInits() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AffineForOpAdaptor : public AffineForOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineForOpGenericAdaptor::AffineForOpGenericAdaptor;
  AffineForOpAdaptor(AffineForOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AffineForOp : public ::mlir::Op<AffineForOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlock, ::mlir::OpTrait::SingleBlockImplicitTerminator<AffineYieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::HasRecursiveMemoryEffects, ::mlir::LoopLikeOpInterface::Trait, ::mlir::RegionBranchOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineForOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineForOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("lowerBoundMap"), ::llvm::StringRef("step"), ::llvm::StringRef("upperBoundMap"), ::llvm::StringRef("operandSegmentSizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getLowerBoundMapAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getLowerBoundMapAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getStepAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getStepAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getUpperBoundMapAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getUpperBoundMapAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
   return (*this)->getName().getAttributeNames().back();
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
   return name.getAttributeNames().back();
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.for");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::operand_range getLowerBoundOperands() {
    return getODSOperands(0);
  }

  ::mlir::Operation::operand_range getUpperBoundOperands() {
    return getODSOperands(1);
  }

  ::mlir::Operation::operand_range getInits() {
    return getODSOperands(2);
  }

  ::mlir::MutableOperandRange getLowerBoundOperandsMutable();
  ::mlir::MutableOperandRange getUpperBoundOperandsMutable();
  ::mlir::MutableOperandRange getInitsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::result_range getResults() {
    return getODSResults(0);
  }

  ::mlir::Region &getRegion() {
    return (*this)->getRegion(0);
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::AffineMapAttr getLowerBoundMapAttr() {
    return ::llvm::cast<::mlir::AffineMapAttr>(getProperties().lowerBoundMap);
  }

  ::mlir::AffineMap getLowerBoundMap();
  ::mlir::AffineMapAttr getUpperBoundMapAttr() {
    return ::llvm::cast<::mlir::AffineMapAttr>(getProperties().upperBoundMap);
  }

  ::mlir::AffineMap getUpperBoundMap();
  ::mlir::IntegerAttr getStepAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().step);
  }

  ::llvm::APInt getStep();
  void setLowerBoundMapAttr(::mlir::AffineMapAttr attr) {
    getProperties().lowerBoundMap = attr;
  }

  void setLowerBoundMap(::mlir::AffineMap attrValue);
  void setUpperBoundMapAttr(::mlir::AffineMapAttr attr) {
    getProperties().upperBoundMap = attr;
  }

  void setUpperBoundMap(::mlir::AffineMap attrValue);
  void setStepAttr(::mlir::IntegerAttr attr) {
    getProperties().step = attr;
  }

  void setStep(::llvm::APInt attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, int64_t lowerBound, int64_t upperBound, int64_t step = 1, ValueRange iterArgs = std::nullopt, function_ref<void(OpBuilder &, Location, Value, ValueRange)> bodyBuilder = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange lbOperands, AffineMap lbMap, ValueRange ubOperands, AffineMap ubMap, int64_t step = 1, ValueRange iterArgs = std::nullopt, function_ref<void(OpBuilder &, Location, Value, ValueRange)> bodyBuilder = nullptr);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verifyRegions();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::llvm::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  ::llvm::SmallVector<::mlir::Region *> getLoopRegions();
  ::std::optional<::llvm::SmallVector<::mlir::Value>> getLoopInductionVars();
  ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> getLoopLowerBounds();
  ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> getLoopSteps();
  ::std::optional<::llvm::SmallVector<::mlir::OpFoldResult>> getLoopUpperBounds();
  std::optional<::llvm::MutableArrayRef<::mlir::OpOperand>> getYieldedValuesMutable();
  ::mlir::FailureOr<::mlir::LoopLikeOpInterface> replaceWithAdditionalYields(::mlir::RewriterBase &rewriter, ::mlir::ValueRange newInitOperands, bool replaceInitOperandUsesInLoop, const ::mlir::NewYieldValuesFn &newYieldValuesFn);
  ::mlir::OperandRange getEntrySuccessorOperands(::mlir::RegionBranchPoint point);
  void getSuccessorRegions(::mlir::RegionBranchPoint point, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// Defining the function type we use for building the body of affine.for.
  using BodyBuilderFn =
      function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

  BlockArgument getInductionVar() { return getBody()->getArgument(0); }
  Block::BlockArgListType getRegionIterArgs() {
    return getBody()->getArguments().drop_front();
  }

  /// Returns operands for the lower and upper bound maps with the operands
  /// for the lower bound map in front of those for the upper bound map.
  operand_range getControlOperands();

  /// Returns information about the lower bound as a single object.
  AffineBound getLowerBound();

  /// Returns information about the upper bound as a single object.
  AffineBound getUpperBound();

  /// Returns loop step.
  int64_t getStepAsInt() { return getStep().getSExtValue(); }

  /// Set lower bound. The new bound must have the same number of operands as
  /// the current bound map. Otherwise, 'replaceForLowerBound' should be used.
  void setLowerBound(ValueRange operands, AffineMap map);
  /// Set upper bound. The new bound must not have more operands than the
  /// current bound map. Otherwise, 'replaceForUpperBound' should be used.
  void setUpperBound(ValueRange operands, AffineMap map);

  /// Set loop step.
  void setStep(int64_t step) {
    assert(step > 0 && "step has to be a positive integer constant");
    setStep(APInt(/*numBits=*/64, step, /*isSigned=*/true));
  }

  /// Returns number of region arguments for loop-carried values.
  unsigned getNumRegionIterArgs() {
    return getBody()->getNumArguments() - 1;
  }

  /// Number of operands controlling the loop: lb and ub.
  unsigned getNumControlOperands() {
    return getOperation()->getNumOperands() - getNumIterOperands();
  }

  /// Get the number of loop-carried values.
  unsigned getNumIterOperands();

  /// Returns true if the lower bound is constant.
  bool hasConstantLowerBound();
  /// Returns true if the upper bound is constant.
  bool hasConstantUpperBound();
  /// Returns true if both bounds are constant.
  bool hasConstantBounds() {
    return hasConstantLowerBound() && hasConstantUpperBound();
  }
  /// Returns the value of the constant lower bound.
  /// Fails assertion if the bound is non-constant.
  int64_t getConstantLowerBound();
  /// Returns the value of the constant upper bound. The upper bound is
  /// exclusive. Fails assertion if the bound is non-constant.
  int64_t getConstantUpperBound();
  /// Sets the lower bound to the given constant value.
  void setConstantLowerBound(int64_t value);
  /// Sets the upper bound to the given constant value.
  void setConstantUpperBound(int64_t value);

  /// Returns true if both the lower and upper bound have the same operand
  /// lists (same operands in the same order).
  bool matchingBoundOperandList();

  /// Interface method for ConditionallySpeculatable.
  Speculation::Speculatability getSpeculatability();
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineForOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineIfOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineIfOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  AffineIfOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("affine.if", odsAttrs.getContext());
  }

  AffineIfOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::Region &getThenRegion() {
    return *odsRegions[0];
  }

  ::mlir::Region &getElseRegion() {
    return *odsRegions[1];
  }

  ::mlir::RegionRange getRegions() {
    return odsRegions;
  }

};
} // namespace detail
template <typename RangeT>
class AffineIfOpGenericAdaptor : public detail::AffineIfOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineIfOpGenericAdaptorBase;
public:
  AffineIfOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AffineIfOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AffineIfOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  template <typename LateInst = AffineIfOp, typename = std::enable_if_t<std::is_same_v<LateInst, AffineIfOp>>>
  AffineIfOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AffineIfOpAdaptor : public AffineIfOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineIfOpGenericAdaptor::AffineIfOpGenericAdaptor;
  AffineIfOpAdaptor(AffineIfOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AffineIfOp : public ::mlir::Op<AffineIfOp, ::mlir::OpTrait::NRegions<2>::Impl, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlock, ::mlir::OpTrait::SingleBlockImplicitTerminator<AffineYieldOp>::Impl, ::mlir::OpTrait::NoRegionArguments, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::RecursivelySpeculatableImplTrait, ::mlir::OpTrait::HasRecursiveMemoryEffects, ::mlir::RegionBranchOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineIfOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineIfOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.if");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::result_range getResults() {
    return getODSResults(0);
  }

  ::mlir::Region &getThenRegion() {
    return (*this)->getRegion(0);
  }

  ::mlir::Region &getElseRegion() {
    return (*this)->getRegion(1);
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, IntegerSet set, ValueRange args, bool withElseRegion);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, IntegerSet set, ValueRange args, bool withElseRegion);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::llvm::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getSuccessorRegions(::mlir::RegionBranchPoint point, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
public:
  static StringRef getConditionAttrStrName() { return "condition"; }

  IntegerSet getIntegerSet();
  void setIntegerSet(IntegerSet newSet);

  /// Sets the integer set with its operands.
  void setConditional(IntegerSet set, ValueRange operands);

  /// Returns true if an else block exists.
  bool hasElse() { return !getElseRegion().empty(); }

  Block *getThenBlock() {
    assert(!getThenRegion().empty() && "Unexpected empty 'then' region.");
    return &getThenRegion().front();
  }

  Block *getElseBlock() {
    assert(hasElse() && "Empty 'else' region.");
    return &getElseRegion().front();
  }

  OpBuilder getThenBodyBuilder() {
    assert(!getThenRegion().empty() && "Unexpected empty 'then' region.");
    Block &body = getThenRegion().front();
    return OpBuilder(&body, std::prev(body.end()));
  }
  OpBuilder getElseBodyBuilder() {
    assert(hasElse() && "No 'else' block");
    Block &body = getElseRegion().front();
    return OpBuilder(&body, std::prev(body.end()));
  }
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineIfOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineLoadOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineLoadOpGenericAdaptorBase {
public:
  struct Properties {
    using mapTy = ::mlir::AffineMapAttr;
    mapTy map;

    auto getMap() {
      auto &propStorage = this->map;
      return ::llvm::cast<::mlir::AffineMapAttr>(propStorage);
    }
    void setMap(const ::mlir::AffineMapAttr &propValue) {
      this->map = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.map == this->map &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  AffineLoadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("affine.load", odsAttrs.getContext());
  }

  AffineLoadOpGenericAdaptorBase(AffineLoadOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::AffineMapAttr getMapAttr() {
    auto attr = ::llvm::cast<::mlir::AffineMapAttr>(getProperties().map);
    return attr;
  }

  ::mlir::AffineMap getMap();
};
} // namespace detail
template <typename RangeT>
class AffineLoadOpGenericAdaptor : public detail::AffineLoadOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineLoadOpGenericAdaptorBase;
public:
  AffineLoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AffineLoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AffineLoadOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  AffineLoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : AffineLoadOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = AffineLoadOp, typename = std::enable_if_t<std::is_same_v<LateInst, AffineLoadOp>>>
  AffineLoadOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getMemref() {
    return (*getODSOperands(0).begin());
  }

  RangeT getIndices() {
    return getODSOperands(1);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AffineLoadOpAdaptor : public AffineLoadOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineLoadOpGenericAdaptor::AffineLoadOpGenericAdaptor;
  AffineLoadOpAdaptor(AffineLoadOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AffineLoadOp : public ::mlir::Op<AffineLoadOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::affine::AffineReadOpInterface::Trait, ::mlir::affine::AffineMapAccessInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineLoadOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineLoadOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("map")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getMapAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getMapAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.load");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::MemRefType> getMemref() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
  }

  ::mlir::Operation::operand_range getIndices() {
    return getODSOperands(1);
  }

  ::mlir::OpOperand &getMemrefMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::AffineMapAttr getMapAttr() {
    return ::llvm::cast<::mlir::AffineMapAttr>(getProperties().map);
  }

  ::mlir::AffineMap getMap();
  void setMapAttr(::mlir::AffineMapAttr attr) {
    getProperties().map = attr;
  }

  void setMap(::mlir::AffineMap attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, AffineMap map, ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value memref, ValueRange indices = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value memref, AffineMap map, ValueRange mapOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::AffineMapAttr map);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::AffineMapAttr map);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::AffineMap map);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::AffineMap map);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// Returns the operand index of the memref.
  unsigned getMemRefOperandIndex() { return 0; }

  void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

  /// Returns the affine map used to index the memref for this operation.
  AffineMapAttr getAffineMapAttr() {
    return getProperties().map;
  }

  static StringRef getMapAttrStrName() { return "map"; }
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineLoadOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineMaxOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineMaxOpGenericAdaptorBase {
public:
  struct Properties {
    using mapTy = ::mlir::AffineMapAttr;
    mapTy map;

    auto getMap() {
      auto &propStorage = this->map;
      return ::llvm::cast<::mlir::AffineMapAttr>(propStorage);
    }
    void setMap(const ::mlir::AffineMapAttr &propValue) {
      this->map = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.map == this->map &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  AffineMaxOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("affine.max", odsAttrs.getContext());
  }

  AffineMaxOpGenericAdaptorBase(AffineMaxOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::AffineMapAttr getMapAttr() {
    auto attr = ::llvm::cast<::mlir::AffineMapAttr>(getProperties().map);
    return attr;
  }

  ::mlir::AffineMap getMap();
};
} // namespace detail
template <typename RangeT>
class AffineMaxOpGenericAdaptor : public detail::AffineMaxOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineMaxOpGenericAdaptorBase;
public:
  AffineMaxOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AffineMaxOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AffineMaxOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  AffineMaxOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : AffineMaxOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = AffineMaxOp, typename = std::enable_if_t<std::is_same_v<LateInst, AffineMaxOp>>>
  AffineMaxOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return getODSOperands(0);
  }

private:
  RangeT odsOperands;
};
class AffineMaxOpAdaptor : public AffineMaxOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineMaxOpGenericAdaptor::AffineMaxOpGenericAdaptor;
  AffineMaxOpAdaptor(AffineMaxOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AffineMaxOp : public ::mlir::Op<AffineMaxOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineMaxOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineMaxOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("map")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getMapAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getMapAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.max");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::operand_range getOperands() {
    return getODSOperands(0);
  }

  ::mlir::MutableOperandRange getOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::AffineMapAttr getMapAttr() {
    return ::llvm::cast<::mlir::AffineMapAttr>(getProperties().map);
  }

  ::mlir::AffineMap getMap();
  void setMapAttr(::mlir::AffineMapAttr attr) {
    getProperties().map = attr;
  }

  void setMap(::mlir::AffineMap attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMap map, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::AffineMap map, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMap map, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  static StringRef getMapAttrStrName() { return "map"; }
  AffineMap getAffineMap() { return getMap(); }
  ValueRange getMapOperands() { return getOperands(); }
  ValueRange getDimOperands() {
    return OperandRange{getOperands().begin(),
                        getOperands().begin() + getMap().getNumDims()};
  }
  ValueRange getSymbolOperands() {
    return OperandRange{getOperands().begin() + getMap().getNumDims(),
                        getOperands().end()};
  }
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineMaxOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineMinOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineMinOpGenericAdaptorBase {
public:
  struct Properties {
    using mapTy = ::mlir::AffineMapAttr;
    mapTy map;

    auto getMap() {
      auto &propStorage = this->map;
      return ::llvm::cast<::mlir::AffineMapAttr>(propStorage);
    }
    void setMap(const ::mlir::AffineMapAttr &propValue) {
      this->map = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.map == this->map &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  AffineMinOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("affine.min", odsAttrs.getContext());
  }

  AffineMinOpGenericAdaptorBase(AffineMinOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::AffineMapAttr getMapAttr() {
    auto attr = ::llvm::cast<::mlir::AffineMapAttr>(getProperties().map);
    return attr;
  }

  ::mlir::AffineMap getMap();
};
} // namespace detail
template <typename RangeT>
class AffineMinOpGenericAdaptor : public detail::AffineMinOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineMinOpGenericAdaptorBase;
public:
  AffineMinOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AffineMinOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AffineMinOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  AffineMinOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : AffineMinOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = AffineMinOp, typename = std::enable_if_t<std::is_same_v<LateInst, AffineMinOp>>>
  AffineMinOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return getODSOperands(0);
  }

private:
  RangeT odsOperands;
};
class AffineMinOpAdaptor : public AffineMinOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineMinOpGenericAdaptor::AffineMinOpGenericAdaptor;
  AffineMinOpAdaptor(AffineMinOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AffineMinOp : public ::mlir::Op<AffineMinOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineMinOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineMinOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("map")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getMapAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getMapAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.min");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::operand_range getOperands() {
    return getODSOperands(0);
  }

  ::mlir::MutableOperandRange getOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::AffineMapAttr getMapAttr() {
    return ::llvm::cast<::mlir::AffineMapAttr>(getProperties().map);
  }

  ::mlir::AffineMap getMap();
  void setMapAttr(::mlir::AffineMapAttr attr) {
    getProperties().map = attr;
  }

  void setMap(::mlir::AffineMap attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMap map, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::AffineMap map, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMap map, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  static StringRef getMapAttrStrName() { return "map"; }
  AffineMap getAffineMap() { return getMap(); }
  ValueRange getMapOperands() { return getOperands(); }
  ValueRange getDimOperands() {
    return OperandRange{getOperands().begin(),
                        getOperands().begin() + getMap().getNumDims()};
  }
  ValueRange getSymbolOperands() {
    return OperandRange{getOperands().begin() + getMap().getNumDims(),
                        getOperands().end()};
  }
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineMinOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineParallelOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineParallelOpGenericAdaptorBase {
public:
  struct Properties {
    using lowerBoundsGroupsTy = ::mlir::DenseIntElementsAttr;
    lowerBoundsGroupsTy lowerBoundsGroups;

    auto getLowerBoundsGroups() {
      auto &propStorage = this->lowerBoundsGroups;
      return ::llvm::cast<::mlir::DenseIntElementsAttr>(propStorage);
    }
    void setLowerBoundsGroups(const ::mlir::DenseIntElementsAttr &propValue) {
      this->lowerBoundsGroups = propValue;
    }
    using lowerBoundsMapTy = ::mlir::AffineMapAttr;
    lowerBoundsMapTy lowerBoundsMap;

    auto getLowerBoundsMap() {
      auto &propStorage = this->lowerBoundsMap;
      return ::llvm::cast<::mlir::AffineMapAttr>(propStorage);
    }
    void setLowerBoundsMap(const ::mlir::AffineMapAttr &propValue) {
      this->lowerBoundsMap = propValue;
    }
    using reductionsTy = ::mlir::ArrayAttr;
    reductionsTy reductions;

    auto getReductions() {
      auto &propStorage = this->reductions;
      return ::llvm::cast<::mlir::ArrayAttr>(propStorage);
    }
    void setReductions(const ::mlir::ArrayAttr &propValue) {
      this->reductions = propValue;
    }
    using stepsTy = ::mlir::ArrayAttr;
    stepsTy steps;

    auto getSteps() {
      auto &propStorage = this->steps;
      return ::llvm::cast<::mlir::ArrayAttr>(propStorage);
    }
    void setSteps(const ::mlir::ArrayAttr &propValue) {
      this->steps = propValue;
    }
    using upperBoundsGroupsTy = ::mlir::DenseIntElementsAttr;
    upperBoundsGroupsTy upperBoundsGroups;

    auto getUpperBoundsGroups() {
      auto &propStorage = this->upperBoundsGroups;
      return ::llvm::cast<::mlir::DenseIntElementsAttr>(propStorage);
    }
    void setUpperBoundsGroups(const ::mlir::DenseIntElementsAttr &propValue) {
      this->upperBoundsGroups = propValue;
    }
    using upperBoundsMapTy = ::mlir::AffineMapAttr;
    upperBoundsMapTy upperBoundsMap;

    auto getUpperBoundsMap() {
      auto &propStorage = this->upperBoundsMap;
      return ::llvm::cast<::mlir::AffineMapAttr>(propStorage);
    }
    void setUpperBoundsMap(const ::mlir::AffineMapAttr &propValue) {
      this->upperBoundsMap = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.lowerBoundsGroups == this->lowerBoundsGroups &&
        rhs.lowerBoundsMap == this->lowerBoundsMap &&
        rhs.reductions == this->reductions &&
        rhs.steps == this->steps &&
        rhs.upperBoundsGroups == this->upperBoundsGroups &&
        rhs.upperBoundsMap == this->upperBoundsMap &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  AffineParallelOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("affine.parallel", odsAttrs.getContext());
  }

  AffineParallelOpGenericAdaptorBase(AffineParallelOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getReductionsAttr() {
    auto attr = ::llvm::cast<::mlir::ArrayAttr>(getProperties().reductions);
    return attr;
  }

  ::mlir::ArrayAttr getReductions();
  ::mlir::AffineMapAttr getLowerBoundsMapAttr() {
    auto attr = ::llvm::cast<::mlir::AffineMapAttr>(getProperties().lowerBoundsMap);
    return attr;
  }

  ::mlir::AffineMap getLowerBoundsMap();
  ::mlir::DenseIntElementsAttr getLowerBoundsGroupsAttr() {
    auto attr = ::llvm::cast<::mlir::DenseIntElementsAttr>(getProperties().lowerBoundsGroups);
    return attr;
  }

  ::mlir::DenseIntElementsAttr getLowerBoundsGroups();
  ::mlir::AffineMapAttr getUpperBoundsMapAttr() {
    auto attr = ::llvm::cast<::mlir::AffineMapAttr>(getProperties().upperBoundsMap);
    return attr;
  }

  ::mlir::AffineMap getUpperBoundsMap();
  ::mlir::DenseIntElementsAttr getUpperBoundsGroupsAttr() {
    auto attr = ::llvm::cast<::mlir::DenseIntElementsAttr>(getProperties().upperBoundsGroups);
    return attr;
  }

  ::mlir::DenseIntElementsAttr getUpperBoundsGroups();
  ::mlir::ArrayAttr getStepsAttr() {
    auto attr = ::llvm::cast<::mlir::ArrayAttr>(getProperties().steps);
    return attr;
  }

  ::llvm::SmallVector<int64_t, 8> getSteps();
  ::mlir::Region &getRegion() {
    return *odsRegions[0];
  }

  ::mlir::RegionRange getRegions() {
    return odsRegions;
  }

};
} // namespace detail
template <typename RangeT>
class AffineParallelOpGenericAdaptor : public detail::AffineParallelOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineParallelOpGenericAdaptorBase;
public:
  AffineParallelOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AffineParallelOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AffineParallelOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  AffineParallelOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : AffineParallelOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = AffineParallelOp, typename = std::enable_if_t<std::is_same_v<LateInst, AffineParallelOp>>>
  AffineParallelOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getMapOperands() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AffineParallelOpAdaptor : public AffineParallelOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineParallelOpGenericAdaptor::AffineParallelOpGenericAdaptor;
  AffineParallelOpAdaptor(AffineParallelOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AffineParallelOp : public ::mlir::Op<AffineParallelOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlock, ::mlir::OpTrait::SingleBlockImplicitTerminator<AffineYieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::RecursivelySpeculatableImplTrait, ::mlir::OpTrait::HasRecursiveMemoryEffects, ::mlir::LoopLikeOpInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineParallelOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineParallelOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("lowerBoundsGroups"), ::llvm::StringRef("lowerBoundsMap"), ::llvm::StringRef("reductions"), ::llvm::StringRef("steps"), ::llvm::StringRef("upperBoundsGroups"), ::llvm::StringRef("upperBoundsMap")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getLowerBoundsGroupsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getLowerBoundsGroupsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getLowerBoundsMapAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getLowerBoundsMapAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getReductionsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getReductionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getStepsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getStepsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getUpperBoundsGroupsAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getUpperBoundsGroupsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getUpperBoundsMapAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getUpperBoundsMapAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.parallel");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::operand_range getMapOperands() {
    return getODSOperands(0);
  }

  ::mlir::MutableOperandRange getMapOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::result_range getResults() {
    return getODSResults(0);
  }

  ::mlir::Region &getRegion() {
    return (*this)->getRegion(0);
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::ArrayAttr getReductionsAttr() {
    return ::llvm::cast<::mlir::ArrayAttr>(getProperties().reductions);
  }

  ::mlir::ArrayAttr getReductions();
  ::mlir::AffineMapAttr getLowerBoundsMapAttr() {
    return ::llvm::cast<::mlir::AffineMapAttr>(getProperties().lowerBoundsMap);
  }

  ::mlir::AffineMap getLowerBoundsMap();
  ::mlir::DenseIntElementsAttr getLowerBoundsGroupsAttr() {
    return ::llvm::cast<::mlir::DenseIntElementsAttr>(getProperties().lowerBoundsGroups);
  }

  ::mlir::DenseIntElementsAttr getLowerBoundsGroups();
  ::mlir::AffineMapAttr getUpperBoundsMapAttr() {
    return ::llvm::cast<::mlir::AffineMapAttr>(getProperties().upperBoundsMap);
  }

  ::mlir::AffineMap getUpperBoundsMap();
  ::mlir::DenseIntElementsAttr getUpperBoundsGroupsAttr() {
    return ::llvm::cast<::mlir::DenseIntElementsAttr>(getProperties().upperBoundsGroups);
  }

  ::mlir::DenseIntElementsAttr getUpperBoundsGroups();
  ::mlir::ArrayAttr getStepsAttr() {
    return ::llvm::cast<::mlir::ArrayAttr>(getProperties().steps);
  }

  ::llvm::SmallVector<int64_t, 8> getSteps();
  void setReductionsAttr(::mlir::ArrayAttr attr) {
    getProperties().reductions = attr;
  }

  void setLowerBoundsMapAttr(::mlir::AffineMapAttr attr) {
    getProperties().lowerBoundsMap = attr;
  }

  void setLowerBoundsMap(::mlir::AffineMap attrValue);
  void setLowerBoundsGroupsAttr(::mlir::DenseIntElementsAttr attr) {
    getProperties().lowerBoundsGroups = attr;
  }

  void setUpperBoundsMapAttr(::mlir::AffineMapAttr attr) {
    getProperties().upperBoundsMap = attr;
  }

  void setUpperBoundsMap(::mlir::AffineMap attrValue);
  void setUpperBoundsGroupsAttr(::mlir::DenseIntElementsAttr attr) {
    getProperties().upperBoundsGroups = attr;
  }

  void setStepsAttr(::mlir::ArrayAttr attr) {
    getProperties().steps = attr;
  }

  void setSteps(::llvm::SmallVector<int64_t, 8> attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, ArrayRef<arith::AtomicRMWKind> reductions, ArrayRef<int64_t> ranges);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, ArrayRef<arith::AtomicRMWKind> reductions, ArrayRef<AffineMap> lbMaps, ValueRange lbArgs, ArrayRef<AffineMap> ubMaps, ValueRange ubArgs, ArrayRef<int64_t> steps);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::ArrayAttr reductions, ::mlir::AffineMapAttr lowerBoundsMap, ::mlir::DenseIntElementsAttr lowerBoundsGroups, ::mlir::AffineMapAttr upperBoundsMap, ::mlir::DenseIntElementsAttr upperBoundsGroups, ::mlir::ArrayAttr steps, ::mlir::ValueRange mapOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::ArrayAttr reductions, ::mlir::AffineMap lowerBoundsMap, ::mlir::DenseIntElementsAttr lowerBoundsGroups, ::mlir::AffineMap upperBoundsMap, ::mlir::DenseIntElementsAttr upperBoundsGroups, ::llvm::SmallVector<int64_t, 8> steps, ::mlir::ValueRange mapOperands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  ::llvm::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  ::llvm::SmallVector<::mlir::Region *> getLoopRegions();
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// Get the number of dimensions.
  unsigned getNumDims();

  /// Get ranges as constants, may fail in dynamic case.
  std::optional<SmallVector<int64_t, 8>> getConstantRanges();

  Block *getBody();
  OpBuilder getBodyBuilder();
  MutableArrayRef<BlockArgument> getIVs() {
    return getBody()->getArguments();
  }

  /// Returns elements of the loop lower bound.
  AffineMap getLowerBoundMap(unsigned pos);
  operand_range getLowerBoundsOperands();
  AffineValueMap getLowerBoundsValueMap();

  /// Sets elements of the loop lower bound.
  void setLowerBounds(ValueRange operands, AffineMap map);

  /// Returns elements of the loop upper bound.
  AffineMap getUpperBoundMap(unsigned pos);
  operand_range getUpperBoundsOperands();
  AffineValueMap getUpperBoundsValueMap();

  /// Sets elements fo the loop upper bound.
  void setUpperBounds(ValueRange operands, AffineMap map);

  void setSteps(ArrayRef<int64_t> newSteps);

  /// Returns attribute names to use in op construction. Not expected to be
  /// used directly.
  static StringRef getReductionsAttrStrName() { return "reductions"; }
  static StringRef getLowerBoundsMapAttrStrName() { return "lowerBoundsMap"; }
  static StringRef getLowerBoundsGroupsAttrStrName() {
    return "lowerBoundsGroups";
  }
  static StringRef getUpperBoundsMapAttrStrName() { return "upperBoundsMap"; }
  static StringRef getUpperBoundsGroupsAttrStrName() {
    return "upperBoundsGroups";
  }
  static StringRef getStepsAttrStrName() { return "steps"; }

  /// Returns `true` if the loop bounds have min/max expressions.
  bool hasMinMaxBounds() {
    return getLowerBoundsMap().getNumResults() != getNumDims() ||
           getUpperBoundsMap().getNumResults() != getNumDims();
  }
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineParallelOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffinePrefetchOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffinePrefetchOpGenericAdaptorBase {
public:
  struct Properties {
    using isDataCacheTy = ::mlir::BoolAttr;
    isDataCacheTy isDataCache;

    auto getIsDataCache() {
      auto &propStorage = this->isDataCache;
      return ::llvm::cast<::mlir::BoolAttr>(propStorage);
    }
    void setIsDataCache(const ::mlir::BoolAttr &propValue) {
      this->isDataCache = propValue;
    }
    using isWriteTy = ::mlir::BoolAttr;
    isWriteTy isWrite;

    auto getIsWrite() {
      auto &propStorage = this->isWrite;
      return ::llvm::cast<::mlir::BoolAttr>(propStorage);
    }
    void setIsWrite(const ::mlir::BoolAttr &propValue) {
      this->isWrite = propValue;
    }
    using localityHintTy = ::mlir::IntegerAttr;
    localityHintTy localityHint;

    auto getLocalityHint() {
      auto &propStorage = this->localityHint;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setLocalityHint(const ::mlir::IntegerAttr &propValue) {
      this->localityHint = propValue;
    }
    using mapTy = ::mlir::AffineMapAttr;
    mapTy map;

    auto getMap() {
      auto &propStorage = this->map;
      return ::llvm::cast<::mlir::AffineMapAttr>(propStorage);
    }
    void setMap(const ::mlir::AffineMapAttr &propValue) {
      this->map = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.isDataCache == this->isDataCache &&
        rhs.isWrite == this->isWrite &&
        rhs.localityHint == this->localityHint &&
        rhs.map == this->map &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  AffinePrefetchOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("affine.prefetch", odsAttrs.getContext());
  }

  AffinePrefetchOpGenericAdaptorBase(AffinePrefetchOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::BoolAttr getIsWriteAttr() {
    auto attr = ::llvm::cast<::mlir::BoolAttr>(getProperties().isWrite);
    return attr;
  }

  bool getIsWrite();
  ::mlir::IntegerAttr getLocalityHintAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().localityHint);
    return attr;
  }

  uint32_t getLocalityHint();
  ::mlir::BoolAttr getIsDataCacheAttr() {
    auto attr = ::llvm::cast<::mlir::BoolAttr>(getProperties().isDataCache);
    return attr;
  }

  bool getIsDataCache();
  ::mlir::AffineMapAttr getMapAttr() {
    auto attr = ::llvm::cast<::mlir::AffineMapAttr>(getProperties().map);
    return attr;
  }

  ::mlir::AffineMap getMap();
};
} // namespace detail
template <typename RangeT>
class AffinePrefetchOpGenericAdaptor : public detail::AffinePrefetchOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffinePrefetchOpGenericAdaptorBase;
public:
  AffinePrefetchOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AffinePrefetchOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AffinePrefetchOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  AffinePrefetchOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : AffinePrefetchOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = AffinePrefetchOp, typename = std::enable_if_t<std::is_same_v<LateInst, AffinePrefetchOp>>>
  AffinePrefetchOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getMemref() {
    return (*getODSOperands(0).begin());
  }

  RangeT getIndices() {
    return getODSOperands(1);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AffinePrefetchOpAdaptor : public AffinePrefetchOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffinePrefetchOpGenericAdaptor::AffinePrefetchOpGenericAdaptor;
  AffinePrefetchOpAdaptor(AffinePrefetchOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AffinePrefetchOp : public ::mlir::Op<AffinePrefetchOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::affine::AffineMapAccessInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffinePrefetchOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffinePrefetchOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("isDataCache"), ::llvm::StringRef("isWrite"), ::llvm::StringRef("localityHint"), ::llvm::StringRef("map")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsDataCacheAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsDataCacheAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIsWriteAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIsWriteAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLocalityHintAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLocalityHintAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getMapAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getMapAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.prefetch");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::MemRefType> getMemref() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
  }

  ::mlir::Operation::operand_range getIndices() {
    return getODSOperands(1);
  }

  ::mlir::OpOperand &getMemrefMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::BoolAttr getIsWriteAttr() {
    return ::llvm::cast<::mlir::BoolAttr>(getProperties().isWrite);
  }

  bool getIsWrite();
  ::mlir::IntegerAttr getLocalityHintAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().localityHint);
  }

  uint32_t getLocalityHint();
  ::mlir::BoolAttr getIsDataCacheAttr() {
    return ::llvm::cast<::mlir::BoolAttr>(getProperties().isDataCache);
  }

  bool getIsDataCache();
  ::mlir::AffineMapAttr getMapAttr() {
    return ::llvm::cast<::mlir::AffineMapAttr>(getProperties().map);
  }

  ::mlir::AffineMap getMap();
  void setIsWriteAttr(::mlir::BoolAttr attr) {
    getProperties().isWrite = attr;
  }

  void setIsWrite(bool attrValue);
  void setLocalityHintAttr(::mlir::IntegerAttr attr) {
    getProperties().localityHint = attr;
  }

  void setLocalityHint(uint32_t attrValue);
  void setIsDataCacheAttr(::mlir::BoolAttr attr) {
    getProperties().isDataCache = attr;
  }

  void setIsDataCache(bool attrValue);
  void setMapAttr(::mlir::AffineMapAttr attr) {
    getProperties().map = attr;
  }

  void setMap(::mlir::AffineMap attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value memref, AffineMap map, ArrayRef<Value> mapOperands, bool isWrite, unsigned localityHint, bool isDataCache);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr isWrite, ::mlir::IntegerAttr localityHint, ::mlir::BoolAttr isDataCache, ::mlir::AffineMapAttr map);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr isWrite, ::mlir::IntegerAttr localityHint, ::mlir::BoolAttr isDataCache, ::mlir::AffineMapAttr map);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, bool isWrite, uint32_t localityHint, bool isDataCache, ::mlir::AffineMap map);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, bool isWrite, uint32_t localityHint, bool isDataCache, ::mlir::AffineMap map);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::llvm::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  MemRefType getMemRefType() {
    return ::llvm::cast<MemRefType>(getMemref().getType());
  }

  /// Returns the affine map used to index the memref for this operation.
  AffineMap getAffineMap() { return getAffineMapAttr().getValue(); }
  AffineMapAttr getAffineMapAttr() {
    return getProperties().map;
  }

  /// Implements the AffineMapAccessInterface.
  /// Returns the AffineMapAttr associated with 'memref'.
  NamedAttribute getAffineMapAttrForMemRef(Value mref) {
    assert(mref == getMemref() &&
           "Expected mref argument to match memref operand");
    return {StringAttr::get(getContext(), getMapAttrStrName()),
      getAffineMapAttr()};
  }

  /// Get affine map operands.
  operand_range getMapOperands() {
    return {operand_begin() + 1, operand_end()};
  }

  static StringRef getMapAttrStrName() { return "map"; }
  static StringRef getLocalityHintAttrStrName() { return "localityHint"; }
  static StringRef getIsWriteAttrStrName() { return "isWrite"; }
  static StringRef getIsDataCacheAttrStrName() { return "isDataCache"; }
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffinePrefetchOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineStoreOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineStoreOpGenericAdaptorBase {
public:
  struct Properties {
    using mapTy = ::mlir::AffineMapAttr;
    mapTy map;

    auto getMap() {
      auto &propStorage = this->map;
      return ::llvm::cast<::mlir::AffineMapAttr>(propStorage);
    }
    void setMap(const ::mlir::AffineMapAttr &propValue) {
      this->map = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.map == this->map &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  AffineStoreOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("affine.store", odsAttrs.getContext());
  }

  AffineStoreOpGenericAdaptorBase(AffineStoreOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::AffineMapAttr getMapAttr() {
    auto attr = ::llvm::cast<::mlir::AffineMapAttr>(getProperties().map);
    return attr;
  }

  ::mlir::AffineMap getMap();
};
} // namespace detail
template <typename RangeT>
class AffineStoreOpGenericAdaptor : public detail::AffineStoreOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineStoreOpGenericAdaptorBase;
public:
  AffineStoreOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AffineStoreOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AffineStoreOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  AffineStoreOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : AffineStoreOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = AffineStoreOp, typename = std::enable_if_t<std::is_same_v<LateInst, AffineStoreOp>>>
  AffineStoreOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getValue() {
    return (*getODSOperands(0).begin());
  }

  ValueT getMemref() {
    return (*getODSOperands(1).begin());
  }

  RangeT getIndices() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AffineStoreOpAdaptor : public AffineStoreOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineStoreOpGenericAdaptor::AffineStoreOpGenericAdaptor;
  AffineStoreOpAdaptor(AffineStoreOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AffineStoreOp : public ::mlir::Op<AffineStoreOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::affine::AffineWriteOpInterface::Trait, ::mlir::affine::AffineMapAccessInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineStoreOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineStoreOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("map")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getMapAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getMapAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.store");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getValue() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::MemRefType> getMemref() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(1).begin());
  }

  ::mlir::Operation::operand_range getIndices() {
    return getODSOperands(2);
  }

  ::mlir::OpOperand &getValueMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getMemrefMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::AffineMapAttr getMapAttr() {
    return ::llvm::cast<::mlir::AffineMapAttr>(getProperties().map);
  }

  ::mlir::AffineMap getMap();
  void setMapAttr(::mlir::AffineMapAttr attr) {
    getProperties().map = attr;
  }

  void setMap(::mlir::AffineMap attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value valueToStore, Value memref, ValueRange indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value valueToStore, Value memref, AffineMap map, ValueRange mapOperands);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::llvm::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// Returns the operand index of the value to be stored.
  unsigned getStoredValOperandIndex() { return 0; }

  /// Returns the operand index of the memref.
  unsigned getMemRefOperandIndex() { return 1; }

  void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

  /// Returns the affine map used to index the memref for this operation.
  AffineMapAttr getAffineMapAttr() {
    return getProperties().map;
  }

  static StringRef getMapAttrStrName() { return "map"; }
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineStoreOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineVectorLoadOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineVectorLoadOpGenericAdaptorBase {
public:
  struct Properties {
    using mapTy = ::mlir::AffineMapAttr;
    mapTy map;

    auto getMap() {
      auto &propStorage = this->map;
      return ::llvm::cast<::mlir::AffineMapAttr>(propStorage);
    }
    void setMap(const ::mlir::AffineMapAttr &propValue) {
      this->map = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.map == this->map &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  AffineVectorLoadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("affine.vector_load", odsAttrs.getContext());
  }

  AffineVectorLoadOpGenericAdaptorBase(AffineVectorLoadOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::AffineMapAttr getMapAttr() {
    auto attr = ::llvm::cast<::mlir::AffineMapAttr>(getProperties().map);
    return attr;
  }

  ::mlir::AffineMap getMap();
};
} // namespace detail
template <typename RangeT>
class AffineVectorLoadOpGenericAdaptor : public detail::AffineVectorLoadOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineVectorLoadOpGenericAdaptorBase;
public:
  AffineVectorLoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AffineVectorLoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AffineVectorLoadOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  AffineVectorLoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : AffineVectorLoadOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = AffineVectorLoadOp, typename = std::enable_if_t<std::is_same_v<LateInst, AffineVectorLoadOp>>>
  AffineVectorLoadOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getMemref() {
    return (*getODSOperands(0).begin());
  }

  RangeT getIndices() {
    return getODSOperands(1);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AffineVectorLoadOpAdaptor : public AffineVectorLoadOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineVectorLoadOpGenericAdaptor::AffineVectorLoadOpGenericAdaptor;
  AffineVectorLoadOpAdaptor(AffineVectorLoadOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AffineVectorLoadOp : public ::mlir::Op<AffineVectorLoadOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::affine::AffineReadOpInterface::Trait, ::mlir::affine::AffineMapAccessInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineVectorLoadOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineVectorLoadOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("map")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getMapAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getMapAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.vector_load");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::MemRefType> getMemref() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
  }

  ::mlir::Operation::operand_range getIndices() {
    return getODSOperands(1);
  }

  ::mlir::OpOperand &getMemrefMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::VectorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::VectorType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::AffineMapAttr getMapAttr() {
    return ::llvm::cast<::mlir::AffineMapAttr>(getProperties().map);
  }

  ::mlir::AffineMap getMap();
  void setMapAttr(::mlir::AffineMapAttr attr) {
    getProperties().map = attr;
  }

  void setMap(::mlir::AffineMap attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, VectorType resultType, AffineMap map, ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, VectorType resultType, Value memref, ValueRange indices = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, VectorType resultType, Value memref, AffineMap map, ValueRange mapOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::AffineMapAttr map);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::AffineMapAttr map);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::AffineMap map);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::AffineMap map);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// Returns the operand index of the memref.
  unsigned getMemRefOperandIndex() { return 0; }

  void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

  /// Returns the affine map used to index the memref for this operation.
  AffineMapAttr getAffineMapAttr() {
    return getProperties().map;
  }

  static StringRef getMapAttrStrName() { return "map"; }

  VectorType getVectorType() {
    return ::llvm::cast<VectorType>(getResult().getType());
  }
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineVectorLoadOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineVectorStoreOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineVectorStoreOpGenericAdaptorBase {
public:
  struct Properties {
    using mapTy = ::mlir::AffineMapAttr;
    mapTy map;

    auto getMap() {
      auto &propStorage = this->map;
      return ::llvm::cast<::mlir::AffineMapAttr>(propStorage);
    }
    void setMap(const ::mlir::AffineMapAttr &propValue) {
      this->map = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.map == this->map &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  AffineVectorStoreOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("affine.vector_store", odsAttrs.getContext());
  }

  AffineVectorStoreOpGenericAdaptorBase(AffineVectorStoreOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::AffineMapAttr getMapAttr() {
    auto attr = ::llvm::cast<::mlir::AffineMapAttr>(getProperties().map);
    return attr;
  }

  ::mlir::AffineMap getMap();
};
} // namespace detail
template <typename RangeT>
class AffineVectorStoreOpGenericAdaptor : public detail::AffineVectorStoreOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineVectorStoreOpGenericAdaptorBase;
public:
  AffineVectorStoreOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AffineVectorStoreOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AffineVectorStoreOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  AffineVectorStoreOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : AffineVectorStoreOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = AffineVectorStoreOp, typename = std::enable_if_t<std::is_same_v<LateInst, AffineVectorStoreOp>>>
  AffineVectorStoreOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getValue() {
    return (*getODSOperands(0).begin());
  }

  ValueT getMemref() {
    return (*getODSOperands(1).begin());
  }

  RangeT getIndices() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AffineVectorStoreOpAdaptor : public AffineVectorStoreOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineVectorStoreOpGenericAdaptor::AffineVectorStoreOpGenericAdaptor;
  AffineVectorStoreOpAdaptor(AffineVectorStoreOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AffineVectorStoreOp : public ::mlir::Op<AffineVectorStoreOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::affine::AffineWriteOpInterface::Trait, ::mlir::affine::AffineMapAccessInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineVectorStoreOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineVectorStoreOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("map")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getMapAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getMapAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.vector_store");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::VectorType> getValue() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::VectorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::MemRefType> getMemref() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(1).begin());
  }

  ::mlir::Operation::operand_range getIndices() {
    return getODSOperands(2);
  }

  ::mlir::OpOperand &getValueMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getMemrefMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::AffineMapAttr getMapAttr() {
    return ::llvm::cast<::mlir::AffineMapAttr>(getProperties().map);
  }

  ::mlir::AffineMap getMap();
  void setMapAttr(::mlir::AffineMapAttr attr) {
    getProperties().map = attr;
  }

  void setMap(::mlir::AffineMap attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value valueToStore, Value memref, ValueRange indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value valueToStore, Value memref, AffineMap map, ValueRange mapOperands);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  /// Returns the operand index of the value to be stored.
  unsigned getStoredValOperandIndex() { return 0; }

  /// Returns the operand index of the memref.
  unsigned getMemRefOperandIndex() { return 1; }

  void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

  /// Returns the affine map used to index the memref for this operation.
  AffineMapAttr getAffineMapAttr() {
    return getProperties().map;
  }

  static StringRef getMapAttrStrName() { return "map"; }

  VectorType getVectorType() {
    return ::llvm::cast<VectorType>(getValue().getType());
  }
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineVectorStoreOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineYieldOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineYieldOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  AffineYieldOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("affine.yield", odsAttrs.getContext());
  }

  AffineYieldOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};
} // namespace detail
template <typename RangeT>
class AffineYieldOpGenericAdaptor : public detail::AffineYieldOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineYieldOpGenericAdaptorBase;
public:
  AffineYieldOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AffineYieldOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AffineYieldOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  template <typename LateInst = AffineYieldOp, typename = std::enable_if_t<std::is_same_v<LateInst, AffineYieldOp>>>
  AffineYieldOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return getODSOperands(0);
  }

private:
  RangeT odsOperands;
};
class AffineYieldOpAdaptor : public AffineYieldOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineYieldOpGenericAdaptor::AffineYieldOpGenericAdaptor;
  AffineYieldOpAdaptor(AffineYieldOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AffineYieldOp : public ::mlir::Op<AffineYieldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::IsTerminator, ::mlir::RegionBranchTerminatorOpInterface::Trait, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::MemRefsNormalizable> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineYieldOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineYieldOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.yield");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::operand_range getOperands() {
    return getODSOperands(0);
  }

  ::mlir::MutableOperandRange getOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  ::mlir::MutableOperandRange getMutableSuccessorOperands(::mlir::RegionBranchPoint point);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineYieldOp)


#endif  // GET_OP_CLASSES

