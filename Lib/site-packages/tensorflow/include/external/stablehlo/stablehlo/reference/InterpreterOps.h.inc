/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: InterpreterOps.td                                                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace stablehlo {
namespace interpreter {
class ProbeOp;
} // namespace interpreter
} // namespace stablehlo
} // namespace mlir
namespace mlir {
namespace stablehlo {
namespace interpreter {
class RunParallelOp;
} // namespace interpreter
} // namespace stablehlo
} // namespace mlir
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace stablehlo {
namespace interpreter {

//===----------------------------------------------------------------------===//
// ::mlir::stablehlo::interpreter::ProbeOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ProbeOpGenericAdaptorBase {
public:
  struct Properties {
    using probe_idTy = ::mlir::StringAttr;
    probe_idTy probe_id;

    auto getProbeId() {
      auto &propStorage = this->probe_id;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setProbeId(const ::mlir::StringAttr &propValue) {
      this->probe_id = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.probe_id == this->probe_id &&
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
  ProbeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("interpreter.probe", odsAttrs.getContext());
  }

  ProbeOpGenericAdaptorBase(ProbeOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::StringAttr getProbeIdAttr() {
    auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().probe_id);
    return attr;
  }

  ::llvm::StringRef getProbeId();
};
} // namespace detail
template <typename RangeT>
class ProbeOpGenericAdaptor : public detail::ProbeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ProbeOpGenericAdaptorBase;
public:
  ProbeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ProbeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ProbeOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ProbeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ProbeOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = ProbeOp, typename = std::enable_if_t<std::is_same_v<LateInst, ProbeOp>>>
  ProbeOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getOperand() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ProbeOpAdaptor : public ProbeOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ProbeOpGenericAdaptor::ProbeOpGenericAdaptor;
  ProbeOpAdaptor(ProbeOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ProbeOp : public ::mlir::Op<ProbeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultType, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ProbeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ProbeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("probe_id")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getProbeIdAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getProbeIdAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("interpreter.probe");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::RankedTensorType> getOperand() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getOperandMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::RankedTensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
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
  ::mlir::StringAttr getProbeIdAttr() {
    return ::llvm::cast<::mlir::StringAttr>(getProperties().probe_id);
  }

  ::llvm::StringRef getProbeId();
  void setProbeIdAttr(::mlir::StringAttr attr) {
    getProperties().probe_id = attr;
  }

  void setProbeId(::llvm::StringRef attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand, ::mlir::StringAttr probe_id);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::StringAttr probe_id);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::StringAttr probe_id);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand, ::llvm::StringRef probe_id);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::llvm::StringRef probe_id);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::llvm::StringRef probe_id);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
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
};
} // namespace interpreter
} // namespace stablehlo
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::stablehlo::interpreter::ProbeOp)

namespace mlir {
namespace stablehlo {
namespace interpreter {

//===----------------------------------------------------------------------===//
// ::mlir::stablehlo::interpreter::RunParallelOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class RunParallelOpGenericAdaptorBase {
public:
  struct Properties {
    using infeedTy = ::mlir::ArrayAttr;
    infeedTy infeed;

    auto getInfeed() {
      auto &propStorage = this->infeed;
      return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(propStorage);
    }
    void setInfeed(const ::mlir::ArrayAttr &propValue) {
      this->infeed = propValue;
    }
    using programsTy = ::mlir::ArrayAttr;
    programsTy programs;

    auto getPrograms() {
      auto &propStorage = this->programs;
      return ::llvm::cast<::mlir::ArrayAttr>(propStorage);
    }
    void setPrograms(const ::mlir::ArrayAttr &propValue) {
      this->programs = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.infeed == this->infeed &&
        rhs.programs == this->programs &&
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
  RunParallelOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("interpreter.run_parallel", odsAttrs.getContext());
  }

  RunParallelOpGenericAdaptorBase(RunParallelOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getInfeedAttr() {
    auto attr = ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(getProperties().infeed);
    return attr;
  }

  ::std::optional< ::mlir::ArrayAttr > getInfeed();
  ::mlir::ArrayAttr getProgramsAttr() {
    auto attr = ::llvm::cast<::mlir::ArrayAttr>(getProperties().programs);
    return attr;
  }

  ::mlir::ArrayAttr getPrograms();
};
} // namespace detail
template <typename RangeT>
class RunParallelOpGenericAdaptor : public detail::RunParallelOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RunParallelOpGenericAdaptorBase;
public:
  RunParallelOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  RunParallelOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : RunParallelOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  RunParallelOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : RunParallelOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  template <typename LateInst = RunParallelOp, typename = std::enable_if_t<std::is_same_v<LateInst, RunParallelOp>>>
  RunParallelOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getInputs() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RunParallelOpAdaptor : public RunParallelOpGenericAdaptor<::mlir::ValueRange> {
public:
  using RunParallelOpGenericAdaptor::RunParallelOpGenericAdaptor;
  RunParallelOpAdaptor(RunParallelOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class RunParallelOp : public ::mlir::Op<RunParallelOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RunParallelOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RunParallelOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("infeed"), ::llvm::StringRef("programs")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getInfeedAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getInfeedAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getProgramsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getProgramsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("interpreter.run_parallel");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::operand_range getInputs() {
    return getODSOperands(0);
  }

  ::mlir::MutableOperandRange getInputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::result_range getResults() {
    return getODSResults(0);
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
  ::mlir::ArrayAttr getInfeedAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(getProperties().infeed);
  }

  ::std::optional< ::mlir::ArrayAttr > getInfeed();
  ::mlir::ArrayAttr getProgramsAttr() {
    return ::llvm::cast<::mlir::ArrayAttr>(getProperties().programs);
  }

  ::mlir::ArrayAttr getPrograms();
  void setInfeedAttr(::mlir::ArrayAttr attr) {
    getProperties().infeed = attr;
  }

  void setProgramsAttr(::mlir::ArrayAttr attr) {
    getProperties().programs = attr;
  }

  ::mlir::Attribute removeInfeedAttr() {
      auto &attr = getProperties().infeed;
      attr = {};
      return attr;
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::ValueRange inputs, /*optional*/::mlir::ArrayAttr infeed, ::mlir::ArrayAttr programs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace interpreter
} // namespace stablehlo
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::stablehlo::interpreter::RunParallelOp)


#endif  // GET_OP_CLASSES

