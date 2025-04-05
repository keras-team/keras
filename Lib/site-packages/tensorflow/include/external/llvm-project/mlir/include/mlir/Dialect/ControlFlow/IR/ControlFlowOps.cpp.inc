/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ControlFlowOps.td                                                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::cf::AssertOp,
::mlir::cf::BranchOp,
::mlir::cf::CondBranchOp,
::mlir::cf::SwitchOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace cf {

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_ControlFlowOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(1)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_ControlFlowOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of any type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_ControlFlowOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::IntegerType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_ControlFlowOps1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::StringAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: string attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_ControlFlowOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ControlFlowOps1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_ControlFlowOps2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::DenseIntElementsAttr>(attr))) && ((true))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: integer elements attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_ControlFlowOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ControlFlowOps2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_ControlFlowOps3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::DenseI32ArrayAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: i32 dense array attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_ControlFlowOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ControlFlowOps3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
} // namespace cf
} // namespace mlir
namespace mlir {
namespace cf {

//===----------------------------------------------------------------------===//
// ::mlir::cf::AssertOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AssertOpGenericAdaptorBase::AssertOpGenericAdaptorBase(AssertOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::llvm::StringRef AssertOpGenericAdaptorBase::getMsg() {
  auto attr = getMsgAttr();
  return attr.getValue();
}

} // namespace detail
AssertOpAdaptor::AssertOpAdaptor(AssertOp op) : AssertOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AssertOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_msg = getProperties().msg; (void)tblgen_msg;
  if (!tblgen_msg) return emitError(loc, "'cf.assert' op ""requires attribute 'msg'");

  if (tblgen_msg && !((::llvm::isa<::mlir::StringAttr>(tblgen_msg))))
    return emitError(loc, "'cf.assert' op ""attribute 'msg' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

::llvm::LogicalResult AssertOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.msg;
       auto attr = dict.get("msg");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `msg` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute AssertOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.msg;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("msg",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code AssertOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.msg.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> AssertOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "msg")
      return prop.msg;
  return std::nullopt;
}

void AssertOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "msg") {
       prop.msg = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.msg)>>(value);
       return;
    }
}

void AssertOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.msg) attrs.append("msg", prop.msg);
}

::llvm::LogicalResult AssertOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getMsgAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ControlFlowOps1(attr, "msg", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult AssertOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.msg)))
    return ::mlir::failure();
  return ::mlir::success();
}

void AssertOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.msg);
}

::llvm::StringRef AssertOp::getMsg() {
  auto attr = getMsgAttr();
  return attr.getValue();
}

void AssertOp::setMsg(::llvm::StringRef attrValue) {
  getProperties().msg = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void AssertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value arg, ::mlir::StringAttr msg) {
  odsState.addOperands(arg);
  odsState.getOrAddProperties<Properties>().msg = msg;
}

void AssertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg, ::mlir::StringAttr msg) {
  odsState.addOperands(arg);
  odsState.getOrAddProperties<Properties>().msg = msg;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AssertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value arg, ::llvm::StringRef msg) {
  odsState.addOperands(arg);
  odsState.getOrAddProperties<Properties>().msg = odsBuilder.getStringAttr(msg);
}

void AssertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg, ::llvm::StringRef msg) {
  odsState.addOperands(arg);
  odsState.getOrAddProperties<Properties>().msg = odsBuilder.getStringAttr(msg);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AssertOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<AssertOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult AssertOp::verifyInvariantsImpl() {
  auto tblgen_msg = getProperties().msg; (void)tblgen_msg;
  if (!tblgen_msg) return emitOpError("requires attribute 'msg'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ControlFlowOps1(*this, tblgen_msg, "msg")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ControlFlowOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult AssertOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void AssertOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.add(canonicalize);
}

::mlir::ParseResult AssertOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(&argRawOperand, 1);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::StringAttr msgAttr;

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(msgAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (msgAttr) result.getOrAddProperties<AssertOp::Properties>().msg = msgAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(argOperands, odsBuildableType0, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AssertOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getMsgAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("msg");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace cf
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::cf::AssertOp)

namespace mlir {
namespace cf {

//===----------------------------------------------------------------------===//
// ::mlir::cf::BranchOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
std::pair<unsigned, unsigned> BranchOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

} // namespace detail
BranchOpAdaptor::BranchOpAdaptor(BranchOp op) : BranchOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult BranchOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BranchOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange BranchOp::getDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

void BranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Block *dest, ValueRange destOperands) {
      odsState.addSuccessors(dest);
      odsState.addOperands(destOperands);
    
}

void BranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange destOperands, ::mlir::Block *dest) {
  odsState.addOperands(destOperands);
  odsState.addSuccessors(dest);
}

void BranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange destOperands, ::mlir::Block *dest) {
  odsState.addOperands(destOperands);
  odsState.addSuccessors(dest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BranchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult BranchOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ControlFlowOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult BranchOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void BranchOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.add(canonicalize);
}

::mlir::ParseResult BranchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Block *destSuccessor = nullptr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> destOperandsOperands;
  ::llvm::SMLoc destOperandsOperandsLoc;
  (void)destOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> destOperandsTypes;

  if (parser.parseSuccessor(destSuccessor))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  destOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(destOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(destOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  result.addSuccessors(destSuccessor);
  if (parser.resolveOperands(destOperandsOperands, destOperandsTypes, destOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BranchOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getDest();
  if (!getDestOperands().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getDestOperands();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getDestOperands().getTypes();
    _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void BranchOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace cf
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::cf::BranchOp)

namespace mlir {
namespace cf {

//===----------------------------------------------------------------------===//
// ::mlir::cf::CondBranchOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CondBranchOpGenericAdaptorBase::CondBranchOpGenericAdaptorBase(CondBranchOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> CondBranchOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

} // namespace detail
CondBranchOpAdaptor::CondBranchOpAdaptor(CondBranchOp op) : CondBranchOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult CondBranchOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CondBranchOp::getODSOperandIndexAndLength(unsigned index) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::MutableOperandRange CondBranchOp::getTrueDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::mlir::MutableOperandRange CondBranchOp::getFalseDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::llvm::LogicalResult CondBranchOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
    {

      auto setFromAttr = [] (auto &propStorage, ::mlir::Attribute propAttr,
               ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) -> ::mlir::LogicalResult {
        return convertFromAttribute(propStorage, propAttr, emitError);
      };
         auto attr = dict.get("operandSegmentSizes");   if (!attr) attr = dict.get("operand_segment_sizes");;
;
      if (attr && ::mlir::failed(setFromAttr(prop.operandSegmentSizes, attr, emitError)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute CondBranchOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.operandSegmentSizes;
      auto attr = [&]() -> ::mlir::Attribute {
        return ::mlir::DenseI32ArrayAttr::get(ctx, propStorage);
      }();
      attrs.push_back(odsBuilder.getNamedAttr("operandSegmentSizes", attr));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code CondBranchOp::computePropertiesHash(const Properties &prop) {
  auto hash_operandSegmentSizes = [] (const auto &propStorage) -> llvm::hash_code {
    return ::llvm::hash_combine_range(std::begin(propStorage), std::end(propStorage));;
  };
  return llvm::hash_combine(
    hash_operandSegmentSizes(prop.operandSegmentSizes));
}

std::optional<mlir::Attribute> CondBranchOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "operand_segment_sizes" || name == "operandSegmentSizes") return [&]() -> ::mlir::Attribute { return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes); }();
  return std::nullopt;
}

void CondBranchOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
        if (name == "operand_segment_sizes" || name == "operandSegmentSizes") {
       auto arrAttr = ::llvm::dyn_cast_or_null<::mlir::DenseI32ArrayAttr>(value);
       if (!arrAttr) return;
       if (arrAttr.size() != sizeof(prop.operandSegmentSizes) / sizeof(int32_t))
         return;
       llvm::copy(arrAttr.asArrayRef(), prop.operandSegmentSizes.begin());
       return;
    }
}

void CondBranchOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
  attrs.append("operandSegmentSizes", [&]() -> ::mlir::Attribute { return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes); }());
}

::llvm::LogicalResult CondBranchOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

::llvm::LogicalResult CondBranchOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (reader.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
    auto &propStorage = prop.operandSegmentSizes;
    ::mlir::DenseI32ArrayAttr attr;
    if (::mlir::failed(reader.readAttribute(attr))) return ::mlir::failure();
    if (attr.size() > static_cast<int64_t>(sizeof(propStorage) / sizeof(int32_t))) {
      reader.emitError("size mismatch for operand/result_segment_size");
      return ::mlir::failure();
    }
    ::llvm::copy(::llvm::ArrayRef<int32_t>(attr), propStorage.begin());
  }

  {
    auto &propStorage = prop.operandSegmentSizes;
    auto readProp = [&]() {

  if (reader.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    return reader.readSparseArray(::llvm::MutableArrayRef(propStorage));
;
      return ::mlir::success();
    };
    if (::mlir::failed(readProp()))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void CondBranchOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

if (writer.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
  auto &propStorage = prop.operandSegmentSizes;
  writer.writeAttribute(::mlir::DenseI32ArrayAttr::get(this->getContext(), propStorage));
}

  {
    auto &propStorage = prop.operandSegmentSizes;

  if (writer.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    writer.writeSparseArray(::llvm::ArrayRef(propStorage));
;
  }
}

void CondBranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value condition, Block *trueDest, ValueRange trueOperands, Block *falseDest, ValueRange falseOperands) {
      build(odsBuilder, odsState, condition, trueOperands, falseOperands, trueDest,
            falseDest);
    
}

void CondBranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value condition, Block *trueDest, Block *falseDest, ValueRange falseOperands) {
      build(odsBuilder, odsState, condition, trueDest, ValueRange(), falseDest,
            falseOperands);
    
}

void CondBranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value condition, ::mlir::ValueRange trueDestOperands, ::mlir::ValueRange falseDestOperands, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(condition);
  odsState.addOperands(trueDestOperands);
  odsState.addOperands(falseDestOperands);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({1, static_cast<int32_t>(trueDestOperands.size()), static_cast<int32_t>(falseDestOperands.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CondBranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::ValueRange trueDestOperands, ::mlir::ValueRange falseDestOperands, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(condition);
  odsState.addOperands(trueDestOperands);
  odsState.addOperands(falseDestOperands);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({1, static_cast<int32_t>(trueDestOperands.size()), static_cast<int32_t>(falseDestOperands.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CondBranchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<CondBranchOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult CondBranchOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ControlFlowOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ControlFlowOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ControlFlowOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult CondBranchOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CondBranchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand conditionRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> conditionOperands(&conditionRawOperand, 1);  ::llvm::SMLoc conditionOperandsLoc;
  (void)conditionOperandsLoc;
  ::mlir::Block *trueDestSuccessor = nullptr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> trueDestOperandsOperands;
  ::llvm::SMLoc trueDestOperandsOperandsLoc;
  (void)trueDestOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> trueDestOperandsTypes;
  ::mlir::Block *falseDestSuccessor = nullptr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> falseDestOperandsOperands;
  ::llvm::SMLoc falseDestOperandsOperandsLoc;
  (void)falseDestOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> falseDestOperandsTypes;

  conditionOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(conditionRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseSuccessor(trueDestSuccessor))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  trueDestOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(trueDestOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(trueDestOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseSuccessor(falseDestSuccessor))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  falseDestOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(falseDestOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(falseDestOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  result.addSuccessors(trueDestSuccessor);
  result.addSuccessors(falseDestSuccessor);
::llvm::copy(::llvm::ArrayRef<int32_t>({1, static_cast<int32_t>(trueDestOperandsOperands.size()), static_cast<int32_t>(falseDestOperandsOperands.size())}), result.getOrAddProperties<CondBranchOp::Properties>().operandSegmentSizes.begin());
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(conditionOperands, odsBuildableType0, conditionOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(trueDestOperandsOperands, trueDestOperandsTypes, trueDestOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(falseDestOperandsOperands, falseDestOperandsTypes, falseDestOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CondBranchOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getCondition();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getTrueDest();
  if (!getTrueDestOperands().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getTrueDestOperands();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getTrueDestOperands().getTypes();
    _odsPrinter << ")";
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getFalseDest();
  if (!getFalseDestOperands().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getFalseDestOperands();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getFalseDestOperands().getTypes();
    _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operandSegmentSizes");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void CondBranchOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace cf
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::cf::CondBranchOp)

namespace mlir {
namespace cf {

//===----------------------------------------------------------------------===//
// ::mlir::cf::SwitchOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchOpGenericAdaptorBase::SwitchOpGenericAdaptorBase(SwitchOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> SwitchOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::std::optional< ::mlir::DenseIntElementsAttr > SwitchOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr ? ::std::optional< ::mlir::DenseIntElementsAttr >(attr) : (::std::nullopt);
}

::llvm::ArrayRef<int32_t> SwitchOpGenericAdaptorBase::getCaseOperandSegments() {
  auto attr = getCaseOperandSegmentsAttr();
  return attr;
}

} // namespace detail
SwitchOpAdaptor::SwitchOpAdaptor(SwitchOp op) : SwitchOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult SwitchOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_case_operand_segments = getProperties().case_operand_segments; (void)tblgen_case_operand_segments;
  if (!tblgen_case_operand_segments) return emitError(loc, "'cf.switch' op ""requires attribute 'case_operand_segments'");
  auto tblgen_case_values = getProperties().case_values; (void)tblgen_case_values;

  if (tblgen_case_values && !(((::llvm::isa<::mlir::DenseIntElementsAttr>(tblgen_case_values))) && ((true))))
    return emitError(loc, "'cf.switch' op ""attribute 'case_values' failed to satisfy constraint: integer elements attribute");

  if (tblgen_case_operand_segments && !((::llvm::isa<::mlir::DenseI32ArrayAttr>(tblgen_case_operand_segments))))
    return emitError(loc, "'cf.switch' op ""attribute 'case_operand_segments' failed to satisfy constraint: i32 dense array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SwitchOp::getODSOperandIndexAndLength(unsigned index) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::MutableOperandRange SwitchOp::getDefaultOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::mlir::MutableOperandRangeRange SwitchOp::getCaseOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange.split(*(*this)->getAttrDictionary().getNamed(getCaseOperandSegmentsAttrName()));
}

::llvm::LogicalResult SwitchOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.case_operand_segments;
       auto attr = dict.get("case_operand_segments");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `case_operand_segments` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.case_values;
       auto attr = dict.get("case_values");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `case_values` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
{

      auto setFromAttr = [] (auto &propStorage, ::mlir::Attribute propAttr,
               ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) -> ::mlir::LogicalResult {
        return convertFromAttribute(propStorage, propAttr, emitError);
      };
         auto attr = dict.get("operandSegmentSizes");   if (!attr) attr = dict.get("operand_segment_sizes");;
;
      if (attr && ::mlir::failed(setFromAttr(prop.operandSegmentSizes, attr, emitError)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute SwitchOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.case_operand_segments;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("case_operand_segments",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.case_values;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("case_values",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.operandSegmentSizes;
      auto attr = [&]() -> ::mlir::Attribute {
        return ::mlir::DenseI32ArrayAttr::get(ctx, propStorage);
      }();
      attrs.push_back(odsBuilder.getNamedAttr("operandSegmentSizes", attr));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code SwitchOp::computePropertiesHash(const Properties &prop) {
  auto hash_operandSegmentSizes = [] (const auto &propStorage) -> llvm::hash_code {
    return ::llvm::hash_combine_range(std::begin(propStorage), std::end(propStorage));;
  };
  return llvm::hash_combine(
    llvm::hash_value(prop.case_operand_segments.getAsOpaquePointer()), 
    llvm::hash_value(prop.case_values.getAsOpaquePointer()), 
    hash_operandSegmentSizes(prop.operandSegmentSizes));
}

std::optional<mlir::Attribute> SwitchOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "case_operand_segments")
      return prop.case_operand_segments;

    if (name == "case_values")
      return prop.case_values;
    if (name == "operand_segment_sizes" || name == "operandSegmentSizes") return [&]() -> ::mlir::Attribute { return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes); }();
  return std::nullopt;
}

void SwitchOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "case_operand_segments") {
       prop.case_operand_segments = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.case_operand_segments)>>(value);
       return;
    }

    if (name == "case_values") {
       prop.case_values = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.case_values)>>(value);
       return;
    }
        if (name == "operand_segment_sizes" || name == "operandSegmentSizes") {
       auto arrAttr = ::llvm::dyn_cast_or_null<::mlir::DenseI32ArrayAttr>(value);
       if (!arrAttr) return;
       if (arrAttr.size() != sizeof(prop.operandSegmentSizes) / sizeof(int32_t))
         return;
       llvm::copy(arrAttr.asArrayRef(), prop.operandSegmentSizes.begin());
       return;
    }
}

void SwitchOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.case_operand_segments) attrs.append("case_operand_segments", prop.case_operand_segments);

    if (prop.case_values) attrs.append("case_values", prop.case_values);
  attrs.append("operandSegmentSizes", [&]() -> ::mlir::Attribute { return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes); }());
}

::llvm::LogicalResult SwitchOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getCaseOperandSegmentsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ControlFlowOps3(attr, "case_operand_segments", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getCaseValuesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ControlFlowOps2(attr, "case_values", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult SwitchOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.case_operand_segments)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.case_values)))
    return ::mlir::failure();

  if (reader.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
    auto &propStorage = prop.operandSegmentSizes;
    ::mlir::DenseI32ArrayAttr attr;
    if (::mlir::failed(reader.readAttribute(attr))) return ::mlir::failure();
    if (attr.size() > static_cast<int64_t>(sizeof(propStorage) / sizeof(int32_t))) {
      reader.emitError("size mismatch for operand/result_segment_size");
      return ::mlir::failure();
    }
    ::llvm::copy(::llvm::ArrayRef<int32_t>(attr), propStorage.begin());
  }

  {
    auto &propStorage = prop.operandSegmentSizes;
    auto readProp = [&]() {

  if (reader.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    return reader.readSparseArray(::llvm::MutableArrayRef(propStorage));
;
      return ::mlir::success();
    };
    if (::mlir::failed(readProp()))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void SwitchOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.case_operand_segments);

  writer.writeOptionalAttribute(prop.case_values);

if (writer.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
  auto &propStorage = prop.operandSegmentSizes;
  writer.writeAttribute(::mlir::DenseI32ArrayAttr::get(this->getContext(), propStorage));
}

  {
    auto &propStorage = prop.operandSegmentSizes;

  if (writer.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    writer.writeSparseArray(::llvm::ArrayRef(propStorage));
;
  }
}

::std::optional< ::mlir::DenseIntElementsAttr > SwitchOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr ? ::std::optional< ::mlir::DenseIntElementsAttr >(attr) : (::std::nullopt);
}

::llvm::ArrayRef<int32_t> SwitchOp::getCaseOperandSegments() {
  auto attr = getCaseOperandSegmentsAttr();
  return attr;
}

void SwitchOp::setCaseOperandSegments(::llvm::ArrayRef<int32_t> attrValue) {
  getProperties().case_operand_segments = ::mlir::Builder((*this)->getContext()).getDenseI32ArrayAttr(attrValue);
}

void SwitchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value flag, ::mlir::ValueRange defaultOperands, ::llvm::ArrayRef<::mlir::ValueRange> caseOperands, /*optional*/::mlir::DenseIntElementsAttr case_values, ::mlir::Block *defaultDestination, ::mlir::BlockRange caseDestinations) {
  odsState.addOperands(flag);
  odsState.addOperands(defaultOperands);
  for (::mlir::ValueRange range : caseOperands)
   odsState.addOperands(range);
  {
    ::llvm::SmallVector<int32_t> rangeSegments;
    for (::mlir::ValueRange range : caseOperands)
      rangeSegments.push_back(range.size());
    auto rangeAttr = odsBuilder.getDenseI32ArrayAttr(rangeSegments);
    odsState.getOrAddProperties<Properties>().case_operand_segments = rangeAttr;  }
  ::llvm::copy(::llvm::ArrayRef<int32_t>({1, static_cast<int32_t>(defaultOperands.size()), static_cast<int32_t>(std::accumulate(caseOperands.begin(), caseOperands.end(), 0, [](int32_t curSum, ::mlir::ValueRange range) { return curSum + static_cast<int32_t>(range.size()); }))}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  if (case_values) {
    odsState.getOrAddProperties<Properties>().case_values = case_values;
  }
  odsState.addSuccessors(defaultDestination);
  odsState.addSuccessors(caseDestinations);
}

void SwitchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value flag, ::mlir::ValueRange defaultOperands, ::llvm::ArrayRef<::mlir::ValueRange> caseOperands, /*optional*/::mlir::DenseIntElementsAttr case_values, ::mlir::Block *defaultDestination, ::mlir::BlockRange caseDestinations) {
  odsState.addOperands(flag);
  odsState.addOperands(defaultOperands);
  for (::mlir::ValueRange range : caseOperands)
   odsState.addOperands(range);
  {
    ::llvm::SmallVector<int32_t> rangeSegments;
    for (::mlir::ValueRange range : caseOperands)
      rangeSegments.push_back(range.size());
    auto rangeAttr = odsBuilder.getDenseI32ArrayAttr(rangeSegments);
    odsState.getOrAddProperties<Properties>().case_operand_segments = rangeAttr;  }
  ::llvm::copy(::llvm::ArrayRef<int32_t>({1, static_cast<int32_t>(defaultOperands.size()), static_cast<int32_t>(std::accumulate(caseOperands.begin(), caseOperands.end(), 0, [](int32_t curSum, ::mlir::ValueRange range) { return curSum + static_cast<int32_t>(range.size()); }))}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  if (case_values) {
    odsState.getOrAddProperties<Properties>().case_values = case_values;
  }
  odsState.addSuccessors(defaultDestination);
  odsState.addSuccessors(caseDestinations);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<SwitchOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult SwitchOp::verifyInvariantsImpl() {
  auto tblgen_case_operand_segments = getProperties().case_operand_segments; (void)tblgen_case_operand_segments;
  if (!tblgen_case_operand_segments) return emitOpError("requires attribute 'case_operand_segments'");
  auto tblgen_case_values = getProperties().case_values; (void)tblgen_case_values;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ControlFlowOps2(*this, tblgen_case_values, "case_values")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ControlFlowOps3(*this, tblgen_case_operand_segments, "case_operand_segments")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ControlFlowOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ControlFlowOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);
    if (::mlir::failed(::mlir::OpTrait::impl::verifyValueSizeAttr(*this, "case_operand_segments", "caseOperands", valueGroup2.size())))
      return ::mlir::failure();

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ControlFlowOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult SwitchOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand flagRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> flagOperands(&flagRawOperand, 1);  ::llvm::SMLoc flagOperandsLoc;
  (void)flagOperandsLoc;
  ::mlir::Type flagRawType{};
  ::llvm::ArrayRef<::mlir::Type> flagTypes(&flagRawType, 1);
  ::mlir::Block *defaultDestinationSuccessor = nullptr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> defaultOperandsOperands;
  ::llvm::SMLoc defaultOperandsOperandsLoc;
  (void)defaultOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> defaultOperandsTypes;
  ::mlir::DenseIntElementsAttr case_valuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> caseDestinationsSuccessors;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> caseOperandsOperands;
    llvm::SmallVector<int32_t> caseOperandsOperandGroupSizes;
  ::llvm::SMLoc caseOperandsOperandsLoc;
  (void)caseOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> caseOperandsTypes;

  flagOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(flagRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    flagRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();
  {
    defaultOperandsOperandsLoc = parser.getCurrentLocation();
    caseOperandsOperandsLoc = parser.getCurrentLocation();
    ::llvm::SmallVector<::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand>> caseOperandsOperandGroups;
    ::llvm::SmallVector<llvm::SmallVector<::mlir::Type>> caseOperandsTypeGroups;
    auto odsResult = parseSwitchOpCases(parser, flagRawType, defaultDestinationSuccessor, defaultOperandsOperands, defaultOperandsTypes, case_valuesAttr, caseDestinationsSuccessors, caseOperandsOperandGroups, caseOperandsTypeGroups);
    if (odsResult) return ::mlir::failure();
    if (case_valuesAttr)
      result.getOrAddProperties<SwitchOp::Properties>().case_values = case_valuesAttr;
    for (const auto &subRange : caseOperandsOperandGroups) {
      caseOperandsOperands.append(subRange.begin(), subRange.end());
      caseOperandsOperandGroupSizes.push_back(subRange.size());
    }
    for (const auto &subRange : caseOperandsTypeGroups)
      caseOperandsTypes.append(subRange.begin(), subRange.end());
  }
  if (parser.parseRSquare())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  result.addSuccessors(defaultDestinationSuccessor);
  result.addSuccessors(caseDestinationsSuccessors);
::llvm::copy(::llvm::ArrayRef<int32_t>({1, static_cast<int32_t>(defaultOperandsOperands.size()), static_cast<int32_t>(caseOperandsOperands.size())}), result.getOrAddProperties<SwitchOp::Properties>().operandSegmentSizes.begin());
  result.getOrAddProperties<SwitchOp::Properties>().case_operand_segments = parser.getBuilder().getDenseI32ArrayAttr(caseOperandsOperandGroupSizes);
  if (parser.resolveOperands(flagOperands, flagTypes, flagOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(defaultOperandsOperands, defaultOperandsTypes, defaultOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(caseOperandsOperands, caseOperandsTypes, caseOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getFlag();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getFlag().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ' << "[";
  _odsPrinter.printNewline();
  printSwitchOpCases(_odsPrinter, *this, getFlag().getType(), getDefaultDestination(), getDefaultOperands(), getDefaultOperands().getTypes(), getCaseValuesAttr(), getCaseDestinations(), getCaseOperands(), getCaseOperands().getTypes());
  _odsPrinter << "]";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operandSegmentSizes");
  elidedAttrs.push_back("case_operand_segments");
  elidedAttrs.push_back("case_values");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void SwitchOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace cf
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::cf::SwitchOp)


#endif  // GET_OP_CLASSES

