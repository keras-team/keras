/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: InterpreterOps.td                                                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::stablehlo::interpreter::ProbeOp,
::mlir::stablehlo::interpreter::RunParallelOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace stablehlo {
namespace interpreter {

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_InterpreterOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::RankedTensorType>(type))) && ([](::mlir::Type elementType) { return (((elementType.isFloat8E3M4())) || ((elementType.isFloat8E4M3())) || ((elementType.isFloat8E4M3FN())) || ((elementType.isFloat8E4M3FNUZ())) || ((elementType.isFloat8E4M3B11FNUZ())) || ((elementType.isFloat8E5M2())) || ((elementType.isFloat8E5M2FNUZ())) || ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())) || ((elementType.isBF16()))) || ((elementType.isSignlessInteger(1))) || ((((elementType.isSignlessInteger(2))) || ((elementType.isSignlessInteger(4))) || ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64)))) || (((elementType.isUnsignedInteger(2))) || ((elementType.isUnsignedInteger(4))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || ((elementType.isUnsignedInteger(32))) || ((elementType.isUnsignedInteger(64))))) || (((::llvm::isa<::mlir::ComplexType>(elementType))) && (((::llvm::cast<::mlir::ComplexType>(elementType).getElementType().isF32())) || ((::llvm::cast<::mlir::ComplexType>(elementType).getElementType().isF64())))) || (((((isa<mlir::quant::UniformQuantizedType>(elementType))) && ((cast<mlir::quant::UniformQuantizedType>(elementType).getStorageTypeIntegralWidth() == 2)) && ((cast<mlir::IntegerType>(cast<mlir::quant::UniformQuantizedType>(elementType).getStorageType()).isSignless())) && ((mlir::hlo::isValidStablehloQuantizedElementType(elementType)))) || (((isa<mlir::quant::UniformQuantizedType>(elementType))) && ((cast<mlir::quant::UniformQuantizedType>(elementType).getStorageTypeIntegralWidth() == 4)) && ((cast<mlir::IntegerType>(cast<mlir::quant::UniformQuantizedType>(elementType).getStorageType()).isSignless())) && ((mlir::hlo::isValidStablehloQuantizedElementType(elementType)))) || (((isa<mlir::quant::UniformQuantizedType>(elementType))) && ((cast<mlir::quant::UniformQuantizedType>(elementType).getStorageTypeIntegralWidth() == 8)) && ((cast<mlir::IntegerType>(cast<mlir::quant::UniformQuantizedType>(elementType).getStorageType()).isSignless())) && ((mlir::hlo::isValidStablehloQuantizedElementType(elementType)))) || (((isa<mlir::quant::UniformQuantizedType>(elementType))) && ((cast<mlir::quant::UniformQuantizedType>(elementType).getStorageTypeIntegralWidth() == 16)) && ((cast<mlir::IntegerType>(cast<mlir::quant::UniformQuantizedType>(elementType).getStorageType()).isSignless())) && ((mlir::hlo::isValidStablehloQuantizedElementType(elementType)))) || (((isa<mlir::quant::UniformQuantizedType>(elementType))) && ((cast<mlir::quant::UniformQuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)) && ((cast<mlir::IntegerType>(cast<mlir::quant::UniformQuantizedType>(elementType).getStorageType()).isSignless())) && ((mlir::hlo::isValidStablehloQuantizedElementType(elementType))))) || ((((isa<mlir::quant::UniformQuantizedType>(elementType))) && ((cast<mlir::quant::UniformQuantizedType>(elementType).getStorageTypeIntegralWidth() == 2)) && ((!cast<mlir::quant::UniformQuantizedType>(elementType).isSigned())) && ((mlir::hlo::isValidStablehloQuantizedElementType(elementType)))) || (((isa<mlir::quant::UniformQuantizedType>(elementType))) && ((cast<mlir::quant::UniformQuantizedType>(elementType).getStorageTypeIntegralWidth() == 4)) && ((!cast<mlir::quant::UniformQuantizedType>(elementType).isSigned())) && ((mlir::hlo::isValidStablehloQuantizedElementType(elementType)))) || (((isa<mlir::quant::UniformQuantizedType>(elementType))) && ((cast<mlir::quant::UniformQuantizedType>(elementType).getStorageTypeIntegralWidth() == 8)) && ((!cast<mlir::quant::UniformQuantizedType>(elementType).isSigned())) && ((mlir::hlo::isValidStablehloQuantizedElementType(elementType)))) || (((isa<mlir::quant::UniformQuantizedType>(elementType))) && ((cast<mlir::quant::UniformQuantizedType>(elementType).getStorageTypeIntegralWidth() == 16)) && ((!cast<mlir::quant::UniformQuantizedType>(elementType).isSigned())) && ((mlir::hlo::isValidStablehloQuantizedElementType(elementType)))) || (((isa<mlir::quant::UniformQuantizedType>(elementType))) && ((cast<mlir::quant::UniformQuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)) && ((!cast<mlir::quant::UniformQuantizedType>(elementType).isSigned())) && ((mlir::hlo::isValidStablehloQuantizedElementType(elementType)))))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be ranked tensor of f8E3M4 type or f8E4M3 type or f8E4M3FN type or f8E4M3FNUZ type or f8E4M3B11FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 2/4/8/16/32/64-bit signless integer or 2/4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 2/4/8/16/32-bit uniform quantized signed integer or 2/4/8/16/32-bit uniform quantized unsigned integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_InterpreterOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::RankedTensorType>(type))) && ([](::mlir::Type elementType) { return (((elementType.isFloat8E3M4())) || ((elementType.isFloat8E4M3())) || ((elementType.isFloat8E4M3FN())) || ((elementType.isFloat8E4M3FNUZ())) || ((elementType.isFloat8E4M3B11FNUZ())) || ((elementType.isFloat8E5M2())) || ((elementType.isFloat8E5M2FNUZ())) || ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())) || ((elementType.isBF16()))) || ((elementType.isSignlessInteger(1))) || ((((elementType.isSignlessInteger(2))) || ((elementType.isSignlessInteger(4))) || ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64)))) || (((elementType.isUnsignedInteger(2))) || ((elementType.isUnsignedInteger(4))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || ((elementType.isUnsignedInteger(32))) || ((elementType.isUnsignedInteger(64))))) || (((::llvm::isa<::mlir::ComplexType>(elementType))) && (((::llvm::cast<::mlir::ComplexType>(elementType).getElementType().isF32())) || ((::llvm::cast<::mlir::ComplexType>(elementType).getElementType().isF64())))) || (((((isa<mlir::quant::UniformQuantizedType>(elementType))) && ((cast<mlir::quant::UniformQuantizedType>(elementType).getStorageTypeIntegralWidth() == 2)) && ((cast<mlir::IntegerType>(cast<mlir::quant::UniformQuantizedType>(elementType).getStorageType()).isSignless())) && ((mlir::hlo::isValidStablehloQuantizedElementType(elementType)))) || (((isa<mlir::quant::UniformQuantizedType>(elementType))) && ((cast<mlir::quant::UniformQuantizedType>(elementType).getStorageTypeIntegralWidth() == 4)) && ((cast<mlir::IntegerType>(cast<mlir::quant::UniformQuantizedType>(elementType).getStorageType()).isSignless())) && ((mlir::hlo::isValidStablehloQuantizedElementType(elementType)))) || (((isa<mlir::quant::UniformQuantizedType>(elementType))) && ((cast<mlir::quant::UniformQuantizedType>(elementType).getStorageTypeIntegralWidth() == 8)) && ((cast<mlir::IntegerType>(cast<mlir::quant::UniformQuantizedType>(elementType).getStorageType()).isSignless())) && ((mlir::hlo::isValidStablehloQuantizedElementType(elementType)))) || (((isa<mlir::quant::UniformQuantizedType>(elementType))) && ((cast<mlir::quant::UniformQuantizedType>(elementType).getStorageTypeIntegralWidth() == 16)) && ((cast<mlir::IntegerType>(cast<mlir::quant::UniformQuantizedType>(elementType).getStorageType()).isSignless())) && ((mlir::hlo::isValidStablehloQuantizedElementType(elementType)))) || (((isa<mlir::quant::UniformQuantizedType>(elementType))) && ((cast<mlir::quant::UniformQuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)) && ((cast<mlir::IntegerType>(cast<mlir::quant::UniformQuantizedType>(elementType).getStorageType()).isSignless())) && ((mlir::hlo::isValidStablehloQuantizedElementType(elementType))))) || ((((isa<mlir::quant::UniformQuantizedType>(elementType))) && ((cast<mlir::quant::UniformQuantizedType>(elementType).getStorageTypeIntegralWidth() == 2)) && ((!cast<mlir::quant::UniformQuantizedType>(elementType).isSigned())) && ((mlir::hlo::isValidStablehloQuantizedElementType(elementType)))) || (((isa<mlir::quant::UniformQuantizedType>(elementType))) && ((cast<mlir::quant::UniformQuantizedType>(elementType).getStorageTypeIntegralWidth() == 4)) && ((!cast<mlir::quant::UniformQuantizedType>(elementType).isSigned())) && ((mlir::hlo::isValidStablehloQuantizedElementType(elementType)))) || (((isa<mlir::quant::UniformQuantizedType>(elementType))) && ((cast<mlir::quant::UniformQuantizedType>(elementType).getStorageTypeIntegralWidth() == 8)) && ((!cast<mlir::quant::UniformQuantizedType>(elementType).isSigned())) && ((mlir::hlo::isValidStablehloQuantizedElementType(elementType)))) || (((isa<mlir::quant::UniformQuantizedType>(elementType))) && ((cast<mlir::quant::UniformQuantizedType>(elementType).getStorageTypeIntegralWidth() == 16)) && ((!cast<mlir::quant::UniformQuantizedType>(elementType).isSigned())) && ((mlir::hlo::isValidStablehloQuantizedElementType(elementType)))) || (((isa<mlir::quant::UniformQuantizedType>(elementType))) && ((cast<mlir::quant::UniformQuantizedType>(elementType).getStorageTypeIntegralWidth() == 32)) && ((!cast<mlir::quant::UniformQuantizedType>(elementType).isSigned())) && ((mlir::hlo::isValidStablehloQuantizedElementType(elementType)))))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || ((isa<TokenType>(type))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of ranked tensor of f8E3M4 type or f8E4M3 type or f8E4M3FN type or f8E4M3FNUZ type or f8E4M3B11FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or pred (AKA boolean or 1-bit integer) or 2/4/8/16/32/64-bit signless integer or 2/4/8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements or 2/4/8/16/32-bit uniform quantized signed integer or 2/4/8/16/32-bit uniform quantized unsigned integer values or token, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_InterpreterOps1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::StringAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: string attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_InterpreterOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_InterpreterOps1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_InterpreterOps2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::FlatSymbolRefAttr>(attr))); }))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: flat symbol ref array attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_InterpreterOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_InterpreterOps2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_InterpreterOps3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::FlatSymbolRefAttr>(attr))); }))); }))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: Array of FlatSymbolRefArrayAttr";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_InterpreterOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_InterpreterOps3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
} // namespace interpreter
} // namespace stablehlo
} // namespace mlir
namespace mlir {
namespace stablehlo {
namespace interpreter {

//===----------------------------------------------------------------------===//
// ::mlir::stablehlo::interpreter::ProbeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ProbeOpGenericAdaptorBase::ProbeOpGenericAdaptorBase(ProbeOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::llvm::StringRef ProbeOpGenericAdaptorBase::getProbeId() {
  auto attr = getProbeIdAttr();
  return attr.getValue();
}

} // namespace detail
ProbeOpAdaptor::ProbeOpAdaptor(ProbeOp op) : ProbeOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ProbeOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_probe_id = getProperties().probe_id; (void)tblgen_probe_id;
  if (!tblgen_probe_id) return emitError(loc, "'interpreter.probe' op ""requires attribute 'probe_id'");

  if (tblgen_probe_id && !((::llvm::isa<::mlir::StringAttr>(tblgen_probe_id))))
    return emitError(loc, "'interpreter.probe' op ""attribute 'probe_id' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

::llvm::LogicalResult ProbeOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.probe_id;
       auto attr = dict.get("probe_id");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `probe_id` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ProbeOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.probe_id;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("probe_id",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ProbeOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.probe_id.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ProbeOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "probe_id")
      return prop.probe_id;
  return std::nullopt;
}

void ProbeOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "probe_id") {
       prop.probe_id = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.probe_id)>>(value);
       return;
    }
}

void ProbeOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.probe_id) attrs.append("probe_id", prop.probe_id);
}

::llvm::LogicalResult ProbeOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getProbeIdAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_InterpreterOps1(attr, "probe_id", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult ProbeOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.probe_id)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ProbeOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.probe_id);
}

::llvm::StringRef ProbeOp::getProbeId() {
  auto attr = getProbeIdAttr();
  return attr.getValue();
}

void ProbeOp::setProbeId(::llvm::StringRef attrValue) {
  getProperties().probe_id = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void ProbeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand, ::mlir::StringAttr probe_id) {
  odsState.addOperands(operand);
  odsState.getOrAddProperties<Properties>().probe_id = probe_id;
  odsState.addTypes(result);
}

void ProbeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::StringAttr probe_id) {
  odsState.addOperands(operand);
  odsState.getOrAddProperties<Properties>().probe_id = probe_id;

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ProbeOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ProbeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::StringAttr probe_id) {
  odsState.addOperands(operand);
  odsState.getOrAddProperties<Properties>().probe_id = probe_id;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ProbeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand, ::llvm::StringRef probe_id) {
  odsState.addOperands(operand);
  odsState.getOrAddProperties<Properties>().probe_id = odsBuilder.getStringAttr(probe_id);
  odsState.addTypes(result);
}

void ProbeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::llvm::StringRef probe_id) {
  odsState.addOperands(operand);
  odsState.getOrAddProperties<Properties>().probe_id = odsBuilder.getStringAttr(probe_id);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ProbeOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ProbeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::llvm::StringRef probe_id) {
  odsState.addOperands(operand);
  odsState.getOrAddProperties<Properties>().probe_id = odsBuilder.getStringAttr(probe_id);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ProbeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ProbeOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void ProbeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ProbeOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ProbeOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::llvm::LogicalResult ProbeOp::verifyInvariantsImpl() {
  auto tblgen_probe_id = getProperties().probe_id; (void)tblgen_probe_id;
  if (!tblgen_probe_id) return emitOpError("requires attribute 'probe_id'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_InterpreterOps1(*this, tblgen_probe_id, "probe_id")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_InterpreterOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_InterpreterOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ProbeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult ProbeOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult ProbeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(&operandRawOperand, 1);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::StringAttr probe_idAttr;
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();
  if (parser.parseKeyword("probe_id"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(probe_idAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (probe_idAttr) result.getOrAddProperties<ProbeOp::Properties>().probe_id = probe_idAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawType = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(operandOperands, resultTypes[0], operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ProbeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter << ",";
  _odsPrinter << ' ' << "probe_id";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getProbeIdAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("probe_id");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::RankedTensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace interpreter
} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::interpreter::ProbeOp)

namespace mlir {
namespace stablehlo {
namespace interpreter {

//===----------------------------------------------------------------------===//
// ::mlir::stablehlo::interpreter::RunParallelOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
RunParallelOpGenericAdaptorBase::RunParallelOpGenericAdaptorBase(RunParallelOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> RunParallelOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::std::optional< ::mlir::ArrayAttr > RunParallelOpGenericAdaptorBase::getInfeed() {
  auto attr = getInfeedAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr RunParallelOpGenericAdaptorBase::getPrograms() {
  auto attr = getProgramsAttr();
  return attr;
}

} // namespace detail
RunParallelOpAdaptor::RunParallelOpAdaptor(RunParallelOp op) : RunParallelOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult RunParallelOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_infeed = getProperties().infeed; (void)tblgen_infeed;
  auto tblgen_programs = getProperties().programs; (void)tblgen_programs;
  if (!tblgen_programs) return emitError(loc, "'interpreter.run_parallel' op ""requires attribute 'programs'");

  if (tblgen_infeed && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_infeed))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_infeed), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::FlatSymbolRefAttr>(attr))); }))))
    return emitError(loc, "'interpreter.run_parallel' op ""attribute 'infeed' failed to satisfy constraint: flat symbol ref array attribute");

  if (tblgen_programs && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_programs))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_programs), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::FlatSymbolRefAttr>(attr))); }))); }))))
    return emitError(loc, "'interpreter.run_parallel' op ""attribute 'programs' failed to satisfy constraint: Array of FlatSymbolRefArrayAttr");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> RunParallelOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::MutableOperandRange RunParallelOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> RunParallelOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::llvm::LogicalResult RunParallelOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.infeed;
       auto attr = dict.get("infeed");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `infeed` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.programs;
       auto attr = dict.get("programs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `programs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute RunParallelOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.infeed;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("infeed",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.programs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("programs",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code RunParallelOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.infeed.getAsOpaquePointer()), 
    llvm::hash_value(prop.programs.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> RunParallelOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "infeed")
      return prop.infeed;

    if (name == "programs")
      return prop.programs;
  return std::nullopt;
}

void RunParallelOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "infeed") {
       prop.infeed = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.infeed)>>(value);
       return;
    }

    if (name == "programs") {
       prop.programs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.programs)>>(value);
       return;
    }
}

void RunParallelOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.infeed) attrs.append("infeed", prop.infeed);

    if (prop.programs) attrs.append("programs", prop.programs);
}

::llvm::LogicalResult RunParallelOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getInfeedAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_InterpreterOps2(attr, "infeed", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getProgramsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_InterpreterOps3(attr, "programs", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult RunParallelOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.infeed)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.programs)))
    return ::mlir::failure();
  return ::mlir::success();
}

void RunParallelOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.infeed);
  writer.writeAttribute(prop.programs);
}

::std::optional< ::mlir::ArrayAttr > RunParallelOp::getInfeed() {
  auto attr = getInfeedAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr RunParallelOp::getPrograms() {
  auto attr = getProgramsAttr();
  return attr;
}

void RunParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::ValueRange inputs, /*optional*/::mlir::ArrayAttr infeed, ::mlir::ArrayAttr programs) {
  odsState.addOperands(inputs);
  if (infeed) {
    odsState.getOrAddProperties<Properties>().infeed = infeed;
  }
  odsState.getOrAddProperties<Properties>().programs = programs;
  odsState.addTypes(results);
}

void RunParallelOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<RunParallelOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult RunParallelOp::verifyInvariantsImpl() {
  auto tblgen_infeed = getProperties().infeed; (void)tblgen_infeed;
  auto tblgen_programs = getProperties().programs; (void)tblgen_programs;
  if (!tblgen_programs) return emitOpError("requires attribute 'programs'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_InterpreterOps2(*this, tblgen_infeed, "infeed")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_InterpreterOps3(*this, tblgen_programs, "programs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_InterpreterOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_InterpreterOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult RunParallelOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace interpreter
} // namespace stablehlo
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::stablehlo::interpreter::RunParallelOp)


#endif  // GET_OP_CLASSES

