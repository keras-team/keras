/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: AffineOps.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::affine::AffineApplyOp,
::mlir::affine::AffineDelinearizeIndexOp,
::mlir::affine::AffineForOp,
::mlir::affine::AffineIfOp,
::mlir::affine::AffineLoadOp,
::mlir::affine::AffineMaxOp,
::mlir::affine::AffineMinOp,
::mlir::affine::AffineParallelOp,
::mlir::affine::AffinePrefetchOp,
::mlir::affine::AffineStoreOp,
::mlir::affine::AffineVectorLoadOp,
::mlir::affine::AffineVectorStoreOp,
::mlir::affine::AffineYieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace affine {

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_AffineOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::IndexType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of index, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_AffineOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::IndexType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be index, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_AffineOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of any type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_AffineOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::MemRefType>(type))) && ([](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_AffineOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_AffineOps6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::VectorType>(type))) && ((::llvm::cast<::mlir::VectorType>(type).getRank() > 0))) && ([](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be vector of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_AffineOps1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::AffineMapAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: AffineMap attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_AffineOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_AffineOps1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_AffineOps2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::isa<::mlir::IndexType>(::llvm::cast<::mlir::IntegerAttr>(attr).getType())))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: index attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_AffineOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_AffineOps2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_AffineOps3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::arith::AtomicRMWKindAttr>(attr))); }))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: Reduction ops";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_AffineOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_AffineOps3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_AffineOps4(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::DenseIntElementsAttr>(attr))) && ((::llvm::cast<::mlir::DenseIntElementsAttr>(attr).getType().getElementType().isSignlessInteger(32)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit signless integer elements attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_AffineOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_AffineOps4(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_AffineOps5(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))); }))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 64-bit integer array attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_AffineOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_AffineOps5(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_AffineOps6(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::BoolAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: bool attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_AffineOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_AffineOps6(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_AffineOps7(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() >= 0)) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getInt() <= 3))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0 whose maximum value is 3";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_AffineOps7(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_AffineOps7(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_region_constraint_AffineOps1(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItems(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with 1 blocks";
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_region_constraint_AffineOps2(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineApplyOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineApplyOpGenericAdaptorBase::AffineApplyOpGenericAdaptorBase(AffineApplyOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> AffineApplyOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::AffineMap AffineApplyOpGenericAdaptorBase::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

} // namespace detail
AffineApplyOpAdaptor::AffineApplyOpAdaptor(AffineApplyOp op) : AffineApplyOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AffineApplyOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_map = getProperties().map; (void)tblgen_map;
  if (!tblgen_map) return emitError(loc, "'affine.apply' op ""requires attribute 'map'");

  if (tblgen_map && !((::llvm::isa<::mlir::AffineMapAttr>(tblgen_map))))
    return emitError(loc, "'affine.apply' op ""attribute 'map' failed to satisfy constraint: AffineMap attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineApplyOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::MutableOperandRange AffineApplyOp::getMapOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult AffineApplyOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.map;
       auto attr = dict.get("map");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `map` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute AffineApplyOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.map;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("map",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code AffineApplyOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.map.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> AffineApplyOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "map")
      return prop.map;
  return std::nullopt;
}

void AffineApplyOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "map") {
       prop.map = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.map)>>(value);
       return;
    }
}

void AffineApplyOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.map) attrs.append("map", prop.map);
}

::llvm::LogicalResult AffineApplyOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getMapAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(attr, "map", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult AffineApplyOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.map)))
    return ::mlir::failure();
  return ::mlir::success();
}

void AffineApplyOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.map);
}

::mlir::AffineMap AffineApplyOp::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

void AffineApplyOp::setMap(::mlir::AffineMap attrValue) {
  getProperties().map = ::mlir::AffineMapAttr::get(attrValue);
}

void AffineApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<AffineExpr>  exprList, ValueRange mapOperands) {
      build(odsBuilder, odsState, odsBuilder.getIndexType(),
            AffineMap::inferFromExprList(exprList, odsBuilder.getContext())
                                        .front(), mapOperands);
    
}

void AffineApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMapAttr map, ::mlir::ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.getOrAddProperties<Properties>().map = map;
  odsState.addTypes(resultType0);
}

void AffineApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::AffineMapAttr map, ::mlir::ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.getOrAddProperties<Properties>().map = map;

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AffineApplyOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AffineApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMapAttr map, ::mlir::ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.getOrAddProperties<Properties>().map = map;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMap map, ::mlir::ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.getOrAddProperties<Properties>().map = ::mlir::AffineMapAttr::get(map);
  odsState.addTypes(resultType0);
}

void AffineApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::AffineMap map, ::mlir::ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.getOrAddProperties<Properties>().map = ::mlir::AffineMapAttr::get(map);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AffineApplyOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AffineApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMap map, ::mlir::ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.getOrAddProperties<Properties>().map = ::mlir::AffineMapAttr::get(map);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineApplyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<AffineApplyOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void AffineApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<AffineApplyOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AffineApplyOp::inferReturnTypes(odsBuilder.getContext(),
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

::llvm::LogicalResult AffineApplyOp::verifyInvariantsImpl() {
  auto tblgen_map = getProperties().map; (void)tblgen_map;
  if (!tblgen_map) return emitOpError("requires attribute 'map'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(*this, tblgen_map, "map")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult AffineApplyOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::llvm::LogicalResult AffineApplyOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = odsBuilder.getIndexType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void AffineApplyOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineApplyOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineDelinearizeIndexOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
std::pair<unsigned, unsigned> AffineDelinearizeIndexOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

} // namespace detail
AffineDelinearizeIndexOpAdaptor::AffineDelinearizeIndexOpAdaptor(AffineDelinearizeIndexOp op) : AffineDelinearizeIndexOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AffineDelinearizeIndexOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineDelinearizeIndexOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange AffineDelinearizeIndexOp::getBasisMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AffineDelinearizeIndexOp::getODSResultIndexAndLength(unsigned index) {
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

void AffineDelinearizeIndexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange multi_index, ::mlir::Value linear_index, ::mlir::ValueRange basis) {
  odsState.addOperands(linear_index);
  odsState.addOperands(basis);
  odsState.addTypes(multi_index);
}

void AffineDelinearizeIndexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value linear_index, ::mlir::ValueRange basis) {
  odsState.addOperands(linear_index);
  odsState.addOperands(basis);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AffineDelinearizeIndexOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AffineDelinearizeIndexOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

void AffineDelinearizeIndexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AffineDelinearizeIndexOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::llvm::LogicalResult AffineDelinearizeIndexOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult AffineDelinearizeIndexOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AffineDelinearizeIndexOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand linear_indexRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> linear_indexOperands(&linear_indexRawOperand, 1);  ::llvm::SMLoc linear_indexOperandsLoc;
  (void)linear_indexOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> basisOperands;
  ::llvm::SMLoc basisOperandsLoc;
  (void)basisOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> multi_indexTypes;

  linear_indexOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(linear_indexRawOperand))
    return ::mlir::failure();
  if (parser.parseKeyword("into"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  basisOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(basisOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(multi_indexTypes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(multi_indexTypes);
  if (parser.resolveOperands(linear_indexOperands, odsBuildableType0, linear_indexOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(basisOperands, odsBuildableType0, basisOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AffineDelinearizeIndexOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLinearIndex();
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  _odsPrinter << "(";
  _odsPrinter << getBasis();
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getMultiIndex().getTypes();
}

void AffineDelinearizeIndexOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineDelinearizeIndexOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineForOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineForOpGenericAdaptorBase::AffineForOpGenericAdaptorBase(AffineForOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> AffineForOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::AffineMap AffineForOpGenericAdaptorBase::getLowerBoundMap() {
  auto attr = getLowerBoundMapAttr();
  return attr.getValue();
}

::mlir::AffineMap AffineForOpGenericAdaptorBase::getUpperBoundMap() {
  auto attr = getUpperBoundMapAttr();
  return attr.getValue();
}

::llvm::APInt AffineForOpGenericAdaptorBase::getStep() {
  auto attr = getStepAttr();
  return attr.getValue();
}

} // namespace detail
AffineForOpAdaptor::AffineForOpAdaptor(AffineForOp op) : AffineForOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AffineForOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_lowerBoundMap = getProperties().lowerBoundMap; (void)tblgen_lowerBoundMap;
  if (!tblgen_lowerBoundMap) return emitError(loc, "'affine.for' op ""requires attribute 'lowerBoundMap'");
  auto tblgen_step = getProperties().step; (void)tblgen_step;
  if (!tblgen_step) return emitError(loc, "'affine.for' op ""requires attribute 'step'");
  auto tblgen_upperBoundMap = getProperties().upperBoundMap; (void)tblgen_upperBoundMap;
  if (!tblgen_upperBoundMap) return emitError(loc, "'affine.for' op ""requires attribute 'upperBoundMap'");

  if (tblgen_lowerBoundMap && !((::llvm::isa<::mlir::AffineMapAttr>(tblgen_lowerBoundMap))))
    return emitError(loc, "'affine.for' op ""attribute 'lowerBoundMap' failed to satisfy constraint: AffineMap attribute");

  if (tblgen_upperBoundMap && !((::llvm::isa<::mlir::AffineMapAttr>(tblgen_upperBoundMap))))
    return emitError(loc, "'affine.for' op ""attribute 'upperBoundMap' failed to satisfy constraint: AffineMap attribute");

  if (tblgen_step && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_step))) && ((::llvm::isa<::mlir::IndexType>(::llvm::cast<::mlir::IntegerAttr>(tblgen_step).getType())))))
    return emitError(loc, "'affine.for' op ""attribute 'step' failed to satisfy constraint: index attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineForOp::getODSOperandIndexAndLength(unsigned index) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::MutableOperandRange AffineForOp::getLowerBoundOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::mlir::MutableOperandRange AffineForOp::getUpperBoundOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::mlir::MutableOperandRange AffineForOp::getInitsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

std::pair<unsigned, unsigned> AffineForOp::getODSResultIndexAndLength(unsigned index) {
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

::llvm::LogicalResult AffineForOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.lowerBoundMap;
       auto attr = dict.get("lowerBoundMap");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `lowerBoundMap` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.step;
       auto attr = dict.get("step");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `step` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.upperBoundMap;
       auto attr = dict.get("upperBoundMap");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `upperBoundMap` in property conversion: " << attr;
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

::mlir::Attribute AffineForOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.lowerBoundMap;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("lowerBoundMap",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.step;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("step",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.upperBoundMap;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("upperBoundMap",
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

llvm::hash_code AffineForOp::computePropertiesHash(const Properties &prop) {
  auto hash_operandSegmentSizes = [] (const auto &propStorage) -> llvm::hash_code {
    return ::llvm::hash_combine_range(std::begin(propStorage), std::end(propStorage));;
  };
  return llvm::hash_combine(
    llvm::hash_value(prop.lowerBoundMap.getAsOpaquePointer()), 
    llvm::hash_value(prop.step.getAsOpaquePointer()), 
    llvm::hash_value(prop.upperBoundMap.getAsOpaquePointer()), 
    hash_operandSegmentSizes(prop.operandSegmentSizes));
}

std::optional<mlir::Attribute> AffineForOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "lowerBoundMap")
      return prop.lowerBoundMap;

    if (name == "step")
      return prop.step;

    if (name == "upperBoundMap")
      return prop.upperBoundMap;
    if (name == "operand_segment_sizes" || name == "operandSegmentSizes") return [&]() -> ::mlir::Attribute { return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes); }();
  return std::nullopt;
}

void AffineForOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "lowerBoundMap") {
       prop.lowerBoundMap = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.lowerBoundMap)>>(value);
       return;
    }

    if (name == "step") {
       prop.step = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.step)>>(value);
       return;
    }

    if (name == "upperBoundMap") {
       prop.upperBoundMap = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.upperBoundMap)>>(value);
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

void AffineForOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.lowerBoundMap) attrs.append("lowerBoundMap", prop.lowerBoundMap);

    if (prop.step) attrs.append("step", prop.step);

    if (prop.upperBoundMap) attrs.append("upperBoundMap", prop.upperBoundMap);
  attrs.append("operandSegmentSizes", [&]() -> ::mlir::Attribute { return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes); }());
}

::llvm::LogicalResult AffineForOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getLowerBoundMapAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(attr, "lowerBoundMap", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getStepAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps2(attr, "step", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getUpperBoundMapAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(attr, "upperBoundMap", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult AffineForOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.lowerBoundMap)))
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

  if (::mlir::failed(reader.readAttribute(prop.step)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.upperBoundMap)))
    return ::mlir::failure();

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

void AffineForOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.lowerBoundMap);

if (writer.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
  auto &propStorage = prop.operandSegmentSizes;
  writer.writeAttribute(::mlir::DenseI32ArrayAttr::get(this->getContext(), propStorage));
}
  writer.writeAttribute(prop.step);
  writer.writeAttribute(prop.upperBoundMap);

  {
    auto &propStorage = prop.operandSegmentSizes;

  if (writer.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    writer.writeSparseArray(::llvm::ArrayRef(propStorage));
;
  }
}

::mlir::AffineMap AffineForOp::getLowerBoundMap() {
  auto attr = getLowerBoundMapAttr();
  return attr.getValue();
}

::mlir::AffineMap AffineForOp::getUpperBoundMap() {
  auto attr = getUpperBoundMapAttr();
  return attr.getValue();
}

::llvm::APInt AffineForOp::getStep() {
  auto attr = getStepAttr();
  return attr.getValue();
}

void AffineForOp::setLowerBoundMap(::mlir::AffineMap attrValue) {
  getProperties().lowerBoundMap = ::mlir::AffineMapAttr::get(attrValue);
}

void AffineForOp::setUpperBoundMap(::mlir::AffineMap attrValue) {
  getProperties().upperBoundMap = ::mlir::AffineMapAttr::get(attrValue);
}

void AffineForOp::setStep(::llvm::APInt attrValue) {
  getProperties().step = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIndexType(), attrValue);
}

::llvm::LogicalResult AffineForOp::verifyInvariantsImpl() {
  auto tblgen_lowerBoundMap = getProperties().lowerBoundMap; (void)tblgen_lowerBoundMap;
  if (!tblgen_lowerBoundMap) return emitOpError("requires attribute 'lowerBoundMap'");
  auto tblgen_step = getProperties().step; (void)tblgen_step;
  if (!tblgen_step) return emitOpError("requires attribute 'step'");
  auto tblgen_upperBoundMap = getProperties().upperBoundMap; (void)tblgen_upperBoundMap;
  if (!tblgen_upperBoundMap) return emitOpError("requires attribute 'upperBoundMap'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(*this, tblgen_lowerBoundMap, "lowerBoundMap")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(*this, tblgen_upperBoundMap, "upperBoundMap")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps2(*this, tblgen_step, "step")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_AffineOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::llvm::LogicalResult AffineForOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineForOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineIfOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
std::pair<unsigned, unsigned> AffineIfOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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
AffineIfOpAdaptor::AffineIfOpAdaptor(AffineIfOp op) : AffineIfOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AffineIfOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineIfOp::getODSOperandIndexAndLength(unsigned index) {
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

std::pair<unsigned, unsigned> AffineIfOp::getODSResultIndexAndLength(unsigned index) {
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

::llvm::LogicalResult AffineIfOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_AffineOps1(*this, region, "thenRegion", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_AffineOps2(*this, region, "elseRegion", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::llvm::LogicalResult AffineIfOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineIfOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineLoadOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineLoadOpGenericAdaptorBase::AffineLoadOpGenericAdaptorBase(AffineLoadOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> AffineLoadOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::AffineMap AffineLoadOpGenericAdaptorBase::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

} // namespace detail
AffineLoadOpAdaptor::AffineLoadOpAdaptor(AffineLoadOp op) : AffineLoadOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AffineLoadOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_map = getProperties().map; (void)tblgen_map;
  if (!tblgen_map) return emitError(loc, "'affine.load' op ""requires attribute 'map'");

  if (tblgen_map && !((::llvm::isa<::mlir::AffineMapAttr>(tblgen_map))))
    return emitError(loc, "'affine.load' op ""attribute 'map' failed to satisfy constraint: AffineMap attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineLoadOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange AffineLoadOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult AffineLoadOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.map;
       auto attr = dict.get("map");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `map` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute AffineLoadOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.map;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("map",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code AffineLoadOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.map.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> AffineLoadOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "map")
      return prop.map;
  return std::nullopt;
}

void AffineLoadOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "map") {
       prop.map = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.map)>>(value);
       return;
    }
}

void AffineLoadOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.map) attrs.append("map", prop.map);
}

::llvm::LogicalResult AffineLoadOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getMapAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(attr, "map", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult AffineLoadOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.map)))
    return ::mlir::failure();
  return ::mlir::success();
}

void AffineLoadOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.map);
}

::mlir::AffineMap AffineLoadOp::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

void AffineLoadOp::setMap(::mlir::AffineMap attrValue) {
  getProperties().map = ::mlir::AffineMapAttr::get(attrValue);
}

void AffineLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::AffineMapAttr map) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.getOrAddProperties<Properties>().map = map;
  odsState.addTypes(result);
}

void AffineLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::AffineMapAttr map) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.getOrAddProperties<Properties>().map = map;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::AffineMap map) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.getOrAddProperties<Properties>().map = ::mlir::AffineMapAttr::get(map);
  odsState.addTypes(result);
}

void AffineLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::AffineMap map) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.getOrAddProperties<Properties>().map = ::mlir::AffineMapAttr::get(map);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<AffineLoadOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult AffineLoadOp::verifyInvariantsImpl() {
  auto tblgen_map = getProperties().map; (void)tblgen_map;
  if (!tblgen_map) return emitOpError("requires attribute 'map'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(*this, tblgen_map, "map")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult AffineLoadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void AffineLoadOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  {
    auto valueRange = getODSOperandIndexAndLength(0);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::SideEffects::DefaultResource::get());
    }
  }
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineLoadOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineMaxOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineMaxOpGenericAdaptorBase::AffineMaxOpGenericAdaptorBase(AffineMaxOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> AffineMaxOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::AffineMap AffineMaxOpGenericAdaptorBase::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

} // namespace detail
AffineMaxOpAdaptor::AffineMaxOpAdaptor(AffineMaxOp op) : AffineMaxOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AffineMaxOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_map = getProperties().map; (void)tblgen_map;
  if (!tblgen_map) return emitError(loc, "'affine.max' op ""requires attribute 'map'");

  if (tblgen_map && !((::llvm::isa<::mlir::AffineMapAttr>(tblgen_map))))
    return emitError(loc, "'affine.max' op ""attribute 'map' failed to satisfy constraint: AffineMap attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineMaxOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::MutableOperandRange AffineMaxOp::getOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult AffineMaxOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.map;
       auto attr = dict.get("map");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `map` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute AffineMaxOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.map;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("map",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code AffineMaxOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.map.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> AffineMaxOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "map")
      return prop.map;
  return std::nullopt;
}

void AffineMaxOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "map") {
       prop.map = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.map)>>(value);
       return;
    }
}

void AffineMaxOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.map) attrs.append("map", prop.map);
}

::llvm::LogicalResult AffineMaxOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getMapAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(attr, "map", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult AffineMaxOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.map)))
    return ::mlir::failure();
  return ::mlir::success();
}

void AffineMaxOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.map);
}

::mlir::AffineMap AffineMaxOp::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

void AffineMaxOp::setMap(::mlir::AffineMap attrValue) {
  getProperties().map = ::mlir::AffineMapAttr::get(attrValue);
}

void AffineMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.getOrAddProperties<Properties>().map = map;
  odsState.addTypes(resultType0);
}

void AffineMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.getOrAddProperties<Properties>().map = map;

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AffineMaxOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AffineMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.getOrAddProperties<Properties>().map = map;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMap map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.getOrAddProperties<Properties>().map = ::mlir::AffineMapAttr::get(map);
  odsState.addTypes(resultType0);
}

void AffineMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::AffineMap map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.getOrAddProperties<Properties>().map = ::mlir::AffineMapAttr::get(map);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AffineMaxOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AffineMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMap map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.getOrAddProperties<Properties>().map = ::mlir::AffineMapAttr::get(map);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineMaxOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<AffineMaxOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void AffineMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<AffineMaxOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AffineMaxOp::inferReturnTypes(odsBuilder.getContext(),
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

::llvm::LogicalResult AffineMaxOp::verifyInvariantsImpl() {
  auto tblgen_map = getProperties().map; (void)tblgen_map;
  if (!tblgen_map) return emitOpError("requires attribute 'map'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(*this, tblgen_map, "map")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult AffineMaxOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::llvm::LogicalResult AffineMaxOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = odsBuilder.getIndexType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void AffineMaxOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineMaxOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineMinOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineMinOpGenericAdaptorBase::AffineMinOpGenericAdaptorBase(AffineMinOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> AffineMinOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::AffineMap AffineMinOpGenericAdaptorBase::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

} // namespace detail
AffineMinOpAdaptor::AffineMinOpAdaptor(AffineMinOp op) : AffineMinOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AffineMinOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_map = getProperties().map; (void)tblgen_map;
  if (!tblgen_map) return emitError(loc, "'affine.min' op ""requires attribute 'map'");

  if (tblgen_map && !((::llvm::isa<::mlir::AffineMapAttr>(tblgen_map))))
    return emitError(loc, "'affine.min' op ""attribute 'map' failed to satisfy constraint: AffineMap attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineMinOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::MutableOperandRange AffineMinOp::getOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult AffineMinOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.map;
       auto attr = dict.get("map");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `map` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute AffineMinOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.map;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("map",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code AffineMinOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.map.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> AffineMinOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "map")
      return prop.map;
  return std::nullopt;
}

void AffineMinOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "map") {
       prop.map = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.map)>>(value);
       return;
    }
}

void AffineMinOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.map) attrs.append("map", prop.map);
}

::llvm::LogicalResult AffineMinOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getMapAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(attr, "map", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult AffineMinOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.map)))
    return ::mlir::failure();
  return ::mlir::success();
}

void AffineMinOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.map);
}

::mlir::AffineMap AffineMinOp::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

void AffineMinOp::setMap(::mlir::AffineMap attrValue) {
  getProperties().map = ::mlir::AffineMapAttr::get(attrValue);
}

void AffineMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.getOrAddProperties<Properties>().map = map;
  odsState.addTypes(resultType0);
}

void AffineMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.getOrAddProperties<Properties>().map = map;

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AffineMinOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AffineMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.getOrAddProperties<Properties>().map = map;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMap map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.getOrAddProperties<Properties>().map = ::mlir::AffineMapAttr::get(map);
  odsState.addTypes(resultType0);
}

void AffineMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::AffineMap map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.getOrAddProperties<Properties>().map = ::mlir::AffineMapAttr::get(map);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AffineMinOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AffineMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMap map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.getOrAddProperties<Properties>().map = ::mlir::AffineMapAttr::get(map);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineMinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<AffineMinOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void AffineMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<AffineMinOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AffineMinOp::inferReturnTypes(odsBuilder.getContext(),
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

::llvm::LogicalResult AffineMinOp::verifyInvariantsImpl() {
  auto tblgen_map = getProperties().map; (void)tblgen_map;
  if (!tblgen_map) return emitOpError("requires attribute 'map'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(*this, tblgen_map, "map")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult AffineMinOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::llvm::LogicalResult AffineMinOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = odsBuilder.getIndexType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void AffineMinOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineMinOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineParallelOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineParallelOpGenericAdaptorBase::AffineParallelOpGenericAdaptorBase(AffineParallelOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> AffineParallelOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::ArrayAttr AffineParallelOpGenericAdaptorBase::getReductions() {
  auto attr = getReductionsAttr();
  return attr;
}

::mlir::AffineMap AffineParallelOpGenericAdaptorBase::getLowerBoundsMap() {
  auto attr = getLowerBoundsMapAttr();
  return attr.getValue();
}

::mlir::DenseIntElementsAttr AffineParallelOpGenericAdaptorBase::getLowerBoundsGroups() {
  auto attr = getLowerBoundsGroupsAttr();
  return attr;
}

::mlir::AffineMap AffineParallelOpGenericAdaptorBase::getUpperBoundsMap() {
  auto attr = getUpperBoundsMapAttr();
  return attr.getValue();
}

::mlir::DenseIntElementsAttr AffineParallelOpGenericAdaptorBase::getUpperBoundsGroups() {
  auto attr = getUpperBoundsGroupsAttr();
  return attr;
}

::llvm::SmallVector<int64_t, 8> AffineParallelOpGenericAdaptorBase::getSteps() {
  auto attr = getStepsAttr();
  return llvm::to_vector<4>(
      llvm::map_range(attr.getAsRange<mlir::IntegerAttr>(),
      [](mlir::IntegerAttr attr) { return attr.getInt(); }));;
}

} // namespace detail
AffineParallelOpAdaptor::AffineParallelOpAdaptor(AffineParallelOp op) : AffineParallelOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AffineParallelOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_lowerBoundsGroups = getProperties().lowerBoundsGroups; (void)tblgen_lowerBoundsGroups;
  if (!tblgen_lowerBoundsGroups) return emitError(loc, "'affine.parallel' op ""requires attribute 'lowerBoundsGroups'");
  auto tblgen_lowerBoundsMap = getProperties().lowerBoundsMap; (void)tblgen_lowerBoundsMap;
  if (!tblgen_lowerBoundsMap) return emitError(loc, "'affine.parallel' op ""requires attribute 'lowerBoundsMap'");
  auto tblgen_reductions = getProperties().reductions; (void)tblgen_reductions;
  if (!tblgen_reductions) return emitError(loc, "'affine.parallel' op ""requires attribute 'reductions'");
  auto tblgen_steps = getProperties().steps; (void)tblgen_steps;
  if (!tblgen_steps) return emitError(loc, "'affine.parallel' op ""requires attribute 'steps'");
  auto tblgen_upperBoundsGroups = getProperties().upperBoundsGroups; (void)tblgen_upperBoundsGroups;
  if (!tblgen_upperBoundsGroups) return emitError(loc, "'affine.parallel' op ""requires attribute 'upperBoundsGroups'");
  auto tblgen_upperBoundsMap = getProperties().upperBoundsMap; (void)tblgen_upperBoundsMap;
  if (!tblgen_upperBoundsMap) return emitError(loc, "'affine.parallel' op ""requires attribute 'upperBoundsMap'");

  if (tblgen_reductions && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_reductions))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_reductions), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::arith::AtomicRMWKindAttr>(attr))); }))))
    return emitError(loc, "'affine.parallel' op ""attribute 'reductions' failed to satisfy constraint: Reduction ops");

  if (tblgen_lowerBoundsMap && !((::llvm::isa<::mlir::AffineMapAttr>(tblgen_lowerBoundsMap))))
    return emitError(loc, "'affine.parallel' op ""attribute 'lowerBoundsMap' failed to satisfy constraint: AffineMap attribute");

  if (tblgen_lowerBoundsGroups && !(((::llvm::isa<::mlir::DenseIntElementsAttr>(tblgen_lowerBoundsGroups))) && ((::llvm::cast<::mlir::DenseIntElementsAttr>(tblgen_lowerBoundsGroups).getType().getElementType().isSignlessInteger(32)))))
    return emitError(loc, "'affine.parallel' op ""attribute 'lowerBoundsGroups' failed to satisfy constraint: 32-bit signless integer elements attribute");

  if (tblgen_upperBoundsMap && !((::llvm::isa<::mlir::AffineMapAttr>(tblgen_upperBoundsMap))))
    return emitError(loc, "'affine.parallel' op ""attribute 'upperBoundsMap' failed to satisfy constraint: AffineMap attribute");

  if (tblgen_upperBoundsGroups && !(((::llvm::isa<::mlir::DenseIntElementsAttr>(tblgen_upperBoundsGroups))) && ((::llvm::cast<::mlir::DenseIntElementsAttr>(tblgen_upperBoundsGroups).getType().getElementType().isSignlessInteger(32)))))
    return emitError(loc, "'affine.parallel' op ""attribute 'upperBoundsGroups' failed to satisfy constraint: 32-bit signless integer elements attribute");

  if (tblgen_steps && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_steps))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_steps), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'affine.parallel' op ""attribute 'steps' failed to satisfy constraint: 64-bit integer array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineParallelOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::MutableOperandRange AffineParallelOp::getMapOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AffineParallelOp::getODSResultIndexAndLength(unsigned index) {
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

::llvm::LogicalResult AffineParallelOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.lowerBoundsGroups;
       auto attr = dict.get("lowerBoundsGroups");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `lowerBoundsGroups` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.lowerBoundsMap;
       auto attr = dict.get("lowerBoundsMap");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `lowerBoundsMap` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.reductions;
       auto attr = dict.get("reductions");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `reductions` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.steps;
       auto attr = dict.get("steps");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `steps` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.upperBoundsGroups;
       auto attr = dict.get("upperBoundsGroups");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `upperBoundsGroups` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.upperBoundsMap;
       auto attr = dict.get("upperBoundsMap");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `upperBoundsMap` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute AffineParallelOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.lowerBoundsGroups;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("lowerBoundsGroups",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.lowerBoundsMap;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("lowerBoundsMap",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.reductions;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("reductions",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.steps;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("steps",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.upperBoundsGroups;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("upperBoundsGroups",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.upperBoundsMap;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("upperBoundsMap",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code AffineParallelOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.lowerBoundsGroups.getAsOpaquePointer()), 
    llvm::hash_value(prop.lowerBoundsMap.getAsOpaquePointer()), 
    llvm::hash_value(prop.reductions.getAsOpaquePointer()), 
    llvm::hash_value(prop.steps.getAsOpaquePointer()), 
    llvm::hash_value(prop.upperBoundsGroups.getAsOpaquePointer()), 
    llvm::hash_value(prop.upperBoundsMap.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> AffineParallelOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "lowerBoundsGroups")
      return prop.lowerBoundsGroups;

    if (name == "lowerBoundsMap")
      return prop.lowerBoundsMap;

    if (name == "reductions")
      return prop.reductions;

    if (name == "steps")
      return prop.steps;

    if (name == "upperBoundsGroups")
      return prop.upperBoundsGroups;

    if (name == "upperBoundsMap")
      return prop.upperBoundsMap;
  return std::nullopt;
}

void AffineParallelOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "lowerBoundsGroups") {
       prop.lowerBoundsGroups = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.lowerBoundsGroups)>>(value);
       return;
    }

    if (name == "lowerBoundsMap") {
       prop.lowerBoundsMap = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.lowerBoundsMap)>>(value);
       return;
    }

    if (name == "reductions") {
       prop.reductions = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.reductions)>>(value);
       return;
    }

    if (name == "steps") {
       prop.steps = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.steps)>>(value);
       return;
    }

    if (name == "upperBoundsGroups") {
       prop.upperBoundsGroups = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.upperBoundsGroups)>>(value);
       return;
    }

    if (name == "upperBoundsMap") {
       prop.upperBoundsMap = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.upperBoundsMap)>>(value);
       return;
    }
}

void AffineParallelOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.lowerBoundsGroups) attrs.append("lowerBoundsGroups", prop.lowerBoundsGroups);

    if (prop.lowerBoundsMap) attrs.append("lowerBoundsMap", prop.lowerBoundsMap);

    if (prop.reductions) attrs.append("reductions", prop.reductions);

    if (prop.steps) attrs.append("steps", prop.steps);

    if (prop.upperBoundsGroups) attrs.append("upperBoundsGroups", prop.upperBoundsGroups);

    if (prop.upperBoundsMap) attrs.append("upperBoundsMap", prop.upperBoundsMap);
}

::llvm::LogicalResult AffineParallelOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getLowerBoundsGroupsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps4(attr, "lowerBoundsGroups", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getLowerBoundsMapAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(attr, "lowerBoundsMap", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getReductionsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps3(attr, "reductions", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getStepsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps5(attr, "steps", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getUpperBoundsGroupsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps4(attr, "upperBoundsGroups", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getUpperBoundsMapAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(attr, "upperBoundsMap", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult AffineParallelOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.lowerBoundsGroups)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.lowerBoundsMap)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.reductions)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.steps)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.upperBoundsGroups)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.upperBoundsMap)))
    return ::mlir::failure();
  return ::mlir::success();
}

void AffineParallelOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.lowerBoundsGroups);
  writer.writeAttribute(prop.lowerBoundsMap);
  writer.writeAttribute(prop.reductions);
  writer.writeAttribute(prop.steps);
  writer.writeAttribute(prop.upperBoundsGroups);
  writer.writeAttribute(prop.upperBoundsMap);
}

::mlir::ArrayAttr AffineParallelOp::getReductions() {
  auto attr = getReductionsAttr();
  return attr;
}

::mlir::AffineMap AffineParallelOp::getLowerBoundsMap() {
  auto attr = getLowerBoundsMapAttr();
  return attr.getValue();
}

::mlir::DenseIntElementsAttr AffineParallelOp::getLowerBoundsGroups() {
  auto attr = getLowerBoundsGroupsAttr();
  return attr;
}

::mlir::AffineMap AffineParallelOp::getUpperBoundsMap() {
  auto attr = getUpperBoundsMapAttr();
  return attr.getValue();
}

::mlir::DenseIntElementsAttr AffineParallelOp::getUpperBoundsGroups() {
  auto attr = getUpperBoundsGroupsAttr();
  return attr;
}

::llvm::SmallVector<int64_t, 8> AffineParallelOp::getSteps() {
  auto attr = getStepsAttr();
  return llvm::to_vector<4>(
      llvm::map_range(attr.getAsRange<mlir::IntegerAttr>(),
      [](mlir::IntegerAttr attr) { return attr.getInt(); }));;
}

void AffineParallelOp::setLowerBoundsMap(::mlir::AffineMap attrValue) {
  getProperties().lowerBoundsMap = ::mlir::AffineMapAttr::get(attrValue);
}

void AffineParallelOp::setUpperBoundsMap(::mlir::AffineMap attrValue) {
  getProperties().upperBoundsMap = ::mlir::AffineMapAttr::get(attrValue);
}

void AffineParallelOp::setSteps(::llvm::SmallVector<int64_t, 8> attrValue) {
  getProperties().steps = ::mlir::Builder((*this)->getContext()).getI64ArrayAttr(attrValue);
}

void AffineParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::ArrayAttr reductions, ::mlir::AffineMapAttr lowerBoundsMap, ::mlir::DenseIntElementsAttr lowerBoundsGroups, ::mlir::AffineMapAttr upperBoundsMap, ::mlir::DenseIntElementsAttr upperBoundsGroups, ::mlir::ArrayAttr steps, ::mlir::ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.getOrAddProperties<Properties>().reductions = reductions;
  odsState.getOrAddProperties<Properties>().lowerBoundsMap = lowerBoundsMap;
  odsState.getOrAddProperties<Properties>().lowerBoundsGroups = lowerBoundsGroups;
  odsState.getOrAddProperties<Properties>().upperBoundsMap = upperBoundsMap;
  odsState.getOrAddProperties<Properties>().upperBoundsGroups = upperBoundsGroups;
  odsState.getOrAddProperties<Properties>().steps = steps;
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void AffineParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::ArrayAttr reductions, ::mlir::AffineMap lowerBoundsMap, ::mlir::DenseIntElementsAttr lowerBoundsGroups, ::mlir::AffineMap upperBoundsMap, ::mlir::DenseIntElementsAttr upperBoundsGroups, ::llvm::SmallVector<int64_t, 8> steps, ::mlir::ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.getOrAddProperties<Properties>().reductions = reductions;
  odsState.getOrAddProperties<Properties>().lowerBoundsMap = ::mlir::AffineMapAttr::get(lowerBoundsMap);
  odsState.getOrAddProperties<Properties>().lowerBoundsGroups = lowerBoundsGroups;
  odsState.getOrAddProperties<Properties>().upperBoundsMap = ::mlir::AffineMapAttr::get(upperBoundsMap);
  odsState.getOrAddProperties<Properties>().upperBoundsGroups = upperBoundsGroups;
  odsState.getOrAddProperties<Properties>().steps = odsBuilder.getI64ArrayAttr(steps);
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void AffineParallelOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<AffineParallelOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult AffineParallelOp::verifyInvariantsImpl() {
  auto tblgen_lowerBoundsGroups = getProperties().lowerBoundsGroups; (void)tblgen_lowerBoundsGroups;
  if (!tblgen_lowerBoundsGroups) return emitOpError("requires attribute 'lowerBoundsGroups'");
  auto tblgen_lowerBoundsMap = getProperties().lowerBoundsMap; (void)tblgen_lowerBoundsMap;
  if (!tblgen_lowerBoundsMap) return emitOpError("requires attribute 'lowerBoundsMap'");
  auto tblgen_reductions = getProperties().reductions; (void)tblgen_reductions;
  if (!tblgen_reductions) return emitOpError("requires attribute 'reductions'");
  auto tblgen_steps = getProperties().steps; (void)tblgen_steps;
  if (!tblgen_steps) return emitOpError("requires attribute 'steps'");
  auto tblgen_upperBoundsGroups = getProperties().upperBoundsGroups; (void)tblgen_upperBoundsGroups;
  if (!tblgen_upperBoundsGroups) return emitOpError("requires attribute 'upperBoundsGroups'");
  auto tblgen_upperBoundsMap = getProperties().upperBoundsMap; (void)tblgen_upperBoundsMap;
  if (!tblgen_upperBoundsMap) return emitOpError("requires attribute 'upperBoundsMap'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps3(*this, tblgen_reductions, "reductions")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(*this, tblgen_lowerBoundsMap, "lowerBoundsMap")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps4(*this, tblgen_lowerBoundsGroups, "lowerBoundsGroups")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(*this, tblgen_upperBoundsMap, "upperBoundsMap")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps4(*this, tblgen_upperBoundsGroups, "upperBoundsGroups")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps5(*this, tblgen_steps, "steps")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_AffineOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::llvm::LogicalResult AffineParallelOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineParallelOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffinePrefetchOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffinePrefetchOpGenericAdaptorBase::AffinePrefetchOpGenericAdaptorBase(AffinePrefetchOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> AffinePrefetchOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

bool AffinePrefetchOpGenericAdaptorBase::getIsWrite() {
  auto attr = getIsWriteAttr();
  return attr.getValue();
}

uint32_t AffinePrefetchOpGenericAdaptorBase::getLocalityHint() {
  auto attr = getLocalityHintAttr();
  return attr.getValue().getZExtValue();
}

bool AffinePrefetchOpGenericAdaptorBase::getIsDataCache() {
  auto attr = getIsDataCacheAttr();
  return attr.getValue();
}

::mlir::AffineMap AffinePrefetchOpGenericAdaptorBase::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

} // namespace detail
AffinePrefetchOpAdaptor::AffinePrefetchOpAdaptor(AffinePrefetchOp op) : AffinePrefetchOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AffinePrefetchOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_isDataCache = getProperties().isDataCache; (void)tblgen_isDataCache;
  if (!tblgen_isDataCache) return emitError(loc, "'affine.prefetch' op ""requires attribute 'isDataCache'");
  auto tblgen_isWrite = getProperties().isWrite; (void)tblgen_isWrite;
  if (!tblgen_isWrite) return emitError(loc, "'affine.prefetch' op ""requires attribute 'isWrite'");
  auto tblgen_localityHint = getProperties().localityHint; (void)tblgen_localityHint;
  if (!tblgen_localityHint) return emitError(loc, "'affine.prefetch' op ""requires attribute 'localityHint'");
  auto tblgen_map = getProperties().map; (void)tblgen_map;
  if (!tblgen_map) return emitError(loc, "'affine.prefetch' op ""requires attribute 'map'");

  if (tblgen_isWrite && !((::llvm::isa<::mlir::BoolAttr>(tblgen_isWrite))))
    return emitError(loc, "'affine.prefetch' op ""attribute 'isWrite' failed to satisfy constraint: bool attribute");

  if (tblgen_localityHint && !((((::llvm::isa<::mlir::IntegerAttr>(tblgen_localityHint))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_localityHint).getType().isSignlessInteger(32)))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_localityHint).getInt() >= 0)) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_localityHint).getInt() <= 3))))
    return emitError(loc, "'affine.prefetch' op ""attribute 'localityHint' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0 whose maximum value is 3");

  if (tblgen_isDataCache && !((::llvm::isa<::mlir::BoolAttr>(tblgen_isDataCache))))
    return emitError(loc, "'affine.prefetch' op ""attribute 'isDataCache' failed to satisfy constraint: bool attribute");

  if (tblgen_map && !((::llvm::isa<::mlir::AffineMapAttr>(tblgen_map))))
    return emitError(loc, "'affine.prefetch' op ""attribute 'map' failed to satisfy constraint: AffineMap attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffinePrefetchOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange AffinePrefetchOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult AffinePrefetchOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.isDataCache;
       auto attr = dict.get("isDataCache");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `isDataCache` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.isWrite;
       auto attr = dict.get("isWrite");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `isWrite` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.localityHint;
       auto attr = dict.get("localityHint");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `localityHint` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.map;
       auto attr = dict.get("map");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `map` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute AffinePrefetchOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.isDataCache;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("isDataCache",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.isWrite;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("isWrite",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.localityHint;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("localityHint",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.map;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("map",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code AffinePrefetchOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.isDataCache.getAsOpaquePointer()), 
    llvm::hash_value(prop.isWrite.getAsOpaquePointer()), 
    llvm::hash_value(prop.localityHint.getAsOpaquePointer()), 
    llvm::hash_value(prop.map.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> AffinePrefetchOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "isDataCache")
      return prop.isDataCache;

    if (name == "isWrite")
      return prop.isWrite;

    if (name == "localityHint")
      return prop.localityHint;

    if (name == "map")
      return prop.map;
  return std::nullopt;
}

void AffinePrefetchOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "isDataCache") {
       prop.isDataCache = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.isDataCache)>>(value);
       return;
    }

    if (name == "isWrite") {
       prop.isWrite = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.isWrite)>>(value);
       return;
    }

    if (name == "localityHint") {
       prop.localityHint = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.localityHint)>>(value);
       return;
    }

    if (name == "map") {
       prop.map = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.map)>>(value);
       return;
    }
}

void AffinePrefetchOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.isDataCache) attrs.append("isDataCache", prop.isDataCache);

    if (prop.isWrite) attrs.append("isWrite", prop.isWrite);

    if (prop.localityHint) attrs.append("localityHint", prop.localityHint);

    if (prop.map) attrs.append("map", prop.map);
}

::llvm::LogicalResult AffinePrefetchOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getIsDataCacheAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps6(attr, "isDataCache", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getIsWriteAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps6(attr, "isWrite", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getLocalityHintAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps7(attr, "localityHint", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getMapAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(attr, "map", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult AffinePrefetchOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.isDataCache)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.isWrite)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.localityHint)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.map)))
    return ::mlir::failure();
  return ::mlir::success();
}

void AffinePrefetchOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.isDataCache);
  writer.writeAttribute(prop.isWrite);
  writer.writeAttribute(prop.localityHint);
  writer.writeAttribute(prop.map);
}

bool AffinePrefetchOp::getIsWrite() {
  auto attr = getIsWriteAttr();
  return attr.getValue();
}

uint32_t AffinePrefetchOp::getLocalityHint() {
  auto attr = getLocalityHintAttr();
  return attr.getValue().getZExtValue();
}

bool AffinePrefetchOp::getIsDataCache() {
  auto attr = getIsDataCacheAttr();
  return attr.getValue();
}

::mlir::AffineMap AffinePrefetchOp::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

void AffinePrefetchOp::setIsWrite(bool attrValue) {
  getProperties().isWrite = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void AffinePrefetchOp::setLocalityHint(uint32_t attrValue) {
  getProperties().localityHint = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void AffinePrefetchOp::setIsDataCache(bool attrValue) {
  getProperties().isDataCache = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void AffinePrefetchOp::setMap(::mlir::AffineMap attrValue) {
  getProperties().map = ::mlir::AffineMapAttr::get(attrValue);
}

void AffinePrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value memref, AffineMap map, ArrayRef<Value> mapOperands, bool isWrite, unsigned localityHint, bool isDataCache) {
      assert(map.getNumInputs() == mapOperands.size()
             && "inconsistent index info");
      auto localityHintAttr = odsBuilder.getI32IntegerAttr(localityHint);
      auto isWriteAttr = odsBuilder.getBoolAttr(isWrite);
      auto isDataCacheAttr = odsBuilder.getBoolAttr(isDataCache);
      odsState.addOperands(memref);
      odsState.addOperands(mapOperands);
      Properties &prop = odsState.getOrAddProperties<Properties>();
      prop.map = AffineMapAttr::get(map);
      prop.localityHint = localityHintAttr;
      prop.isWrite = isWriteAttr;
      prop.isDataCache = isDataCacheAttr;
    
}

void AffinePrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr isWrite, ::mlir::IntegerAttr localityHint, ::mlir::BoolAttr isDataCache, ::mlir::AffineMapAttr map) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.getOrAddProperties<Properties>().isWrite = isWrite;
  odsState.getOrAddProperties<Properties>().localityHint = localityHint;
  odsState.getOrAddProperties<Properties>().isDataCache = isDataCache;
  odsState.getOrAddProperties<Properties>().map = map;
}

void AffinePrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr isWrite, ::mlir::IntegerAttr localityHint, ::mlir::BoolAttr isDataCache, ::mlir::AffineMapAttr map) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.getOrAddProperties<Properties>().isWrite = isWrite;
  odsState.getOrAddProperties<Properties>().localityHint = localityHint;
  odsState.getOrAddProperties<Properties>().isDataCache = isDataCache;
  odsState.getOrAddProperties<Properties>().map = map;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffinePrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, bool isWrite, uint32_t localityHint, bool isDataCache, ::mlir::AffineMap map) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.getOrAddProperties<Properties>().isWrite = odsBuilder.getBoolAttr(isWrite);
  odsState.getOrAddProperties<Properties>().localityHint = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), localityHint);
  odsState.getOrAddProperties<Properties>().isDataCache = odsBuilder.getBoolAttr(isDataCache);
  odsState.getOrAddProperties<Properties>().map = ::mlir::AffineMapAttr::get(map);
}

void AffinePrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, bool isWrite, uint32_t localityHint, bool isDataCache, ::mlir::AffineMap map) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.getOrAddProperties<Properties>().isWrite = odsBuilder.getBoolAttr(isWrite);
  odsState.getOrAddProperties<Properties>().localityHint = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), localityHint);
  odsState.getOrAddProperties<Properties>().isDataCache = odsBuilder.getBoolAttr(isDataCache);
  odsState.getOrAddProperties<Properties>().map = ::mlir::AffineMapAttr::get(map);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffinePrefetchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<AffinePrefetchOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult AffinePrefetchOp::verifyInvariantsImpl() {
  auto tblgen_isDataCache = getProperties().isDataCache; (void)tblgen_isDataCache;
  if (!tblgen_isDataCache) return emitOpError("requires attribute 'isDataCache'");
  auto tblgen_isWrite = getProperties().isWrite; (void)tblgen_isWrite;
  if (!tblgen_isWrite) return emitOpError("requires attribute 'isWrite'");
  auto tblgen_localityHint = getProperties().localityHint; (void)tblgen_localityHint;
  if (!tblgen_localityHint) return emitOpError("requires attribute 'localityHint'");
  auto tblgen_map = getProperties().map; (void)tblgen_map;
  if (!tblgen_map) return emitOpError("requires attribute 'map'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps6(*this, tblgen_isWrite, "isWrite")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps7(*this, tblgen_localityHint, "localityHint")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps6(*this, tblgen_isDataCache, "isDataCache")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(*this, tblgen_map, "map")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult AffinePrefetchOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffinePrefetchOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineStoreOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineStoreOpGenericAdaptorBase::AffineStoreOpGenericAdaptorBase(AffineStoreOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> AffineStoreOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::AffineMap AffineStoreOpGenericAdaptorBase::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

} // namespace detail
AffineStoreOpAdaptor::AffineStoreOpAdaptor(AffineStoreOp op) : AffineStoreOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AffineStoreOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_map = getProperties().map; (void)tblgen_map;
  if (!tblgen_map) return emitError(loc, "'affine.store' op ""requires attribute 'map'");

  if (tblgen_map && !((::llvm::isa<::mlir::AffineMapAttr>(tblgen_map))))
    return emitError(loc, "'affine.store' op ""attribute 'map' failed to satisfy constraint: AffineMap attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineStoreOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange AffineStoreOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult AffineStoreOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.map;
       auto attr = dict.get("map");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `map` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute AffineStoreOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.map;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("map",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code AffineStoreOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.map.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> AffineStoreOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "map")
      return prop.map;
  return std::nullopt;
}

void AffineStoreOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "map") {
       prop.map = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.map)>>(value);
       return;
    }
}

void AffineStoreOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.map) attrs.append("map", prop.map);
}

::llvm::LogicalResult AffineStoreOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getMapAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(attr, "map", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult AffineStoreOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.map)))
    return ::mlir::failure();
  return ::mlir::success();
}

void AffineStoreOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.map);
}

::mlir::AffineMap AffineStoreOp::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

void AffineStoreOp::setMap(::mlir::AffineMap attrValue) {
  getProperties().map = ::mlir::AffineMapAttr::get(attrValue);
}

::llvm::LogicalResult AffineStoreOp::verifyInvariantsImpl() {
  auto tblgen_map = getProperties().map; (void)tblgen_map;
  if (!tblgen_map) return emitOpError("requires attribute 'map'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(*this, tblgen_map, "map")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult AffineStoreOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void AffineStoreOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  {
    auto valueRange = getODSOperandIndexAndLength(1);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Write::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::SideEffects::DefaultResource::get());
    }
  }
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineStoreOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineVectorLoadOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineVectorLoadOpGenericAdaptorBase::AffineVectorLoadOpGenericAdaptorBase(AffineVectorLoadOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> AffineVectorLoadOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::AffineMap AffineVectorLoadOpGenericAdaptorBase::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

} // namespace detail
AffineVectorLoadOpAdaptor::AffineVectorLoadOpAdaptor(AffineVectorLoadOp op) : AffineVectorLoadOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AffineVectorLoadOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_map = getProperties().map; (void)tblgen_map;
  if (!tblgen_map) return emitError(loc, "'affine.vector_load' op ""requires attribute 'map'");

  if (tblgen_map && !((::llvm::isa<::mlir::AffineMapAttr>(tblgen_map))))
    return emitError(loc, "'affine.vector_load' op ""attribute 'map' failed to satisfy constraint: AffineMap attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineVectorLoadOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange AffineVectorLoadOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult AffineVectorLoadOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.map;
       auto attr = dict.get("map");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `map` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute AffineVectorLoadOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.map;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("map",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code AffineVectorLoadOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.map.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> AffineVectorLoadOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "map")
      return prop.map;
  return std::nullopt;
}

void AffineVectorLoadOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "map") {
       prop.map = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.map)>>(value);
       return;
    }
}

void AffineVectorLoadOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.map) attrs.append("map", prop.map);
}

::llvm::LogicalResult AffineVectorLoadOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getMapAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(attr, "map", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult AffineVectorLoadOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.map)))
    return ::mlir::failure();
  return ::mlir::success();
}

void AffineVectorLoadOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.map);
}

::mlir::AffineMap AffineVectorLoadOp::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

void AffineVectorLoadOp::setMap(::mlir::AffineMap attrValue) {
  getProperties().map = ::mlir::AffineMapAttr::get(attrValue);
}

void AffineVectorLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::AffineMapAttr map) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.getOrAddProperties<Properties>().map = map;
  odsState.addTypes(result);
}

void AffineVectorLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::AffineMapAttr map) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.getOrAddProperties<Properties>().map = map;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineVectorLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::AffineMap map) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.getOrAddProperties<Properties>().map = ::mlir::AffineMapAttr::get(map);
  odsState.addTypes(result);
}

void AffineVectorLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::AffineMap map) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.getOrAddProperties<Properties>().map = ::mlir::AffineMapAttr::get(map);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineVectorLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<AffineVectorLoadOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult AffineVectorLoadOp::verifyInvariantsImpl() {
  auto tblgen_map = getProperties().map; (void)tblgen_map;
  if (!tblgen_map) return emitOpError("requires attribute 'map'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(*this, tblgen_map, "map")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps6(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult AffineVectorLoadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void AffineVectorLoadOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  {
    auto valueRange = getODSOperandIndexAndLength(0);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Read::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::SideEffects::DefaultResource::get());
    }
  }
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineVectorLoadOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineVectorStoreOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineVectorStoreOpGenericAdaptorBase::AffineVectorStoreOpGenericAdaptorBase(AffineVectorStoreOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> AffineVectorStoreOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::AffineMap AffineVectorStoreOpGenericAdaptorBase::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

} // namespace detail
AffineVectorStoreOpAdaptor::AffineVectorStoreOpAdaptor(AffineVectorStoreOp op) : AffineVectorStoreOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AffineVectorStoreOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_map = getProperties().map; (void)tblgen_map;
  if (!tblgen_map) return emitError(loc, "'affine.vector_store' op ""requires attribute 'map'");

  if (tblgen_map && !((::llvm::isa<::mlir::AffineMapAttr>(tblgen_map))))
    return emitError(loc, "'affine.vector_store' op ""attribute 'map' failed to satisfy constraint: AffineMap attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineVectorStoreOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange AffineVectorStoreOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult AffineVectorStoreOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.map;
       auto attr = dict.get("map");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `map` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute AffineVectorStoreOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.map;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("map",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code AffineVectorStoreOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.map.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> AffineVectorStoreOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "map")
      return prop.map;
  return std::nullopt;
}

void AffineVectorStoreOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "map") {
       prop.map = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.map)>>(value);
       return;
    }
}

void AffineVectorStoreOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.map) attrs.append("map", prop.map);
}

::llvm::LogicalResult AffineVectorStoreOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getMapAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(attr, "map", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult AffineVectorStoreOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.map)))
    return ::mlir::failure();
  return ::mlir::success();
}

void AffineVectorStoreOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.map);
}

::mlir::AffineMap AffineVectorStoreOp::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

void AffineVectorStoreOp::setMap(::mlir::AffineMap attrValue) {
  getProperties().map = ::mlir::AffineMapAttr::get(attrValue);
}

::llvm::LogicalResult AffineVectorStoreOp::verifyInvariantsImpl() {
  auto tblgen_map = getProperties().map; (void)tblgen_map;
  if (!tblgen_map) return emitOpError("requires attribute 'map'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(*this, tblgen_map, "map")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult AffineVectorStoreOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void AffineVectorStoreOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  {
    auto valueRange = getODSOperandIndexAndLength(1);
    for (unsigned idx = valueRange.first; idx < valueRange.first + valueRange.second; idx++) {
      effects.emplace_back(::mlir::MemoryEffects::Write::get(), &getOperation()->getOpOperand(idx), 0, false, ::mlir::SideEffects::DefaultResource::get());
    }
  }
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineVectorStoreOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineYieldOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
std::pair<unsigned, unsigned> AffineYieldOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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
AffineYieldOpAdaptor::AffineYieldOpAdaptor(AffineYieldOp op) : AffineYieldOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AffineYieldOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineYieldOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::MutableOperandRange AffineYieldOp::getOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

void AffineYieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
 build(odsBuilder, odsState, std::nullopt); 
}

void AffineYieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
}

void AffineYieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult AffineYieldOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult AffineYieldOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AffineYieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> operandsOperands;
  ::llvm::SMLoc operandsOperandsLoc;
  (void)operandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> operandsTypes;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }

  operandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandsOperands))
    return ::mlir::failure();
  if (!operandsOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(operandsTypes))
    return ::mlir::failure();
  }
  if (parser.resolveOperands(operandsOperands, operandsTypes, operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AffineYieldOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  if (!getOperands().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << getOperands();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getOperands().getTypes();
  }
}

void AffineYieldOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

::mlir::MutableOperandRange AffineYieldOp::getMutableSuccessorOperands(
  ::mlir::RegionBranchPoint point) {
  return ::mlir::MutableOperandRange(*this);
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineYieldOp)


#endif  // GET_OP_CLASSES

