/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ArmNeon.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::arm_neon::SMullOp,
::mlir::arm_neon::Sdot2dOp,
::mlir::arm_neon::SdotOp,
::mlir::arm_neon::SmmlaOp,
::mlir::arm_neon::UmmlaOp,
::mlir::arm_neon::UsmmlaOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace arm_neon {

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_ArmNeon1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((((::llvm::isa<::mlir::VectorType>(type))) && ((::llvm::cast<::mlir::VectorType>(type).getRank() > 0))) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) && ((((::llvm::isa<::mlir::VectorType>(type))) && ((::llvm::cast<::mlir::VectorType>(type).getRank() > 0))) && (((::llvm::cast<::mlir::VectorType>(type).getNumElements()
                           == 8)) || ((::llvm::cast<::mlir::VectorType>(type).getNumElements()
                           == 4)) || ((::llvm::cast<::mlir::VectorType>(type).getNumElements()
                           == 2)))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be vector of 8-bit signless integer or 16-bit signless integer or 32-bit signless integer values of length 8/4/2, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_ArmNeon2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((((::llvm::isa<::mlir::VectorType>(type))) && ((::llvm::cast<::mlir::VectorType>(type).getRank() > 0))) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) && ((((::llvm::isa<::mlir::VectorType>(type))) && ((::llvm::cast<::mlir::VectorType>(type).getRank() > 0))) && (((::llvm::cast<::mlir::VectorType>(type).getNumElements()
                           == 8)) || ((::llvm::cast<::mlir::VectorType>(type).getNumElements()
                           == 4)) || ((::llvm::cast<::mlir::VectorType>(type).getNumElements()
                           == 2)))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be vector of 16-bit signless integer or 32-bit signless integer or 64-bit signless integer values of length 8/4/2, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_ArmNeon3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((((::llvm::isa<::mlir::VectorType>(type))) && ((::llvm::cast<::mlir::VectorType>(type).getRank() > 0))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) && ((((::llvm::isa<::mlir::VectorType>(type))) && ((::llvm::cast<::mlir::VectorType>(type).getRank() > 0))) && (((::llvm::cast<::mlir::VectorType>(type).getNumElements()
                           == 4)) || ((::llvm::cast<::mlir::VectorType>(type).getNumElements()
                           == 2)))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be vector of 32-bit signless integer values of length 4/2, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_ArmNeon4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((((::llvm::isa<::mlir::VectorType>(type))) && ((::llvm::cast<::mlir::VectorType>(type).getRank() > 0))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) && ((((::llvm::isa<::mlir::VectorType>(type))) && ((::llvm::cast<::mlir::VectorType>(type).getRank() > 0))) && (((::llvm::cast<::mlir::VectorType>(type).getNumElements()
                           == 16)) || ((::llvm::cast<::mlir::VectorType>(type).getNumElements()
                           == 8)))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be vector of 8-bit signless integer values of length 16/8, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_ArmNeon5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::cast<::mlir::VectorType>(type).getShape() == ArrayRef<int64_t>({4}))) && ((::llvm::isa<::mlir::VectorType>(type) &&
                                  !::llvm::cast<VectorType>(type).isScalable()))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be a vector with length 4 of 32-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_ArmNeon6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::cast<::mlir::VectorType>(type).getShape() == ArrayRef<int64_t>({16}))) && ((::llvm::isa<::mlir::VectorType>(type) &&
                                  !::llvm::cast<VectorType>(type).isScalable()))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be a vector with length 16 of 8-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}
} // namespace arm_neon
} // namespace mlir
namespace mlir {
namespace arm_neon {

//===----------------------------------------------------------------------===//
// ::mlir::arm_neon::SMullOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
SMullOpAdaptor::SMullOpAdaptor(SMullOp op) : SMullOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult SMullOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void SMullOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void SMullOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SMullOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult SMullOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((((*this->getODSOperands(0).begin()).getType()) == ((*this->getODSOperands(1).begin()).getType()) && ((*this->getODSOperands(1).begin()).getType()) == ((*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that all of {a, b} have same type");
  if (!((std::equal_to<>()(::llvm::cast<VectorType>((*this->getODSOperands(0).begin()).getType()).scaleElementBitwidth(2), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that res has same vector shape and element bitwidth scaled by 2 as a");
  return ::mlir::success();
}

::llvm::LogicalResult SMullOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult SMullOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand aRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> aOperands(&aRawOperand, 1);  ::llvm::SMLoc aOperandsLoc;
  (void)aOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand bRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> bOperands(&bRawOperand, 1);  ::llvm::SMLoc bOperandsLoc;
  (void)bOperandsLoc;
  ::mlir::Type aRawType{};
  ::llvm::ArrayRef<::mlir::Type> aTypes(&aRawType, 1);
  ::mlir::Type resRawType{};
  ::llvm::ArrayRef<::mlir::Type> resTypes(&resRawType, 1);

  aOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  bOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    aRawType = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawType = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(aOperands, aTypes, aOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(bOperands, aTypes[0], bOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SMullOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getA();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getB();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getA().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::VectorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::VectorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void SMullOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace arm_neon
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_neon::SMullOp)

namespace mlir {
namespace arm_neon {

//===----------------------------------------------------------------------===//
// ::mlir::arm_neon::Sdot2dOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
Sdot2dOpAdaptor::Sdot2dOpAdaptor(Sdot2dOp op) : Sdot2dOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult Sdot2dOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void Sdot2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  odsState.addTypes(res);
}

void Sdot2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Sdot2dOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult Sdot2dOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((((*this->getODSOperands(1).begin()).getType()) == ((*this->getODSOperands(2).begin()).getType()) && ((*this->getODSOperands(2).begin()).getType()) == ((*this->getODSOperands(1).begin()).getType()))))
    return emitOpError("failed to verify that all of {b, c} have same type");
  if (!((((*this->getODSOperands(0).begin()).getType()) == ((*this->getODSResults(0).begin()).getType()) && ((*this->getODSResults(0).begin()).getType()) == ((*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that all of {a, res} have same type");
  if (!((::llvm::cast<VectorType>(getA().getType()).getShape().size() == 1)))
    return emitOpError("failed to verify that operand `a` should be 1-dimensional");
  if (!((::llvm::cast<VectorType>(getB().getType()).getShape().size() == 2)))
    return emitOpError("failed to verify that operand `b` should be 2-dimensional");
  if (!((::llvm::cast<VectorType>(getB().getType()).getShape()[1] == 4)))
    return emitOpError("failed to verify that operand `b` should have 4 columns");
  if (!((::llvm::cast<VectorType>(getB().getType()).getShape()[0] == ::llvm::cast<VectorType>(getA().getType()).getShape()[0])))
    return emitOpError("failed to verify that operand `b` should have as many rows as the size of operand `a`");
  return ::mlir::success();
}

::llvm::LogicalResult Sdot2dOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult Sdot2dOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand aRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> aOperands(&aRawOperand, 1);  ::llvm::SMLoc aOperandsLoc;
  (void)aOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand bRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> bOperands(&bRawOperand, 1);  ::llvm::SMLoc bOperandsLoc;
  (void)bOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand cRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> cOperands(&cRawOperand, 1);  ::llvm::SMLoc cOperandsLoc;
  (void)cOperandsLoc;
  ::mlir::Type bRawType{};
  ::llvm::ArrayRef<::mlir::Type> bTypes(&bRawType, 1);
  ::mlir::Type cRawType{};
  ::llvm::ArrayRef<::mlir::Type> cTypes(&cRawType, 1);
  ::mlir::Type resRawType{};
  ::llvm::ArrayRef<::mlir::Type> resTypes(&resRawType, 1);

  aOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  bOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  cOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(cRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    bRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    cRawType = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawType = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(aOperands, resTypes[0], aOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(bOperands, bTypes, bOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(cOperands, cTypes, cOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void Sdot2dOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getA();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getB();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getC();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getB().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::VectorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getC().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::VectorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::VectorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void Sdot2dOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace arm_neon
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_neon::Sdot2dOp)

namespace mlir {
namespace arm_neon {

//===----------------------------------------------------------------------===//
// ::mlir::arm_neon::SdotOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
SdotOpAdaptor::SdotOpAdaptor(SdotOp op) : SdotOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult SdotOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void SdotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  odsState.addTypes(res);
}

void SdotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SdotOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult SdotOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((((*this->getODSOperands(1).begin()).getType()) == ((*this->getODSOperands(2).begin()).getType()) && ((*this->getODSOperands(2).begin()).getType()) == ((*this->getODSOperands(1).begin()).getType()))))
    return emitOpError("failed to verify that all of {b, c} have same type");
  if (!((((*this->getODSOperands(0).begin()).getType()) == ((*this->getODSResults(0).begin()).getType()) && ((*this->getODSResults(0).begin()).getType()) == ((*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that all of {a, res} have same type");
  if (!((std::equal_to<>()(VectorType::get({::llvm::cast<VectorType>((*this->getODSOperands(1).begin()).getType()).getShape()[0] / 4},IntegerType::get((*this->getODSOperands(1).begin()).getType().getContext(), 32)), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that res has the same number of elements as operand b");
  return ::mlir::success();
}

::llvm::LogicalResult SdotOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult SdotOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand aRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> aOperands(&aRawOperand, 1);  ::llvm::SMLoc aOperandsLoc;
  (void)aOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand bRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> bOperands(&bRawOperand, 1);  ::llvm::SMLoc bOperandsLoc;
  (void)bOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand cRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> cOperands(&cRawOperand, 1);  ::llvm::SMLoc cOperandsLoc;
  (void)cOperandsLoc;
  ::mlir::Type bRawType{};
  ::llvm::ArrayRef<::mlir::Type> bTypes(&bRawType, 1);
  ::mlir::Type cRawType{};
  ::llvm::ArrayRef<::mlir::Type> cTypes(&cRawType, 1);
  ::mlir::Type resRawType{};
  ::llvm::ArrayRef<::mlir::Type> resTypes(&resRawType, 1);

  aOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  bOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  cOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(cRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    bRawType = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    cRawType = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawType = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(aOperands, resTypes[0], aOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(bOperands, bTypes, bOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(cOperands, cTypes, cOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SdotOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getA();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getB();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getC();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getB().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::VectorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getC().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::VectorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::VectorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void SdotOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace arm_neon
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_neon::SdotOp)

namespace mlir {
namespace arm_neon {

//===----------------------------------------------------------------------===//
// ::mlir::arm_neon::SmmlaOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
SmmlaOpAdaptor::SmmlaOpAdaptor(SmmlaOp op) : SmmlaOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult SmmlaOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void SmmlaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(acc);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(res);
}

void SmmlaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(acc);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SmmlaOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult SmmlaOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((((*this->getODSOperands(1).begin()).getType()) == ((*this->getODSOperands(2).begin()).getType()) && ((*this->getODSOperands(2).begin()).getType()) == ((*this->getODSOperands(1).begin()).getType()))))
    return emitOpError("failed to verify that all of {src1, src2} have same type");
  if (!((((*this->getODSOperands(0).begin()).getType()) == ((*this->getODSResults(0).begin()).getType()) && ((*this->getODSResults(0).begin()).getType()) == ((*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that all of {acc, res} have same type");
  return ::mlir::success();
}

::llvm::LogicalResult SmmlaOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult SmmlaOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand accRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> accOperands(&accRawOperand, 1);  ::llvm::SMLoc accOperandsLoc;
  (void)accOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand src1RawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> src1Operands(&src1RawOperand, 1);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand src2RawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> src2Operands(&src2RawOperand, 1);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type src1RawType{};
  ::llvm::ArrayRef<::mlir::Type> src1Types(&src1RawType, 1);
  ::mlir::Type resRawType{};
  ::llvm::ArrayRef<::mlir::Type> resTypes(&resRawType, 1);

  accOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(accRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    src1RawType = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawType = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(accOperands, resTypes[0], accOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src1Operands, src1Types, src1OperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, src1Types[0], src2OperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SmmlaOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAcc();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getSrc1();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getSrc2();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSrc1().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::VectorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::VectorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void SmmlaOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace arm_neon
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_neon::SmmlaOp)

namespace mlir {
namespace arm_neon {

//===----------------------------------------------------------------------===//
// ::mlir::arm_neon::UmmlaOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
UmmlaOpAdaptor::UmmlaOpAdaptor(UmmlaOp op) : UmmlaOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult UmmlaOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void UmmlaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(acc);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(res);
}

void UmmlaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(acc);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UmmlaOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult UmmlaOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((((*this->getODSOperands(1).begin()).getType()) == ((*this->getODSOperands(2).begin()).getType()) && ((*this->getODSOperands(2).begin()).getType()) == ((*this->getODSOperands(1).begin()).getType()))))
    return emitOpError("failed to verify that all of {src1, src2} have same type");
  if (!((((*this->getODSOperands(0).begin()).getType()) == ((*this->getODSResults(0).begin()).getType()) && ((*this->getODSResults(0).begin()).getType()) == ((*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that all of {acc, res} have same type");
  return ::mlir::success();
}

::llvm::LogicalResult UmmlaOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult UmmlaOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand accRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> accOperands(&accRawOperand, 1);  ::llvm::SMLoc accOperandsLoc;
  (void)accOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand src1RawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> src1Operands(&src1RawOperand, 1);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand src2RawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> src2Operands(&src2RawOperand, 1);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type src1RawType{};
  ::llvm::ArrayRef<::mlir::Type> src1Types(&src1RawType, 1);
  ::mlir::Type resRawType{};
  ::llvm::ArrayRef<::mlir::Type> resTypes(&resRawType, 1);

  accOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(accRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    src1RawType = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawType = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(accOperands, resTypes[0], accOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src1Operands, src1Types, src1OperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, src1Types[0], src2OperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void UmmlaOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAcc();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getSrc1();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getSrc2();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSrc1().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::VectorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::VectorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void UmmlaOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace arm_neon
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_neon::UmmlaOp)

namespace mlir {
namespace arm_neon {

//===----------------------------------------------------------------------===//
// ::mlir::arm_neon::UsmmlaOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
UsmmlaOpAdaptor::UsmmlaOpAdaptor(UsmmlaOp op) : UsmmlaOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult UsmmlaOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void UsmmlaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(acc);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  odsState.addTypes(res);
}

void UsmmlaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value acc, ::mlir::Value src1, ::mlir::Value src2) {
  odsState.addOperands(acc);
  odsState.addOperands(src1);
  odsState.addOperands(src2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UsmmlaOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult UsmmlaOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ArmNeon5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((((*this->getODSOperands(1).begin()).getType()) == ((*this->getODSOperands(2).begin()).getType()) && ((*this->getODSOperands(2).begin()).getType()) == ((*this->getODSOperands(1).begin()).getType()))))
    return emitOpError("failed to verify that all of {src1, src2} have same type");
  if (!((((*this->getODSOperands(0).begin()).getType()) == ((*this->getODSResults(0).begin()).getType()) && ((*this->getODSResults(0).begin()).getType()) == ((*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that all of {acc, res} have same type");
  return ::mlir::success();
}

::llvm::LogicalResult UsmmlaOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult UsmmlaOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand accRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> accOperands(&accRawOperand, 1);  ::llvm::SMLoc accOperandsLoc;
  (void)accOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand src1RawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> src1Operands(&src1RawOperand, 1);  ::llvm::SMLoc src1OperandsLoc;
  (void)src1OperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand src2RawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> src2Operands(&src2RawOperand, 1);  ::llvm::SMLoc src2OperandsLoc;
  (void)src2OperandsLoc;
  ::mlir::Type src1RawType{};
  ::llvm::ArrayRef<::mlir::Type> src1Types(&src1RawType, 1);
  ::mlir::Type resRawType{};
  ::llvm::ArrayRef<::mlir::Type> resTypes(&resRawType, 1);

  accOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(accRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src1OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src1RawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  src2OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(src2RawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    src1RawType = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawType = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(accOperands, resTypes[0], accOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src1Operands, src1Types, src1OperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(src2Operands, src1Types[0], src2OperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void UsmmlaOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAcc();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getSrc1();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getSrc2();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSrc1().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::VectorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::VectorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void UsmmlaOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace arm_neon
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::arm_neon::UsmmlaOp)


#endif  // GET_OP_CLASSES

