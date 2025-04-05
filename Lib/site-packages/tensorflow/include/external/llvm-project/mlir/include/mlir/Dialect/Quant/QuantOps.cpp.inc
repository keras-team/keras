/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: QuantOps.td                                                          *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::quant::DequantizeCastOp,
::mlir::quant::QuantizeCastOp,
::mlir::quant::StorageCastOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace quant {

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_QuantOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::FloatType>(type))) || ((::llvm::isa<mlir::quant::QuantizedType>(type)))) || (((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return ((::llvm::isa<::mlir::FloatType>(elementType))) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || ((((::llvm::isa<::mlir::VectorType>(type))) && ((::llvm::cast<::mlir::VectorType>(type).getRank() > 0))) && ([](::mlir::Type elementType) { return ((::llvm::isa<::mlir::FloatType>(elementType))) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be primitive/tensor/vector of real valued primitive (float or quantized type), but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_QuantOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((((::llvm::isa<::mlir::FloatType>(type))) || ((::llvm::isa<mlir::quant::QuantizedType>(type)))) || (((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return ((::llvm::isa<::mlir::FloatType>(elementType))) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || ((((::llvm::isa<::mlir::VectorType>(type))) && ((::llvm::cast<::mlir::VectorType>(type).getRank() > 0))) && ([](::mlir::Type elementType) { return ((::llvm::isa<::mlir::FloatType>(elementType))) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) || ((((type.isSignlessInteger())) || ((::llvm::isa<mlir::quant::QuantizedType>(type)))) || (((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))) || ((((::llvm::isa<::mlir::VectorType>(type))) && ((::llvm::cast<::mlir::VectorType>(type).getRank() > 0))) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((::llvm::isa<mlir::quant::QuantizedType>(elementType))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be real valued or storage primitive or container type, but got " << type;
  }
  return ::mlir::success();
}
} // namespace quant
} // namespace mlir
namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::DequantizeCastOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
DequantizeCastOpAdaptor::DequantizeCastOpAdaptor(DequantizeCastOp op) : DequantizeCastOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult DequantizeCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void DequantizeCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void DequantizeCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DequantizeCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult DequantizeCastOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult DequantizeCastOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void DequantizeCastOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace quant
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::quant::DequantizeCastOp)

namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::QuantizeCastOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
QuantizeCastOpAdaptor::QuantizeCastOpAdaptor(QuantizeCastOp op) : QuantizeCastOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult QuantizeCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void QuantizeCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void QuantizeCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void QuantizeCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult QuantizeCastOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult QuantizeCastOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void QuantizeCastOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace quant
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::quant::QuantizeCastOp)

namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::StorageCastOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
StorageCastOpAdaptor::StorageCastOpAdaptor(StorageCastOp op) : StorageCastOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult StorageCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void StorageCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void StorageCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StorageCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult StorageCastOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult StorageCastOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void StorageCastOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace quant
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::quant::StorageCastOp)


#endif  // GET_OP_CLASSES

