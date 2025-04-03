/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: PDLInterpOps.td                                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::pdl_interp::ApplyConstraintOp,
::mlir::pdl_interp::ApplyRewriteOp,
::mlir::pdl_interp::AreEqualOp,
::mlir::pdl_interp::BranchOp,
::mlir::pdl_interp::CheckAttributeOp,
::mlir::pdl_interp::CheckOperandCountOp,
::mlir::pdl_interp::CheckOperationNameOp,
::mlir::pdl_interp::CheckResultCountOp,
::mlir::pdl_interp::CheckTypeOp,
::mlir::pdl_interp::CheckTypesOp,
::mlir::pdl_interp::ContinueOp,
::mlir::pdl_interp::CreateAttributeOp,
::mlir::pdl_interp::CreateOperationOp,
::mlir::pdl_interp::CreateRangeOp,
::mlir::pdl_interp::CreateTypeOp,
::mlir::pdl_interp::CreateTypesOp,
::mlir::pdl_interp::EraseOp,
::mlir::pdl_interp::ExtractOp,
::mlir::pdl_interp::FinalizeOp,
::mlir::pdl_interp::ForEachOp,
::mlir::pdl_interp::FuncOp,
::mlir::pdl_interp::GetAttributeOp,
::mlir::pdl_interp::GetAttributeTypeOp,
::mlir::pdl_interp::GetDefiningOpOp,
::mlir::pdl_interp::GetOperandOp,
::mlir::pdl_interp::GetOperandsOp,
::mlir::pdl_interp::GetResultOp,
::mlir::pdl_interp::GetResultsOp,
::mlir::pdl_interp::GetUsersOp,
::mlir::pdl_interp::GetValueTypeOp,
::mlir::pdl_interp::IsNotNullOp,
::mlir::pdl_interp::RecordMatchOp,
::mlir::pdl_interp::ReplaceOp,
::mlir::pdl_interp::SwitchAttributeOp,
::mlir::pdl_interp::SwitchOperandCountOp,
::mlir::pdl_interp::SwitchOperationNameOp,
::mlir::pdl_interp::SwitchResultCountOp,
::mlir::pdl_interp::SwitchTypeOp,
::mlir::pdl_interp::SwitchTypesOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace pdl_interp {

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::pdl::PDLType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of pdl type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::pdl::PDLType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be pdl type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::pdl::AttributeType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle to an `mlir::Attribute`, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::pdl::OperationType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle to an `mlir::Operation *`, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::pdl::TypeType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle to an `mlir::Type`, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::pdl::RangeType>(type))) && ((::llvm::isa<::mlir::pdl::TypeType>(::llvm::cast<::mlir::pdl::RangeType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be range of PDL handle to an `mlir::Type` values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::pdl::ValueType>(type))) || (((::llvm::isa<::mlir::pdl::RangeType>(type))) && ((::llvm::isa<::mlir::pdl::ValueType>(::llvm::cast<::mlir::pdl::RangeType>(type).getElementType())))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of single element or range of PDL handle for an `mlir::Value`, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps8(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::pdl::AttributeType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of PDL handle to an `mlir::Attribute`, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps9(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::pdl::TypeType>(type))) || (((::llvm::isa<::mlir::pdl::RangeType>(type))) && ((::llvm::isa<::mlir::pdl::TypeType>(::llvm::cast<::mlir::pdl::RangeType>(type).getElementType())))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of single element or range of PDL handle to an `mlir::Type`, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps10(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::pdl::RangeType>(type))) && (((::llvm::isa<::mlir::pdl::TypeType>(::llvm::cast<::mlir::pdl::RangeType>(type).getElementType()))) || ((::llvm::isa<::mlir::pdl::ValueType>(::llvm::cast<::mlir::pdl::RangeType>(type).getElementType())))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be range of PDL handle to an `mlir::Type` or PDL handle for an `mlir::Value` values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps11(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::pdl::RangeType>(type))) && ((::llvm::isa<::mlir::pdl::PDLType>(::llvm::cast<::mlir::pdl::RangeType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be range of pdl type values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps12(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::pdl::ValueType>(type))) || (((::llvm::isa<::mlir::pdl::RangeType>(type))) && ((::llvm::isa<::mlir::pdl::ValueType>(::llvm::cast<::mlir::pdl::RangeType>(type).getElementType())))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be single element or range of PDL handle for an `mlir::Value`, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps13(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::pdl::ValueType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle for an `mlir::Value`, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps14(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::pdl::RangeType>(type))) && ((::llvm::isa<::mlir::pdl::OperationType>(::llvm::cast<::mlir::pdl::RangeType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be range of PDL handle to an `mlir::Operation *` values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps15(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::pdl::TypeType>(type))) || (((::llvm::isa<::mlir::pdl::RangeType>(type))) && ((::llvm::isa<::mlir::pdl::TypeType>(::llvm::cast<::mlir::pdl::RangeType>(type).getElementType())))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be single element or range of PDL handle to an `mlir::Type`, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps16(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::pdl::OperationType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of PDL handle to an `mlir::Operation *`, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::StringAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: string attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::BoolAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: bool attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((true)))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: any attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps4(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && ((!::llvm::cast<::mlir::IntegerAttr>(attr).getValue().isNegative()))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps4(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps5(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::UnitAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: unit attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps5(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps6(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: any type attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps6(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps7(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: type array attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps7(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps7(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps8(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::StringAttr>(attr))); }))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: string array attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps8(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps8(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps9(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::FunctionType>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((::llvm::isa<::mlir::FunctionType>(::llvm::cast<::mlir::TypeAttr>(attr).getValue())))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: type attribute of function type";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps9(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps9(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps10(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::DictionaryAttr>(attr))); }))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: Array of dictionary attributes";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps10(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps10(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps11(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::SymbolRefAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: symbol reference attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps11(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps11(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps12(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(16)))) && ((!::llvm::cast<::mlir::IntegerAttr>(attr).getValue().isNegative()))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 16-bit signless integer attribute whose value is non-negative";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps12(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps12(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps13(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::ArrayAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: array attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps13(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps13(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps14(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::DenseIntElementsAttr>(attr))) && ((::llvm::cast<::mlir::DenseIntElementsAttr>(attr).getType().getElementType().isSignlessInteger(32)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit signless integer elements attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps14(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps14(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps15(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))); }))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: type-array array attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps15(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps15(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_region_constraint_PDLInterpOps1(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_region_constraint_PDLInterpOps2(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItemsOrMore(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with at least 1 blocks";
  }
  return ::mlir::success();
}
} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ApplyConstraintOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ApplyConstraintOpGenericAdaptorBase::ApplyConstraintOpGenericAdaptorBase(ApplyConstraintOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> ApplyConstraintOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::llvm::StringRef ApplyConstraintOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

::mlir::BoolAttr ApplyConstraintOpGenericAdaptorBase::getIsNegatedAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::BoolAttr>(getProperties().isNegated);
  return attr;
}

bool ApplyConstraintOpGenericAdaptorBase::getIsNegated() {
  auto attr = getIsNegatedAttr();
  return attr.getValue();
}

} // namespace detail
ApplyConstraintOpAdaptor::ApplyConstraintOpAdaptor(ApplyConstraintOp op) : ApplyConstraintOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ApplyConstraintOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_isNegated = getProperties().isNegated; (void)tblgen_isNegated;
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitError(loc, "'pdl_interp.apply_constraint' op ""requires attribute 'name'");

  if (tblgen_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_name))))
    return emitError(loc, "'pdl_interp.apply_constraint' op ""attribute 'name' failed to satisfy constraint: string attribute");

  if (tblgen_isNegated && !((::llvm::isa<::mlir::BoolAttr>(tblgen_isNegated))))
    return emitError(loc, "'pdl_interp.apply_constraint' op ""attribute 'isNegated' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ApplyConstraintOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::MutableOperandRange ApplyConstraintOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ApplyConstraintOp::getODSResultIndexAndLength(unsigned index) {
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

::llvm::LogicalResult ApplyConstraintOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.isNegated;
       auto attr = dict.get("isNegated");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `isNegated` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.name;
       auto attr = dict.get("name");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `name` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ApplyConstraintOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.isNegated;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("isNegated",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.name;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("name",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ApplyConstraintOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.isNegated.getAsOpaquePointer()), 
    llvm::hash_value(prop.name.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ApplyConstraintOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "isNegated")
      return prop.isNegated;

    if (name == "name")
      return prop.name;
  return std::nullopt;
}

void ApplyConstraintOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "isNegated") {
       prop.isNegated = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.isNegated)>>(value);
       return;
    }

    if (name == "name") {
       prop.name = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.name)>>(value);
       return;
    }
}

void ApplyConstraintOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.isNegated) attrs.append("isNegated", prop.isNegated);

    if (prop.name) attrs.append("name", prop.name);
}

::llvm::LogicalResult ApplyConstraintOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getIsNegatedAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(attr, "isNegated", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(attr, "name", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult ApplyConstraintOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.isNegated)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.name)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ApplyConstraintOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.isNegated);
  writer.writeAttribute(prop.name);
}

::llvm::StringRef ApplyConstraintOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

bool ApplyConstraintOp::getIsNegated() {
  auto attr = getIsNegatedAttr();
  return attr.getValue();
}

void ApplyConstraintOp::setName(::llvm::StringRef attrValue) {
  getProperties().name = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void ApplyConstraintOp::setIsNegated(bool attrValue) {
  getProperties().isNegated = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void ApplyConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::StringAttr name, ::mlir::ValueRange args, ::mlir::BoolAttr isNegated, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().name = name;
  if (isNegated) {
    odsState.getOrAddProperties<Properties>().isNegated = isNegated;
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  odsState.addTypes(results);
}

void ApplyConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::llvm::StringRef name, ::mlir::ValueRange args, bool isNegated, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  odsState.getOrAddProperties<Properties>().isNegated = odsBuilder.getBoolAttr(isNegated);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  odsState.addTypes(results);
}

void ApplyConstraintOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ApplyConstraintOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

void ApplyConstraintOp::populateDefaultProperties(::mlir::OperationName opName, Properties &properties) {
  ::mlir::Builder odsBuilder(opName.getContext());
  if (!properties.isNegated)
    properties.isNegated = odsBuilder.getBoolAttr(false);
}

::llvm::LogicalResult ApplyConstraintOp::verifyInvariantsImpl() {
  auto tblgen_isNegated = getProperties().isNegated; (void)tblgen_isNegated;
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitOpError("requires attribute 'name'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(*this, tblgen_name, "name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_isNegated, "isNegated")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult ApplyConstraintOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ApplyConstraintOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> argsTypes;
  ::llvm::SmallVector<::mlir::Type, 1> resultsTypes;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (nameAttr) result.getOrAddProperties<ApplyConstraintOp::Properties>().name = nameAttr;
  if (parser.parseLParen())
    return ::mlir::failure();

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(argsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalColon())) {

  if (parser.parseTypeList(resultsTypes))
    return ::mlir::failure();
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  result.addTypes(resultsTypes);
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ApplyConstraintOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  _odsPrinter << "(";
  _odsPrinter << getArgs();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getArgs().getTypes();
  _odsPrinter << ")";
  if (!getResults().empty()) {
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getResults().getTypes();
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getIsNegatedAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("isNegated");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ApplyConstraintOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ApplyRewriteOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ApplyRewriteOpGenericAdaptorBase::ApplyRewriteOpGenericAdaptorBase(ApplyRewriteOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> ApplyRewriteOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::llvm::StringRef ApplyRewriteOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

} // namespace detail
ApplyRewriteOpAdaptor::ApplyRewriteOpAdaptor(ApplyRewriteOp op) : ApplyRewriteOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ApplyRewriteOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitError(loc, "'pdl_interp.apply_rewrite' op ""requires attribute 'name'");

  if (tblgen_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_name))))
    return emitError(loc, "'pdl_interp.apply_rewrite' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ApplyRewriteOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::MutableOperandRange ApplyRewriteOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ApplyRewriteOp::getODSResultIndexAndLength(unsigned index) {
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

::llvm::LogicalResult ApplyRewriteOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.name;
       auto attr = dict.get("name");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `name` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ApplyRewriteOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.name;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("name",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ApplyRewriteOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.name.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ApplyRewriteOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "name")
      return prop.name;
  return std::nullopt;
}

void ApplyRewriteOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "name") {
       prop.name = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.name)>>(value);
       return;
    }
}

void ApplyRewriteOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.name) attrs.append("name", prop.name);
}

::llvm::LogicalResult ApplyRewriteOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(attr, "name", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult ApplyRewriteOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.name)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ApplyRewriteOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.name);
}

::llvm::StringRef ApplyRewriteOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void ApplyRewriteOp::setName(::llvm::StringRef attrValue) {
  getProperties().name = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void ApplyRewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::StringAttr name, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().name = name;
  odsState.addTypes(results);
}

void ApplyRewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::llvm::StringRef name, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  odsState.addTypes(results);
}

void ApplyRewriteOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ApplyRewriteOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult ApplyRewriteOp::verifyInvariantsImpl() {
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitOpError("requires attribute 'name'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ApplyRewriteOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ApplyRewriteOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> argsTypes;
  ::llvm::SmallVector<::mlir::Type, 1> resultsTypes;

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (nameAttr) result.getOrAddProperties<ApplyRewriteOp::Properties>().name = nameAttr;
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(argsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalColon())) {

  if (parser.parseTypeList(resultsTypes))
    return ::mlir::failure();
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  result.addTypes(resultsTypes);
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ApplyRewriteOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  if (!getArgs().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getArgs();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getArgs().getTypes();
    _odsPrinter << ")";
  }
  if (!getResults().empty()) {
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getResults().getTypes();
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ApplyRewriteOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::AreEqualOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
AreEqualOpAdaptor::AreEqualOpAdaptor(AreEqualOp op) : AreEqualOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AreEqualOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void AreEqualOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void AreEqualOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AreEqualOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult AreEqualOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult AreEqualOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AreEqualOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allOperands;
  ::mlir::Type lhsRawType{};
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(&lhsRawType, 1);
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;
  [[maybe_unused]] ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawType = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  if (parser.resolveOperands(allOperands, ::llvm::concat<const ::mlir::Type>(::llvm::ArrayRef<::mlir::Type>(lhsTypes), ::llvm::ArrayRef<::mlir::Type>(lhsTypes[0])), allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AreEqualOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getLhs().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::pdl::PDLType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void AreEqualOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::AreEqualOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::BranchOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
BranchOpAdaptor::BranchOpAdaptor(BranchOp op) : BranchOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult BranchOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void BranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Block *dest) {
  odsState.addSuccessors(dest);
}

void BranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Block *dest) {
  odsState.addSuccessors(dest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BranchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult BranchOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult BranchOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BranchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Block *destSuccessor = nullptr;

  if (parser.parseSuccessor(destSuccessor))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  result.addSuccessors(destSuccessor);
  return ::mlir::success();
}

void BranchOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getDest();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void BranchOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::BranchOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckAttributeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CheckAttributeOpGenericAdaptorBase::CheckAttributeOpGenericAdaptorBase(CheckAttributeOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::Attribute CheckAttributeOpGenericAdaptorBase::getConstantValue() {
  auto attr = getConstantValueAttr();
  return attr;
}

} // namespace detail
CheckAttributeOpAdaptor::CheckAttributeOpAdaptor(CheckAttributeOp op) : CheckAttributeOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult CheckAttributeOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_constantValue = getProperties().constantValue; (void)tblgen_constantValue;
  if (!tblgen_constantValue) return emitError(loc, "'pdl_interp.check_attribute' op ""requires attribute 'constantValue'");

  if (tblgen_constantValue && !((true)))
    return emitError(loc, "'pdl_interp.check_attribute' op ""attribute 'constantValue' failed to satisfy constraint: any attribute");
  return ::mlir::success();
}

::llvm::LogicalResult CheckAttributeOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.constantValue;
       auto attr = dict.get("constantValue");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `constantValue` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CheckAttributeOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.constantValue;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("constantValue",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code CheckAttributeOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.constantValue.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CheckAttributeOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "constantValue")
      return prop.constantValue;
  return std::nullopt;
}

void CheckAttributeOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "constantValue") {
       prop.constantValue = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.constantValue)>>(value);
       return;
    }
}

void CheckAttributeOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.constantValue) attrs.append("constantValue", prop.constantValue);
}

::llvm::LogicalResult CheckAttributeOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getConstantValueAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps3(attr, "constantValue", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult CheckAttributeOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.constantValue)))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckAttributeOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.constantValue);
}

::mlir::Attribute CheckAttributeOp::getConstantValue() {
  auto attr = getConstantValueAttr();
  return attr;
}

void CheckAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value attribute, ::mlir::Attribute constantValue, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(attribute);
  odsState.getOrAddProperties<Properties>().constantValue = constantValue;
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value attribute, ::mlir::Attribute constantValue, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(attribute);
  odsState.getOrAddProperties<Properties>().constantValue = constantValue;
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckAttributeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<CheckAttributeOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult CheckAttributeOp::verifyInvariantsImpl() {
  auto tblgen_constantValue = getProperties().constantValue; (void)tblgen_constantValue;
  if (!tblgen_constantValue) return emitOpError("requires attribute 'constantValue'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps3(*this, tblgen_constantValue, "constantValue")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult CheckAttributeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CheckAttributeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand attributeRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> attributeOperands(&attributeRawOperand, 1);  ::llvm::SMLoc attributeOperandsLoc;
  (void)attributeOperandsLoc;
  ::mlir::Attribute constantValueAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;

  attributeOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(attributeRawOperand))
    return ::mlir::failure();
  if (parser.parseKeyword("is"))
    return ::mlir::failure();

  if (parser.parseAttribute(constantValueAttr, ::mlir::Type{}))
    return ::mlir::failure();
  if (constantValueAttr) result.getOrAddProperties<CheckAttributeOp::Properties>().constantValue = constantValueAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  if (parser.resolveOperands(attributeOperands, odsBuildableType0, attributeOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckAttributeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAttribute();
  _odsPrinter << ' ' << "is";
  _odsPrinter << ' ';
  _odsPrinter.printAttribute(getConstantValueAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("constantValue");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckAttributeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckAttributeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckOperandCountOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CheckOperandCountOpGenericAdaptorBase::CheckOperandCountOpGenericAdaptorBase(CheckOperandCountOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint32_t CheckOperandCountOpGenericAdaptorBase::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr CheckOperandCountOpGenericAdaptorBase::getCompareAtLeastAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::UnitAttr>(getProperties().compareAtLeast);
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool CheckOperandCountOpGenericAdaptorBase::getCompareAtLeast() {
  auto attr = getCompareAtLeastAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
CheckOperandCountOpAdaptor::CheckOperandCountOpAdaptor(CheckOperandCountOp op) : CheckOperandCountOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult CheckOperandCountOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_compareAtLeast = getProperties().compareAtLeast; (void)tblgen_compareAtLeast;
  auto tblgen_count = getProperties().count; (void)tblgen_count;
  if (!tblgen_count) return emitError(loc, "'pdl_interp.check_operand_count' op ""requires attribute 'count'");

  if (tblgen_count && !((((::llvm::isa<::mlir::IntegerAttr>(tblgen_count))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_count).getType().isSignlessInteger(32)))) && ((!::llvm::cast<::mlir::IntegerAttr>(tblgen_count).getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.check_operand_count' op ""attribute 'count' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");

  if (tblgen_compareAtLeast && !((::llvm::isa<::mlir::UnitAttr>(tblgen_compareAtLeast))))
    return emitError(loc, "'pdl_interp.check_operand_count' op ""attribute 'compareAtLeast' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

::llvm::LogicalResult CheckOperandCountOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.compareAtLeast;
       auto attr = dict.get("compareAtLeast");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `compareAtLeast` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.count;
       auto attr = dict.get("count");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `count` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CheckOperandCountOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.compareAtLeast;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("compareAtLeast",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.count;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("count",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code CheckOperandCountOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.compareAtLeast.getAsOpaquePointer()), 
    llvm::hash_value(prop.count.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CheckOperandCountOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "compareAtLeast")
      return prop.compareAtLeast;

    if (name == "count")
      return prop.count;
  return std::nullopt;
}

void CheckOperandCountOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "compareAtLeast") {
       prop.compareAtLeast = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.compareAtLeast)>>(value);
       return;
    }

    if (name == "count") {
       prop.count = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.count)>>(value);
       return;
    }
}

void CheckOperandCountOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.compareAtLeast) attrs.append("compareAtLeast", prop.compareAtLeast);

    if (prop.count) attrs.append("count", prop.count);
}

::llvm::LogicalResult CheckOperandCountOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getCompareAtLeastAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps5(attr, "compareAtLeast", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getCountAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(attr, "count", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult CheckOperandCountOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.compareAtLeast)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.count)))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckOperandCountOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.compareAtLeast);
  writer.writeAttribute(prop.count);
}

uint32_t CheckOperandCountOp::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

bool CheckOperandCountOp::getCompareAtLeast() {
  auto attr = getCompareAtLeastAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void CheckOperandCountOp::setCount(uint32_t attrValue) {
  getProperties().count = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void CheckOperandCountOp::setCompareAtLeast(bool attrValue) {
    auto &odsProp = getProperties().compareAtLeast;
    if (attrValue)
      odsProp = ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr);
    else
      odsProp = nullptr;
}

void CheckOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::IntegerAttr count, /*optional*/::mlir::UnitAttr compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().count = count;
  if (compareAtLeast) {
    odsState.getOrAddProperties<Properties>().compareAtLeast = compareAtLeast;
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::IntegerAttr count, /*optional*/::mlir::UnitAttr compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().count = count;
  if (compareAtLeast) {
    odsState.getOrAddProperties<Properties>().compareAtLeast = compareAtLeast;
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, uint32_t count, /*optional*/bool compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().count = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count);
  if (compareAtLeast) {
    odsState.getOrAddProperties<Properties>().compareAtLeast = ((compareAtLeast) ? odsBuilder.getUnitAttr() : nullptr);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, uint32_t count, /*optional*/bool compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().count = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count);
  if (compareAtLeast) {
    odsState.getOrAddProperties<Properties>().compareAtLeast = ((compareAtLeast) ? odsBuilder.getUnitAttr() : nullptr);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckOperandCountOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<CheckOperandCountOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult CheckOperandCountOp::verifyInvariantsImpl() {
  auto tblgen_compareAtLeast = getProperties().compareAtLeast; (void)tblgen_compareAtLeast;
  auto tblgen_count = getProperties().count; (void)tblgen_count;
  if (!tblgen_count) return emitOpError("requires attribute 'count'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(*this, tblgen_count, "count")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps5(*this, tblgen_compareAtLeast, "compareAtLeast")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult CheckOperandCountOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CheckOperandCountOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(&inputOpRawOperand, 1);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::IntegerAttr countAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperand))
    return ::mlir::failure();
  if (parser.parseKeyword("is"))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("at_least"))) {
    result.getOrAddProperties<CheckOperandCountOp::Properties>().compareAtLeast = parser.getBuilder().getUnitAttr();  }

  if (parser.parseCustomAttributeWithFallback(countAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (countAttr) result.getOrAddProperties<CheckOperandCountOp::Properties>().count = countAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckOperandCountOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "is";
  if ((getCompareAtLeastAttr() && getCompareAtLeastAttr() != ((false) ? ::mlir::OpBuilder((*this)->getContext()).getUnitAttr() : nullptr))) {
    _odsPrinter << ' ' << "at_least";
  }
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCountAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("compareAtLeast");
  elidedAttrs.push_back("count");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getCompareAtLeastAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("compareAtLeast");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckOperandCountOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckOperandCountOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckOperationNameOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CheckOperationNameOpGenericAdaptorBase::CheckOperationNameOpGenericAdaptorBase(CheckOperationNameOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::llvm::StringRef CheckOperationNameOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

} // namespace detail
CheckOperationNameOpAdaptor::CheckOperationNameOpAdaptor(CheckOperationNameOp op) : CheckOperationNameOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult CheckOperationNameOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitError(loc, "'pdl_interp.check_operation_name' op ""requires attribute 'name'");

  if (tblgen_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_name))))
    return emitError(loc, "'pdl_interp.check_operation_name' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

::llvm::LogicalResult CheckOperationNameOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.name;
       auto attr = dict.get("name");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `name` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CheckOperationNameOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.name;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("name",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code CheckOperationNameOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.name.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CheckOperationNameOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "name")
      return prop.name;
  return std::nullopt;
}

void CheckOperationNameOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "name") {
       prop.name = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.name)>>(value);
       return;
    }
}

void CheckOperationNameOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.name) attrs.append("name", prop.name);
}

::llvm::LogicalResult CheckOperationNameOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(attr, "name", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult CheckOperationNameOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.name)))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckOperationNameOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.name);
}

::llvm::StringRef CheckOperationNameOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void CheckOperationNameOp::setName(::llvm::StringRef attrValue) {
  getProperties().name = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void CheckOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::StringAttr name, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().name = name;
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::StringAttr name, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().name = name;
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::llvm::StringRef name, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::llvm::StringRef name, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckOperationNameOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<CheckOperationNameOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult CheckOperationNameOp::verifyInvariantsImpl() {
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitOpError("requires attribute 'name'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult CheckOperationNameOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CheckOperationNameOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(&inputOpRawOperand, 1);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::StringAttr nameAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperand))
    return ::mlir::failure();
  if (parser.parseKeyword("is"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (nameAttr) result.getOrAddProperties<CheckOperationNameOp::Properties>().name = nameAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckOperationNameOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "is";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckOperationNameOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckOperationNameOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckResultCountOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CheckResultCountOpGenericAdaptorBase::CheckResultCountOpGenericAdaptorBase(CheckResultCountOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint32_t CheckResultCountOpGenericAdaptorBase::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr CheckResultCountOpGenericAdaptorBase::getCompareAtLeastAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::UnitAttr>(getProperties().compareAtLeast);
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool CheckResultCountOpGenericAdaptorBase::getCompareAtLeast() {
  auto attr = getCompareAtLeastAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
CheckResultCountOpAdaptor::CheckResultCountOpAdaptor(CheckResultCountOp op) : CheckResultCountOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult CheckResultCountOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_compareAtLeast = getProperties().compareAtLeast; (void)tblgen_compareAtLeast;
  auto tblgen_count = getProperties().count; (void)tblgen_count;
  if (!tblgen_count) return emitError(loc, "'pdl_interp.check_result_count' op ""requires attribute 'count'");

  if (tblgen_count && !((((::llvm::isa<::mlir::IntegerAttr>(tblgen_count))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_count).getType().isSignlessInteger(32)))) && ((!::llvm::cast<::mlir::IntegerAttr>(tblgen_count).getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.check_result_count' op ""attribute 'count' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");

  if (tblgen_compareAtLeast && !((::llvm::isa<::mlir::UnitAttr>(tblgen_compareAtLeast))))
    return emitError(loc, "'pdl_interp.check_result_count' op ""attribute 'compareAtLeast' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

::llvm::LogicalResult CheckResultCountOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.compareAtLeast;
       auto attr = dict.get("compareAtLeast");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `compareAtLeast` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.count;
       auto attr = dict.get("count");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `count` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CheckResultCountOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.compareAtLeast;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("compareAtLeast",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.count;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("count",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code CheckResultCountOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.compareAtLeast.getAsOpaquePointer()), 
    llvm::hash_value(prop.count.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CheckResultCountOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "compareAtLeast")
      return prop.compareAtLeast;

    if (name == "count")
      return prop.count;
  return std::nullopt;
}

void CheckResultCountOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "compareAtLeast") {
       prop.compareAtLeast = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.compareAtLeast)>>(value);
       return;
    }

    if (name == "count") {
       prop.count = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.count)>>(value);
       return;
    }
}

void CheckResultCountOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.compareAtLeast) attrs.append("compareAtLeast", prop.compareAtLeast);

    if (prop.count) attrs.append("count", prop.count);
}

::llvm::LogicalResult CheckResultCountOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getCompareAtLeastAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps5(attr, "compareAtLeast", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getCountAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(attr, "count", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult CheckResultCountOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.compareAtLeast)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.count)))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckResultCountOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.compareAtLeast);
  writer.writeAttribute(prop.count);
}

uint32_t CheckResultCountOp::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

bool CheckResultCountOp::getCompareAtLeast() {
  auto attr = getCompareAtLeastAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void CheckResultCountOp::setCount(uint32_t attrValue) {
  getProperties().count = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void CheckResultCountOp::setCompareAtLeast(bool attrValue) {
    auto &odsProp = getProperties().compareAtLeast;
    if (attrValue)
      odsProp = ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr);
    else
      odsProp = nullptr;
}

void CheckResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::IntegerAttr count, /*optional*/::mlir::UnitAttr compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().count = count;
  if (compareAtLeast) {
    odsState.getOrAddProperties<Properties>().compareAtLeast = compareAtLeast;
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::IntegerAttr count, /*optional*/::mlir::UnitAttr compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().count = count;
  if (compareAtLeast) {
    odsState.getOrAddProperties<Properties>().compareAtLeast = compareAtLeast;
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, uint32_t count, /*optional*/bool compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().count = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count);
  if (compareAtLeast) {
    odsState.getOrAddProperties<Properties>().compareAtLeast = ((compareAtLeast) ? odsBuilder.getUnitAttr() : nullptr);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, uint32_t count, /*optional*/bool compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().count = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count);
  if (compareAtLeast) {
    odsState.getOrAddProperties<Properties>().compareAtLeast = ((compareAtLeast) ? odsBuilder.getUnitAttr() : nullptr);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckResultCountOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<CheckResultCountOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult CheckResultCountOp::verifyInvariantsImpl() {
  auto tblgen_compareAtLeast = getProperties().compareAtLeast; (void)tblgen_compareAtLeast;
  auto tblgen_count = getProperties().count; (void)tblgen_count;
  if (!tblgen_count) return emitOpError("requires attribute 'count'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(*this, tblgen_count, "count")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps5(*this, tblgen_compareAtLeast, "compareAtLeast")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult CheckResultCountOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CheckResultCountOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(&inputOpRawOperand, 1);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::IntegerAttr countAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperand))
    return ::mlir::failure();
  if (parser.parseKeyword("is"))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("at_least"))) {
    result.getOrAddProperties<CheckResultCountOp::Properties>().compareAtLeast = parser.getBuilder().getUnitAttr();  }

  if (parser.parseCustomAttributeWithFallback(countAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (countAttr) result.getOrAddProperties<CheckResultCountOp::Properties>().count = countAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckResultCountOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "is";
  if ((getCompareAtLeastAttr() && getCompareAtLeastAttr() != ((false) ? ::mlir::OpBuilder((*this)->getContext()).getUnitAttr() : nullptr))) {
    _odsPrinter << ' ' << "at_least";
  }
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCountAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("compareAtLeast");
  elidedAttrs.push_back("count");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getCompareAtLeastAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("compareAtLeast");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckResultCountOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckResultCountOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckTypeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CheckTypeOpGenericAdaptorBase::CheckTypeOpGenericAdaptorBase(CheckTypeOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::Type CheckTypeOpGenericAdaptorBase::getType() {
  auto attr = getTypeAttr();
  return ::llvm::cast<::mlir::Type>(attr.getValue());
}

} // namespace detail
CheckTypeOpAdaptor::CheckTypeOpAdaptor(CheckTypeOp op) : CheckTypeOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult CheckTypeOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_type = getProperties().type; (void)tblgen_type;
  if (!tblgen_type) return emitError(loc, "'pdl_interp.check_type' op ""requires attribute 'type'");

  if (tblgen_type && !(((::llvm::isa<::mlir::TypeAttr>(tblgen_type))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(tblgen_type).getValue()))) && ((true))))
    return emitError(loc, "'pdl_interp.check_type' op ""attribute 'type' failed to satisfy constraint: any type attribute");
  return ::mlir::success();
}

::llvm::LogicalResult CheckTypeOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.type;
       auto attr = dict.get("type");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `type` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CheckTypeOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.type;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("type",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code CheckTypeOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.type.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CheckTypeOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "type")
      return prop.type;
  return std::nullopt;
}

void CheckTypeOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "type") {
       prop.type = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.type)>>(value);
       return;
    }
}

void CheckTypeOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.type) attrs.append("type", prop.type);
}

::llvm::LogicalResult CheckTypeOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getTypeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps6(attr, "type", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult CheckTypeOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.type)))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckTypeOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.type);
}

::mlir::Type CheckTypeOp::getType() {
  auto attr = getTypeAttr();
  return ::llvm::cast<::mlir::Type>(attr.getValue());
}

void CheckTypeOp::setType(::mlir::Type attrValue) {
  getProperties().type = ::mlir::TypeAttr::get(attrValue);
}

void CheckTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::TypeAttr type, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().type = type;
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::TypeAttr type, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().type = type;
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Type type, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().type = ::mlir::TypeAttr::get(type);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Type type, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().type = ::mlir::TypeAttr::get(type);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<CheckTypeOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult CheckTypeOp::verifyInvariantsImpl() {
  auto tblgen_type = getProperties().type; (void)tblgen_type;
  if (!tblgen_type) return emitOpError("requires attribute 'type'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps6(*this, tblgen_type, "type")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult CheckTypeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CheckTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(&valueRawOperand, 1);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::TypeAttr typeAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperand))
    return ::mlir::failure();
  if (parser.parseKeyword("is"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(typeAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (typeAttr) result.getOrAddProperties<CheckTypeOp::Properties>().type = typeAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  if (parser.resolveOperands(valueOperands, odsBuildableType0, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckTypeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << "is";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getTypeAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("type");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckTypeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckTypeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckTypesOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CheckTypesOpGenericAdaptorBase::CheckTypesOpGenericAdaptorBase(CheckTypesOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::ArrayAttr CheckTypesOpGenericAdaptorBase::getTypes() {
  auto attr = getTypesAttr();
  return attr;
}

} // namespace detail
CheckTypesOpAdaptor::CheckTypesOpAdaptor(CheckTypesOp op) : CheckTypesOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult CheckTypesOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_types = getProperties().types; (void)tblgen_types;
  if (!tblgen_types) return emitError(loc, "'pdl_interp.check_types' op ""requires attribute 'types'");

  if (tblgen_types && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_types))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_types), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'pdl_interp.check_types' op ""attribute 'types' failed to satisfy constraint: type array attribute");
  return ::mlir::success();
}

::llvm::LogicalResult CheckTypesOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.types;
       auto attr = dict.get("types");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `types` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CheckTypesOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.types;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("types",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code CheckTypesOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.types.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CheckTypesOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "types")
      return prop.types;
  return std::nullopt;
}

void CheckTypesOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "types") {
       prop.types = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.types)>>(value);
       return;
    }
}

void CheckTypesOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.types) attrs.append("types", prop.types);
}

::llvm::LogicalResult CheckTypesOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getTypesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps7(attr, "types", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult CheckTypesOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.types)))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckTypesOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.types);
}

::mlir::ArrayAttr CheckTypesOp::getTypes() {
  auto attr = getTypesAttr();
  return attr;
}

void CheckTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::ArrayAttr types, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().types = types;
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::ArrayAttr types, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().types = types;
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckTypesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<CheckTypesOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult CheckTypesOp::verifyInvariantsImpl() {
  auto tblgen_types = getProperties().types; (void)tblgen_types;
  if (!tblgen_types) return emitOpError("requires attribute 'types'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps7(*this, tblgen_types, "types")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult CheckTypesOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CheckTypesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(&valueRawOperand, 1);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::ArrayAttr typesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperand))
    return ::mlir::failure();
  if (parser.parseKeyword("are"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(typesAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (typesAttr) result.getOrAddProperties<CheckTypesOp::Properties>().types = typesAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::TypeType>());
  if (parser.resolveOperands(valueOperands, odsBuildableType0, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckTypesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << "are";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getTypesAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("types");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckTypesOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckTypesOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ContinueOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
ContinueOpAdaptor::ContinueOpAdaptor(ContinueOp op) : ContinueOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ContinueOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void ContinueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void ContinueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ContinueOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult ContinueOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::llvm::LogicalResult ContinueOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ContinueOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void ContinueOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void ContinueOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ContinueOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateAttributeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CreateAttributeOpGenericAdaptorBase::CreateAttributeOpGenericAdaptorBase(CreateAttributeOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::Attribute CreateAttributeOpGenericAdaptorBase::getValue() {
  auto attr = getValueAttr();
  return attr;
}

} // namespace detail
CreateAttributeOpAdaptor::CreateAttributeOpAdaptor(CreateAttributeOp op) : CreateAttributeOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult CreateAttributeOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitError(loc, "'pdl_interp.create_attribute' op ""requires attribute 'value'");

  if (tblgen_value && !((true)))
    return emitError(loc, "'pdl_interp.create_attribute' op ""attribute 'value' failed to satisfy constraint: any attribute");
  return ::mlir::success();
}

::llvm::LogicalResult CreateAttributeOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.value;
       auto attr = dict.get("value");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `value` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CreateAttributeOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.value;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("value",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code CreateAttributeOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.value.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CreateAttributeOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "value")
      return prop.value;
  return std::nullopt;
}

void CreateAttributeOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "value") {
       prop.value = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.value)>>(value);
       return;
    }
}

void CreateAttributeOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.value) attrs.append("value", prop.value);
}

::llvm::LogicalResult CreateAttributeOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getValueAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps3(attr, "value", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult CreateAttributeOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.value)))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateAttributeOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.value);
}

::mlir::Attribute CreateAttributeOp::getValue() {
  auto attr = getValueAttr();
  return attr;
}

void CreateAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Attribute value) {
      build(odsBuilder, odsState, odsBuilder.getType<pdl::AttributeType>(), value);
    
}

void CreateAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type attribute, ::mlir::Attribute value) {
  odsState.getOrAddProperties<Properties>().value = value;
  odsState.addTypes(attribute);
}

void CreateAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Attribute value) {
  odsState.getOrAddProperties<Properties>().value = value;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateAttributeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<CreateAttributeOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult CreateAttributeOp::verifyInvariantsImpl() {
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitOpError("requires attribute 'value'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps3(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult CreateAttributeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CreateAttributeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Attribute valueAttr;

  if (parser.parseAttribute(valueAttr, ::mlir::Type{}))
    return ::mlir::failure();
  if (valueAttr) result.getOrAddProperties<CreateAttributeOp::Properties>().value = valueAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void CreateAttributeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttribute(getValueAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("value");
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
}

void CreateAttributeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CreateAttributeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateOperationOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CreateOperationOpGenericAdaptorBase::CreateOperationOpGenericAdaptorBase(CreateOperationOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> CreateOperationOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::llvm::StringRef CreateOperationOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

::mlir::ArrayAttr CreateOperationOpGenericAdaptorBase::getInputAttributeNames() {
  auto attr = getInputAttributeNamesAttr();
  return attr;
}

::mlir::UnitAttr CreateOperationOpGenericAdaptorBase::getInferredResultTypesAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::UnitAttr>(getProperties().inferredResultTypes);
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool CreateOperationOpGenericAdaptorBase::getInferredResultTypes() {
  auto attr = getInferredResultTypesAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
CreateOperationOpAdaptor::CreateOperationOpAdaptor(CreateOperationOp op) : CreateOperationOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult CreateOperationOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_inferredResultTypes = getProperties().inferredResultTypes; (void)tblgen_inferredResultTypes;
  auto tblgen_inputAttributeNames = getProperties().inputAttributeNames; (void)tblgen_inputAttributeNames;
  if (!tblgen_inputAttributeNames) return emitError(loc, "'pdl_interp.create_operation' op ""requires attribute 'inputAttributeNames'");
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitError(loc, "'pdl_interp.create_operation' op ""requires attribute 'name'");

  if (tblgen_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_name))))
    return emitError(loc, "'pdl_interp.create_operation' op ""attribute 'name' failed to satisfy constraint: string attribute");

  if (tblgen_inputAttributeNames && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_inputAttributeNames))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_inputAttributeNames), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::StringAttr>(attr))); }))))
    return emitError(loc, "'pdl_interp.create_operation' op ""attribute 'inputAttributeNames' failed to satisfy constraint: string array attribute");

  if (tblgen_inferredResultTypes && !((::llvm::isa<::mlir::UnitAttr>(tblgen_inferredResultTypes))))
    return emitError(loc, "'pdl_interp.create_operation' op ""attribute 'inferredResultTypes' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CreateOperationOp::getODSOperandIndexAndLength(unsigned index) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::MutableOperandRange CreateOperationOp::getInputOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::mlir::MutableOperandRange CreateOperationOp::getInputAttributesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::mlir::MutableOperandRange CreateOperationOp::getInputResultTypesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::llvm::LogicalResult CreateOperationOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.inferredResultTypes;
       auto attr = dict.get("inferredResultTypes");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `inferredResultTypes` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.inputAttributeNames;
       auto attr = dict.get("inputAttributeNames");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `inputAttributeNames` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.name;
       auto attr = dict.get("name");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `name` in property conversion: " << attr;
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

::mlir::Attribute CreateOperationOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.inferredResultTypes;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("inferredResultTypes",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.inputAttributeNames;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("inputAttributeNames",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.name;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("name",
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

llvm::hash_code CreateOperationOp::computePropertiesHash(const Properties &prop) {
  auto hash_operandSegmentSizes = [] (const auto &propStorage) -> llvm::hash_code {
    return ::llvm::hash_combine_range(std::begin(propStorage), std::end(propStorage));;
  };
  return llvm::hash_combine(
    llvm::hash_value(prop.inferredResultTypes.getAsOpaquePointer()), 
    llvm::hash_value(prop.inputAttributeNames.getAsOpaquePointer()), 
    llvm::hash_value(prop.name.getAsOpaquePointer()), 
    hash_operandSegmentSizes(prop.operandSegmentSizes));
}

std::optional<mlir::Attribute> CreateOperationOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "inferredResultTypes")
      return prop.inferredResultTypes;

    if (name == "inputAttributeNames")
      return prop.inputAttributeNames;

    if (name == "name")
      return prop.name;
    if (name == "operand_segment_sizes" || name == "operandSegmentSizes") return [&]() -> ::mlir::Attribute { return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes); }();
  return std::nullopt;
}

void CreateOperationOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "inferredResultTypes") {
       prop.inferredResultTypes = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.inferredResultTypes)>>(value);
       return;
    }

    if (name == "inputAttributeNames") {
       prop.inputAttributeNames = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.inputAttributeNames)>>(value);
       return;
    }

    if (name == "name") {
       prop.name = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.name)>>(value);
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

void CreateOperationOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.inferredResultTypes) attrs.append("inferredResultTypes", prop.inferredResultTypes);

    if (prop.inputAttributeNames) attrs.append("inputAttributeNames", prop.inputAttributeNames);

    if (prop.name) attrs.append("name", prop.name);
  attrs.append("operandSegmentSizes", [&]() -> ::mlir::Attribute { return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes); }());
}

::llvm::LogicalResult CreateOperationOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getInferredResultTypesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps5(attr, "inferredResultTypes", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getInputAttributeNamesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps8(attr, "inputAttributeNames", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(attr, "name", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult CreateOperationOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.inferredResultTypes)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.inputAttributeNames)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.name)))
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

void CreateOperationOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.inferredResultTypes);
  writer.writeAttribute(prop.inputAttributeNames);
  writer.writeAttribute(prop.name);

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

::llvm::StringRef CreateOperationOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

::mlir::ArrayAttr CreateOperationOp::getInputAttributeNames() {
  auto attr = getInputAttributeNamesAttr();
  return attr;
}

bool CreateOperationOp::getInferredResultTypes() {
  auto attr = getInferredResultTypesAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void CreateOperationOp::setName(::llvm::StringRef attrValue) {
  getProperties().name = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void CreateOperationOp::setInferredResultTypes(bool attrValue) {
    auto &odsProp = getProperties().inferredResultTypes;
    if (attrValue)
      odsProp = ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr);
    else
      odsProp = nullptr;
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringRef name, ValueRange types, bool inferredResultTypes, ValueRange operands, ValueRange attributes, ArrayAttr attributeNames) {
      build(odsBuilder, odsState, odsBuilder.getType<pdl::OperationType>(), name,
            operands, attributes, attributeNames, types, inferredResultTypes);
    
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultOp, ::mlir::StringAttr name, ::mlir::ValueRange inputOperands, ::mlir::ValueRange inputAttributes, ::mlir::ArrayAttr inputAttributeNames, ::mlir::ValueRange inputResultTypes, /*optional*/::mlir::UnitAttr inferredResultTypes) {
  odsState.addOperands(inputOperands);
  odsState.addOperands(inputAttributes);
  odsState.addOperands(inputResultTypes);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(inputOperands.size()), static_cast<int32_t>(inputAttributes.size()), static_cast<int32_t>(inputResultTypes.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().name = name;
  odsState.getOrAddProperties<Properties>().inputAttributeNames = inputAttributeNames;
  if (inferredResultTypes) {
    odsState.getOrAddProperties<Properties>().inferredResultTypes = inferredResultTypes;
  }
  odsState.addTypes(resultOp);
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::ValueRange inputOperands, ::mlir::ValueRange inputAttributes, ::mlir::ArrayAttr inputAttributeNames, ::mlir::ValueRange inputResultTypes, /*optional*/::mlir::UnitAttr inferredResultTypes) {
  odsState.addOperands(inputOperands);
  odsState.addOperands(inputAttributes);
  odsState.addOperands(inputResultTypes);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(inputOperands.size()), static_cast<int32_t>(inputAttributes.size()), static_cast<int32_t>(inputResultTypes.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().name = name;
  odsState.getOrAddProperties<Properties>().inputAttributeNames = inputAttributeNames;
  if (inferredResultTypes) {
    odsState.getOrAddProperties<Properties>().inferredResultTypes = inferredResultTypes;
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultOp, ::llvm::StringRef name, ::mlir::ValueRange inputOperands, ::mlir::ValueRange inputAttributes, ::mlir::ArrayAttr inputAttributeNames, ::mlir::ValueRange inputResultTypes, /*optional*/bool inferredResultTypes) {
  odsState.addOperands(inputOperands);
  odsState.addOperands(inputAttributes);
  odsState.addOperands(inputResultTypes);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(inputOperands.size()), static_cast<int32_t>(inputAttributes.size()), static_cast<int32_t>(inputResultTypes.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  odsState.getOrAddProperties<Properties>().inputAttributeNames = inputAttributeNames;
  if (inferredResultTypes) {
    odsState.getOrAddProperties<Properties>().inferredResultTypes = ((inferredResultTypes) ? odsBuilder.getUnitAttr() : nullptr);
  }
  odsState.addTypes(resultOp);
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, ::mlir::ValueRange inputOperands, ::mlir::ValueRange inputAttributes, ::mlir::ArrayAttr inputAttributeNames, ::mlir::ValueRange inputResultTypes, /*optional*/bool inferredResultTypes) {
  odsState.addOperands(inputOperands);
  odsState.addOperands(inputAttributes);
  odsState.addOperands(inputResultTypes);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(inputOperands.size()), static_cast<int32_t>(inputAttributes.size()), static_cast<int32_t>(inputResultTypes.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  odsState.getOrAddProperties<Properties>().inputAttributeNames = inputAttributeNames;
  if (inferredResultTypes) {
    odsState.getOrAddProperties<Properties>().inferredResultTypes = ((inferredResultTypes) ? odsBuilder.getUnitAttr() : nullptr);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateOperationOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<CreateOperationOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult CreateOperationOp::verifyInvariantsImpl() {
  auto tblgen_inferredResultTypes = getProperties().inferredResultTypes; (void)tblgen_inferredResultTypes;
  auto tblgen_inputAttributeNames = getProperties().inputAttributeNames; (void)tblgen_inputAttributeNames;
  if (!tblgen_inputAttributeNames) return emitOpError("requires attribute 'inputAttributeNames'");
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitOpError("requires attribute 'name'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(*this, tblgen_name, "name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps8(*this, tblgen_inputAttributeNames, "inputAttributeNames")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps5(*this, tblgen_inferredResultTypes, "inferredResultTypes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult CreateOperationOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CreateOperationOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputOperandsOperands;
  ::llvm::SMLoc inputOperandsOperandsLoc;
  (void)inputOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> inputOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputAttributesOperands;
  ::llvm::SMLoc inputAttributesOperandsLoc;
  (void)inputAttributesOperandsLoc;
  ::mlir::ArrayAttr inputAttributeNamesAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputResultTypesOperands;
  ::llvm::SMLoc inputResultTypesOperandsLoc;
  (void)inputResultTypesOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> inputResultTypesTypes;
  ::mlir::UnitAttr inferredResultTypesAttr;

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (nameAttr) result.getOrAddProperties<CreateOperationOp::Properties>().name = nameAttr;
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  inputOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(inputOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  {
    inputAttributesOperandsLoc = parser.getCurrentLocation();
    auto odsResult = parseCreateOperationOpAttributes(parser, inputAttributesOperands, inputAttributeNamesAttr);
    if (odsResult) return ::mlir::failure();
    result.getOrAddProperties<CreateOperationOp::Properties>().inputAttributeNames = inputAttributeNamesAttr;
  }
  {
    inputResultTypesOperandsLoc = parser.getCurrentLocation();
    auto odsResult = parseCreateOperationOpResults(parser, inputResultTypesOperands, inputResultTypesTypes, inferredResultTypesAttr);
    if (odsResult) return ::mlir::failure();
    if (inferredResultTypesAttr)
      result.getOrAddProperties<CreateOperationOp::Properties>().inferredResultTypes = inferredResultTypesAttr;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(inputOperandsOperands.size()), static_cast<int32_t>(inputAttributesOperands.size()), static_cast<int32_t>(inputResultTypesOperands.size())}), result.getOrAddProperties<CreateOperationOp::Properties>().operandSegmentSizes.begin());
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(inputOperandsOperands, inputOperandsTypes, inputOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(inputAttributesOperands, odsBuildableType1, inputAttributesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(inputResultTypesOperands, inputResultTypesTypes, inputResultTypesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateOperationOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  if (!getInputOperands().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getInputOperands();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getInputOperands().getTypes();
    _odsPrinter << ")";
  }
  printCreateOperationOpAttributes(_odsPrinter, *this, getInputAttributes(), getInputAttributeNamesAttr());
  _odsPrinter << ' ';
  printCreateOperationOpResults(_odsPrinter, *this, getInputResultTypes(), getInputResultTypes().getTypes(), getInferredResultTypesAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operandSegmentSizes");
  elidedAttrs.push_back("name");
  elidedAttrs.push_back("inputAttributeNames");
  elidedAttrs.push_back("inferredResultTypes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getInferredResultTypesAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("inferredResultTypes");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CreateOperationOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateRangeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
std::pair<unsigned, unsigned> CreateRangeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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
CreateRangeOpAdaptor::CreateRangeOpAdaptor(CreateRangeOp op) : CreateRangeOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult CreateRangeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CreateRangeOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::MutableOperandRange CreateRangeOp::getArgumentsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

void CreateRangeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange arguments) {
  odsState.addOperands(arguments);
  odsState.addTypes(result);
}

void CreateRangeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult CreateRangeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult CreateRangeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CreateRangeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argumentsOperands;
  ::llvm::SMLoc argumentsOperandsLoc;
  (void)argumentsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> argumentsTypes;
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);

  argumentsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argumentsOperands))
    return ::mlir::failure();
  if (!argumentsOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(argumentsTypes))
    return ::mlir::failure();
  }
  {
    auto odsResult = parseRangeType(parser, argumentsTypes, resultRawType);
    if (odsResult) return ::mlir::failure();
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(argumentsOperands, argumentsTypes, argumentsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateRangeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!getArguments().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << getArguments();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getArguments().getTypes();
  }
  _odsPrinter << ' ';
  printRangeType(_odsPrinter, *this, getArguments().getTypes(), getResult().getType());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void CreateRangeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CreateRangeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateTypeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CreateTypeOpGenericAdaptorBase::CreateTypeOpGenericAdaptorBase(CreateTypeOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::Type CreateTypeOpGenericAdaptorBase::getValue() {
  auto attr = getValueAttr();
  return ::llvm::cast<::mlir::Type>(attr.getValue());
}

} // namespace detail
CreateTypeOpAdaptor::CreateTypeOpAdaptor(CreateTypeOp op) : CreateTypeOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult CreateTypeOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitError(loc, "'pdl_interp.create_type' op ""requires attribute 'value'");

  if (tblgen_value && !(((::llvm::isa<::mlir::TypeAttr>(tblgen_value))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(tblgen_value).getValue()))) && ((true))))
    return emitError(loc, "'pdl_interp.create_type' op ""attribute 'value' failed to satisfy constraint: any type attribute");
  return ::mlir::success();
}

::llvm::LogicalResult CreateTypeOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.value;
       auto attr = dict.get("value");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `value` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CreateTypeOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.value;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("value",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code CreateTypeOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.value.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CreateTypeOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "value")
      return prop.value;
  return std::nullopt;
}

void CreateTypeOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "value") {
       prop.value = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.value)>>(value);
       return;
    }
}

void CreateTypeOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.value) attrs.append("value", prop.value);
}

::llvm::LogicalResult CreateTypeOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getValueAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps6(attr, "value", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult CreateTypeOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.value)))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateTypeOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.value);
}

::mlir::Type CreateTypeOp::getValue() {
  auto attr = getValueAttr();
  return ::llvm::cast<::mlir::Type>(attr.getValue());
}

void CreateTypeOp::setValue(::mlir::Type attrValue) {
  getProperties().value = ::mlir::TypeAttr::get(attrValue);
}

void CreateTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeAttr type) {
      build(odsBuilder, odsState, odsBuilder.getType<pdl::TypeType>(), type);
    
}

void CreateTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::TypeAttr value) {
  odsState.getOrAddProperties<Properties>().value = value;
  odsState.addTypes(result);
}

void CreateTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::TypeAttr value) {
  odsState.getOrAddProperties<Properties>().value = value;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Type value) {
  odsState.getOrAddProperties<Properties>().value = ::mlir::TypeAttr::get(value);
  odsState.addTypes(result);
}

void CreateTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Type value) {
  odsState.getOrAddProperties<Properties>().value = ::mlir::TypeAttr::get(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<CreateTypeOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult CreateTypeOp::verifyInvariantsImpl() {
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitOpError("requires attribute 'value'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps6(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult CreateTypeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CreateTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::TypeAttr valueAttr;

  if (parser.parseCustomAttributeWithFallback(valueAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (valueAttr) result.getOrAddProperties<CreateTypeOp::Properties>().value = valueAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void CreateTypeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getValueAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("value");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void CreateTypeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CreateTypeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateTypesOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CreateTypesOpGenericAdaptorBase::CreateTypesOpGenericAdaptorBase(CreateTypesOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::ArrayAttr CreateTypesOpGenericAdaptorBase::getValue() {
  auto attr = getValueAttr();
  return attr;
}

} // namespace detail
CreateTypesOpAdaptor::CreateTypesOpAdaptor(CreateTypesOp op) : CreateTypesOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult CreateTypesOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitError(loc, "'pdl_interp.create_types' op ""requires attribute 'value'");

  if (tblgen_value && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_value))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_value), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'pdl_interp.create_types' op ""attribute 'value' failed to satisfy constraint: type array attribute");
  return ::mlir::success();
}

::llvm::LogicalResult CreateTypesOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.value;
       auto attr = dict.get("value");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `value` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CreateTypesOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.value;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("value",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code CreateTypesOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.value.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CreateTypesOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "value")
      return prop.value;
  return std::nullopt;
}

void CreateTypesOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "value") {
       prop.value = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.value)>>(value);
       return;
    }
}

void CreateTypesOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.value) attrs.append("value", prop.value);
}

::llvm::LogicalResult CreateTypesOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getValueAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps7(attr, "value", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult CreateTypesOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.value)))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateTypesOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.value);
}

::mlir::ArrayAttr CreateTypesOp::getValue() {
  auto attr = getValueAttr();
  return attr;
}

void CreateTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayAttr type) {
      build(odsBuilder, odsState,
            pdl::RangeType::get(odsBuilder.getType<pdl::TypeType>()), type);
    
}

void CreateTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ArrayAttr value) {
  odsState.getOrAddProperties<Properties>().value = value;
  odsState.addTypes(result);
}

void CreateTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ArrayAttr value) {
  odsState.getOrAddProperties<Properties>().value = value;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateTypesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<CreateTypesOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult CreateTypesOp::verifyInvariantsImpl() {
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitOpError("requires attribute 'value'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps7(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps6(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult CreateTypesOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CreateTypesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::ArrayAttr valueAttr;

  if (parser.parseCustomAttributeWithFallback(valueAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (valueAttr) result.getOrAddProperties<CreateTypesOp::Properties>().value = valueAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::TypeType>());
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void CreateTypesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getValueAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("value");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void CreateTypesOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CreateTypesOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::EraseOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
EraseOpAdaptor::EraseOpAdaptor(EraseOp op) : EraseOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult EraseOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void EraseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp) {
  odsState.addOperands(inputOp);
}

void EraseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp) {
  odsState.addOperands(inputOp);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EraseOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult EraseOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult EraseOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult EraseOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(&inputOpRawOperand, 1);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void EraseOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::EraseOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ExtractOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ExtractOpGenericAdaptorBase::ExtractOpGenericAdaptorBase(ExtractOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint32_t ExtractOpGenericAdaptorBase::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ExtractOpAdaptor::ExtractOpAdaptor(ExtractOp op) : ExtractOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ExtractOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_index = getProperties().index; (void)tblgen_index;
  if (!tblgen_index) return emitError(loc, "'pdl_interp.extract' op ""requires attribute 'index'");

  if (tblgen_index && !((((::llvm::isa<::mlir::IntegerAttr>(tblgen_index))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_index).getType().isSignlessInteger(32)))) && ((!::llvm::cast<::mlir::IntegerAttr>(tblgen_index).getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.extract' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  return ::mlir::success();
}

::llvm::LogicalResult ExtractOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.index;
       auto attr = dict.get("index");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `index` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ExtractOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.index;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("index",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ExtractOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.index.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ExtractOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "index")
      return prop.index;
  return std::nullopt;
}

void ExtractOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "index") {
       prop.index = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.index)>>(value);
       return;
    }
}

void ExtractOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.index) attrs.append("index", prop.index);
}

::llvm::LogicalResult ExtractOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getIndexAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(attr, "index", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult ExtractOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.index)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExtractOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.index);
}

uint32_t ExtractOp::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

void ExtractOp::setIndex(uint32_t attrValue) {
  getProperties().index = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value range, unsigned index) {
      build(odsBuilder, odsState,
            ::llvm::cast<pdl::RangeType>(range.getType()).getElementType(),
            range, index);
    
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value range, ::mlir::IntegerAttr index) {
  odsState.addOperands(range);
  odsState.getOrAddProperties<Properties>().index = index;
  odsState.addTypes(result);
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value range, ::mlir::IntegerAttr index) {
  odsState.addOperands(range);
  odsState.getOrAddProperties<Properties>().index = index;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value range, uint32_t index) {
  odsState.addOperands(range);
  odsState.getOrAddProperties<Properties>().index = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index);
  odsState.addTypes(result);
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value range, uint32_t index) {
  odsState.addOperands(range);
  odsState.getOrAddProperties<Properties>().index = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ExtractOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult ExtractOp::verifyInvariantsImpl() {
  auto tblgen_index = getProperties().index; (void)tblgen_index;
  if (!tblgen_index) return emitOpError("requires attribute 'index'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(*this, tblgen_index, "index")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()(pdl::RangeType::get((*this->getODSResults(0).begin()).getType()), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that `range` is a PDL range whose element type matches type of `result`");
  return ::mlir::success();
}

::llvm::LogicalResult ExtractOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ExtractOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::UnresolvedOperand rangeRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rangeOperands(&rangeRawOperand, 1);  ::llvm::SMLoc rangeOperandsLoc;
  (void)rangeOperandsLoc;
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);

  if (parser.parseCustomAttributeWithFallback(indexAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (indexAttr) result.getOrAddProperties<ExtractOp::Properties>().index = indexAttr;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  rangeOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rangeRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawType = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  for (::mlir::Type type : resultTypes) {
    (void)type;
    if (!((::llvm::isa<::mlir::pdl::PDLType>(type)))) {
      return parser.emitError(parser.getNameLoc()) << "'result' must be pdl type, but got " << type;
    }
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(rangeOperands, pdl::RangeType::get(resultTypes[0]), rangeOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExtractOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getIndexAttr());
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getRange();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::pdl::PDLType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("index");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void ExtractOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ExtractOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::FinalizeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
FinalizeOpAdaptor::FinalizeOpAdaptor(FinalizeOp op) : FinalizeOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult FinalizeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void FinalizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void FinalizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FinalizeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult FinalizeOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::llvm::LogicalResult FinalizeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FinalizeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void FinalizeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void FinalizeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::FinalizeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ForEachOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
ForEachOpAdaptor::ForEachOpAdaptor(ForEachOp op) : ForEachOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ForEachOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void ForEachOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value values, ::mlir::Block *successor) {
  odsState.addOperands(values);
  (void)odsState.addRegion();
  odsState.addSuccessors(successor);
}

void ForEachOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value values, ::mlir::Block *successor) {
  odsState.addOperands(values);
  (void)odsState.addRegion();
  odsState.addSuccessors(successor);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ForEachOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult ForEachOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_PDLInterpOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult ForEachOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ForEachOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::FuncOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FuncOpGenericAdaptorBase::FuncOpGenericAdaptorBase(FuncOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::llvm::StringRef FuncOpGenericAdaptorBase::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::FunctionType FuncOpGenericAdaptorBase::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return ::llvm::cast<::mlir::FunctionType>(attr.getValue());
}

::std::optional< ::mlir::ArrayAttr > FuncOpGenericAdaptorBase::getArgAttrs() {
  auto attr = getArgAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > FuncOpGenericAdaptorBase::getResAttrs() {
  auto attr = getResAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

} // namespace detail
FuncOpAdaptor::FuncOpAdaptor(FuncOp op) : FuncOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult FuncOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_arg_attrs = getProperties().arg_attrs; (void)tblgen_arg_attrs;
  auto tblgen_function_type = getProperties().function_type; (void)tblgen_function_type;
  if (!tblgen_function_type) return emitError(loc, "'pdl_interp.func' op ""requires attribute 'function_type'");
  auto tblgen_res_attrs = getProperties().res_attrs; (void)tblgen_res_attrs;
  auto tblgen_sym_name = getProperties().sym_name; (void)tblgen_sym_name;
  if (!tblgen_sym_name) return emitError(loc, "'pdl_interp.func' op ""requires attribute 'sym_name'");

  if (tblgen_sym_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_sym_name))))
    return emitError(loc, "'pdl_interp.func' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_function_type && !(((::llvm::isa<::mlir::TypeAttr>(tblgen_function_type))) && ((::llvm::isa<::mlir::FunctionType>(::llvm::cast<::mlir::TypeAttr>(tblgen_function_type).getValue()))) && ((::llvm::isa<::mlir::FunctionType>(::llvm::cast<::mlir::TypeAttr>(tblgen_function_type).getValue())))))
    return emitError(loc, "'pdl_interp.func' op ""attribute 'function_type' failed to satisfy constraint: type attribute of function type");

  if (tblgen_arg_attrs && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_arg_attrs))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_arg_attrs), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::DictionaryAttr>(attr))); }))))
    return emitError(loc, "'pdl_interp.func' op ""attribute 'arg_attrs' failed to satisfy constraint: Array of dictionary attributes");

  if (tblgen_res_attrs && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_res_attrs))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_res_attrs), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::DictionaryAttr>(attr))); }))))
    return emitError(loc, "'pdl_interp.func' op ""attribute 'res_attrs' failed to satisfy constraint: Array of dictionary attributes");
  return ::mlir::success();
}

::llvm::LogicalResult FuncOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.arg_attrs;
       auto attr = dict.get("arg_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `arg_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.function_type;
       auto attr = dict.get("function_type");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `function_type` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.res_attrs;
       auto attr = dict.get("res_attrs");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `res_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.sym_name;
       auto attr = dict.get("sym_name");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `sym_name` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute FuncOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.arg_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("arg_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.function_type;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("function_type",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.res_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("res_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.sym_name;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("sym_name",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code FuncOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.arg_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.function_type.getAsOpaquePointer()), 
    llvm::hash_value(prop.res_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.sym_name.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> FuncOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "arg_attrs")
      return prop.arg_attrs;

    if (name == "function_type")
      return prop.function_type;

    if (name == "res_attrs")
      return prop.res_attrs;

    if (name == "sym_name")
      return prop.sym_name;
  return std::nullopt;
}

void FuncOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "arg_attrs") {
       prop.arg_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.arg_attrs)>>(value);
       return;
    }

    if (name == "function_type") {
       prop.function_type = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.function_type)>>(value);
       return;
    }

    if (name == "res_attrs") {
       prop.res_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.res_attrs)>>(value);
       return;
    }

    if (name == "sym_name") {
       prop.sym_name = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.sym_name)>>(value);
       return;
    }
}

void FuncOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.arg_attrs) attrs.append("arg_attrs", prop.arg_attrs);

    if (prop.function_type) attrs.append("function_type", prop.function_type);

    if (prop.res_attrs) attrs.append("res_attrs", prop.res_attrs);

    if (prop.sym_name) attrs.append("sym_name", prop.sym_name);
}

::llvm::LogicalResult FuncOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getArgAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps10(attr, "arg_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getFunctionTypeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps9(attr, "function_type", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getResAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps10(attr, "res_attrs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getSymNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(attr, "sym_name", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult FuncOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.arg_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.function_type)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.res_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.sym_name)))
    return ::mlir::failure();
  return ::mlir::success();
}

void FuncOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.arg_attrs);
  writer.writeAttribute(prop.function_type);

  writer.writeOptionalAttribute(prop.res_attrs);
  writer.writeAttribute(prop.sym_name);
}

::llvm::StringRef FuncOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::FunctionType FuncOp::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return ::llvm::cast<::mlir::FunctionType>(attr.getValue());
}

::std::optional< ::mlir::ArrayAttr > FuncOp::getArgAttrs() {
  auto attr = getArgAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > FuncOp::getResAttrs() {
  auto attr = getResAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void FuncOp::setSymName(::llvm::StringRef attrValue) {
  getProperties().sym_name = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void FuncOp::setFunctionType(::mlir::FunctionType attrValue) {
  getProperties().function_type = ::mlir::TypeAttr::get(attrValue);
}

::llvm::LogicalResult FuncOp::verifyInvariantsImpl() {
  auto tblgen_arg_attrs = getProperties().arg_attrs; (void)tblgen_arg_attrs;
  auto tblgen_function_type = getProperties().function_type; (void)tblgen_function_type;
  if (!tblgen_function_type) return emitOpError("requires attribute 'function_type'");
  auto tblgen_res_attrs = getProperties().res_attrs; (void)tblgen_res_attrs;
  auto tblgen_sym_name = getProperties().sym_name; (void)tblgen_sym_name;
  if (!tblgen_sym_name) return emitOpError("requires attribute 'sym_name'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps9(*this, tblgen_function_type, "function_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps10(*this, tblgen_arg_attrs, "arg_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps10(*this, tblgen_res_attrs, "res_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_PDLInterpOps2(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::llvm::LogicalResult FuncOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::FuncOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetAttributeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetAttributeOpGenericAdaptorBase::GetAttributeOpGenericAdaptorBase(GetAttributeOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::llvm::StringRef GetAttributeOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

} // namespace detail
GetAttributeOpAdaptor::GetAttributeOpAdaptor(GetAttributeOp op) : GetAttributeOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult GetAttributeOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitError(loc, "'pdl_interp.get_attribute' op ""requires attribute 'name'");

  if (tblgen_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_name))))
    return emitError(loc, "'pdl_interp.get_attribute' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

::llvm::LogicalResult GetAttributeOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.name;
       auto attr = dict.get("name");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `name` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute GetAttributeOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.name;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("name",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code GetAttributeOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.name.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> GetAttributeOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "name")
      return prop.name;
  return std::nullopt;
}

void GetAttributeOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "name") {
       prop.name = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.name)>>(value);
       return;
    }
}

void GetAttributeOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.name) attrs.append("name", prop.name);
}

::llvm::LogicalResult GetAttributeOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(attr, "name", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult GetAttributeOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.name)))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetAttributeOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.name);
}

::llvm::StringRef GetAttributeOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void GetAttributeOp::setName(::llvm::StringRef attrValue) {
  getProperties().name = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void GetAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type attribute, ::mlir::Value inputOp, ::mlir::StringAttr name) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().name = name;
  odsState.addTypes(attribute);
}

void GetAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::StringAttr name) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().name = name;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type attribute, ::mlir::Value inputOp, ::llvm::StringRef name) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  odsState.addTypes(attribute);
}

void GetAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::llvm::StringRef name) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetAttributeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<GetAttributeOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult GetAttributeOp::verifyInvariantsImpl() {
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitOpError("requires attribute 'name'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult GetAttributeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetAttributeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(&inputOpRawOperand, 1);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (nameAttr) result.getOrAddProperties<GetAttributeOp::Properties>().name = nameAttr;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperand))
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
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(inputOpOperands, odsBuildableType1, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetAttributeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetAttributeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetAttributeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetAttributeTypeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
GetAttributeTypeOpAdaptor::GetAttributeTypeOpAdaptor(GetAttributeTypeOp op) : GetAttributeTypeOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult GetAttributeTypeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void GetAttributeTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value) {
      build(odsBuilder, odsState, odsBuilder.getType<pdl::TypeType>(), value);
    
}

void GetAttributeTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value value) {
  odsState.addOperands(value);
  odsState.addTypes(result);
}

void GetAttributeTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetAttributeTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult GetAttributeTypeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult GetAttributeTypeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetAttributeTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(&valueRawOperand, 1);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(valueOperands, odsBuildableType1, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetAttributeTypeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetAttributeTypeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetAttributeTypeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetDefiningOpOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
GetDefiningOpOpAdaptor::GetDefiningOpOpAdaptor(GetDefiningOpOp op) : GetDefiningOpOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult GetDefiningOpOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void GetDefiningOpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type inputOp, ::mlir::Value value) {
  odsState.addOperands(value);
  odsState.addTypes(inputOp);
}

void GetDefiningOpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetDefiningOpOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult GetDefiningOpOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult GetDefiningOpOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetDefiningOpOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(&valueRawOperand, 1);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type valueRawType{};
  ::llvm::ArrayRef<::mlir::Type> valueTypes(&valueRawType, 1);
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawType = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetDefiningOpOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::pdl::PDLType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetDefiningOpOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetDefiningOpOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetOperandOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetOperandOpGenericAdaptorBase::GetOperandOpGenericAdaptorBase(GetOperandOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint32_t GetOperandOpGenericAdaptorBase::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
GetOperandOpAdaptor::GetOperandOpAdaptor(GetOperandOp op) : GetOperandOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult GetOperandOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_index = getProperties().index; (void)tblgen_index;
  if (!tblgen_index) return emitError(loc, "'pdl_interp.get_operand' op ""requires attribute 'index'");

  if (tblgen_index && !((((::llvm::isa<::mlir::IntegerAttr>(tblgen_index))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_index).getType().isSignlessInteger(32)))) && ((!::llvm::cast<::mlir::IntegerAttr>(tblgen_index).getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.get_operand' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  return ::mlir::success();
}

::llvm::LogicalResult GetOperandOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.index;
       auto attr = dict.get("index");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `index` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute GetOperandOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.index;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("index",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code GetOperandOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.index.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> GetOperandOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "index")
      return prop.index;
  return std::nullopt;
}

void GetOperandOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "index") {
       prop.index = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.index)>>(value);
       return;
    }
}

void GetOperandOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.index) attrs.append("index", prop.index);
}

::llvm::LogicalResult GetOperandOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getIndexAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(attr, "index", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult GetOperandOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.index)))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetOperandOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.index);
}

uint32_t GetOperandOp::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

void GetOperandOp::setIndex(uint32_t attrValue) {
  getProperties().index = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void GetOperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value inputOp, ::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().index = index;
  odsState.addTypes(value);
}

void GetOperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().index = index;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetOperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value inputOp, uint32_t index) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().index = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index);
  odsState.addTypes(value);
}

void GetOperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, uint32_t index) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().index = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetOperandOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<GetOperandOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult GetOperandOp::verifyInvariantsImpl() {
  auto tblgen_index = getProperties().index; (void)tblgen_index;
  if (!tblgen_index) return emitOpError("requires attribute 'index'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(*this, tblgen_index, "index")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps13(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult GetOperandOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetOperandOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(&inputOpRawOperand, 1);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;

  if (parser.parseCustomAttributeWithFallback(indexAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (indexAttr) result.getOrAddProperties<GetOperandOp::Properties>().index = indexAttr;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperand))
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
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::ValueType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(inputOpOperands, odsBuildableType1, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetOperandOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getIndexAttr());
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("index");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetOperandOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetOperandOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetOperandsOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetOperandsOpGenericAdaptorBase::GetOperandsOpGenericAdaptorBase(GetOperandsOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::std::optional<uint32_t> GetOperandsOpGenericAdaptorBase::getIndex() {
  auto attr = getIndexAttr();
  return attr ? ::std::optional<uint32_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

} // namespace detail
GetOperandsOpAdaptor::GetOperandsOpAdaptor(GetOperandsOp op) : GetOperandsOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult GetOperandsOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_index = getProperties().index; (void)tblgen_index;

  if (tblgen_index && !((((::llvm::isa<::mlir::IntegerAttr>(tblgen_index))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_index).getType().isSignlessInteger(32)))) && ((!::llvm::cast<::mlir::IntegerAttr>(tblgen_index).getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.get_operands' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  return ::mlir::success();
}

::llvm::LogicalResult GetOperandsOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.index;
       auto attr = dict.get("index");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `index` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute GetOperandsOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.index;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("index",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code GetOperandsOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.index.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> GetOperandsOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "index")
      return prop.index;
  return std::nullopt;
}

void GetOperandsOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "index") {
       prop.index = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.index)>>(value);
       return;
    }
}

void GetOperandsOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.index) attrs.append("index", prop.index);
}

::llvm::LogicalResult GetOperandsOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getIndexAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(attr, "index", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult GetOperandsOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.index)))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetOperandsOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.index);
}

::std::optional<uint32_t> GetOperandsOp::getIndex() {
  auto attr = getIndexAttr();
  return attr ? ::std::optional<uint32_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

void GetOperandsOp::setIndex(::std::optional<uint32_t> attrValue) {
    auto &odsProp = getProperties().index;
    if (attrValue)
      odsProp = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), *attrValue);
    else
      odsProp = nullptr;
}

void GetOperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value inputOp, std::optional<unsigned> index) {
      build(odsBuilder, odsState, resultType, inputOp,
            index ? odsBuilder.getI32IntegerAttr(*index) : IntegerAttr());
    
}

void GetOperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value inputOp, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  if (index) {
    odsState.getOrAddProperties<Properties>().index = index;
  }
  odsState.addTypes(value);
}

void GetOperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  if (index) {
    odsState.getOrAddProperties<Properties>().index = index;
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetOperandsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<GetOperandsOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult GetOperandsOp::verifyInvariantsImpl() {
  auto tblgen_index = getProperties().index; (void)tblgen_index;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(*this, tblgen_index, "index")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps12(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult GetOperandsOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetOperandsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(&inputOpRawOperand, 1);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::Type valueRawType{};
  ::llvm::ArrayRef<::mlir::Type> valueTypes(&valueRawType, 1);

  ::mlir::OptionalParseResult parseResultindexAttr =
    parser.parseOptionalAttribute(indexAttr, parser.getBuilder().getIntegerType(32));
  if (parseResultindexAttr.has_value() && failed(*parseResultindexAttr))
    return ::mlir::failure();
  if (parseResultindexAttr.has_value() && succeeded(*parseResultindexAttr))
  if (indexAttr) result.getOrAddProperties<GetOperandsOp::Properties>().index = indexAttr;
  if (indexAttr) {
  }
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawType = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(valueTypes);
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetOperandsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (getIndexAttr()) {
    _odsPrinter << ' ';
    _odsPrinter.printAttributeWithoutType(getIndexAttr());
  }
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::pdl::PDLType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("index");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetOperandsOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetOperandsOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetResultOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetResultOpGenericAdaptorBase::GetResultOpGenericAdaptorBase(GetResultOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint32_t GetResultOpGenericAdaptorBase::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
GetResultOpAdaptor::GetResultOpAdaptor(GetResultOp op) : GetResultOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult GetResultOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_index = getProperties().index; (void)tblgen_index;
  if (!tblgen_index) return emitError(loc, "'pdl_interp.get_result' op ""requires attribute 'index'");

  if (tblgen_index && !((((::llvm::isa<::mlir::IntegerAttr>(tblgen_index))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_index).getType().isSignlessInteger(32)))) && ((!::llvm::cast<::mlir::IntegerAttr>(tblgen_index).getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.get_result' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  return ::mlir::success();
}

::llvm::LogicalResult GetResultOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.index;
       auto attr = dict.get("index");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `index` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute GetResultOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.index;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("index",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code GetResultOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.index.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> GetResultOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "index")
      return prop.index;
  return std::nullopt;
}

void GetResultOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "index") {
       prop.index = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.index)>>(value);
       return;
    }
}

void GetResultOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.index) attrs.append("index", prop.index);
}

::llvm::LogicalResult GetResultOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getIndexAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(attr, "index", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult GetResultOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.index)))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetResultOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.index);
}

uint32_t GetResultOp::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

void GetResultOp::setIndex(uint32_t attrValue) {
  getProperties().index = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value inputOp, ::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().index = index;
  odsState.addTypes(value);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().index = index;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value inputOp, uint32_t index) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().index = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index);
  odsState.addTypes(value);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, uint32_t index) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().index = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetResultOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<GetResultOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult GetResultOp::verifyInvariantsImpl() {
  auto tblgen_index = getProperties().index; (void)tblgen_index;
  if (!tblgen_index) return emitOpError("requires attribute 'index'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(*this, tblgen_index, "index")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps13(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult GetResultOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetResultOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(&inputOpRawOperand, 1);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;

  if (parser.parseCustomAttributeWithFallback(indexAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (indexAttr) result.getOrAddProperties<GetResultOp::Properties>().index = indexAttr;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperand))
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
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::ValueType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(inputOpOperands, odsBuildableType1, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetResultOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getIndexAttr());
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("index");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetResultOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetResultOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetResultsOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetResultsOpGenericAdaptorBase::GetResultsOpGenericAdaptorBase(GetResultsOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::std::optional<uint32_t> GetResultsOpGenericAdaptorBase::getIndex() {
  auto attr = getIndexAttr();
  return attr ? ::std::optional<uint32_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

} // namespace detail
GetResultsOpAdaptor::GetResultsOpAdaptor(GetResultsOp op) : GetResultsOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult GetResultsOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_index = getProperties().index; (void)tblgen_index;

  if (tblgen_index && !((((::llvm::isa<::mlir::IntegerAttr>(tblgen_index))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_index).getType().isSignlessInteger(32)))) && ((!::llvm::cast<::mlir::IntegerAttr>(tblgen_index).getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.get_results' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  return ::mlir::success();
}

::llvm::LogicalResult GetResultsOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.index;
       auto attr = dict.get("index");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `index` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute GetResultsOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.index;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("index",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code GetResultsOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.index.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> GetResultsOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "index")
      return prop.index;
  return std::nullopt;
}

void GetResultsOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "index") {
       prop.index = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.index)>>(value);
       return;
    }
}

void GetResultsOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.index) attrs.append("index", prop.index);
}

::llvm::LogicalResult GetResultsOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getIndexAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(attr, "index", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult GetResultsOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.index)))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetResultsOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.index);
}

::std::optional<uint32_t> GetResultsOp::getIndex() {
  auto attr = getIndexAttr();
  return attr ? ::std::optional<uint32_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

void GetResultsOp::setIndex(::std::optional<uint32_t> attrValue) {
    auto &odsProp = getProperties().index;
    if (attrValue)
      odsProp = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), *attrValue);
    else
      odsProp = nullptr;
}

void GetResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value inputOp, std::optional<unsigned> index) {
      build(odsBuilder, odsState, resultType, inputOp,
            index ? odsBuilder.getI32IntegerAttr(*index) : IntegerAttr());
    
}

void GetResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value inputOp) {
      build(odsBuilder, odsState,
            pdl::RangeType::get(odsBuilder.getType<pdl::ValueType>()), inputOp,
            IntegerAttr());
    
}

void GetResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value inputOp, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  if (index) {
    odsState.getOrAddProperties<Properties>().index = index;
  }
  odsState.addTypes(value);
}

void GetResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  if (index) {
    odsState.getOrAddProperties<Properties>().index = index;
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetResultsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<GetResultsOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult GetResultsOp::verifyInvariantsImpl() {
  auto tblgen_index = getProperties().index; (void)tblgen_index;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(*this, tblgen_index, "index")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps12(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult GetResultsOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetResultsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(&inputOpRawOperand, 1);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::Type valueRawType{};
  ::llvm::ArrayRef<::mlir::Type> valueTypes(&valueRawType, 1);

  ::mlir::OptionalParseResult parseResultindexAttr =
    parser.parseOptionalAttribute(indexAttr, parser.getBuilder().getIntegerType(32));
  if (parseResultindexAttr.has_value() && failed(*parseResultindexAttr))
    return ::mlir::failure();
  if (parseResultindexAttr.has_value() && succeeded(*parseResultindexAttr))
  if (indexAttr) result.getOrAddProperties<GetResultsOp::Properties>().index = indexAttr;
  if (indexAttr) {
  }
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawType = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(valueTypes);
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetResultsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (getIndexAttr()) {
    _odsPrinter << ' ';
    _odsPrinter.printAttributeWithoutType(getIndexAttr());
  }
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::pdl::PDLType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("index");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetResultsOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetResultsOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetUsersOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
GetUsersOpAdaptor::GetUsersOpAdaptor(GetUsersOp op) : GetUsersOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult GetUsersOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void GetUsersOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value) {
      build(odsBuilder, odsState,
            pdl::RangeType::get(odsBuilder.getType<pdl::OperationType>()),
            value);
    
}

void GetUsersOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type operations, ::mlir::Value value) {
  odsState.addOperands(value);
  odsState.addTypes(operations);
}

void GetUsersOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetUsersOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult GetUsersOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult GetUsersOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetUsersOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(&valueRawOperand, 1);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type valueRawType{};
  ::llvm::ArrayRef<::mlir::Type> valueTypes(&valueRawType, 1);
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawType = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::OperationType>());
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetUsersOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::pdl::PDLType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetUsersOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetUsersOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetValueTypeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
GetValueTypeOpAdaptor::GetValueTypeOpAdaptor(GetValueTypeOp op) : GetValueTypeOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult GetValueTypeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void GetValueTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value) {
      Type valType = value.getType();
      Type typeType = odsBuilder.getType<pdl::TypeType>();
      build(odsBuilder, odsState,
            ::llvm::isa<pdl::RangeType>(valType) ? pdl::RangeType::get(typeType)
                                          : typeType,
            value);
    
}

void GetValueTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value value) {
  odsState.addOperands(value);
  odsState.addTypes(result);
}

void GetValueTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetValueTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult GetValueTypeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps15(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()(getGetValueTypeOpValueType((*this->getODSResults(0).begin()).getType()), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that `value` type matches arity of `result`");
  return ::mlir::success();
}

::llvm::LogicalResult GetValueTypeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetValueTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(&valueRawOperand, 1);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawType = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  for (::mlir::Type type : resultTypes) {
    (void)type;
    if (!(((::llvm::isa<::mlir::pdl::TypeType>(type))) || (((::llvm::isa<::mlir::pdl::RangeType>(type))) && ((::llvm::isa<::mlir::pdl::TypeType>(::llvm::cast<::mlir::pdl::RangeType>(type).getElementType())))))) {
      return parser.emitError(parser.getNameLoc()) << "'result' must be single element or range of PDL handle to an `mlir::Type`, but got " << type;
    }
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(valueOperands, getGetValueTypeOpValueType(resultTypes[0]), valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetValueTypeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::pdl::PDLType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetValueTypeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetValueTypeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::IsNotNullOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
IsNotNullOpAdaptor::IsNotNullOpAdaptor(IsNotNullOp op) : IsNotNullOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult IsNotNullOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void IsNotNullOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void IsNotNullOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IsNotNullOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult IsNotNullOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult IsNotNullOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult IsNotNullOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(&valueRawOperand, 1);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type valueRawType{};
  ::llvm::ArrayRef<::mlir::Type> valueTypes(&valueRawType, 1);
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperand))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawType = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void IsNotNullOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::pdl::PDLType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void IsNotNullOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::IsNotNullOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::RecordMatchOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
RecordMatchOpGenericAdaptorBase::RecordMatchOpGenericAdaptorBase(RecordMatchOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> RecordMatchOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::SymbolRefAttr RecordMatchOpGenericAdaptorBase::getRewriter() {
  auto attr = getRewriterAttr();
  return attr;
}

::std::optional< ::llvm::StringRef > RecordMatchOpGenericAdaptorBase::getRootKind() {
  auto attr = getRootKindAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > RecordMatchOpGenericAdaptorBase::getGeneratedOps() {
  auto attr = getGeneratedOpsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

uint16_t RecordMatchOpGenericAdaptorBase::getBenefit() {
  auto attr = getBenefitAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
RecordMatchOpAdaptor::RecordMatchOpAdaptor(RecordMatchOp op) : RecordMatchOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult RecordMatchOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_benefit = getProperties().benefit; (void)tblgen_benefit;
  if (!tblgen_benefit) return emitError(loc, "'pdl_interp.record_match' op ""requires attribute 'benefit'");
  auto tblgen_generatedOps = getProperties().generatedOps; (void)tblgen_generatedOps;
  auto tblgen_rewriter = getProperties().rewriter; (void)tblgen_rewriter;
  if (!tblgen_rewriter) return emitError(loc, "'pdl_interp.record_match' op ""requires attribute 'rewriter'");
  auto tblgen_rootKind = getProperties().rootKind; (void)tblgen_rootKind;

  if (tblgen_rewriter && !((::llvm::isa<::mlir::SymbolRefAttr>(tblgen_rewriter))))
    return emitError(loc, "'pdl_interp.record_match' op ""attribute 'rewriter' failed to satisfy constraint: symbol reference attribute");

  if (tblgen_rootKind && !((::llvm::isa<::mlir::StringAttr>(tblgen_rootKind))))
    return emitError(loc, "'pdl_interp.record_match' op ""attribute 'rootKind' failed to satisfy constraint: string attribute");

  if (tblgen_generatedOps && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_generatedOps))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_generatedOps), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::StringAttr>(attr))); }))))
    return emitError(loc, "'pdl_interp.record_match' op ""attribute 'generatedOps' failed to satisfy constraint: string array attribute");

  if (tblgen_benefit && !((((::llvm::isa<::mlir::IntegerAttr>(tblgen_benefit))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_benefit).getType().isSignlessInteger(16)))) && ((!::llvm::cast<::mlir::IntegerAttr>(tblgen_benefit).getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.record_match' op ""attribute 'benefit' failed to satisfy constraint: 16-bit signless integer attribute whose value is non-negative");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> RecordMatchOp::getODSOperandIndexAndLength(unsigned index) {
  ::llvm::ArrayRef<int32_t> sizeAttr = getProperties().operandSegmentSizes;

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::MutableOperandRange RecordMatchOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::mlir::MutableOperandRange RecordMatchOp::getMatchedOpsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, {getOperandSegmentSizesAttrName(), ::mlir::DenseI32ArrayAttr::get(getContext(), getProperties().operandSegmentSizes)}));
  return mutableRange;
}

::llvm::LogicalResult RecordMatchOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.benefit;
       auto attr = dict.get("benefit");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `benefit` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.generatedOps;
       auto attr = dict.get("generatedOps");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `generatedOps` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.rewriter;
       auto attr = dict.get("rewriter");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `rewriter` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.rootKind;
       auto attr = dict.get("rootKind");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `rootKind` in property conversion: " << attr;
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

::mlir::Attribute RecordMatchOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.benefit;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("benefit",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.generatedOps;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("generatedOps",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.rewriter;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("rewriter",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.rootKind;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("rootKind",
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

llvm::hash_code RecordMatchOp::computePropertiesHash(const Properties &prop) {
  auto hash_operandSegmentSizes = [] (const auto &propStorage) -> llvm::hash_code {
    return ::llvm::hash_combine_range(std::begin(propStorage), std::end(propStorage));;
  };
  return llvm::hash_combine(
    llvm::hash_value(prop.benefit.getAsOpaquePointer()), 
    llvm::hash_value(prop.generatedOps.getAsOpaquePointer()), 
    llvm::hash_value(prop.rewriter.getAsOpaquePointer()), 
    llvm::hash_value(prop.rootKind.getAsOpaquePointer()), 
    hash_operandSegmentSizes(prop.operandSegmentSizes));
}

std::optional<mlir::Attribute> RecordMatchOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "benefit")
      return prop.benefit;

    if (name == "generatedOps")
      return prop.generatedOps;

    if (name == "rewriter")
      return prop.rewriter;

    if (name == "rootKind")
      return prop.rootKind;
    if (name == "operand_segment_sizes" || name == "operandSegmentSizes") return [&]() -> ::mlir::Attribute { return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes); }();
  return std::nullopt;
}

void RecordMatchOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "benefit") {
       prop.benefit = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.benefit)>>(value);
       return;
    }

    if (name == "generatedOps") {
       prop.generatedOps = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.generatedOps)>>(value);
       return;
    }

    if (name == "rewriter") {
       prop.rewriter = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.rewriter)>>(value);
       return;
    }

    if (name == "rootKind") {
       prop.rootKind = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.rootKind)>>(value);
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

void RecordMatchOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.benefit) attrs.append("benefit", prop.benefit);

    if (prop.generatedOps) attrs.append("generatedOps", prop.generatedOps);

    if (prop.rewriter) attrs.append("rewriter", prop.rewriter);

    if (prop.rootKind) attrs.append("rootKind", prop.rootKind);
  attrs.append("operandSegmentSizes", [&]() -> ::mlir::Attribute { return ::mlir::DenseI32ArrayAttr::get(ctx, prop.operandSegmentSizes); }());
}

::llvm::LogicalResult RecordMatchOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getBenefitAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps12(attr, "benefit", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getGeneratedOpsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps8(attr, "generatedOps", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getRewriterAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps11(attr, "rewriter", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getRootKindAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(attr, "rootKind", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult RecordMatchOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.benefit)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.generatedOps)))
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

  if (::mlir::failed(reader.readAttribute(prop.rewriter)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.rootKind)))
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

void RecordMatchOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.benefit);

  writer.writeOptionalAttribute(prop.generatedOps);

if (writer.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
  auto &propStorage = prop.operandSegmentSizes;
  writer.writeAttribute(::mlir::DenseI32ArrayAttr::get(this->getContext(), propStorage));
}
  writer.writeAttribute(prop.rewriter);

  writer.writeOptionalAttribute(prop.rootKind);

  {
    auto &propStorage = prop.operandSegmentSizes;

  if (writer.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6)
    writer.writeSparseArray(::llvm::ArrayRef(propStorage));
;
  }
}

::mlir::SymbolRefAttr RecordMatchOp::getRewriter() {
  auto attr = getRewriterAttr();
  return attr;
}

::std::optional< ::llvm::StringRef > RecordMatchOp::getRootKind() {
  auto attr = getRootKindAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::std::optional< ::mlir::ArrayAttr > RecordMatchOp::getGeneratedOps() {
  auto attr = getGeneratedOpsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

uint16_t RecordMatchOp::getBenefit() {
  auto attr = getBenefitAttr();
  return attr.getValue().getZExtValue();
}

void RecordMatchOp::setRootKind(::std::optional<::llvm::StringRef> attrValue) {
    auto &odsProp = getProperties().rootKind;
    if (attrValue)
      odsProp = ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue);
    else
      odsProp = nullptr;
}

void RecordMatchOp::setBenefit(uint16_t attrValue) {
  getProperties().benefit = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(16), attrValue);
}

void RecordMatchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::ValueRange matchedOps, ::mlir::SymbolRefAttr rewriter, /*optional*/::mlir::StringAttr rootKind, /*optional*/::mlir::ArrayAttr generatedOps, ::mlir::IntegerAttr benefit, ::mlir::Block *dest) {
  odsState.addOperands(inputs);
  odsState.addOperands(matchedOps);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(matchedOps.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().rewriter = rewriter;
  if (rootKind) {
    odsState.getOrAddProperties<Properties>().rootKind = rootKind;
  }
  if (generatedOps) {
    odsState.getOrAddProperties<Properties>().generatedOps = generatedOps;
  }
  odsState.getOrAddProperties<Properties>().benefit = benefit;
  odsState.addSuccessors(dest);
}

void RecordMatchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::ValueRange matchedOps, ::mlir::SymbolRefAttr rewriter, /*optional*/::mlir::StringAttr rootKind, /*optional*/::mlir::ArrayAttr generatedOps, ::mlir::IntegerAttr benefit, ::mlir::Block *dest) {
  odsState.addOperands(inputs);
  odsState.addOperands(matchedOps);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(matchedOps.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().rewriter = rewriter;
  if (rootKind) {
    odsState.getOrAddProperties<Properties>().rootKind = rootKind;
  }
  if (generatedOps) {
    odsState.getOrAddProperties<Properties>().generatedOps = generatedOps;
  }
  odsState.getOrAddProperties<Properties>().benefit = benefit;
  odsState.addSuccessors(dest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RecordMatchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::ValueRange matchedOps, ::mlir::SymbolRefAttr rewriter, /*optional*/::mlir::StringAttr rootKind, /*optional*/::mlir::ArrayAttr generatedOps, uint16_t benefit, ::mlir::Block *dest) {
  odsState.addOperands(inputs);
  odsState.addOperands(matchedOps);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(matchedOps.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().rewriter = rewriter;
  if (rootKind) {
    odsState.getOrAddProperties<Properties>().rootKind = rootKind;
  }
  if (generatedOps) {
    odsState.getOrAddProperties<Properties>().generatedOps = generatedOps;
  }
  odsState.getOrAddProperties<Properties>().benefit = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(16), benefit);
  odsState.addSuccessors(dest);
}

void RecordMatchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::ValueRange matchedOps, ::mlir::SymbolRefAttr rewriter, /*optional*/::mlir::StringAttr rootKind, /*optional*/::mlir::ArrayAttr generatedOps, uint16_t benefit, ::mlir::Block *dest) {
  odsState.addOperands(inputs);
  odsState.addOperands(matchedOps);
  ::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(matchedOps.size())}), odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().rewriter = rewriter;
  if (rootKind) {
    odsState.getOrAddProperties<Properties>().rootKind = rootKind;
  }
  if (generatedOps) {
    odsState.getOrAddProperties<Properties>().generatedOps = generatedOps;
  }
  odsState.getOrAddProperties<Properties>().benefit = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(16), benefit);
  odsState.addSuccessors(dest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RecordMatchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<RecordMatchOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult RecordMatchOp::verifyInvariantsImpl() {
  auto tblgen_benefit = getProperties().benefit; (void)tblgen_benefit;
  if (!tblgen_benefit) return emitOpError("requires attribute 'benefit'");
  auto tblgen_generatedOps = getProperties().generatedOps; (void)tblgen_generatedOps;
  auto tblgen_rewriter = getProperties().rewriter; (void)tblgen_rewriter;
  if (!tblgen_rewriter) return emitOpError("requires attribute 'rewriter'");
  auto tblgen_rootKind = getProperties().rootKind; (void)tblgen_rootKind;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps11(*this, tblgen_rewriter, "rewriter")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(*this, tblgen_rootKind, "rootKind")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps8(*this, tblgen_generatedOps, "generatedOps")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps12(*this, tblgen_benefit, "benefit")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps16(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult RecordMatchOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult RecordMatchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SymbolRefAttr rewriterAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputsOperands;
  ::llvm::SMLoc inputsOperandsLoc;
  (void)inputsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> inputsTypes;
  ::mlir::IntegerAttr benefitAttr;
  ::mlir::ArrayAttr generatedOpsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> matchedOpsOperands;
  ::llvm::SMLoc matchedOpsOperandsLoc;
  (void)matchedOpsOperandsLoc;
  ::mlir::StringAttr rootKindAttr;
  ::mlir::Block *destSuccessor = nullptr;

  if (parser.parseCustomAttributeWithFallback(rewriterAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (rewriterAttr) result.getOrAddProperties<RecordMatchOp::Properties>().rewriter = rewriterAttr;
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  inputsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(inputsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseKeyword("benefit"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(benefitAttr, parser.getBuilder().getIntegerType(16))) {
    return ::mlir::failure();
  }
  if (benefitAttr) result.getOrAddProperties<RecordMatchOp::Properties>().benefit = benefitAttr;
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("generatedOps"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(generatedOpsAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (generatedOpsAttr) result.getOrAddProperties<RecordMatchOp::Properties>().generatedOps = generatedOpsAttr;
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();
  }
  if (parser.parseKeyword("loc"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  matchedOpsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(matchedOpsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalComma())) {
  if (parser.parseKeyword("root"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(rootKindAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (rootKindAttr) result.getOrAddProperties<RecordMatchOp::Properties>().rootKind = rootKindAttr;
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(destSuccessor))
    return ::mlir::failure();
  result.addSuccessors(destSuccessor);
::llvm::copy(::llvm::ArrayRef<int32_t>({static_cast<int32_t>(inputsOperands.size()), static_cast<int32_t>(matchedOpsOperands.size())}), result.getOrAddProperties<RecordMatchOp::Properties>().operandSegmentSizes.begin());
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputsOperands, inputsTypes, inputsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(matchedOpsOperands, odsBuildableType0, matchedOpsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void RecordMatchOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getRewriterAttr());
  if (!getInputs().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getInputs();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getInputs().getTypes();
    _odsPrinter << ")";
  }
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "benefit";
  _odsPrinter << "(";
  _odsPrinter.printAttributeWithoutType(getBenefitAttr());
  _odsPrinter << ")";
  _odsPrinter << ",";
  if (getGeneratedOpsAttr()) {
    _odsPrinter << ' ' << "generatedOps";
    _odsPrinter << "(";
    _odsPrinter.printAttributeWithoutType(getGeneratedOpsAttr());
    _odsPrinter << ")";
    _odsPrinter << ",";
  }
  _odsPrinter << ' ' << "loc";
  _odsPrinter << "(";
  _odsPrinter << "[";
  _odsPrinter << getMatchedOps();
  _odsPrinter << "]";
  _odsPrinter << ")";
  if (getRootKindAttr()) {
    _odsPrinter << ",";
    _odsPrinter << ' ' << "root";
    _odsPrinter << "(";
    _odsPrinter.printAttributeWithoutType(getRootKindAttr());
    _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operandSegmentSizes");
  elidedAttrs.push_back("rewriter");
  elidedAttrs.push_back("benefit");
  elidedAttrs.push_back("generatedOps");
  elidedAttrs.push_back("rootKind");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDest();
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::RecordMatchOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ReplaceOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
std::pair<unsigned, unsigned> ReplaceOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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
ReplaceOpAdaptor::ReplaceOpAdaptor(ReplaceOp op) : ReplaceOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ReplaceOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReplaceOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::MutableOperandRange ReplaceOp::getReplValuesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

void ReplaceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::ValueRange replValues) {
  odsState.addOperands(inputOp);
  odsState.addOperands(replValues);
}

void ReplaceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::ValueRange replValues) {
  odsState.addOperands(inputOp);
  odsState.addOperands(replValues);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReplaceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult ReplaceOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ReplaceOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ReplaceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(&inputOpRawOperand, 1);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> replValuesOperands;
  ::llvm::SMLoc replValuesOperandsLoc;
  (void)replValuesOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> replValuesTypes;

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperand))
    return ::mlir::failure();
  if (parser.parseKeyword("with"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  replValuesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(replValuesOperands))
    return ::mlir::failure();
  if (!replValuesOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(replValuesTypes))
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(replValuesOperands, replValuesTypes, replValuesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReplaceOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "with";
  _odsPrinter << ' ';
  _odsPrinter << "(";
  if (!getReplValues().empty()) {
    _odsPrinter << getReplValues();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getReplValues().getTypes();
  }
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ReplaceOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchAttributeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchAttributeOpGenericAdaptorBase::SwitchAttributeOpGenericAdaptorBase(SwitchAttributeOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::ArrayAttr SwitchAttributeOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

} // namespace detail
SwitchAttributeOpAdaptor::SwitchAttributeOpAdaptor(SwitchAttributeOp op) : SwitchAttributeOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult SwitchAttributeOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitError(loc, "'pdl_interp.switch_attribute' op ""requires attribute 'caseValues'");

  if (tblgen_caseValues && !((::llvm::isa<::mlir::ArrayAttr>(tblgen_caseValues))))
    return emitError(loc, "'pdl_interp.switch_attribute' op ""attribute 'caseValues' failed to satisfy constraint: array attribute");
  return ::mlir::success();
}

::llvm::LogicalResult SwitchAttributeOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.caseValues;
       auto attr = dict.get("caseValues");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `caseValues` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute SwitchAttributeOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.caseValues;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("caseValues",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code SwitchAttributeOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.caseValues.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> SwitchAttributeOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "caseValues")
      return prop.caseValues;
  return std::nullopt;
}

void SwitchAttributeOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "caseValues") {
       prop.caseValues = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.caseValues)>>(value);
       return;
    }
}

void SwitchAttributeOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.caseValues) attrs.append("caseValues", prop.caseValues);
}

::llvm::LogicalResult SwitchAttributeOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getCaseValuesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps13(attr, "caseValues", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult SwitchAttributeOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.caseValues)))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchAttributeOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.caseValues);
}

::mlir::ArrayAttr SwitchAttributeOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

void SwitchAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value attribute, ArrayRef<Attribute> caseValues, Block *defaultDest, BlockRange dests) {
    build(odsBuilder, odsState, attribute, odsBuilder.getArrayAttr(caseValues),
          defaultDest, dests);
  
}

void SwitchAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value attribute, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(attribute);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value attribute, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(attribute);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchAttributeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<SwitchAttributeOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult SwitchAttributeOp::verifyInvariantsImpl() {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitOpError("requires attribute 'caseValues'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps13(*this, tblgen_caseValues, "caseValues")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult SwitchAttributeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchAttributeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand attributeRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> attributeOperands(&attributeRawOperand, 1);  ::llvm::SMLoc attributeOperandsLoc;
  (void)attributeOperandsLoc;
  ::mlir::ArrayAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;

  attributeOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(attributeRawOperand))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(caseValuesAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (caseValuesAttr) result.getOrAddProperties<SwitchAttributeOp::Properties>().caseValues = caseValuesAttr;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
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
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  if (parser.resolveOperands(attributeOperands, odsBuildableType0, attributeOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchAttributeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAttribute();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCaseValuesAttr());
  _odsPrinter << "(";
  ::llvm::interleaveComma(getCases(), _odsPrinter);
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("caseValues");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchAttributeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchAttributeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchOperandCountOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchOperandCountOpGenericAdaptorBase::SwitchOperandCountOpGenericAdaptorBase(SwitchOperandCountOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::DenseIntElementsAttr SwitchOperandCountOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

} // namespace detail
SwitchOperandCountOpAdaptor::SwitchOperandCountOpAdaptor(SwitchOperandCountOp op) : SwitchOperandCountOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult SwitchOperandCountOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitError(loc, "'pdl_interp.switch_operand_count' op ""requires attribute 'caseValues'");

  if (tblgen_caseValues && !(((::llvm::isa<::mlir::DenseIntElementsAttr>(tblgen_caseValues))) && ((::llvm::cast<::mlir::DenseIntElementsAttr>(tblgen_caseValues).getType().getElementType().isSignlessInteger(32)))))
    return emitError(loc, "'pdl_interp.switch_operand_count' op ""attribute 'caseValues' failed to satisfy constraint: 32-bit signless integer elements attribute");
  return ::mlir::success();
}

::llvm::LogicalResult SwitchOperandCountOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.caseValues;
       auto attr = dict.get("caseValues");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `caseValues` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute SwitchOperandCountOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.caseValues;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("caseValues",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code SwitchOperandCountOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.caseValues.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> SwitchOperandCountOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "caseValues")
      return prop.caseValues;
  return std::nullopt;
}

void SwitchOperandCountOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "caseValues") {
       prop.caseValues = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.caseValues)>>(value);
       return;
    }
}

void SwitchOperandCountOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.caseValues) attrs.append("caseValues", prop.caseValues);
}

::llvm::LogicalResult SwitchOperandCountOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getCaseValuesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps14(attr, "caseValues", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult SwitchOperandCountOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.caseValues)))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchOperandCountOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.caseValues);
}

::mlir::DenseIntElementsAttr SwitchOperandCountOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

void SwitchOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value inputOp, ArrayRef<int32_t> counts, Block *defaultDest, BlockRange dests) {
    build(odsBuilder, odsState, inputOp, odsBuilder.getI32VectorAttr(counts),
          defaultDest, dests);
  
}

void SwitchOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::DenseIntElementsAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::DenseIntElementsAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchOperandCountOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<SwitchOperandCountOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult SwitchOperandCountOp::verifyInvariantsImpl() {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitOpError("requires attribute 'caseValues'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps14(*this, tblgen_caseValues, "caseValues")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult SwitchOperandCountOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchOperandCountOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(&inputOpRawOperand, 1);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::DenseIntElementsAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperand))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(caseValuesAttr, ::mlir::Type{})) {
    return ::mlir::failure();
  }
  if (caseValuesAttr) result.getOrAddProperties<SwitchOperandCountOp::Properties>().caseValues = caseValuesAttr;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
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
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchOperandCountOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getCaseValuesAttr());
  _odsPrinter << "(";
  ::llvm::interleaveComma(getCases(), _odsPrinter);
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("caseValues");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchOperandCountOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchOperandCountOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchOperationNameOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchOperationNameOpGenericAdaptorBase::SwitchOperationNameOpGenericAdaptorBase(SwitchOperationNameOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::ArrayAttr SwitchOperationNameOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

} // namespace detail
SwitchOperationNameOpAdaptor::SwitchOperationNameOpAdaptor(SwitchOperationNameOp op) : SwitchOperationNameOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult SwitchOperationNameOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitError(loc, "'pdl_interp.switch_operation_name' op ""requires attribute 'caseValues'");

  if (tblgen_caseValues && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_caseValues))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_caseValues), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::StringAttr>(attr))); }))))
    return emitError(loc, "'pdl_interp.switch_operation_name' op ""attribute 'caseValues' failed to satisfy constraint: string array attribute");
  return ::mlir::success();
}

::llvm::LogicalResult SwitchOperationNameOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.caseValues;
       auto attr = dict.get("caseValues");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `caseValues` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute SwitchOperationNameOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.caseValues;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("caseValues",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code SwitchOperationNameOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.caseValues.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> SwitchOperationNameOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "caseValues")
      return prop.caseValues;
  return std::nullopt;
}

void SwitchOperationNameOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "caseValues") {
       prop.caseValues = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.caseValues)>>(value);
       return;
    }
}

void SwitchOperationNameOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.caseValues) attrs.append("caseValues", prop.caseValues);
}

::llvm::LogicalResult SwitchOperationNameOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getCaseValuesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps8(attr, "caseValues", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult SwitchOperationNameOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.caseValues)))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchOperationNameOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.caseValues);
}

::mlir::ArrayAttr SwitchOperationNameOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

void SwitchOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value inputOp, ArrayRef<OperationName> names, Block *defaultDest, BlockRange dests) {
      auto stringNames = llvm::to_vector<8>(llvm::map_range(names,
          [](OperationName name) { return name.getStringRef(); }));
      build(odsBuilder, odsState, inputOp, odsBuilder.getStrArrayAttr(stringNames),
            defaultDest, dests);
    
}

void SwitchOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchOperationNameOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<SwitchOperationNameOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult SwitchOperationNameOp::verifyInvariantsImpl() {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitOpError("requires attribute 'caseValues'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps8(*this, tblgen_caseValues, "caseValues")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult SwitchOperationNameOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchOperationNameOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(&inputOpRawOperand, 1);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::ArrayAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperand))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(caseValuesAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (caseValuesAttr) result.getOrAddProperties<SwitchOperationNameOp::Properties>().caseValues = caseValuesAttr;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
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
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchOperationNameOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCaseValuesAttr());
  _odsPrinter << "(";
  ::llvm::interleaveComma(getCases(), _odsPrinter);
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("caseValues");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchOperationNameOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchOperationNameOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchResultCountOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchResultCountOpGenericAdaptorBase::SwitchResultCountOpGenericAdaptorBase(SwitchResultCountOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::DenseIntElementsAttr SwitchResultCountOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

} // namespace detail
SwitchResultCountOpAdaptor::SwitchResultCountOpAdaptor(SwitchResultCountOp op) : SwitchResultCountOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult SwitchResultCountOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitError(loc, "'pdl_interp.switch_result_count' op ""requires attribute 'caseValues'");

  if (tblgen_caseValues && !(((::llvm::isa<::mlir::DenseIntElementsAttr>(tblgen_caseValues))) && ((::llvm::cast<::mlir::DenseIntElementsAttr>(tblgen_caseValues).getType().getElementType().isSignlessInteger(32)))))
    return emitError(loc, "'pdl_interp.switch_result_count' op ""attribute 'caseValues' failed to satisfy constraint: 32-bit signless integer elements attribute");
  return ::mlir::success();
}

::llvm::LogicalResult SwitchResultCountOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.caseValues;
       auto attr = dict.get("caseValues");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `caseValues` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute SwitchResultCountOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.caseValues;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("caseValues",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code SwitchResultCountOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.caseValues.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> SwitchResultCountOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "caseValues")
      return prop.caseValues;
  return std::nullopt;
}

void SwitchResultCountOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "caseValues") {
       prop.caseValues = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.caseValues)>>(value);
       return;
    }
}

void SwitchResultCountOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.caseValues) attrs.append("caseValues", prop.caseValues);
}

::llvm::LogicalResult SwitchResultCountOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getCaseValuesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps14(attr, "caseValues", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult SwitchResultCountOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.caseValues)))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchResultCountOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.caseValues);
}

::mlir::DenseIntElementsAttr SwitchResultCountOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

void SwitchResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value inputOp, ArrayRef<int32_t> counts, Block *defaultDest, BlockRange dests) {
    build(odsBuilder, odsState, inputOp, odsBuilder.getI32VectorAttr(counts),
          defaultDest, dests);
  
}

void SwitchResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::DenseIntElementsAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::DenseIntElementsAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchResultCountOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<SwitchResultCountOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult SwitchResultCountOp::verifyInvariantsImpl() {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitOpError("requires attribute 'caseValues'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps14(*this, tblgen_caseValues, "caseValues")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult SwitchResultCountOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchResultCountOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(&inputOpRawOperand, 1);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::DenseIntElementsAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperand))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(caseValuesAttr, ::mlir::Type{})) {
    return ::mlir::failure();
  }
  if (caseValuesAttr) result.getOrAddProperties<SwitchResultCountOp::Properties>().caseValues = caseValuesAttr;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
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
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchResultCountOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getCaseValuesAttr());
  _odsPrinter << "(";
  ::llvm::interleaveComma(getCases(), _odsPrinter);
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("caseValues");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchResultCountOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchResultCountOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchTypeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchTypeOpGenericAdaptorBase::SwitchTypeOpGenericAdaptorBase(SwitchTypeOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::ArrayAttr SwitchTypeOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

} // namespace detail
SwitchTypeOpAdaptor::SwitchTypeOpAdaptor(SwitchTypeOp op) : SwitchTypeOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult SwitchTypeOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitError(loc, "'pdl_interp.switch_type' op ""requires attribute 'caseValues'");

  if (tblgen_caseValues && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_caseValues))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_caseValues), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'pdl_interp.switch_type' op ""attribute 'caseValues' failed to satisfy constraint: type array attribute");
  return ::mlir::success();
}

::llvm::LogicalResult SwitchTypeOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.caseValues;
       auto attr = dict.get("caseValues");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `caseValues` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute SwitchTypeOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.caseValues;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("caseValues",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code SwitchTypeOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.caseValues.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> SwitchTypeOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "caseValues")
      return prop.caseValues;
  return std::nullopt;
}

void SwitchTypeOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "caseValues") {
       prop.caseValues = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.caseValues)>>(value);
       return;
    }
}

void SwitchTypeOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.caseValues) attrs.append("caseValues", prop.caseValues);
}

::llvm::LogicalResult SwitchTypeOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getCaseValuesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps7(attr, "caseValues", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult SwitchTypeOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.caseValues)))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchTypeOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.caseValues);
}

::mlir::ArrayAttr SwitchTypeOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

void SwitchTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value edge, ArrayRef<Attribute> types, Block *defaultDest, BlockRange dests) {
      build(odsBuilder, odsState, edge, odsBuilder.getArrayAttr(types),
            defaultDest, dests);
    
}

void SwitchTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<SwitchTypeOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult SwitchTypeOp::verifyInvariantsImpl() {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitOpError("requires attribute 'caseValues'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps7(*this, tblgen_caseValues, "caseValues")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult SwitchTypeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(&valueRawOperand, 1);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::ArrayAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperand))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(caseValuesAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (caseValuesAttr) result.getOrAddProperties<SwitchTypeOp::Properties>().caseValues = caseValuesAttr;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
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
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  if (parser.resolveOperands(valueOperands, odsBuildableType0, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchTypeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCaseValuesAttr());
  _odsPrinter << "(";
  ::llvm::interleaveComma(getCases(), _odsPrinter);
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("caseValues");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchTypeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchTypeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchTypesOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchTypesOpGenericAdaptorBase::SwitchTypesOpGenericAdaptorBase(SwitchTypesOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::ArrayAttr SwitchTypesOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

} // namespace detail
SwitchTypesOpAdaptor::SwitchTypesOpAdaptor(SwitchTypesOp op) : SwitchTypesOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult SwitchTypesOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitError(loc, "'pdl_interp.switch_types' op ""requires attribute 'caseValues'");

  if (tblgen_caseValues && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_caseValues))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_caseValues), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))); }))))
    return emitError(loc, "'pdl_interp.switch_types' op ""attribute 'caseValues' failed to satisfy constraint: type-array array attribute");
  return ::mlir::success();
}

::llvm::LogicalResult SwitchTypesOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.caseValues;
       auto attr = dict.get("caseValues");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `caseValues` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute SwitchTypesOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.caseValues;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("caseValues",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code SwitchTypesOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.caseValues.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> SwitchTypesOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "caseValues")
      return prop.caseValues;
  return std::nullopt;
}

void SwitchTypesOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "caseValues") {
       prop.caseValues = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.caseValues)>>(value);
       return;
    }
}

void SwitchTypesOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.caseValues) attrs.append("caseValues", prop.caseValues);
}

::llvm::LogicalResult SwitchTypesOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getCaseValuesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps15(attr, "caseValues", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult SwitchTypesOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.caseValues)))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchTypesOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.caseValues);
}

::mlir::ArrayAttr SwitchTypesOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

void SwitchTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value edge, ArrayRef<Attribute> types, Block *defaultDest, BlockRange dests) {
      build(odsBuilder, odsState, edge, odsBuilder.getArrayAttr(types),
            defaultDest, dests);
    
}

void SwitchTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchTypesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<SwitchTypesOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

::llvm::LogicalResult SwitchTypesOp::verifyInvariantsImpl() {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitOpError("requires attribute 'caseValues'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps15(*this, tblgen_caseValues, "caseValues")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::llvm::LogicalResult SwitchTypesOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchTypesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(&valueRawOperand, 1);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::ArrayAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperand))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(caseValuesAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (caseValuesAttr) result.getOrAddProperties<SwitchTypesOp::Properties>().caseValues = caseValuesAttr;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
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
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::TypeType>());
  if (parser.resolveOperands(valueOperands, odsBuildableType0, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchTypesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCaseValuesAttr());
  _odsPrinter << "(";
  ::llvm::interleaveComma(getCases(), _odsPrinter);
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("caseValues");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchTypesOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchTypesOp)


#endif  // GET_OP_CLASSES

