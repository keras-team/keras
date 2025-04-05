/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: lower_tf.td                                                          *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


static ::llvm::LogicalResult __mlir_ods_local_type_constraint_lower_tf1(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Type type,
    ::llvm::StringRef failureStr) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return ((((elementType.isF16())) || ((elementType.isa<mlir::TF::HalfRefType>()))) || (((elementType.isF32())) || ((elementType.isa<mlir::TF::FloatRefType>()))) || (((elementType.isF64())) || ((elementType.isa<mlir::TF::DoubleRefType>()))) || (((elementType.isBF16())) || ((elementType.isa<mlir::TF::Bfloat16RefType>()))) || (((elementType.isFloat8E4M3FN())) || ((elementType.isa<mlir::TF::Float8E4M3FNRefType>()))) || (((elementType.isFloat8E5M2())) || ((elementType.isa<mlir::TF::Float8E5M2RefType>())))) || ((((elementType.isSignedInteger(4))) || ((elementType.isa<mlir::TF::Int4RefType>()))) || (((elementType.isSignlessInteger(8))) || ((elementType.isa<mlir::TF::Int8RefType>()))) || (((elementType.isSignlessInteger(16))) || ((elementType.isa<mlir::TF::Int16RefType>()))) || (((elementType.isSignlessInteger(32))) || ((elementType.isa<mlir::TF::Int32RefType>()))) || (((elementType.isSignlessInteger(64))) || ((elementType.isa<mlir::TF::Int64RefType>())))) || (((((::llvm::isa<::mlir::ComplexType>(elementType))) && ((::llvm::cast<::mlir::ComplexType>(elementType).getElementType().isF32()))) || ((elementType.isa<mlir::TF::Complex64RefType>()))) || ((((::llvm::isa<::mlir::ComplexType>(elementType))) && ((::llvm::cast<::mlir::ComplexType>(elementType).getElementType().isF64()))) || ((elementType.isa<mlir::TF::Complex128RefType>())))) || (((elementType.isUnsignedInteger(8))) || ((elementType.isa<mlir::TF::Uint8RefType>()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": tensor of floating-point or signed integer or complex or 8-bit unsigned integer values";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_lower_tf2(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Type type,
    ::llvm::StringRef failureStr) {
  if (!(((::llvm::isa<::mlir::RankedTensorType>(type))) && ([](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": ranked tensor of any type values";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_lower_tf3(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Type type,
    ::llvm::StringRef failureStr) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (((elementType.isF16())) || ((elementType.isa<mlir::TF::HalfRefType>()))) || (((elementType.isF32())) || ((elementType.isa<mlir::TF::FloatRefType>()))) || (((elementType.isF64())) || ((elementType.isa<mlir::TF::DoubleRefType>()))) || (((elementType.isBF16())) || ((elementType.isa<mlir::TF::Bfloat16RefType>()))) || (((elementType.isFloat8E4M3FN())) || ((elementType.isa<mlir::TF::Float8E4M3FNRefType>()))) || (((elementType.isFloat8E5M2())) || ((elementType.isa<mlir::TF::Float8E5M2RefType>()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": tensor of floating-point values";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_lower_tf4(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Type type,
    ::llvm::StringRef failureStr) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return ((::llvm::isa<::mlir::IntegerType>(elementType))) || ((::llvm::isa<::mlir::FloatType>(elementType))) || ((::llvm::isa<::mlir::ComplexType>(elementType))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": tensor of integer or floating-point or complex-type values";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_lower_tf5(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Type type,
    ::llvm::StringRef failureStr) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((::llvm::isa<::mlir::FloatType>(elementType))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": tensor of signless integer or floating-point values";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_lower_tf6(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Type type,
    ::llvm::StringRef failureStr) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return ((((elementType.isSignedInteger(4))) || ((elementType.isa<mlir::TF::Int4RefType>()))) || (((elementType.isSignlessInteger(8))) || ((elementType.isa<mlir::TF::Int8RefType>()))) || (((elementType.isSignlessInteger(16))) || ((elementType.isa<mlir::TF::Int16RefType>()))) || (((elementType.isSignlessInteger(32))) || ((elementType.isa<mlir::TF::Int32RefType>()))) || (((elementType.isSignlessInteger(64))) || ((elementType.isa<mlir::TF::Int64RefType>())))) || ((((elementType.isUnsignedInteger(4))) || ((elementType.isa<mlir::TF::Uint4RefType>()))) || (((elementType.isUnsignedInteger(8))) || ((elementType.isa<mlir::TF::Uint8RefType>()))) || (((elementType.isUnsignedInteger(16))) || ((elementType.isa<mlir::TF::Uint16RefType>()))) || (((elementType.isUnsignedInteger(32))) || ((elementType.isa<mlir::TF::Uint32RefType>()))) || (((elementType.isUnsignedInteger(64))) || ((elementType.isa<mlir::TF::Uint64RefType>())))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": tensor of integer values";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_lower_tf7(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Type type,
    ::llvm::StringRef failureStr) {
  if (!((((::llvm::isa<::mlir::RankedTensorType>(type))) && ((::llvm::cast<::mlir::ShapedType>(type).hasStaticShape()))) && ([](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": statically shaped tensor of any type values";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_lower_tf8(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Type type,
    ::llvm::StringRef failureStr) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (::llvm::isa<::mlir::ComplexType>(elementType)); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": tensor of complex-type values";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_lower_tf1(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((attr.cast<StringAttr>().getValue() == "MIN_COMBINED"))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": string attribute whose value is MIN_COMBINED";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_lower_tf2(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((attr == rewriter.getBoolAttr(false)))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": constant attribute false";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_lower_tf3(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((attr == rewriter.getIntegerAttr(rewriter.getIntegerType(64), -1)))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": constant attribute -1";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_lower_tf4(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((attr.cast<::mlir::BoolAttr>().getValue()))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": ";
    });
  }
  return ::mlir::success();
}
static ::llvm::LogicalResult static_dag_matcher_0(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &input, ::mlir::Operation::operand_range &indices, ::mlir::Operation::operand_range &updates, ::mlir::StringAttr &bad_indices_policy) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::TensorScatterUpdateOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::TF::TensorScatterUpdateOp type";
    });
  }
  input = castedOp1.getODSOperands(0);
  indices = castedOp1.getODSOperands(1);
  updates = castedOp1.getODSOperands(2);
  {
    auto tblgen_attr = op0->getAttrOfType<::mlir::StringAttr>("bad_indices_policy");(void)tblgen_attr;
    if (!tblgen_attr) tblgen_attr = rewriter.getStringAttr("");
    bad_indices_policy = tblgen_attr;
  }
  return ::mlir::success();
}

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:67
*/
struct LowerAddOp : public ::mlir::RewritePattern {
  LowerAddOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Add", 1, context, {"tf.AddV2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::AddOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'tf.Add' failed to satisfy constraint: 'tensor of floating-point or signed integer or complex or 8-bit unsigned integer values'"))) {
      return ::mlir::failure();
    }
    x = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf1(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'tf.Add' failed to satisfy constraint: 'tensor of floating-point or signed integer or complex or 8-bit unsigned integer values'"))) {
      return ::mlir::failure();
    }
    y = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::AddV2Op tblgen_AddV2Op_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_AddV2Op_0 = rewriter.create<::mlir::TF::AddV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AddV2Op_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:79
*/
struct LowerBiasAddGradOp : public ::mlir::RewritePattern {
  LowerBiasAddGradOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.BiasAddGrad", 1, context, {"tf.Const", "tf.Sum"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range out_backprop(op0->getOperands());
    ::mlir::StringAttr data_format;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::BiasAddGradOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf2(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'tf.BiasAddGrad' failed to satisfy constraint: 'ranked tensor of any type values'"))) {
      return ::mlir::failure();
    }
    out_backprop = castedOp0.getODSOperands(0);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::StringAttr>("data_format");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getStringAttr("NHWC");
      data_format = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = GetBiasAddGradReductionIndices((*out_backprop.begin()).getType().cast<RankedTensorType>().getRank(), data_format, &rewriter); (void)nativeVar_0;
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::TF::SumOp tblgen_SumOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*out_backprop.begin()));
      tblgen_values.push_back((*tblgen_ConstOp_1.getODSResults(0).begin()));
      if (auto tmpAttr = rewriter.getBoolAttr(false)) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("keep_dims"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SumOp_2 = rewriter.create<::mlir::TF::SumOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SumOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:168
*/
struct LowerDequantizeOp : public ::mlir::RewritePattern {
  LowerDequantizeOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Dequantize", 1, context, {"tf.AddV2", "tf.Cast", "tf.Const", "tf.Div", "tf.Mul", "tf.Sub"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::IntegerAttr axis;
    ::mlir::BoolAttr narrow_range;
    ::mlir::TF::DequantizeOp result;
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range min_range(op0->getOperands());
    ::mlir::Operation::operand_range max_range(op0->getOperands());
    ::mlir::StringAttr mode;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::DequantizeOp>(op0); (void)castedOp0;
    result = castedOp0;
    input = castedOp0.getODSOperands(0);
    min_range = castedOp0.getODSOperands(1);
    max_range = castedOp0.getODSOperands(2);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::StringAttr>("mode");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getStringAttr("MIN_COMBINED");
      if (!tblgen_attr) return ::mlir::failure();
      if(::mlir::failed(__mlir_ods_local_attr_constraint_lower_tf1(rewriter, op0, tblgen_attr, "op 'tf.Dequantize' attribute 'mode' failed to satisfy constraint: 'string attribute whose value is MIN_COMBINED'"))) {
        return ::mlir::failure();
      }
      mode = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("narrow_range");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
      if (!tblgen_attr) return ::mlir::failure();
      if(::mlir::failed(__mlir_ods_local_attr_constraint_lower_tf2(rewriter, op0, tblgen_attr, "op 'tf.Dequantize' attribute 'narrow_range' failed to satisfy constraint: 'constant attribute false'"))) {
        return ::mlir::failure();
      }
      narrow_range = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::IntegerAttr>("axis");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getIntegerAttr(rewriter.getIntegerType(64), -1);
      if (!tblgen_attr) return ::mlir::failure();
      if(::mlir::failed(__mlir_ods_local_attr_constraint_lower_tf3(rewriter, op0, tblgen_attr, "op 'tf.Dequantize' attribute 'axis' failed to satisfy constraint: 'constant attribute -1'"))) {
        return ::mlir::failure();
      }
      axis = tblgen_attr;
    }
    if (!(((::llvm::isa<::mlir::TensorType>(((*input.begin()).getType())))) && ([](::mlir::Type elementType) { return (((elementType.isa<mlir::TF::Qint8Type>())) || ((elementType.isa<mlir::TF::Qint8RefType>()))) || (((elementType.isa<mlir::TF::Quint8Type>())) || ((elementType.isa<mlir::TF::Quint8RefType>()))); }(::llvm::cast<::mlir::ShapedType>(((*input.begin()).getType())).getElementType())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "value entity 'input' failed to satisfy constraint: 'tensor of 8-bit quantized integer or 8-bit quantized unsigned integer values'";
      });
    }
    if (!(((::llvm::isa<::mlir::TensorType>(((*min_range.begin()).getType())))) && ([](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isa<mlir::TF::FloatRefType>())); }(::llvm::cast<::mlir::ShapedType>(((*min_range.begin()).getType())).getElementType())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "value entity 'min_range' failed to satisfy constraint: 'tensor of 32-bit float values'";
      });
    }
    if (!(((::llvm::isa<::mlir::TensorType>(((*max_range.begin()).getType())))) && ([](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isa<mlir::TF::FloatRefType>())); }(::llvm::cast<::mlir::ShapedType>(((*max_range.begin()).getType())).getElementType())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "value entity 'max_range' failed to satisfy constraint: 'tensor of 32-bit float values'";
      });
    }
    if (!(((::llvm::isa<::mlir::TensorType>(((*result.getODSResults(0).begin()).getType())))) && ([](::mlir::Type elementType) { return (((elementType.isF32())) || ((elementType.isa<mlir::TF::FloatRefType>()))) || (((elementType.isBF16())) || ((elementType.isa<mlir::TF::Bfloat16RefType>()))); }(::llvm::cast<::mlir::ShapedType>(((*result.getODSResults(0).begin()).getType())).getElementType())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "value entity 'result' failed to satisfy constraint: 'tensor of 32-bit float or bfloat16 values'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = CreateTFCastOpF32(&rewriter, (*result.getODSResults(0).begin()).getLoc(), (*input.begin()), rewriter.getBoolAttr(false)); (void)nativeVar_0;
    auto nativeVar_1 = DequantizeHalfRange(&rewriter, (*input.begin())); (void)nativeVar_1;
    ::mlir::TF::ConstOp tblgen_ConstOp_2;
    {
      tblgen_ConstOp_2 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_1
      );
    }
    ::mlir::TF::AddV2Op tblgen_AddV2Op_3;
    {
      ::mlir::Value tblgen_value_0 = nativeVar_0;
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstOp_2.getODSResults(0).begin());
      tblgen_AddV2Op_3 = rewriter.create<::mlir::TF::AddV2Op>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::SubOp tblgen_SubOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*max_range.begin());
      ::mlir::Value tblgen_value_1 = (*min_range.begin());
      tblgen_SubOp_4 = rewriter.create<::mlir::TF::SubOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    auto nativeVar_5 = GetF32Scalar(&rewriter, 255); (void)nativeVar_5;
    ::mlir::TF::ConstOp tblgen_ConstOp_6;
    {
      tblgen_ConstOp_6 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_5
      );
    }
    ::mlir::TF::DivOp tblgen_DivOp_7;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_SubOp_4.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstOp_6.getODSResults(0).begin());
      tblgen_DivOp_7 = rewriter.create<::mlir::TF::DivOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_8;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_AddV2Op_3.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_DivOp_7.getODSResults(0).begin());
      tblgen_MulOp_8 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::AddV2Op tblgen_AddV2Op_9;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_8.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*min_range.begin());
      tblgen_AddV2Op_9 = rewriter.create<::mlir::TF::AddV2Op>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::CastOp tblgen_CastOp_10;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_AddV2Op_9.getODSResults(0).begin()));
      if (auto tmpAttr = rewriter.getBoolAttr(false)) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("Truncate"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_CastOp_10 = rewriter.create<::mlir::TF::CastOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_CastOp_10.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:218
*/
struct LowerDivNoNanOp : public ::mlir::RewritePattern {
  LowerDivNoNanOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.DivNoNan", 1, context, {"tf.Const", "tf.Div", "tf.Equal", "tf.SelectV2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range l(op0->getOperands());
    ::mlir::Operation::operand_range r(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::DivNoNanOp>(op0); (void)castedOp0;
    l = castedOp0.getODSOperands(0);
    r = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = GetScalarOfType(getElementTypeOrSelf((*r.begin())),0); (void)nativeVar_0;
    ::mlir::TF::ConstOp zero;
    {
      zero = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::TF::EqualOp tblgen_EqualOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*r.begin());
      ::mlir::Value tblgen_value_1 = (*zero.getODSResults(0).begin());
      tblgen_EqualOp_1 = rewriter.create<::mlir::TF::EqualOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1,
        rewriter.getBoolAttr(true)
      );
    }
    ::mlir::TF::DivOp tblgen_DivOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*l.begin());
      ::mlir::Value tblgen_value_1 = (*r.begin());
      tblgen_DivOp_2 = rewriter.create<::mlir::TF::DivOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::SelectV2Op tblgen_SelectV2Op_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_EqualOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*zero.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_DivOp_2.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SelectV2Op_3 = rewriter.create<::mlir::TF::SelectV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SelectV2Op_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:254
*/
struct LowerEmptyOp : public ::mlir::RewritePattern {
  LowerEmptyOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Empty", 1, context, {"tf.BroadcastTo", "tf.Const"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::EmptyOp result;
    ::mlir::Operation::operand_range dims(op0->getOperands());
    ::mlir::BoolAttr init;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::EmptyOp>(op0); (void)castedOp0;
    result = castedOp0;
    dims = castedOp0.getODSOperands(0);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("init");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
      if (!tblgen_attr) return ::mlir::failure();
      if(::mlir::failed(__mlir_ods_local_attr_constraint_lower_tf4(rewriter, op0, tblgen_attr, "op 'tf.Empty' attribute 'init' failed to satisfy constraint: ''"))) {
        return ::mlir::failure();
      }
      init = tblgen_attr;
    }
    if (!(((::llvm::isa<::mlir::TensorType>(((*result.getODSResults(0).begin()).getType())))) && ([](::mlir::Type elementType) { return ((((elementType.isSignedInteger(4))) || ((elementType.isa<mlir::TF::Int4RefType>()))) || (((elementType.isSignlessInteger(8))) || ((elementType.isa<mlir::TF::Int8RefType>()))) || (((elementType.isSignlessInteger(16))) || ((elementType.isa<mlir::TF::Int16RefType>()))) || (((elementType.isSignlessInteger(32))) || ((elementType.isa<mlir::TF::Int32RefType>()))) || (((elementType.isSignlessInteger(64))) || ((elementType.isa<mlir::TF::Int64RefType>())))) || ((((elementType.isF16())) || ((elementType.isa<mlir::TF::HalfRefType>()))) || (((elementType.isF32())) || ((elementType.isa<mlir::TF::FloatRefType>()))) || (((elementType.isF64())) || ((elementType.isa<mlir::TF::DoubleRefType>()))) || (((elementType.isBF16())) || ((elementType.isa<mlir::TF::Bfloat16RefType>()))) || (((elementType.isFloat8E4M3FN())) || ((elementType.isa<mlir::TF::Float8E4M3FNRefType>()))) || (((elementType.isFloat8E5M2())) || ((elementType.isa<mlir::TF::Float8E5M2RefType>())))); }(::llvm::cast<::mlir::ShapedType>(((*result.getODSResults(0).begin()).getType())).getElementType())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "value entity 'result' failed to satisfy constraint: 'tensor of signed integer or floating-point values'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = GetScalarOfType(getElementTypeOrSelf((*result.getODSResults(0).begin())),0); (void)nativeVar_0;
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::TF::BroadcastToOp tblgen_BroadcastToOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ConstOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*dims.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_BroadcastToOp_2 = rewriter.create<::mlir::TF::BroadcastToOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_BroadcastToOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:613
*/
struct LowerExp1mOp : public ::mlir::RewritePattern {
  LowerExp1mOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Expm1", 1, context, {"tf.Const", "tf.Exp", "tf.Sub"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::Expm1Op>(op0); (void)castedOp0;
    x = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::ExpOp tblgen_ExpOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*x.begin());
      tblgen_ExpOp_0 = rewriter.create<::mlir::TF::ExpOp>(odsLoc,
        /*x=*/tblgen_value_0
      );
    }
    auto nativeVar_1 = GetScalarOfType(getElementTypeOrSelf((*x.begin())),1); (void)nativeVar_1;
    ::mlir::TF::ConstOp tblgen_ConstOp_2;
    {
      tblgen_ConstOp_2 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_1
      );
    }
    ::mlir::TF::SubOp tblgen_SubOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ExpOp_0.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_ConstOp_2.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SubOp_3 = rewriter.create<::mlir::TF::SubOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SubOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:448
*/
struct LowerFakeQuantWithMinMaxArgs : public ::mlir::RewritePattern {
  LowerFakeQuantWithMinMaxArgs(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.FakeQuantWithMinMaxArgs", 1, context, {"tf.Const", "tf.FakeQuantWithMinMaxVars"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::FloatAttr min;
    ::mlir::IntegerAttr bits;
    ::mlir::FloatAttr max;
    ::mlir::BoolAttr narrow_range;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::FakeQuantWithMinMaxArgsOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf3(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'tf.FakeQuantWithMinMaxArgs' failed to satisfy constraint: 'tensor of floating-point values'"))) {
      return ::mlir::failure();
    }
    input = castedOp0.getODSOperands(0);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::FloatAttr>("min");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getFloatAttr(rewriter.getF32Type(), -6.0f);
      min = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::FloatAttr>("max");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getFloatAttr(rewriter.getF32Type(), 6.0f);
      max = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::IntegerAttr>("num_bits");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getIntegerAttr(rewriter.getIntegerType(64), 8);
      bits = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("narrow_range");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
      narrow_range = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::ConstOp tblgen_ConstOp_0;
    {
      tblgen_ConstOp_0 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/min
      );
    }
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/max
      );
    }
    ::mlir::TF::FakeQuantWithMinMaxVarsOp tblgen_FakeQuantWithMinMaxVarsOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*tblgen_ConstOp_0.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_ConstOp_1.getODSResults(0).begin()));
      if (auto tmpAttr = bits) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("num_bits"), tmpAttr);
      }
      if (auto tmpAttr = narrow_range) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("narrow_range"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_FakeQuantWithMinMaxVarsOp_2 = rewriter.create<::mlir::TF::FakeQuantWithMinMaxVarsOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_FakeQuantWithMinMaxVarsOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:247
*/
struct LowerFillOp : public ::mlir::RewritePattern {
  LowerFillOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Fill", 1, context, {"tf.BroadcastTo"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range dims(op0->getOperands());
    ::mlir::Operation::operand_range value(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::FillOp>(op0); (void)castedOp0;
    dims = castedOp0.getODSOperands(0);
    value = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::BroadcastToOp tblgen_BroadcastToOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*value.begin()));
      tblgen_values.push_back((*dims.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_BroadcastToOp_0 = rewriter.create<::mlir::TF::BroadcastToOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_BroadcastToOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:263
*/
struct LowerInv : public ::mlir::RewritePattern {
  LowerInv(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Inv", 1, context, {"tf.Reciprocal"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::InvOp>(op0); (void)castedOp0;
    x = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::ReciprocalOp tblgen_ReciprocalOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ReciprocalOp_0 = rewriter.create<::mlir::TF::ReciprocalOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ReciprocalOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:512
*/
struct LowerIsFiniteOp : public ::mlir::RewritePattern {
  LowerIsFiniteOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.IsFinite", 1, context, {"tf.Const", "tf.Equal", "tf.Sub"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::IsFiniteOp>(op0); (void)castedOp0;
    x = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::SubOp tblgen_SubOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*x.begin());
      ::mlir::Value tblgen_value_1 = (*x.begin());
      tblgen_SubOp_0 = rewriter.create<::mlir::TF::SubOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    auto nativeVar_1 = GetScalarOfType(getElementTypeOrSelf((*x.begin())),0); (void)nativeVar_1;
    ::mlir::TF::ConstOp tblgen_ConstOp_2;
    {
      tblgen_ConstOp_2 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_1
      );
    }
    ::mlir::TF::EqualOp tblgen_EqualOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_SubOp_0.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_ConstOp_2.getODSResults(0).begin()));
      if (auto tmpAttr = rewriter.getBoolAttr(true)) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("incompatible_shape_error"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_EqualOp_3 = rewriter.create<::mlir::TF::EqualOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_EqualOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:269
*/
struct LowerIsInfOp : public ::mlir::RewritePattern {
  LowerIsInfOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.IsInf", 1, context, {"tf.Abs", "tf.Const", "tf.Equal"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::IsInfOp>(op0); (void)castedOp0;
    x = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::AbsOp abs;
    {
      ::mlir::Value tblgen_value_0 = (*x.begin());
      abs = rewriter.create<::mlir::TF::AbsOp>(odsLoc,
        /*x=*/tblgen_value_0
      );
    }
    auto nativeVar_0 = GetScalarOfType(getElementTypeOrSelf((*x.begin())), std::numeric_limits<double>::infinity()); (void)nativeVar_0;
    ::mlir::TF::ConstOp inf;
    {
      inf = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::TF::EqualOp tblgen_EqualOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*abs.getODSResults(0).begin()));
      tblgen_values.push_back((*inf.getODSResults(0).begin()));
      if (auto tmpAttr = rewriter.getBoolAttr(true)) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("incompatible_shape_error"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_EqualOp_1 = rewriter.create<::mlir::TF::EqualOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_EqualOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:278
*/
struct LowerIsNanOp : public ::mlir::RewritePattern {
  LowerIsNanOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.IsNan", 1, context, {"tf.NotEqual"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::IsNanOp>(op0); (void)castedOp0;
    x = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::NotEqualOp tblgen_NotEqualOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*x.begin()));
      if (auto tmpAttr = rewriter.getBoolAttr(true)) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("incompatible_shape_error"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_NotEqualOp_0 = rewriter.create<::mlir::TF::NotEqualOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_NotEqualOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:295
*/
struct LowerL2LossOp : public ::mlir::RewritePattern {
  LowerL2LossOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.L2Loss", 1, context, {"tf.Const", "tf.Div", "tf.Mul", "tf.Sum"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::L2LossOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf2(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'tf.L2Loss' failed to satisfy constraint: 'ranked tensor of any type values'"))) {
      return ::mlir::failure();
    }
    input = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::MulOp tblgen_MulOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*input.begin());
      ::mlir::Value tblgen_value_1 = (*input.begin());
      tblgen_MulOp_0 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    auto nativeVar_1 = GetI64ElementsAttrForSeq(0, (*input.begin()).getType().cast<RankedTensorType>().getRank(), &rewriter); (void)nativeVar_1;
    ::mlir::TF::ConstOp tblgen_ConstOp_2;
    {
      tblgen_ConstOp_2 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_1
      );
    }
    ::mlir::TF::SumOp tblgen_SumOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_0.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstOp_2.getODSResults(0).begin());
      tblgen_SumOp_3 = rewriter.create<::mlir::TF::SumOp>(odsLoc,
        /*input=*/tblgen_value_0,
        /*reduction_indices=*/tblgen_value_1,
        rewriter.getBoolAttr(false)
      );
    }
    auto nativeVar_4 = GetScalarOfType(getElementTypeOrSelf((*input.begin())),2); (void)nativeVar_4;
    ::mlir::TF::ConstOp tblgen_ConstOp_5;
    {
      tblgen_ConstOp_5 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_4
      );
    }
    ::mlir::TF::DivOp tblgen_DivOp_6;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_SumOp_3.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_ConstOp_5.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_DivOp_6 = rewriter.create<::mlir::TF::DivOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_DivOp_6.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:636
*/
struct LowerMatrixBandPartOp : public ::mlir::RewritePattern {
  LowerMatrixBandPartOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.MatrixBandPart", 1, context, {"tf.Const", "tf.ExpandDims", "tf.GreaterEqual", "tf.Less", "tf.LessEqual", "tf.LogicalAnd", "tf.Neg", "tf.Range", "tf.SelectV2", "tf.Sub", "tf.ZerosLike"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range num_lower(op0->getOperands());
    ::mlir::Operation::operand_range num_upper(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::MatrixBandPartOp>(op0); (void)castedOp0;
    input = castedOp0.getODSOperands(0);
    num_lower = castedOp0.getODSOperands(1);
    num_upper = castedOp0.getODSOperands(2);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    auto nativeVar_0 = GetScalarOfType(getElementTypeOrSelf((*num_lower.begin())),0); (void)nativeVar_0;
    ::mlir::TF::ConstOp zero;
    {
      zero = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    auto nativeVar_1 = GetScalarOfType(getElementTypeOrSelf((*num_lower.begin())),1); (void)nativeVar_1;
    ::mlir::TF::ConstOp one;
    {
      one = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_1
      );
    }
    auto nativeVar_2 = GetScalarOfType(getElementTypeOrSelf((*num_lower.begin())),-1); (void)nativeVar_2;
    ::mlir::TF::ConstOp neg_one;
    {
      neg_one = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_2
      );
    }
    auto nativeVar_3 = rewriter.getBoolAttr(getElementTypeOrSelf((*num_lower.begin()).getType()).isInteger(32)); (void)nativeVar_3;
    auto m = GetDimensionSize(&rewriter, odsLoc, (*input.begin()), -2, nativeVar_3); (void)m;
    auto nativeVar_4 = rewriter.getBoolAttr(getElementTypeOrSelf((*num_upper.begin()).getType()).isInteger(32)); (void)nativeVar_4;
    auto n = GetDimensionSize(&rewriter, odsLoc, (*input.begin()), -1, nativeVar_4); (void)n;
    ::mlir::TF::LessOp tblgen_LessOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*num_lower.begin());
      ::mlir::Value tblgen_value_1 = (*zero.getODSResults(0).begin());
      tblgen_LessOp_5 = rewriter.create<::mlir::TF::LessOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::SelectV2Op num_lower_or_m;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_LessOp_5.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = m;
      ::mlir::Value tblgen_value_2 = (*num_lower.begin());
      num_lower_or_m = rewriter.create<::mlir::TF::SelectV2Op>(odsLoc,
        /*condition=*/tblgen_value_0,
        /*then_value=*/tblgen_value_1,
        /*else_value=*/tblgen_value_2
      );
    }
    ::mlir::TF::LessOp tblgen_LessOp_6;
    {
      ::mlir::Value tblgen_value_0 = (*num_upper.begin());
      ::mlir::Value tblgen_value_1 = (*zero.getODSResults(0).begin());
      tblgen_LessOp_6 = rewriter.create<::mlir::TF::LessOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::SelectV2Op num_upper_or_n;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_LessOp_6.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = n;
      ::mlir::Value tblgen_value_2 = (*num_upper.begin());
      num_upper_or_n = rewriter.create<::mlir::TF::SelectV2Op>(odsLoc,
        /*condition=*/tblgen_value_0,
        /*then_value=*/tblgen_value_1,
        /*else_value=*/tblgen_value_2
      );
    }
    ::mlir::TF::RangeOp range_m;
    {
      ::mlir::Value tblgen_value_0 = (*zero.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = m;
      ::mlir::Value tblgen_value_2 = (*one.getODSResults(0).begin());
      range_m = rewriter.create<::mlir::TF::RangeOp>(odsLoc,
        /*start=*/tblgen_value_0,
        /*limit=*/tblgen_value_1,
        /*delta=*/tblgen_value_2
      );
    }
    ::mlir::TF::RangeOp range_n;
    {
      ::mlir::Value tblgen_value_0 = (*zero.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = n;
      ::mlir::Value tblgen_value_2 = (*one.getODSResults(0).begin());
      range_n = rewriter.create<::mlir::TF::RangeOp>(odsLoc,
        /*start=*/tblgen_value_0,
        /*limit=*/tblgen_value_1,
        /*delta=*/tblgen_value_2
      );
    }
    ::mlir::TF::ExpandDimsOp tblgen_ExpandDimsOp_7;
    {
      ::mlir::Value tblgen_value_0 = (*range_m.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*neg_one.getODSResults(0).begin());
      tblgen_ExpandDimsOp_7 = rewriter.create<::mlir::TF::ExpandDimsOp>(odsLoc,
        /*input=*/tblgen_value_0,
        /*dim=*/tblgen_value_1
      );
    }
    ::mlir::TF::SubOp offset;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ExpandDimsOp_7.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*range_n.getODSResults(0).begin());
      offset = rewriter.create<::mlir::TF::SubOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::LessEqualOp tblgen_LessEqualOp_8;
    {
      ::mlir::Value tblgen_value_0 = (*offset.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*num_lower_or_m.getODSResults(0).begin());
      tblgen_LessEqualOp_8 = rewriter.create<::mlir::TF::LessEqualOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::NegOp tblgen_NegOp_9;
    {
      ::mlir::Value tblgen_value_0 = (*num_upper_or_n.getODSResults(0).begin());
      tblgen_NegOp_9 = rewriter.create<::mlir::TF::NegOp>(odsLoc,
        /*x=*/tblgen_value_0
      );
    }
    ::mlir::TF::GreaterEqualOp tblgen_GreaterEqualOp_10;
    {
      ::mlir::Value tblgen_value_0 = (*offset.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_NegOp_9.getODSResults(0).begin());
      tblgen_GreaterEqualOp_10 = rewriter.create<::mlir::TF::GreaterEqualOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::LogicalAndOp indicator;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_LessEqualOp_8.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_GreaterEqualOp_10.getODSResults(0).begin());
      indicator = rewriter.create<::mlir::TF::LogicalAndOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::ZerosLikeOp tblgen_ZerosLikeOp_11;
    {
      ::mlir::Value tblgen_value_0 = (*input.begin());
      tblgen_ZerosLikeOp_11 = rewriter.create<::mlir::TF::ZerosLikeOp>(odsLoc,
        /*x=*/tblgen_value_0
      );
    }
    ::mlir::TF::SelectV2Op tblgen_SelectV2Op_12;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*indicator.getODSResults(0).begin()));
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*tblgen_ZerosLikeOp_11.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SelectV2Op_12 = rewriter.create<::mlir::TF::SelectV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SelectV2Op_12.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:220
*/
struct LowerMulNoNanOp : public ::mlir::RewritePattern {
  LowerMulNoNanOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.MulNoNan", 1, context, {"tf.Const", "tf.Equal", "tf.Mul", "tf.SelectV2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range l(op0->getOperands());
    ::mlir::Operation::operand_range r(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::MulNoNanOp>(op0); (void)castedOp0;
    l = castedOp0.getODSOperands(0);
    r = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = GetScalarOfType(getElementTypeOrSelf((*r.begin())),0); (void)nativeVar_0;
    ::mlir::TF::ConstOp zero;
    {
      zero = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::TF::EqualOp tblgen_EqualOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*r.begin());
      ::mlir::Value tblgen_value_1 = (*zero.getODSResults(0).begin());
      tblgen_EqualOp_1 = rewriter.create<::mlir::TF::EqualOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1,
        rewriter.getBoolAttr(true)
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*l.begin());
      ::mlir::Value tblgen_value_1 = (*r.begin());
      tblgen_MulOp_2 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::SelectV2Op tblgen_SelectV2Op_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_EqualOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*zero.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_MulOp_2.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SelectV2Op_3 = rewriter.create<::mlir::TF::SelectV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SelectV2Op_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:466
*/
struct LowerOnesLikeOp : public ::mlir::RewritePattern {
  LowerOnesLikeOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.OnesLike", 1, context, {"tf.BroadcastTo", "tf.Const"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::OnesLikeOp src_op;
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::OnesLikeOp>(op0); (void)castedOp0;
    src_op = castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf4(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'tf.OnesLike' failed to satisfy constraint: 'tensor of integer or floating-point or complex-type values'"))) {
      return ::mlir::failure();
    }
    input = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = GetScalarOfType(getElementTypeOrSelf((*input.begin())),1); (void)nativeVar_0;
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    auto nativeVar_2 = rewriter.create<TF::ShapeOp>((*src_op.getODSResults(0).begin()).getLoc(), (*input.begin()), rewriter.getBoolAttr(false)); (void)nativeVar_2;
    ::mlir::TF::BroadcastToOp tblgen_BroadcastToOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ConstOp_1.getODSResults(0).begin()));
      tblgen_values.push_back(nativeVar_2);
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_BroadcastToOp_3 = rewriter.create<::mlir::TF::BroadcastToOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_BroadcastToOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:307
*/
struct LowerPadOp : public ::mlir::RewritePattern {
  LowerPadOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Pad", 1, context, {"tf.Const", "tf.PadV2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range paddings(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::PadOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf5(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'tf.Pad' failed to satisfy constraint: 'tensor of signless integer or floating-point values'"))) {
      return ::mlir::failure();
    }
    input = castedOp0.getODSOperands(0);
    paddings = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = GetScalarOfType(getElementTypeOrSelf((*input.begin())),0); (void)nativeVar_0;
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::TF::PadV2Op tblgen_PadV2Op_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*paddings.begin()));
      tblgen_values.push_back((*tblgen_ConstOp_1.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_PadV2Op_2 = rewriter.create<::mlir::TF::PadV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_PadV2Op_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:319
*/
struct LowerReciprocal : public ::mlir::RewritePattern {
  LowerReciprocal(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Reciprocal", 1, context, {"tf.Const", "tf.Div"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::ReciprocalOp>(op0); (void)castedOp0;
    x = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = GetScalarOfType(getElementTypeOrSelf((*x.begin())),1); (void)nativeVar_0;
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::TF::DivOp tblgen_DivOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ConstOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*x.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_DivOp_2 = rewriter.create<::mlir::TF::DivOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_DivOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:328
*/
struct LowerRintOp : public ::mlir::RewritePattern {
  LowerRintOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Rint", 1, context, {"tf.Round"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::RintOp res;
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::RintOp>(op0); (void)castedOp0;
    res = castedOp0;
    input = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::RoundOp tblgen_RoundOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_RoundOp_0 = rewriter.create<::mlir::TF::RoundOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_RoundOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:337
*/
struct LowerRoundOpOnFloatTensor : public ::mlir::RewritePattern {
  LowerRoundOpOnFloatTensor(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Round", 1, context, {"tf.AddV2", "tf.Const", "tf.Equal", "tf.Floor", "tf.Greater", "tf.LogicalAnd", "tf.LogicalOr", "tf.Mul", "tf.SelectV2", "tf.Sub"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::RoundOp res;
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::RoundOp>(op0); (void)castedOp0;
    res = castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf3(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'tf.Round' failed to satisfy constraint: 'tensor of floating-point values'"))) {
      return ::mlir::failure();
    }
    input = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = GetScalarOfType(getElementTypeOrSelf((*input.begin())),0.0); (void)nativeVar_0;
    ::mlir::TF::ConstOp zero;
    {
      zero = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::TF::FloorOp round_val;
    {
      ::mlir::Value tblgen_value_0 = (*input.begin());
      round_val = rewriter.create<::mlir::TF::FloorOp>(odsLoc,
        /*x=*/tblgen_value_0
      );
    }
    ::mlir::TF::SubOp fraction;
    {
      ::mlir::Value tblgen_value_0 = (*input.begin());
      ::mlir::Value tblgen_value_1 = (*round_val.getODSResults(0).begin());
      fraction = rewriter.create<::mlir::TF::SubOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    auto nativeVar_1 = GetScalarOfType(getElementTypeOrSelf((*input.begin())),0.5); (void)nativeVar_1;
    ::mlir::TF::ConstOp half;
    {
      half = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_1
      );
    }
    ::mlir::TF::GreaterOp tblgen_GreaterOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*fraction.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*half.getODSResults(0).begin());
      tblgen_GreaterOp_2 = rewriter.create<::mlir::TF::GreaterOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::EqualOp tblgen_EqualOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*fraction.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*half.getODSResults(0).begin());
      tblgen_EqualOp_3 = rewriter.create<::mlir::TF::EqualOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1,
        rewriter.getBoolAttr(true)
      );
    }
    auto nativeVar_4 = GetScalarOfType(getElementTypeOrSelf((*input.begin())),2); (void)nativeVar_4;
    ::mlir::TF::ConstOp tblgen_ConstOp_5;
    {
      tblgen_ConstOp_5 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_4
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_6;
    {
      ::mlir::Value tblgen_value_0 = (*half.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*input.begin());
      tblgen_MulOp_6 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::FloorOp tblgen_FloorOp_7;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_6.getODSResults(0).begin());
      tblgen_FloorOp_7 = rewriter.create<::mlir::TF::FloorOp>(odsLoc,
        /*x=*/tblgen_value_0
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_8;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ConstOp_5.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_FloorOp_7.getODSResults(0).begin());
      tblgen_MulOp_8 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::SubOp tblgen_SubOp_9;
    {
      ::mlir::Value tblgen_value_0 = (*round_val.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_MulOp_8.getODSResults(0).begin());
      tblgen_SubOp_9 = rewriter.create<::mlir::TF::SubOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    auto nativeVar_10 = GetScalarOfType(getElementTypeOrSelf((*input.begin())),1); (void)nativeVar_10;
    ::mlir::TF::ConstOp one;
    {
      one = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_10
      );
    }
    ::mlir::TF::EqualOp tblgen_EqualOp_11;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_SubOp_9.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*one.getODSResults(0).begin());
      tblgen_EqualOp_11 = rewriter.create<::mlir::TF::EqualOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1,
        rewriter.getBoolAttr(true)
      );
    }
    ::mlir::TF::LogicalAndOp tblgen_LogicalAndOp_12;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_EqualOp_3.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_EqualOp_11.getODSResults(0).begin());
      tblgen_LogicalAndOp_12 = rewriter.create<::mlir::TF::LogicalAndOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::LogicalOrOp tblgen_LogicalOrOp_13;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_GreaterOp_2.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_LogicalAndOp_12.getODSResults(0).begin());
      tblgen_LogicalOrOp_13 = rewriter.create<::mlir::TF::LogicalOrOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::AddV2Op tblgen_AddV2Op_14;
    {
      ::mlir::Value tblgen_value_0 = (*round_val.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*one.getODSResults(0).begin());
      tblgen_AddV2Op_14 = rewriter.create<::mlir::TF::AddV2Op>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::SelectV2Op rounded;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_LogicalOrOp_13.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_AddV2Op_14.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*round_val.getODSResults(0).begin());
      rounded = rewriter.create<::mlir::TF::SelectV2Op>(odsLoc,
        /*condition=*/tblgen_value_0,
        /*then_value=*/tblgen_value_1,
        /*else_value=*/tblgen_value_2
      );
    }
    ::mlir::TF::EqualOp tblgen_EqualOp_15;
    {
      ::mlir::Value tblgen_value_0 = (*zero.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*rounded.getODSResults(0).begin());
      tblgen_EqualOp_15 = rewriter.create<::mlir::TF::EqualOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1,
        rewriter.getBoolAttr(true)
      );
    }
    ::mlir::TF::SelectV2Op tblgen_SelectV2Op_16;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_EqualOp_15.getODSResults(0).begin()));
      tblgen_values.push_back((*zero.getODSResults(0).begin()));
      tblgen_values.push_back((*rounded.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SelectV2Op_16 = rewriter.create<::mlir::TF::SelectV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SelectV2Op_16.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:331
*/
struct LowerRoundOpOnIntTensor : public ::mlir::RewritePattern {
  LowerRoundOpOnIntTensor(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Round", 1, context, {"tf.Identity"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::RoundOp res;
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::RoundOp>(op0); (void)castedOp0;
    res = castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf6(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'tf.Round' failed to satisfy constraint: 'tensor of integer values'"))) {
      return ::mlir::failure();
    }
    input = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::IdentityOp tblgen_IdentityOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_IdentityOp_0 = rewriter.create<::mlir::TF::IdentityOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_IdentityOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:383
*/
struct LowerRsqrtGradOp : public ::mlir::RewritePattern {
  LowerRsqrtGradOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.RsqrtGrad", 1, context, {"tf.Const", "tf.Div", "tf.Mul"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::RsqrtGradOp>(op0); (void)castedOp0;
    lhs = castedOp0.getODSOperands(0);
    rhs = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::MulOp tblgen_MulOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*lhs.begin());
      ::mlir::Value tblgen_value_1 = (*lhs.begin());
      tblgen_MulOp_0 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_0.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*lhs.begin());
      tblgen_MulOp_1 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    auto nativeVar_2 = GetScalarOfType(getElementTypeOrSelf((*rhs.begin())),-2); (void)nativeVar_2;
    ::mlir::TF::ConstOp tblgen_ConstOp_3;
    {
      tblgen_ConstOp_3 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_2
      );
    }
    ::mlir::TF::DivOp tblgen_DivOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*rhs.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstOp_3.getODSResults(0).begin());
      tblgen_DivOp_4 = rewriter.create<::mlir::TF::DivOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_5;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_MulOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_DivOp_4.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_MulOp_5 = rewriter.create<::mlir::TF::MulOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_MulOp_5.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:468
*/
struct LowerScatterNdOp : public ::mlir::RewritePattern {
  LowerScatterNdOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.ScatterNd", 1, context, {"tf.Const", "tf.Fill", "tf.TensorScatterAdd"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range indices(op0->getOperands());
    ::mlir::Operation::operand_range updates(op0->getOperands());
    ::mlir::Operation::operand_range shape(op0->getOperands());
    ::mlir::StringAttr bad_indices_policy;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::ScatterNdOp>(op0); (void)castedOp0;
    indices = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf4(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'tf.ScatterNd' failed to satisfy constraint: 'tensor of integer or floating-point or complex-type values'"))) {
      return ::mlir::failure();
    }
    updates = castedOp0.getODSOperands(1);
    shape = castedOp0.getODSOperands(2);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::StringAttr>("bad_indices_policy");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getStringAttr("");
      bad_indices_policy = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = GetScalarOfType(getElementTypeOrSelf((*updates.begin())),0); (void)nativeVar_0;
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::TF::FillOp tblgen_FillOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*shape.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstOp_1.getODSResults(0).begin());
      tblgen_FillOp_2 = rewriter.create<::mlir::TF::FillOp>(odsLoc,
        /*dims=*/tblgen_value_0,
        /*value=*/tblgen_value_1
      );
    }
    ::mlir::TF::TensorScatterAddOp tblgen_TensorScatterAddOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_FillOp_2.getODSResults(0).begin()));
      tblgen_values.push_back((*indices.begin()));
      tblgen_values.push_back((*updates.begin()));
      if (auto tmpAttr = bad_indices_policy) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("bad_indices_policy"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_TensorScatterAddOp_3 = rewriter.create<::mlir::TF::TensorScatterAddOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_TensorScatterAddOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:591
*/
struct LowerSeluGradOp : public ::mlir::RewritePattern {
  LowerSeluGradOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.SeluGrad", 1, context, {"tf.AddV2", "tf.Const", "tf.Greater", "tf.Mul", "tf.SelectV2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range features(op0->getOperands());
    ::mlir::Operation::operand_range gradients(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::SeluGradOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf7(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'tf.SeluGrad' failed to satisfy constraint: 'statically shaped tensor of any type values'"))) {
      return ::mlir::failure();
    }
    gradients = castedOp0.getODSOperands(0);
    features = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = GetScalarOfType(getElementTypeOrSelf((*features.begin())),0); (void)nativeVar_0;
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::TF::GreaterOp tblgen_GreaterOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*features.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstOp_1.getODSResults(0).begin());
      tblgen_GreaterOp_2 = rewriter.create<::mlir::TF::GreaterOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    auto nativeVar_3 = GetScalarOfType(getElementTypeOrSelf((*features.begin())), 1.0507009873554804934193349852946); (void)nativeVar_3;
    ::mlir::TF::ConstOp tblgen_ConstOp_4;
    {
      tblgen_ConstOp_4 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_3
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*gradients.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstOp_4.getODSResults(0).begin());
      tblgen_MulOp_5 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    auto nativeVar_6 = GetScalarOfType(getElementTypeOrSelf((*features.begin())), 1.7580993408473768599402175208123); (void)nativeVar_6;
    ::mlir::TF::ConstOp tblgen_ConstOp_7;
    {
      tblgen_ConstOp_7 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_6
      );
    }
    ::mlir::TF::AddV2Op tblgen_AddV2Op_8;
    {
      ::mlir::Value tblgen_value_0 = (*features.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstOp_7.getODSResults(0).begin());
      tblgen_AddV2Op_8 = rewriter.create<::mlir::TF::AddV2Op>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_9;
    {
      ::mlir::Value tblgen_value_0 = (*gradients.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_AddV2Op_8.getODSResults(0).begin());
      tblgen_MulOp_9 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::SelectV2Op tblgen_SelectV2Op_10;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_GreaterOp_2.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_MulOp_5.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_MulOp_9.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SelectV2Op_10 = rewriter.create<::mlir::TF::SelectV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SelectV2Op_10.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:574
*/
struct LowerSeluOp : public ::mlir::RewritePattern {
  LowerSeluOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Selu", 1, context, {"tf.Const", "tf.Expm1", "tf.Greater", "tf.Mul", "tf.SelectV2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range features(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::SeluOp>(op0); (void)castedOp0;
    features = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = GetScalarOfType(getElementTypeOrSelf((*features.begin())),0); (void)nativeVar_0;
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::TF::GreaterOp tblgen_GreaterOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*features.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstOp_1.getODSResults(0).begin());
      tblgen_GreaterOp_2 = rewriter.create<::mlir::TF::GreaterOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    auto nativeVar_3 = GetScalarOfType(getElementTypeOrSelf((*features.begin())), 1.0507009873554804934193349852946); (void)nativeVar_3;
    ::mlir::TF::ConstOp tblgen_ConstOp_4;
    {
      tblgen_ConstOp_4 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_3
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ConstOp_4.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*features.begin());
      tblgen_MulOp_5 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    auto nativeVar_6 = GetScalarOfType(getElementTypeOrSelf((*features.begin())), 1.7580993408473768599402175208123); (void)nativeVar_6;
    ::mlir::TF::ConstOp tblgen_ConstOp_7;
    {
      tblgen_ConstOp_7 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_6
      );
    }
    ::mlir::TF::Expm1Op tblgen_Expm1Op_8;
    {
      ::mlir::Value tblgen_value_0 = (*features.begin());
      tblgen_Expm1Op_8 = rewriter.create<::mlir::TF::Expm1Op>(odsLoc,
        /*x=*/tblgen_value_0
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_9;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ConstOp_7.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_Expm1Op_8.getODSResults(0).begin());
      tblgen_MulOp_9 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::SelectV2Op tblgen_SelectV2Op_10;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_GreaterOp_2.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_MulOp_5.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_MulOp_9.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SelectV2Op_10 = rewriter.create<::mlir::TF::SelectV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SelectV2Op_10.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:403
*/
struct LowerSizeOp : public ::mlir::RewritePattern {
  LowerSizeOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Size", 1, context, {"tf.Const", "tf.Prod"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::SizeOp res;
    ::mlir::Operation::operand_range arg(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::SizeOp>(op0); (void)castedOp0;
    res = castedOp0;
    arg = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = rewriter.getBoolAttr(getElementTypeOrSelf((*res.getODSResults(0).begin()).getType()).isInteger(32)); (void)nativeVar_0;
    auto nativeVar_1 = rewriter.create<TF::ShapeOp>((*res.getODSResults(0).begin()).getLoc(), (*arg.begin()), nativeVar_0); (void)nativeVar_1;
    auto nativeVar_2 = GetScalarOfType(getElementTypeOrSelf((*res.getODSResults(0).begin())),0); (void)nativeVar_2;
    ::mlir::TF::ConstOp tblgen_ConstOp_3;
    {
      tblgen_ConstOp_3 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_2
      );
    }
    ::mlir::TF::ProdOp tblgen_ProdOp_4;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back(nativeVar_1);
      tblgen_values.push_back((*tblgen_ConstOp_3.getODSResults(0).begin()));
      if (auto tmpAttr = rewriter.getBoolAttr(false)) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("keep_dims"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ProdOp_4 = rewriter.create<::mlir::TF::ProdOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ProdOp_4.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:106
*/
struct LowerSoftmaxCrossEntropyWithLogitsOp : public ::mlir::RewritePattern {
  LowerSoftmaxCrossEntropyWithLogitsOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.SoftmaxCrossEntropyWithLogits", 1, context, {"tf.Const", "tf.LogSoftmax", "tf.MulNoNan", "tf.Neg", "tf.Softmax", "tf.Sub", "tf.Sum"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range labels(op0->getOperands());
    ::mlir::Operation::operand_range features(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::SoftmaxCrossEntropyWithLogitsOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf2(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'tf.SoftmaxCrossEntropyWithLogits' failed to satisfy constraint: 'ranked tensor of any type values'"))) {
      return ::mlir::failure();
    }
    features = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf2(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'tf.SoftmaxCrossEntropyWithLogits' failed to satisfy constraint: 'ranked tensor of any type values'"))) {
      return ::mlir::failure();
    }
    labels = castedOp0.getODSOperands(1);
    if (!(!(((::llvm::cast<::mlir::ShapedType>(((*features.begin()).getType())).hasRank())) && ((::llvm::cast<::mlir::ShapedType>(((*features.begin()).getType())).getRank()
                             == 0))))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "value entity 'features' failed to satisfy constraint: 'Non scalar type'";
      });
    }
    if (!(!(((::llvm::cast<::mlir::ShapedType>(((*labels.begin()).getType())).hasRank())) && ((::llvm::cast<::mlir::ShapedType>(((*labels.begin()).getType())).getRank()
                             == 0))))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "value entity 'labels' failed to satisfy constraint: 'Non scalar type'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::LogSoftmaxOp tblgen_LogSoftmaxOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*features.begin());
      tblgen_LogSoftmaxOp_0 = rewriter.create<::mlir::TF::LogSoftmaxOp>(odsLoc,
        /*logits=*/tblgen_value_0
      );
    }
    ::mlir::TF::NegOp tblgen_NegOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*labels.begin());
      tblgen_NegOp_1 = rewriter.create<::mlir::TF::NegOp>(odsLoc,
        /*x=*/tblgen_value_0
      );
    }
    ::mlir::TF::MulNoNanOp sum_input;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_LogSoftmaxOp_0.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_NegOp_1.getODSResults(0).begin());
      sum_input = rewriter.create<::mlir::TF::MulNoNanOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    auto nativeVar_2 = GetI64ElementsAttr({-1}, &rewriter); (void)nativeVar_2;
    ::mlir::TF::ConstOp tblgen_ConstOp_3;
    {
      tblgen_ConstOp_3 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_2
      );
    }
    ::mlir::TF::SumOp tblgen_SumOp_4;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*sum_input.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_ConstOp_3.getODSResults(0).begin()));
      if (auto tmpAttr = rewriter.getBoolAttr(false)) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("keep_dims"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SumOp_4 = rewriter.create<::mlir::TF::SumOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SumOp_4.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    ::mlir::TF::SoftmaxOp tblgen_SoftmaxOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*features.begin());
      tblgen_SoftmaxOp_5 = rewriter.create<::mlir::TF::SoftmaxOp>(odsLoc,
        /*logits=*/tblgen_value_0
      );
    }
    ::mlir::TF::SubOp tblgen_SubOp_6;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_SoftmaxOp_5.getODSResults(0).begin()));
      tblgen_values.push_back((*labels.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(1)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SubOp_6 = rewriter.create<::mlir::TF::SubOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SubOp_6.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:137
*/
struct LowerSparseSoftmaxCrossEntropyWithLogitsOp : public ::mlir::RewritePattern {
  LowerSparseSoftmaxCrossEntropyWithLogitsOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.SparseSoftmaxCrossEntropyWithLogits", 1, context, {"tf.AddV2", "tf.Const", "tf.ExpandDims", "tf.Less", "tf.LessEqual", "tf.LogicalAnd", "tf.OneHot", "tf.SelectV2", "tf.SoftmaxCrossEntropyWithLogits"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range sparse_labels(op0->getOperands());
    ::mlir::TF::SparseSoftmaxCrossEntropyWithLogitsOp src_op;
    ::mlir::Operation::operand_range features(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::SparseSoftmaxCrossEntropyWithLogitsOp>(op0); (void)castedOp0;
    src_op = castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf7(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'tf.SparseSoftmaxCrossEntropyWithLogits' failed to satisfy constraint: 'statically shaped tensor of any type values'"))) {
      return ::mlir::failure();
    }
    features = castedOp0.getODSOperands(0);
    sparse_labels = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    auto nativeVar_0 = GetScalarOfType(rewriter.getIntegerType(32), (*features.begin()).getType().cast<RankedTensorType>().getDimSize(1)); (void)nativeVar_0;
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    auto nativeVar_2 = GetScalarOfType(getElementTypeOrSelf((*features.begin())),1); (void)nativeVar_2;
    ::mlir::TF::ConstOp tblgen_ConstOp_3;
    {
      tblgen_ConstOp_3 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_2
      );
    }
    auto nativeVar_4 = GetScalarOfType(getElementTypeOrSelf((*features.begin())),0); (void)nativeVar_4;
    ::mlir::TF::ConstOp tblgen_ConstOp_5;
    {
      tblgen_ConstOp_5 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_4
      );
    }
    ::mlir::TF::OneHotOp labels;
    {
      ::mlir::Value tblgen_value_0 = (*sparse_labels.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstOp_1.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*tblgen_ConstOp_3.getODSResults(0).begin());
      ::mlir::Value tblgen_value_3 = (*tblgen_ConstOp_5.getODSResults(0).begin());
      labels = rewriter.create<::mlir::TF::OneHotOp>(odsLoc,
        /*indices=*/tblgen_value_0,
        /*depth=*/tblgen_value_1,
        /*on_value=*/tblgen_value_2,
        /*off_value=*/tblgen_value_3,
        rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1)
      );
    }
    auto nativeVar_6 = GetScalarOfType(getElementTypeOrSelf((*sparse_labels.begin())),0); (void)nativeVar_6;
    ::mlir::TF::ConstOp tblgen_ConstOp_7;
    {
      tblgen_ConstOp_7 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_6
      );
    }
    ::mlir::TF::LessEqualOp tblgen_LessEqualOp_8;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ConstOp_7.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*sparse_labels.begin());
      tblgen_LessEqualOp_8 = rewriter.create<::mlir::TF::LessEqualOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    auto nativeVar_9 = GetScalarOfType(getElementTypeOrSelf((*sparse_labels.begin())), (*features.begin()).getType().cast<RankedTensorType>().getDimSize(1)); (void)nativeVar_9;
    ::mlir::TF::ConstOp tblgen_ConstOp_10;
    {
      tblgen_ConstOp_10 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_9
      );
    }
    ::mlir::TF::LessOp tblgen_LessOp_11;
    {
      ::mlir::Value tblgen_value_0 = (*sparse_labels.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstOp_10.getODSResults(0).begin());
      tblgen_LessOp_11 = rewriter.create<::mlir::TF::LessOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::LogicalAndOp tblgen_LogicalAndOp_12;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_LessEqualOp_8.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_LessOp_11.getODSResults(0).begin());
      tblgen_LogicalAndOp_12 = rewriter.create<::mlir::TF::LogicalAndOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    auto nativeVar_13 = GetScalarOfType(getElementTypeOrSelf((*features.begin())),0); (void)nativeVar_13;
    ::mlir::TF::ConstOp tblgen_ConstOp_14;
    {
      tblgen_ConstOp_14 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_13
      );
    }
    auto nativeVar_15 = GetScalarOfType(getElementTypeOrSelf((*labels.getODSResults(0).begin())), std::numeric_limits<double>::quiet_NaN()); (void)nativeVar_15;
    ::mlir::TF::ConstOp tblgen_ConstOp_16;
    {
      tblgen_ConstOp_16 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_15
      );
    }
    ::mlir::TF::SelectV2Op zero_or_nan;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_LogicalAndOp_12.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstOp_14.getODSResults(0).begin());
      ::mlir::Value tblgen_value_2 = (*tblgen_ConstOp_16.getODSResults(0).begin());
      zero_or_nan = rewriter.create<::mlir::TF::SelectV2Op>(odsLoc,
        /*condition=*/tblgen_value_0,
        /*then_value=*/tblgen_value_1,
        /*else_value=*/tblgen_value_2
      );
    }
    auto nativeVar_17 = GetI64ElementsAttr({-1}, &rewriter); (void)nativeVar_17;
    ::mlir::TF::ConstOp tblgen_ConstOp_18;
    {
      tblgen_ConstOp_18 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_17
      );
    }
    ::mlir::TF::ExpandDimsOp tblgen_ExpandDimsOp_19;
    {
      ::mlir::Value tblgen_value_0 = (*zero_or_nan.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstOp_18.getODSResults(0).begin());
      tblgen_ExpandDimsOp_19 = rewriter.create<::mlir::TF::ExpandDimsOp>(odsLoc,
        /*input=*/tblgen_value_0,
        /*dim=*/tblgen_value_1
      );
    }
    ::mlir::TF::AddV2Op adjusted_labels;
    {
      ::mlir::Value tblgen_value_0 = (*labels.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ExpandDimsOp_19.getODSResults(0).begin());
      adjusted_labels = rewriter.create<::mlir::TF::AddV2Op>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::SoftmaxCrossEntropyWithLogitsOp tblgen_SoftmaxCrossEntropyWithLogitsOp_20;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*features.begin()));
      tblgen_values.push_back((*adjusted_labels.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      for (auto v: castedOp0.getODSResults(1)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SoftmaxCrossEntropyWithLogitsOp_20 = rewriter.create<::mlir::TF::SoftmaxCrossEntropyWithLogitsOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SoftmaxCrossEntropyWithLogitsOp_20.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SoftmaxCrossEntropyWithLogitsOp_20.getODSResults(1) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:424
*/
struct LowerSqrtGradOp : public ::mlir::RewritePattern {
  LowerSqrtGradOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.SqrtGrad", 1, context, {"tf.Const", "tf.Div", "tf.Mul"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::mlir::Operation::operand_range dy(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::SqrtGradOp>(op0); (void)castedOp0;
    y = castedOp0.getODSOperands(0);
    dy = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = GetScalarOfType(getElementTypeOrSelf((*dy.begin())),0.5); (void)nativeVar_0;
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*dy.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstOp_1.getODSResults(0).begin());
      tblgen_MulOp_2 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::DivOp tblgen_DivOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_MulOp_2.getODSResults(0).begin()));
      tblgen_values.push_back((*y.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_DivOp_3 = rewriter.create<::mlir::TF::DivOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_DivOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:197
*/
struct LowerSquareOp : public ::mlir::RewritePattern {
  LowerSquareOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Square", 1, context, {"tf.Mul"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range val(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::SquareOp>(op0); (void)castedOp0;
    val = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::MulOp tblgen_MulOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*val.begin()));
      tblgen_values.push_back((*val.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_MulOp_0 = rewriter.create<::mlir::TF::MulOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_MulOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:199
*/
struct LowerSquaredDifferenceOpOnRealTensors : public ::mlir::RewritePattern {
  LowerSquaredDifferenceOpOnRealTensors(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.SquaredDifference", 1, context, {"tf.Square", "tf.Sub"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::SquaredDifferenceOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf5(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'tf.SquaredDifference' failed to satisfy constraint: 'tensor of signless integer or floating-point values'"))) {
      return ::mlir::failure();
    }
    lhs = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf5(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'tf.SquaredDifference' failed to satisfy constraint: 'tensor of signless integer or floating-point values'"))) {
      return ::mlir::failure();
    }
    rhs = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::SubOp tblgen_SubOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*lhs.begin());
      ::mlir::Value tblgen_value_1 = (*rhs.begin());
      tblgen_SubOp_0 = rewriter.create<::mlir::TF::SubOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::SquareOp tblgen_SquareOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_SubOp_0.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SquareOp_1 = rewriter.create<::mlir::TF::SquareOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SquareOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:203
*/
struct LowerSquaredDifferenceOpOneComplexTensors : public ::mlir::RewritePattern {
  LowerSquaredDifferenceOpOneComplexTensors(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.SquaredDifference", 1, context, {"tf.Conj", "tf.Mul", "tf.Sub"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::SquaredDifferenceOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf8(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'tf.SquaredDifference' failed to satisfy constraint: 'tensor of complex-type values'"))) {
      return ::mlir::failure();
    }
    lhs = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf8(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'tf.SquaredDifference' failed to satisfy constraint: 'tensor of complex-type values'"))) {
      return ::mlir::failure();
    }
    rhs = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::SubOp diff;
    {
      ::mlir::Value tblgen_value_0 = (*lhs.begin());
      ::mlir::Value tblgen_value_1 = (*rhs.begin());
      diff = rewriter.create<::mlir::TF::SubOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::ConjOp tblgen_ConjOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*diff.getODSResults(0).begin());
      tblgen_ConjOp_0 = rewriter.create<::mlir::TF::ConjOp>(odsLoc,
        /*input=*/tblgen_value_0
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*diff.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_ConjOp_0.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_MulOp_1 = rewriter.create<::mlir::TF::MulOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_MulOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:437
*/
struct LowerTanhGradOp : public ::mlir::RewritePattern {
  LowerTanhGradOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.TanhGrad", 1, context, {"tf.Const", "tf.Mul", "tf.Square", "tf.Sub"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::mlir::Operation::operand_range dy(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::TanhGradOp>(op0); (void)castedOp0;
    y = castedOp0.getODSOperands(0);
    dy = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = GetScalarOfType(getElementTypeOrSelf((*y.begin())),1); (void)nativeVar_0;
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::TF::SquareOp tblgen_SquareOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*y.begin());
      tblgen_SquareOp_2 = rewriter.create<::mlir::TF::SquareOp>(odsLoc,
        /*x=*/tblgen_value_0
      );
    }
    ::mlir::TF::SubOp tblgen_SubOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ConstOp_1.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_SquareOp_2.getODSResults(0).begin());
      tblgen_SubOp_3 = rewriter.create<::mlir::TF::SubOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_4;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*dy.begin()));
      tblgen_values.push_back((*tblgen_SubOp_3.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_MulOp_4 = rewriter.create<::mlir::TF::MulOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_MulOp_4.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:522
*/
struct LowerTensorScatterUpdate_1 : public ::mlir::RewritePattern {
  LowerTensorScatterUpdate_1(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.TensorScatterUpdate", 1, context, {"tf.Cast"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range indices(op0->getOperands());
    ::mlir::Operation::operand_range updates(op0->getOperands());
    ::mlir::StringAttr bad_indices_policy;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    if(::mlir::failed(static_dag_matcher_0(rewriter, op0, tblgen_ops, input, indices, updates, bad_indices_policy))) {
      return ::mlir::failure();
    }
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::TF::TensorScatterUpdateOp>(op0); (void)castedOp0;
    if (!(((::llvm::isa<::mlir::TensorType>(((*input.begin()).getType())))) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || ((elementType.isa<mlir::TF::BoolRefType>())); }(::llvm::cast<::mlir::ShapedType>(((*input.begin()).getType())).getElementType())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "value entity 'input' failed to satisfy constraint: 'tensor of bool values'";
      });
    }
    if (!(((::llvm::isa<::mlir::TensorType>(((*updates.begin()).getType())))) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger(1))) || ((elementType.isa<mlir::TF::BoolRefType>())); }(::llvm::cast<::mlir::ShapedType>(((*updates.begin()).getType())).getElementType())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "value entity 'updates' failed to satisfy constraint: 'tensor of bool values'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = CreateTFCastOpI32(&rewriter, (*input.begin()).getLoc(), (*input.begin()), rewriter.getBoolAttr(false)); (void)nativeVar_0;
    auto nativeVar_1 = CreateTFCastOpI32(&rewriter, (*updates.begin()).getLoc(), (*updates.begin()), rewriter.getBoolAttr(false)); (void)nativeVar_1;
    auto nativeVar_2 = rewriter.create<TF::TensorScatterUpdateOp>(nativeVar_0.getLoc(), nativeVar_0.getType(), nativeVar_0, (*indices.begin()), nativeVar_1, bad_indices_policy); (void)nativeVar_2;
    ::mlir::TF::CastOp tblgen_CastOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back(nativeVar_2);
      if (auto tmpAttr = rewriter.getBoolAttr(false)) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("Truncate"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_CastOp_3 = rewriter.create<::mlir::TF::CastOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_CastOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:538
*/
struct LowerTensorScatterUpdate_2 : public ::mlir::RewritePattern {
  LowerTensorScatterUpdate_2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.TensorScatterUpdate", 1, context, {"tf.AddV2", "tf.Const", "tf.Mul", "tf.OnesLike", "tf.Sub"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range indices(op0->getOperands());
    ::mlir::Operation::operand_range updates(op0->getOperands());
    ::mlir::StringAttr bad_indices_policy;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    if(::mlir::failed(static_dag_matcher_0(rewriter, op0, tblgen_ops, input, indices, updates, bad_indices_policy))) {
      return ::mlir::failure();
    }
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::TF::TensorScatterUpdateOp>(op0); (void)castedOp0;
    if (!(((::llvm::isa<::mlir::TensorType>(((*updates.begin()).getType())))) && ([](::mlir::Type elementType) { return (((((elementType.isSignedInteger(4))) || ((elementType.isa<mlir::TF::Int4RefType>()))) || (((elementType.isSignlessInteger(8))) || ((elementType.isa<mlir::TF::Int8RefType>()))) || (((elementType.isSignlessInteger(16))) || ((elementType.isa<mlir::TF::Int16RefType>()))) || (((elementType.isSignlessInteger(32))) || ((elementType.isa<mlir::TF::Int32RefType>()))) || (((elementType.isSignlessInteger(64))) || ((elementType.isa<mlir::TF::Int64RefType>())))) || ((((elementType.isUnsignedInteger(4))) || ((elementType.isa<mlir::TF::Uint4RefType>()))) || (((elementType.isUnsignedInteger(8))) || ((elementType.isa<mlir::TF::Uint8RefType>()))) || (((elementType.isUnsignedInteger(16))) || ((elementType.isa<mlir::TF::Uint16RefType>()))) || (((elementType.isUnsignedInteger(32))) || ((elementType.isa<mlir::TF::Uint32RefType>()))) || (((elementType.isUnsignedInteger(64))) || ((elementType.isa<mlir::TF::Uint64RefType>()))))) || ((((elementType.isF16())) || ((elementType.isa<mlir::TF::HalfRefType>()))) || (((elementType.isF32())) || ((elementType.isa<mlir::TF::FloatRefType>()))) || (((elementType.isF64())) || ((elementType.isa<mlir::TF::DoubleRefType>()))) || (((elementType.isBF16())) || ((elementType.isa<mlir::TF::Bfloat16RefType>()))) || (((elementType.isFloat8E4M3FN())) || ((elementType.isa<mlir::TF::Float8E4M3FNRefType>()))) || (((elementType.isFloat8E5M2())) || ((elementType.isa<mlir::TF::Float8E5M2RefType>())))) || (((((::llvm::isa<::mlir::ComplexType>(elementType))) && ((::llvm::cast<::mlir::ComplexType>(elementType).getElementType().isF32()))) || ((elementType.isa<mlir::TF::Complex64RefType>()))) || ((((::llvm::isa<::mlir::ComplexType>(elementType))) && ((::llvm::cast<::mlir::ComplexType>(elementType).getElementType().isF64()))) || ((elementType.isa<mlir::TF::Complex128RefType>())))); }(::llvm::cast<::mlir::ShapedType>(((*updates.begin()).getType())).getElementType())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "value entity 'updates' failed to satisfy constraint: 'tensor of integer or floating-point or complex values'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = GetScalarOfType(getElementTypeOrSelf((*updates.begin())),1); (void)nativeVar_0;
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::TF::OnesLikeOp tblgen_OnesLikeOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*updates.begin());
      tblgen_OnesLikeOp_2 = rewriter.create<::mlir::TF::OnesLikeOp>(odsLoc,
        /*x=*/tblgen_value_0
      );
    }
    auto nativeVar_3 = rewriter.create<TF::ShapeOp>((*input.begin()).getLoc(), (*input.begin()), rewriter.getBoolAttr(true)); (void)nativeVar_3;
    auto nativeVar_4 = rewriter.create<TF::ScatterNdOp>((*input.begin()).getLoc(), (*input.begin()).getType(), (*indices.begin()), tblgen_OnesLikeOp_2, nativeVar_3, bad_indices_policy); (void)nativeVar_4;
    ::mlir::TF::SubOp tblgen_SubOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ConstOp_1.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = nativeVar_4;
      tblgen_SubOp_5 = rewriter.create<::mlir::TF::SubOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_6;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_SubOp_5.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*input.begin());
      tblgen_MulOp_6 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    auto nativeVar_7 = rewriter.create<TF::ShapeOp>((*input.begin()).getLoc(), (*input.begin()), rewriter.getBoolAttr(true)); (void)nativeVar_7;
    auto nativeVar_8 = rewriter.create<TF::ScatterNdOp>((*input.begin()).getLoc(), (*input.begin()).getType(), (*indices.begin()), (*updates.begin()), nativeVar_7, bad_indices_policy); (void)nativeVar_8;
    ::mlir::TF::AddV2Op tblgen_AddV2Op_9;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_MulOp_6.getODSResults(0).begin()));
      tblgen_values.push_back(nativeVar_8);
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_AddV2Op_9 = rewriter.create<::mlir::TF::AddV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AddV2Op_9.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:232
*/
struct LowerTruncateDivOpOnFloatTensors : public ::mlir::RewritePattern {
  LowerTruncateDivOpOnFloatTensors(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.TruncateDiv", 1, context, {"tf.Ceil", "tf.Const", "tf.Div", "tf.Floor", "tf.Less", "tf.SelectV2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::TruncateDivOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf3(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'tf.TruncateDiv' failed to satisfy constraint: 'tensor of floating-point values'"))) {
      return ::mlir::failure();
    }
    lhs = castedOp0.getODSOperands(0);
    rhs = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::DivOp div;
    {
      ::mlir::Value tblgen_value_0 = (*lhs.begin());
      ::mlir::Value tblgen_value_1 = (*rhs.begin());
      div = rewriter.create<::mlir::TF::DivOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    auto nativeVar_0 = GetScalarOfType(getElementTypeOrSelf((*lhs.begin())),0.0); (void)nativeVar_0;
    ::mlir::TF::ConstOp zero;
    {
      zero = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::TF::LessOp tblgen_LessOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*div.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*zero.getODSResults(0).begin());
      tblgen_LessOp_1 = rewriter.create<::mlir::TF::LessOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::CeilOp tblgen_CeilOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*div.getODSResults(0).begin());
      tblgen_CeilOp_2 = rewriter.create<::mlir::TF::CeilOp>(odsLoc,
        /*x=*/tblgen_value_0
      );
    }
    ::mlir::TF::FloorOp tblgen_FloorOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*div.getODSResults(0).begin());
      tblgen_FloorOp_3 = rewriter.create<::mlir::TF::FloorOp>(odsLoc,
        /*x=*/tblgen_value_0
      );
    }
    ::mlir::TF::SelectV2Op tblgen_SelectV2Op_4;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_LessOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_CeilOp_2.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_FloorOp_3.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SelectV2Op_4 = rewriter.create<::mlir::TF::SelectV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SelectV2Op_4.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:226
*/
struct LowerTruncateDivOpOnIntTensors : public ::mlir::RewritePattern {
  LowerTruncateDivOpOnIntTensors(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.TruncateDiv", 1, context, {"tf.Div"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::TruncateDivOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf6(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'tf.TruncateDiv' failed to satisfy constraint: 'tensor of integer values'"))) {
      return ::mlir::failure();
    }
    lhs = castedOp0.getODSOperands(0);
    rhs = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::DivOp tblgen_DivOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*lhs.begin()));
      tblgen_values.push_back((*rhs.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_DivOp_0 = rewriter.create<::mlir::TF::DivOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_DivOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:496
*/
struct LowerXdivyOp : public ::mlir::RewritePattern {
  LowerXdivyOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Xdivy", 1, context, {"tf.Const", "tf.Div", "tf.Equal", "tf.SelectV2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::XdivyOp>(op0); (void)castedOp0;
    x = castedOp0.getODSOperands(0);
    y = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = GetScalarOfType(getElementTypeOrSelf((*x.begin())),0); (void)nativeVar_0;
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::TF::EqualOp tblgen_EqualOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*x.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstOp_1.getODSResults(0).begin());
      tblgen_EqualOp_2 = rewriter.create<::mlir::TF::EqualOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1,
        rewriter.getBoolAttr(true)
      );
    }
    ::mlir::TF::DivOp tblgen_DivOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*x.begin());
      ::mlir::Value tblgen_value_1 = (*y.begin());
      tblgen_DivOp_3 = rewriter.create<::mlir::TF::DivOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::SelectV2Op tblgen_SelectV2Op_4;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_EqualOp_2.getODSResults(0).begin()));
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*tblgen_DivOp_3.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SelectV2Op_4 = rewriter.create<::mlir::TF::SelectV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SelectV2Op_4.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:500
*/
struct LowerXlog1pyOp : public ::mlir::RewritePattern {
  LowerXlog1pyOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Xlog1py", 1, context, {"tf.Const", "tf.Equal", "tf.Log1p", "tf.Mul", "tf.SelectV2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::Xlog1pyOp>(op0); (void)castedOp0;
    x = castedOp0.getODSOperands(0);
    y = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = GetScalarOfType(getElementTypeOrSelf((*x.begin())),0); (void)nativeVar_0;
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::TF::EqualOp tblgen_EqualOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*x.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstOp_1.getODSResults(0).begin());
      tblgen_EqualOp_2 = rewriter.create<::mlir::TF::EqualOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1,
        rewriter.getBoolAttr(true)
      );
    }
    ::mlir::TF::Log1pOp tblgen_Log1pOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*y.begin());
      tblgen_Log1pOp_3 = rewriter.create<::mlir::TF::Log1pOp>(odsLoc,
        /*x=*/tblgen_value_0
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*x.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_Log1pOp_3.getODSResults(0).begin());
      tblgen_MulOp_4 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::SelectV2Op tblgen_SelectV2Op_5;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_EqualOp_2.getODSResults(0).begin()));
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*tblgen_MulOp_4.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SelectV2Op_5 = rewriter.create<::mlir::TF::SelectV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SelectV2Op_5.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:504
*/
struct LowerXlogyOp : public ::mlir::RewritePattern {
  LowerXlogyOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Xlogy", 1, context, {"tf.Const", "tf.Equal", "tf.Log", "tf.Mul", "tf.SelectV2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::XlogyOp>(op0); (void)castedOp0;
    x = castedOp0.getODSOperands(0);
    y = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = GetScalarOfType(getElementTypeOrSelf((*x.begin())),0); (void)nativeVar_0;
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::TF::EqualOp tblgen_EqualOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*x.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstOp_1.getODSResults(0).begin());
      tblgen_EqualOp_2 = rewriter.create<::mlir::TF::EqualOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1,
        rewriter.getBoolAttr(true)
      );
    }
    ::mlir::TF::LogOp tblgen_LogOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*y.begin());
      tblgen_LogOp_3 = rewriter.create<::mlir::TF::LogOp>(odsLoc,
        /*x=*/tblgen_value_0
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*x.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_LogOp_3.getODSResults(0).begin());
      tblgen_MulOp_4 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::SelectV2Op tblgen_SelectV2Op_5;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_EqualOp_2.getODSResults(0).begin()));
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*tblgen_MulOp_4.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SelectV2Op_5 = rewriter.create<::mlir::TF::SelectV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SelectV2Op_5.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.td:465
*/
struct LowerZerosLikeOp : public ::mlir::RewritePattern {
  LowerZerosLikeOp(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.ZerosLike", 1, context, {"tf.BroadcastTo", "tf.Const"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::ZerosLikeOp src_op;
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::ZerosLikeOp>(op0); (void)castedOp0;
    src_op = castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_tf4(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'tf.ZerosLike' failed to satisfy constraint: 'tensor of integer or floating-point or complex-type values'"))) {
      return ::mlir::failure();
    }
    input = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = GetScalarOfType(getElementTypeOrSelf((*input.begin())),0); (void)nativeVar_0;
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    auto nativeVar_2 = rewriter.create<TF::ShapeOp>((*src_op.getODSResults(0).begin()).getLoc(), (*input.begin()), rewriter.getBoolAttr(false)); (void)nativeVar_2;
    ::mlir::TF::BroadcastToOp tblgen_BroadcastToOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_ConstOp_1.getODSResults(0).begin()));
      tblgen_values.push_back(nativeVar_2);
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_BroadcastToOp_3 = rewriter.create<::mlir::TF::BroadcastToOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_BroadcastToOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<LowerAddOp>(patterns.getContext());
  patterns.add<LowerBiasAddGradOp>(patterns.getContext());
  patterns.add<LowerDequantizeOp>(patterns.getContext());
  patterns.add<LowerDivNoNanOp>(patterns.getContext());
  patterns.add<LowerEmptyOp>(patterns.getContext());
  patterns.add<LowerExp1mOp>(patterns.getContext());
  patterns.add<LowerFakeQuantWithMinMaxArgs>(patterns.getContext());
  patterns.add<LowerFillOp>(patterns.getContext());
  patterns.add<LowerInv>(patterns.getContext());
  patterns.add<LowerIsFiniteOp>(patterns.getContext());
  patterns.add<LowerIsInfOp>(patterns.getContext());
  patterns.add<LowerIsNanOp>(patterns.getContext());
  patterns.add<LowerL2LossOp>(patterns.getContext());
  patterns.add<LowerMatrixBandPartOp>(patterns.getContext());
  patterns.add<LowerMulNoNanOp>(patterns.getContext());
  patterns.add<LowerOnesLikeOp>(patterns.getContext());
  patterns.add<LowerPadOp>(patterns.getContext());
  patterns.add<LowerReciprocal>(patterns.getContext());
  patterns.add<LowerRintOp>(patterns.getContext());
  patterns.add<LowerRoundOpOnFloatTensor>(patterns.getContext());
  patterns.add<LowerRoundOpOnIntTensor>(patterns.getContext());
  patterns.add<LowerRsqrtGradOp>(patterns.getContext());
  patterns.add<LowerScatterNdOp>(patterns.getContext());
  patterns.add<LowerSeluGradOp>(patterns.getContext());
  patterns.add<LowerSeluOp>(patterns.getContext());
  patterns.add<LowerSizeOp>(patterns.getContext());
  patterns.add<LowerSoftmaxCrossEntropyWithLogitsOp>(patterns.getContext());
  patterns.add<LowerSparseSoftmaxCrossEntropyWithLogitsOp>(patterns.getContext());
  patterns.add<LowerSqrtGradOp>(patterns.getContext());
  patterns.add<LowerSquareOp>(patterns.getContext());
  patterns.add<LowerSquaredDifferenceOpOnRealTensors>(patterns.getContext());
  patterns.add<LowerSquaredDifferenceOpOneComplexTensors>(patterns.getContext());
  patterns.add<LowerTanhGradOp>(patterns.getContext());
  patterns.add<LowerTensorScatterUpdate_1>(patterns.getContext());
  patterns.add<LowerTensorScatterUpdate_2>(patterns.getContext());
  patterns.add<LowerTruncateDivOpOnFloatTensors>(patterns.getContext());
  patterns.add<LowerTruncateDivOpOnIntTensors>(patterns.getContext());
  patterns.add<LowerXdivyOp>(patterns.getContext());
  patterns.add<LowerXlog1pyOp>(patterns.getContext());
  patterns.add<LowerXlogyOp>(patterns.getContext());
  patterns.add<LowerZerosLikeOp>(patterns.getContext());
}
