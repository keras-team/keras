/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: canonicalize.td                                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


static ::llvm::LogicalResult __mlir_ods_local_type_constraint_canonicalize1(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Type type,
    ::llvm::StringRef failureStr) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (((((elementType.isSignedInteger(4))) || ((elementType.isa<mlir::TF::Int4RefType>()))) || (((elementType.isSignlessInteger(8))) || ((elementType.isa<mlir::TF::Int8RefType>()))) || (((elementType.isSignlessInteger(16))) || ((elementType.isa<mlir::TF::Int16RefType>()))) || (((elementType.isSignlessInteger(32))) || ((elementType.isa<mlir::TF::Int32RefType>()))) || (((elementType.isSignlessInteger(64))) || ((elementType.isa<mlir::TF::Int64RefType>())))) || ((((elementType.isUnsignedInteger(4))) || ((elementType.isa<mlir::TF::Uint4RefType>()))) || (((elementType.isUnsignedInteger(8))) || ((elementType.isa<mlir::TF::Uint8RefType>()))) || (((elementType.isUnsignedInteger(16))) || ((elementType.isa<mlir::TF::Uint16RefType>()))) || (((elementType.isUnsignedInteger(32))) || ((elementType.isa<mlir::TF::Uint32RefType>()))) || (((elementType.isUnsignedInteger(64))) || ((elementType.isa<mlir::TF::Uint64RefType>()))))) || ((((elementType.isF16())) || ((elementType.isa<mlir::TF::HalfRefType>()))) || (((elementType.isF32())) || ((elementType.isa<mlir::TF::FloatRefType>()))) || (((elementType.isF64())) || ((elementType.isa<mlir::TF::DoubleRefType>()))) || (((elementType.isBF16())) || ((elementType.isa<mlir::TF::Bfloat16RefType>()))) || (((elementType.isFloat8E4M3FN())) || ((elementType.isa<mlir::TF::Float8E4M3FNRefType>()))) || (((elementType.isFloat8E5M2())) || ((elementType.isa<mlir::TF::Float8E5M2RefType>())))) || ((((elementType.isa<mlir::TF::Qint8Type>())) || ((elementType.isa<mlir::TF::Qint8RefType>()))) || (((elementType.isa<mlir::TF::Qint16Type>())) || ((elementType.isa<mlir::TF::Qint16RefType>()))) || (((elementType.isa<mlir::TF::Qint32Type>())) || ((elementType.isa<mlir::TF::Qint32RefType>()))) || (((elementType.isa<mlir::TF::Quint8Type>())) || ((elementType.isa<mlir::TF::Quint8RefType>()))) || (((elementType.isa<mlir::TF::Quint16Type>())) || ((elementType.isa<mlir::TF::Quint16RefType>())))) || (((((::llvm::isa<::mlir::ComplexType>(elementType))) && ((::llvm::cast<::mlir::ComplexType>(elementType).getElementType().isF32()))) || ((elementType.isa<mlir::TF::Complex64RefType>()))) || ((((::llvm::isa<::mlir::ComplexType>(elementType))) && ((::llvm::cast<::mlir::ComplexType>(elementType).getElementType().isF64()))) || ((elementType.isa<mlir::TF::Complex128RefType>())))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": tensor of number values";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_canonicalize2(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Type type,
    ::llvm::StringRef failureStr) {
  if (!((((::llvm::isa<::mlir::RankedTensorType>(type))) && ((::llvm::cast<::mlir::ShapedType>(type).hasStaticShape()))) && ([](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": statically shaped tensor of any type values";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_canonicalize3(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Type type,
    ::llvm::StringRef failureStr) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (((elementType.isBF16())) || ((elementType.isa<mlir::TF::Bfloat16RefType>()))) || ((((::llvm::isa<::mlir::ComplexType>(elementType))) && ((::llvm::cast<::mlir::ComplexType>(elementType).getElementType().isF64()))) || ((elementType.isa<mlir::TF::Complex128RefType>()))) || ((((::llvm::isa<::mlir::ComplexType>(elementType))) && ((::llvm::cast<::mlir::ComplexType>(elementType).getElementType().isF32()))) || ((elementType.isa<mlir::TF::Complex64RefType>()))) || (((elementType.isF16())) || ((elementType.isa<mlir::TF::HalfRefType>()))) || (((elementType.isF32())) || ((elementType.isa<mlir::TF::FloatRefType>()))) || (((elementType.isSignlessInteger(32))) || ((elementType.isa<mlir::TF::Int32RefType>()))) || (((elementType.isSignlessInteger(64))) || ((elementType.isa<mlir::TF::Int64RefType>()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": tensor of bfloat16 or 128-bit complex or 64-bit complex or 16-bit float or 32-bit float or 32-bit integer or 64-bit integer values";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_canonicalize1(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((attr == ::llvm::cast<::mlir::DenseFPElementsAttr>(::mlir::DenseElementsAttr::get(::mlir::RankedTensorType::get({}, rewriter.getF32Type()), ::llvm::ArrayRef(1.0f)))))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": constant attribute 1.0f";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_canonicalize2(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((::llvm::isa<::mlir::BoolAttr>(attr)))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": bool attribute";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_canonicalize3(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((::llvm::isa<::mlir::DenseFPElementsAttr>(attr) &&::llvm::cast<::mlir::DenseElementsAttr>(attr).getType().getElementType().isF32()))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": 32-bit float elements attribute";
    });
  }
  return ::mlir::success();
}
static ::llvm::LogicalResult static_dag_matcher_0(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &arg1) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::NegOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::TF::NegOp type";
    });
  }
  arg1 = castedOp1.getODSOperands(0);
  return ::mlir::success();
}

static ::llvm::LogicalResult static_dag_matcher_1(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::Operation::operand_range &arg1) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::SqrtOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::TF::SqrtOp type";
    });
  }
  arg1 = castedOp1.getODSOperands(0);
  return ::mlir::success();
}

static ::llvm::LogicalResult static_dag_matcher_2(::mlir::PatternRewriter &rewriter, ::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation *, 4> &tblgen_ops, ::mlir::StringAttr &container, ::mlir::BoolAttr &use_node_name_sharing, ::mlir::StringAttr &shared_name, ::mlir::TypeAttr &key_dtype, ::mlir::TypeAttr &value_dtype) {
  (void)tblgen_ops;
  auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::HashTableOp>(op0); (void)castedOp1;
  if (!(castedOp1)){
    return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
      diag << "castedOp1 is not ::mlir::TF::HashTableOp type";
    });
  }
  {
    auto tblgen_attr = op0->getAttrOfType<::mlir::StringAttr>("container");(void)tblgen_attr;
    if (!tblgen_attr) tblgen_attr = rewriter.getStringAttr("");
    container = tblgen_attr;
  }
  {
    auto tblgen_attr = op0->getAttrOfType<::mlir::StringAttr>("shared_name");(void)tblgen_attr;
    if (!tblgen_attr) tblgen_attr = rewriter.getStringAttr("");
    shared_name = tblgen_attr;
  }
  {
    auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("use_node_name_sharing");(void)tblgen_attr;
    if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
    use_node_name_sharing = tblgen_attr;
  }
  {
    auto tblgen_attr = op0->getAttrOfType<::mlir::TypeAttr>("key_dtype");(void)tblgen_attr;
    if (!(tblgen_attr)){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "expected op 'tf.HashTable' to have attribute 'key_dtype' of type '::mlir::TypeAttr'";
      });
    }
    key_dtype = tblgen_attr;
  }
  {
    auto tblgen_attr = op0->getAttrOfType<::mlir::TypeAttr>("value_dtype");(void)tblgen_attr;
    if (!(tblgen_attr)){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "expected op 'tf.HashTable' to have attribute 'value_dtype' of type '::mlir::TypeAttr'";
      });
    }
    value_dtype = tblgen_attr;
  }
  return ::mlir::success();
}

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:44
*/
struct AddToAddV2 : public ::mlir::RewritePattern {
  AddToAddV2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Add", 1, context, {"tf.AddV2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::AddOp src;
    ::mlir::Operation::operand_range arg0(op0->getOperands());
    ::mlir::Operation::operand_range arg1(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::AddOp>(op0); (void)castedOp0;
    src = castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_canonicalize1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'tf.Add' failed to satisfy constraint: 'tensor of number values'"))) {
      return ::mlir::failure();
    }
    arg0 = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_canonicalize1(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'tf.Add' failed to satisfy constraint: 'tensor of number values'"))) {
      return ::mlir::failure();
    }
    arg1 = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::AddV2Op dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg0.begin()));
      tblgen_values.push_back((*arg1.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::AddV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:52
*/
struct AddV2OfNegLeft : public ::mlir::RewritePattern {
  AddV2OfNegLeft(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.AddV2", 2, context, {"tf.Sub"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::AddV2Op src;
    ::mlir::Operation::operand_range arg0(op0->getOperands());
    ::mlir::Operation::operand_range arg1(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::AddV2Op>(op0); (void)castedOp0;
    src = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::NegOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::TF::NegOp type";
        });
      }
      arg0 = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }
    arg1 = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::SubOp dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg1.begin()));
      tblgen_values.push_back((*arg0.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::SubOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:56
*/
struct AddV2OfNegRight : public ::mlir::RewritePattern {
  AddV2OfNegRight(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.AddV2", 2, context, {"tf.Sub"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::AddV2Op src;
    ::mlir::Operation::operand_range arg0(op0->getOperands());
    ::mlir::Operation::operand_range arg1(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::AddV2Op>(op0); (void)castedOp0;
    src = castedOp0;
    arg0 = castedOp0.getODSOperands(0);
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_0(rewriter, op1, tblgen_ops, arg1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::SubOp dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg0.begin()));
      tblgen_values.push_back((*arg1.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::SubOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:94
*/
struct BatchMatMulToMatMul : public ::mlir::RewritePattern {
  BatchMatMulToMatMul(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.BatchMatMul", 1, context, {"tf.MatMul"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::BoolAttr grad_y;
    ::mlir::BoolAttr adj_x;
    ::mlir::TF::BatchMatMulOp src;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::mlir::BoolAttr grad_x;
    ::mlir::BoolAttr adj_y;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::BatchMatMulOp>(op0); (void)castedOp0;
    src = castedOp0;
    x = castedOp0.getODSOperands(0);
    y = castedOp0.getODSOperands(1);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("adj_x");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
      adj_x = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("adj_y");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
      adj_y = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("grad_x");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
      grad_x = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("grad_y");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
      grad_y = tblgen_attr;
    }
    if (!(((::llvm::cast<::mlir::ShapedType>(((*x.begin()).getType())).hasRank())) && ((::llvm::cast<::mlir::ShapedType>(((*x.begin()).getType())).getRank()
                             == 2)))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "value entity 'x' failed to satisfy constraint: 'Rank 2 tensor'";
      });
    }
    if (!(((::llvm::cast<::mlir::ShapedType>(((*y.begin()).getType())).hasRank())) && ((::llvm::cast<::mlir::ShapedType>(((*y.begin()).getType())).getRank()
                             == 2)))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "value entity 'y' failed to satisfy constraint: 'Rank 2 tensor'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::MatMulOp dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      if (auto tmpAttr = adj_x) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("transpose_a"), tmpAttr);
      }
      if (auto tmpAttr = adj_y) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("transpose_b"), tmpAttr);
      }
      if (auto tmpAttr = grad_x) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("grad_a"), tmpAttr);
      }
      if (auto tmpAttr = grad_y) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("grad_b"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::MatMulOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:88
*/
struct BatchMatMulToV2 : public ::mlir::RewritePattern {
  BatchMatMulToV2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.BatchMatMul", 1, context, {"tf.BatchMatMulV2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::BoolAttr grad_y;
    ::mlir::BoolAttr adj_x;
    ::mlir::TF::BatchMatMulOp src;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::mlir::BoolAttr grad_x;
    ::mlir::BoolAttr adj_y;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::BatchMatMulOp>(op0); (void)castedOp0;
    src = castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_canonicalize2(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'tf.BatchMatMul' failed to satisfy constraint: 'statically shaped tensor of any type values'"))) {
      return ::mlir::failure();
    }
    x = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_canonicalize2(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'tf.BatchMatMul' failed to satisfy constraint: 'statically shaped tensor of any type values'"))) {
      return ::mlir::failure();
    }
    y = castedOp0.getODSOperands(1);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("adj_x");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
      adj_x = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("adj_y");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
      adj_y = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("grad_x");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
      grad_x = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("grad_y");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
      grad_y = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::BatchMatMulV2Op dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      if (auto tmpAttr = adj_x) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("adj_x"), tmpAttr);
      }
      if (auto tmpAttr = adj_y) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("adj_y"), tmpAttr);
      }
      if (auto tmpAttr = grad_x) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("grad_x"), tmpAttr);
      }
      if (auto tmpAttr = grad_y) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("grad_y"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::BatchMatMulV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:104
*/
struct BatchMatMulV2ToMatMul : public ::mlir::RewritePattern {
  BatchMatMulV2ToMatMul(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.BatchMatMulV2", 1, context, {"tf.MatMul"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::BoolAttr grad_y;
    ::mlir::BoolAttr adj_x;
    ::mlir::TF::BatchMatMulV2Op src;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::mlir::BoolAttr grad_x;
    ::mlir::BoolAttr adj_y;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::BatchMatMulV2Op>(op0); (void)castedOp0;
    src = castedOp0;
    x = castedOp0.getODSOperands(0);
    y = castedOp0.getODSOperands(1);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("adj_x");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
      adj_x = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("adj_y");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
      adj_y = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("grad_x");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
      grad_x = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("grad_y");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
      grad_y = tblgen_attr;
    }
    if (!(((::llvm::cast<::mlir::ShapedType>(((*x.begin()).getType())).hasRank())) && ((::llvm::cast<::mlir::ShapedType>(((*x.begin()).getType())).getRank()
                             == 2)))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "value entity 'x' failed to satisfy constraint: 'Rank 2 tensor'";
      });
    }
    if (!(((::llvm::cast<::mlir::ShapedType>(((*y.begin()).getType())).hasRank())) && ((::llvm::cast<::mlir::ShapedType>(((*y.begin()).getType())).getRank()
                             == 2)))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "value entity 'y' failed to satisfy constraint: 'Rank 2 tensor'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::MatMulOp dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      tblgen_values.push_back((*y.begin()));
      if (auto tmpAttr = adj_x) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("transpose_a"), tmpAttr);
      }
      if (auto tmpAttr = adj_y) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("transpose_b"), tmpAttr);
      }
      if (auto tmpAttr = grad_x) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("grad_a"), tmpAttr);
      }
      if (auto tmpAttr = grad_y) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("grad_b"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::MatMulOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:118
*/
struct BatchToSpaceToBatchToSpaceND : public ::mlir::RewritePattern {
  BatchToSpaceToBatchToSpaceND(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.BatchToSpace", 1, context, {"tf.BatchToSpaceND", "tf.Const"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::BatchToSpaceOp src;
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range crops(op0->getOperands());
    ::mlir::IntegerAttr block_size;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::BatchToSpaceOp>(op0); (void)castedOp0;
    src = castedOp0;
    input = castedOp0.getODSOperands(0);
    crops = castedOp0.getODSOperands(1);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::IntegerAttr>("block_size");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'tf.BatchToSpace' to have attribute 'block_size' of type '::mlir::IntegerAttr'";
        });
      }
      block_size = tblgen_attr;
    }
    if (!(((::llvm::cast<::mlir::ShapedType>(((*input.begin()).getType())).hasRank())) && ((::llvm::cast<::mlir::ShapedType>(((*input.begin()).getType())).getRank()
                             == 4)))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "value entity 'input' failed to satisfy constraint: 'Rank 4 tensor'";
      });
    }
    if (!(((::llvm::cast<::mlir::ShapedType>(((*crops.begin()).getType())).hasRank())) && ((::llvm::cast<::mlir::ShapedType>(((*crops.begin()).getType())).getRank()
                             == 2)))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "value entity 'crops' failed to satisfy constraint: 'Rank 2 tensor'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = DenseElementsAttr::get(RankedTensorType::get({2}, rewriter.getI64Type()), ArrayRef<APInt>{block_size.getValue(), block_size.getValue()}); (void)nativeVar_0;
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    ::mlir::TF::BatchToSpaceNDOp dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*tblgen_ConstOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*crops.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::BatchToSpaceNDOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:128
*/
struct BiasAddV1ToBiasAdd : public ::mlir::RewritePattern {
  BiasAddV1ToBiasAdd(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.BiasAddV1", 1, context, {"tf.BiasAdd"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::BiasAddV1Op src;
    ::mlir::Operation::operand_range arg0(op0->getOperands());
    ::mlir::Operation::operand_range arg1(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::BiasAddV1Op>(op0); (void)castedOp0;
    src = castedOp0;
    arg0 = castedOp0.getODSOperands(0);
    arg1 = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::BiasAddOp dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg0.begin()));
      tblgen_values.push_back((*arg1.begin()));
      if (auto tmpAttr = rewriter.getStringAttr("NHWC")) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("data_format"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::BiasAddOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:146
*/
struct BitcastNested : public ::mlir::RewritePattern {
  BitcastNested(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Bitcast", 2, context, {"tf.Bitcast"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::BitcastOp src;
    ::mlir::Operation::operand_range arg(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::BitcastOp>(op0); (void)castedOp0;
    src = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::BitcastOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::TF::BitcastOp type";
        });
      }
      arg = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::BitcastOp dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::BitcastOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:141
*/
struct BitcastSameType : public ::mlir::RewritePattern {
  BitcastSameType(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Bitcast", 1, context, {}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::BitcastOp res;
    ::mlir::Operation::operand_range arg(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::BitcastOp>(op0); (void)castedOp0;
    res = castedOp0;
    arg = castedOp0.getODSOperands(0);
    if (!((getElementTypeOrSelf((*res.getODSResults(0).begin())) == getElementTypeOrSelf((*arg.begin()))))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'res, arg' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ arg }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:154
*/
struct ConvertToConcatV2 : public ::mlir::RewritePattern {
  ConvertToConcatV2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Concat", 1, context, {"tf.ConcatV2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::ConcatOp src;
    ::mlir::Operation::operand_range axis(op0->getOperands());
    ::mlir::Operation::operand_range inputs(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::ConcatOp>(op0); (void)castedOp0;
    src = castedOp0;
    axis = castedOp0.getODSOperands(0);
    inputs = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::ConcatV2Op dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      for (auto v: inputs) {
        tblgen_values.push_back(v);
      }
      tblgen_values.push_back((*axis.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::ConcatV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:163
*/
struct DivWithSqrtDivisor : public ::mlir::RewritePattern {
  DivWithSqrtDivisor(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Div", 2, context, {"tf.Mul", "tf.Rsqrt"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::DivOp src;
    ::mlir::Operation::operand_range arg0(op0->getOperands());
    ::mlir::Operation::operand_range arg1(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::DivOp>(op0); (void)castedOp0;
    src = castedOp0;
    arg0 = castedOp0.getODSOperands(0);
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_1(rewriter, op1, tblgen_ops, arg1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::RsqrtOp dest2;
    {
      ::mlir::Value tblgen_value_0 = (*arg1.begin());
      dest2 = rewriter.create<::mlir::TF::RsqrtOp>(odsLoc,
        /*x=*/tblgen_value_0
      );
    }
    ::mlir::TF::MulOp dest1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg0.begin()));
      tblgen_values.push_back((*dest2.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest1 = rewriter.create<::mlir::TF::MulOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest1.getODSResults(0).begin()));
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest2.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:73
*/
struct GatherToV2 : public ::mlir::RewritePattern {
  GatherToV2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Gather", 1, context, {"tf.Const", "tf.GatherV2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::GatherOp src;
    ::mlir::Operation::operand_range params(op0->getOperands());
    ::mlir::BoolAttr ignored_validate_indices;
    ::mlir::Operation::operand_range indices(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::GatherOp>(op0); (void)castedOp0;
    src = castedOp0;
    params = castedOp0.getODSOperands(0);
    indices = castedOp0.getODSOperands(1);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("validate_indices");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(true);
      ignored_validate_indices = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = rewriter.getI32IntegerAttr(0); (void)nativeVar_0;
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    auto nativeVar_2 = rewriter.getI64IntegerAttr(0); (void)nativeVar_2;
    ::mlir::TF::GatherV2Op dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*params.begin()));
      tblgen_values.push_back((*indices.begin()));
      tblgen_values.push_back((*tblgen_ConstOp_1.getODSResults(0).begin()));
      if (auto tmpAttr = nativeVar_2) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("batch_dims"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::GatherV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:348
*/
struct HashTableAndInitializeTableToV2 : public ::mlir::RewritePattern {
  HashTableAndInitializeTableToV2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.InitializeTableFromTextFile", 2, context, {"tf.HashTableV2", "tf.InitializeTableFromTextFileV2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TypeAttr value_dtype;
    ::mlir::TF::InitializeTableFromTextFileOp src;
    ::mlir::Operation::operand_range filename(op0->getOperands());
    ::mlir::StringAttr container;
    ::mlir::StringAttr delimiter;
    ::mlir::BoolAttr use_node_name_sharing;
    ::mlir::IntegerAttr offset;
    ::mlir::StringAttr shared_name;
    ::mlir::IntegerAttr key_index;
    ::mlir::TypeAttr key_dtype;
    ::mlir::IntegerAttr value_index;
    ::mlir::IntegerAttr vocab_size;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::InitializeTableFromTextFileOp>(op0); (void)castedOp0;
    src = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, container, use_node_name_sharing, shared_name, key_dtype, value_dtype))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    filename = castedOp0.getODSOperands(1);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::IntegerAttr>("key_index");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'tf.InitializeTableFromTextFile' to have attribute 'key_index' of type '::mlir::IntegerAttr'";
        });
      }
      key_index = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::IntegerAttr>("value_index");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'tf.InitializeTableFromTextFile' to have attribute 'value_index' of type '::mlir::IntegerAttr'";
        });
      }
      value_index = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::IntegerAttr>("vocab_size");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getIntegerAttr(rewriter.getIntegerType(64), -1);
      vocab_size = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::StringAttr>("delimiter");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getStringAttr("\t");
      delimiter = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::IntegerAttr>("offset");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getIntegerAttr(rewriter.getIntegerType(64), 0);
      offset = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::mlir::TF::HashTableV2Op dest2;
    {
      dest2 = rewriter.create<::mlir::TF::HashTableV2Op>(odsLoc,
        /*container=*/container,
        /*shared_name=*/shared_name,
        /*use_node_name_sharing=*/use_node_name_sharing,
        /*key_dtype=*/key_dtype,
        /*value_dtype=*/value_dtype
      );
    }
    ::mlir::TF::InitializeTableFromTextFileV2Op dest1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*dest2.getODSResults(0).begin()));
      tblgen_values.push_back((*filename.begin()));
      if (auto tmpAttr = key_index) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("key_index"), tmpAttr);
      }
      if (auto tmpAttr = value_index) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("value_index"), tmpAttr);
      }
      if (auto tmpAttr = vocab_size) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("vocab_size"), tmpAttr);
      }
      if (auto tmpAttr = delimiter) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("delimiter"), tmpAttr);
      }
      if (auto tmpAttr = offset) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("offset"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      dest1 = rewriter.create<::mlir::TF::InitializeTableFromTextFileV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor(src, dest1);
    CopyDeviceAndUnderscoredAttributesAdaptor(src, (*dest2.getODSResults(0).begin()));
    rewriter.eraseOp(op0);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:364
*/
struct HashTableAndLookupTableFindToV2 : public ::mlir::RewritePattern {
  HashTableAndLookupTableFindToV2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.LookupTableFind", 2, context, {"tf.HashTableV2", "tf.LookupTableFindV2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TypeAttr value_dtype;
    ::mlir::TF::LookupTableFindOp src;
    ::mlir::Operation::operand_range default_value(op0->getOperands());
    ::mlir::StringAttr container;
    ::mlir::BoolAttr use_node_name_sharing;
    ::mlir::StringAttr shared_name;
    ::mlir::TypeAttr key_dtype;
    ::mlir::Operation::operand_range keys(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::LookupTableFindOp>(op0); (void)castedOp0;
    src = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, container, use_node_name_sharing, shared_name, key_dtype, value_dtype))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }
    keys = castedOp0.getODSOperands(1);
    default_value = castedOp0.getODSOperands(2);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::HashTableV2Op dest2;
    {
      dest2 = rewriter.create<::mlir::TF::HashTableV2Op>(odsLoc,
        /*container=*/container,
        /*shared_name=*/shared_name,
        /*use_node_name_sharing=*/use_node_name_sharing,
        /*key_dtype=*/key_dtype,
        /*value_dtype=*/value_dtype
      );
    }
    ::mlir::TF::LookupTableFindV2Op dest1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*dest2.getODSResults(0).begin()));
      tblgen_values.push_back((*keys.begin()));
      tblgen_values.push_back((*default_value.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest1 = rewriter.create<::mlir::TF::LookupTableFindV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest1.getODSResults(0).begin()));
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest2.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:357
*/
struct HashTableAndLookupTableSizeToV2 : public ::mlir::RewritePattern {
  HashTableAndLookupTableSizeToV2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.LookupTableSize", 2, context, {"tf.HashTableV2", "tf.LookupTableSizeV2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TypeAttr value_dtype;
    ::mlir::TF::LookupTableSizeOp src;
    ::mlir::StringAttr container;
    ::mlir::BoolAttr use_node_name_sharing;
    ::mlir::StringAttr shared_name;
    ::mlir::TypeAttr key_dtype;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::LookupTableSizeOp>(op0); (void)castedOp0;
    src = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_2(rewriter, op1, tblgen_ops, container, use_node_name_sharing, shared_name, key_dtype, value_dtype))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::HashTableV2Op dest2;
    {
      dest2 = rewriter.create<::mlir::TF::HashTableV2Op>(odsLoc,
        /*container=*/container,
        /*shared_name=*/shared_name,
        /*use_node_name_sharing=*/use_node_name_sharing,
        /*key_dtype=*/key_dtype,
        /*value_dtype=*/value_dtype
      );
    }
    ::mlir::TF::LookupTableSizeV2Op dest1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*dest2.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest1 = rewriter.create<::mlir::TF::LookupTableSizeV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest1.getODSResults(0).begin()));
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest2.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:172
*/
struct LogOfSoftmax : public ::mlir::RewritePattern {
  LogOfSoftmax(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Log", 2, context, {"tf.LogSoftmax"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::LogOp src;
    ::mlir::Operation::operand_range arg(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::LogOp>(op0); (void)castedOp0;
    src = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::SoftmaxOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::TF::SoftmaxOp type";
        });
      }
      arg = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::LogSoftmaxOp dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::LogSoftmaxOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:182
*/
struct LogToLog1p : public ::mlir::RewritePattern {
  LogToLog1p(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Log", 3, context, {"tf.Log1p"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::LogOp src;
    ::mlir::Operation::operand_range arg(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::LogOp>(op0); (void)castedOp0;
    src = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::AddV2Op>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::TF::AddV2Op type";
        });
      }
      arg = castedOp1.getODSOperands(0);
      {
        auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
        if (!(op2)){
          return rewriter.notifyMatchFailure(castedOp1, [&](::mlir::Diagnostic &diag) {
            diag << "There's no operation that defines operand 1 of castedOp1";
          });
        }
        auto castedOp2 = ::llvm::dyn_cast<::mlir::TF::ConstOp>(op2); (void)castedOp2;
        if (!(castedOp2)){
          return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
            diag << "castedOp2 is not ::mlir::TF::ConstOp type";
          });
        }
        {
          auto tblgen_attr = op2->getAttrOfType<::mlir::ElementsAttr>("value");(void)tblgen_attr;
          if (!(tblgen_attr)){
            return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
              diag << "expected op 'tf.Const' to have attribute 'value' of type '::mlir::ElementsAttr'";
            });
          }
          if(::mlir::failed(__mlir_ods_local_attr_constraint_canonicalize1(rewriter, op2, tblgen_attr, "op 'tf.Const' attribute 'value' failed to satisfy constraint: 'constant attribute 1.0f'"))) {
            return ::mlir::failure();
          }
        }
        tblgen_ops.push_back(op2);
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::Log1pOp dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::Log1pOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:191
*/
struct LogicalNotOfEqual : public ::mlir::RewritePattern {
  LogicalNotOfEqual(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.LogicalNot", 2, context, {"tf.NotEqual"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::LogicalNotOp src;
    ::mlir::Operation::operand_range arg0(op0->getOperands());
    ::mlir::Operation::operand_range arg1(op0->getOperands());
    ::mlir::BoolAttr shape_error;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::LogicalNotOp>(op0); (void)castedOp0;
    src = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::EqualOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::TF::EqualOp type";
        });
      }
      arg0 = castedOp1.getODSOperands(0);
      arg1 = castedOp1.getODSOperands(1);
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::BoolAttr>("incompatible_shape_error");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(true);
        shape_error = tblgen_attr;
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::NotEqualOp dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg0.begin()));
      tblgen_values.push_back((*arg1.begin()));
      if (auto tmpAttr = shape_error) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("incompatible_shape_error"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::NotEqualOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:200
*/
struct LogicalNotOfGreater : public ::mlir::RewritePattern {
  LogicalNotOfGreater(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.LogicalNot", 2, context, {"tf.LessEqual"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::LogicalNotOp src;
    ::mlir::Operation::operand_range arg0(op0->getOperands());
    ::mlir::Operation::operand_range arg1(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::LogicalNotOp>(op0); (void)castedOp0;
    src = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::GreaterOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::TF::GreaterOp type";
        });
      }
      arg0 = castedOp1.getODSOperands(0);
      arg1 = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::LessEqualOp dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg0.begin()));
      tblgen_values.push_back((*arg1.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::LessEqualOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:204
*/
struct LogicalNotOfGreaterEqual : public ::mlir::RewritePattern {
  LogicalNotOfGreaterEqual(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.LogicalNot", 2, context, {"tf.Less"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::LogicalNotOp src;
    ::mlir::Operation::operand_range arg0(op0->getOperands());
    ::mlir::Operation::operand_range arg1(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::LogicalNotOp>(op0); (void)castedOp0;
    src = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::GreaterEqualOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::TF::GreaterEqualOp type";
        });
      }
      arg0 = castedOp1.getODSOperands(0);
      arg1 = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::LessOp dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg0.begin()));
      tblgen_values.push_back((*arg1.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::LessOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:208
*/
struct LogicalNotOfLess : public ::mlir::RewritePattern {
  LogicalNotOfLess(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.LogicalNot", 2, context, {"tf.GreaterEqual"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::LogicalNotOp src;
    ::mlir::Operation::operand_range arg0(op0->getOperands());
    ::mlir::Operation::operand_range arg1(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::LogicalNotOp>(op0); (void)castedOp0;
    src = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::LessOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::TF::LessOp type";
        });
      }
      arg0 = castedOp1.getODSOperands(0);
      arg1 = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::GreaterEqualOp dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg0.begin()));
      tblgen_values.push_back((*arg1.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::GreaterEqualOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:212
*/
struct LogicalNotOfLessEqual : public ::mlir::RewritePattern {
  LogicalNotOfLessEqual(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.LogicalNot", 2, context, {"tf.Greater"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::LogicalNotOp src;
    ::mlir::Operation::operand_range arg0(op0->getOperands());
    ::mlir::Operation::operand_range arg1(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::LogicalNotOp>(op0); (void)castedOp0;
    src = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::LessEqualOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::TF::LessEqualOp type";
        });
      }
      arg0 = castedOp1.getODSOperands(0);
      arg1 = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::GreaterOp dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg0.begin()));
      tblgen_values.push_back((*arg1.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::GreaterOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:196
*/
struct LogicalNotOfNotEqual : public ::mlir::RewritePattern {
  LogicalNotOfNotEqual(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.LogicalNot", 2, context, {"tf.Equal"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::LogicalNotOp src;
    ::mlir::Operation::operand_range arg0(op0->getOperands());
    ::mlir::Operation::operand_range arg1(op0->getOperands());
    ::mlir::BoolAttr shape_error;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::LogicalNotOp>(op0); (void)castedOp0;
    src = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::NotEqualOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::TF::NotEqualOp type";
        });
      }
      arg0 = castedOp1.getODSOperands(0);
      arg1 = castedOp1.getODSOperands(1);
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::BoolAttr>("incompatible_shape_error");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(true);
        shape_error = tblgen_attr;
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::EqualOp dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg0.begin()));
      tblgen_values.push_back((*arg1.begin()));
      if (auto tmpAttr = shape_error) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("incompatible_shape_error"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::EqualOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:239
*/
struct MatrixDiagToV3 : public ::mlir::RewritePattern {
  MatrixDiagToV3(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.MatrixDiag", 1, context, {"tf.Const", "tf.MatrixDiagV3"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::MatrixDiagOp src;
    ::mlir::Operation::operand_range diag(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::MatrixDiagOp>(op0); (void)castedOp0;
    src = castedOp0;
    diag = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = rewriter.getI32IntegerAttr(0); (void)nativeVar_0;
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    auto nativeVar_2 = rewriter.getI32IntegerAttr(-1); (void)nativeVar_2;
    ::mlir::TF::ConstOp tblgen_ConstOp_3;
    {
      tblgen_ConstOp_3 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_2
      );
    }
    auto nativeVar_4 = rewriter.getI32IntegerAttr(-1); (void)nativeVar_4;
    ::mlir::TF::ConstOp tblgen_ConstOp_5;
    {
      tblgen_ConstOp_5 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_4
      );
    }
    auto nativeVar_6 = GetScalarOfType(getElementTypeOrSelf((*diag.begin())),0); (void)nativeVar_6;
    ::mlir::TF::ConstOp tblgen_ConstOp_7;
    {
      tblgen_ConstOp_7 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_6
      );
    }
    auto nativeVar_8 = rewriter.getStringAttr("RIGHT_LEFT"); (void)nativeVar_8;
    ::mlir::TF::MatrixDiagV3Op dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*diag.begin()));
      tblgen_values.push_back((*tblgen_ConstOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_ConstOp_3.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_ConstOp_5.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_ConstOp_7.getODSResults(0).begin()));
      if (auto tmpAttr = nativeVar_8) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("align"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::MatrixDiagV3Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:223
*/
struct MatrixSetDiagToV3 : public ::mlir::RewritePattern {
  MatrixSetDiagToV3(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.MatrixSetDiag", 1, context, {"tf.Const", "tf.MatrixSetDiagV3"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::MatrixSetDiagOp src;
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range diag(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::MatrixSetDiagOp>(op0); (void)castedOp0;
    src = castedOp0;
    input = castedOp0.getODSOperands(0);
    diag = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = rewriter.getI32IntegerAttr(0); (void)nativeVar_0;
    ::mlir::TF::ConstOp tblgen_ConstOp_1;
    {
      tblgen_ConstOp_1 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/nativeVar_0
      );
    }
    auto nativeVar_2 = rewriter.getStringAttr("RIGHT_LEFT"); (void)nativeVar_2;
    ::mlir::TF::MatrixSetDiagV3Op dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*diag.begin()));
      tblgen_values.push_back((*tblgen_ConstOp_1.getODSResults(0).begin()));
      if (auto tmpAttr = nativeVar_2) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("align"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::MatrixSetDiagV3Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:230
*/
struct MatrixSetDiagV2ToV3 : public ::mlir::RewritePattern {
  MatrixSetDiagV2ToV3(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.MatrixSetDiagV2", 1, context, {"tf.MatrixSetDiagV3"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::MatrixSetDiagV2Op src;
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation::operand_range diag(op0->getOperands());
    ::mlir::Operation::operand_range k(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::MatrixSetDiagV2Op>(op0); (void)castedOp0;
    src = castedOp0;
    input = castedOp0.getODSOperands(0);
    diag = castedOp0.getODSOperands(1);
    k = castedOp0.getODSOperands(2);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = rewriter.getStringAttr("LEFT_LEFT"); (void)nativeVar_0;
    ::mlir::TF::MatrixSetDiagV3Op dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*diag.begin()));
      tblgen_values.push_back((*k.begin()));
      if (auto tmpAttr = nativeVar_0) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("align"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::MatrixSetDiagV3Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:385
*/
struct MaximumOfZeroToRelu : public ::mlir::RewritePattern {
  MaximumOfZeroToRelu(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Maximum", 1, context, {"tf.Relu"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::MaximumOp maximum_op;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::MaximumOp>(op0); (void)castedOp0;
    maximum_op = castedOp0;
    x = castedOp0.getODSOperands(0);
    y = castedOp0.getODSOperands(1);
    if (!((IsConstantValueOf((*y.begin()),0)))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'y' failed to satisfy constraint: ''";
      });
    }
    if (!(!(((IsOnGpuDevice((*maximum_op.getODSResults(0).begin()).getOwner()))) && ((getElementTypeOrSelf((*maximum_op.getODSResults(0).begin()).getType()).isInteger(32)))))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'maximum_op' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::ReluOp dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::ReluOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*maximum_op.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:250
*/
struct QuantizeAndDequantizeV2ToQuantizeAndDequantizeV4 : public ::mlir::RewritePattern {
  QuantizeAndDequantizeV2ToQuantizeAndDequantizeV4(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.QuantizeAndDequantizeV2", 1, context, {"tf.QuantizeAndDequantizeV4"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range max(op0->getOperands());
    ::mlir::TF::QuantizeAndDequantizeV2Op src;
    ::mlir::BoolAttr range_given;
    ::mlir::Operation::operand_range inputs(op0->getOperands());
    ::mlir::StringAttr round_mode;
    ::mlir::Operation::operand_range min(op0->getOperands());
    ::mlir::BoolAttr signed_input;
    ::mlir::BoolAttr narrow_range;
    ::mlir::IntegerAttr num_bits;
    ::mlir::IntegerAttr axis;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::QuantizeAndDequantizeV2Op>(op0); (void)castedOp0;
    src = castedOp0;
    inputs = castedOp0.getODSOperands(0);
    min = castedOp0.getODSOperands(1);
    max = castedOp0.getODSOperands(2);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("signed_input");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(true);
      signed_input = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::IntegerAttr>("num_bits");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getIntegerAttr(rewriter.getIntegerType(64), 8);
      num_bits = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("range_given");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
      range_given = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::StringAttr>("round_mode");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getStringAttr("HALF_TO_EVEN");
      round_mode = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("narrow_range");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
      narrow_range = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::IntegerAttr>("axis");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getIntegerAttr(rewriter.getIntegerType(64), -1);
      axis = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::QuantizeAndDequantizeV4Op dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*inputs.begin()));
      tblgen_values.push_back((*min.begin()));
      tblgen_values.push_back((*max.begin()));
      if (auto tmpAttr = signed_input) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("signed_input"), tmpAttr);
      }
      if (auto tmpAttr = num_bits) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("num_bits"), tmpAttr);
      }
      if (auto tmpAttr = range_given) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("range_given"), tmpAttr);
      }
      if (auto tmpAttr = round_mode) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("round_mode"), tmpAttr);
      }
      if (auto tmpAttr = narrow_range) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("narrow_range"), tmpAttr);
      }
      if (auto tmpAttr = axis) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("axis"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::QuantizeAndDequantizeV4Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:330
*/
struct ReadVariableOfCast : public ::mlir::RewritePattern {
  ReadVariableOfCast(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.ReadVariableOp", 2, context, {"tf.ReadVariableOp"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::ReadVariableOp src;
    ::mlir::TF::CastOp output;
    ::mlir::BoolAttr Truncate;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::ReadVariableOp>(op0); (void)castedOp0;
    src = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::CastOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::TF::CastOp type";
        });
      }
      output = castedOp1;
      x = castedOp1.getODSOperands(0);
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::BoolAttr>("Truncate");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
        if (!tblgen_attr) return ::mlir::failure();
        if(::mlir::failed(__mlir_ods_local_attr_constraint_canonicalize2(rewriter, op1, tblgen_attr, "op 'tf.Cast' attribute 'Truncate' failed to satisfy constraint: 'bool attribute'"))) {
          return ::mlir::failure();
        }
        Truncate = tblgen_attr;
      }
      tblgen_ops.push_back(op1);
    }
    if (!((llvm::all_of((*output.getODSResults(0).begin()).getUsers(), [](mlir::OpOperand op) { return llvm::isa<mlir::TF::ReadVariableOp>(op.getOwner()); })))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'output' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::ReadVariableOp dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::ReadVariableOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:269
*/
struct RealDivWithConstDivisor : public ::mlir::RewritePattern {
  RealDivWithConstDivisor(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.RealDiv", 2, context, {"tf.Const", "tf.Mul", "tf.Reciprocal"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::RealDivOp src;
    ::mlir::Operation::operand_range arg0(op0->getOperands());
    ::mlir::ElementsAttr value;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::RealDivOp>(op0); (void)castedOp0;
    src = castedOp0;
    arg0 = castedOp0.getODSOperands(0);
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::ConstOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::TF::ConstOp type";
        });
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::ElementsAttr>("value");(void)tblgen_attr;
        if (!(tblgen_attr)){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "expected op 'tf.Const' to have attribute 'value' of type '::mlir::ElementsAttr'";
          });
        }
        if(::mlir::failed(__mlir_ods_local_attr_constraint_canonicalize3(rewriter, op1, tblgen_attr, "op 'tf.Const' attribute 'value' failed to satisfy constraint: '32-bit float elements attribute'"))) {
          return ::mlir::failure();
        }
        value = tblgen_attr;
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::ConstOp tblgen_ConstOp_0;
    {
      tblgen_ConstOp_0 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/value
      );
    }
    ::mlir::TF::ReciprocalOp dest2;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ConstOp_0.getODSResults(0).begin());
      dest2 = rewriter.create<::mlir::TF::ReciprocalOp>(odsLoc,
        /*x=*/tblgen_value_0
      );
    }
    ::mlir::TF::MulOp dest1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg0.begin()));
      tblgen_values.push_back((*dest2.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest1 = rewriter.create<::mlir::TF::MulOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest1.getODSResults(0).begin()));
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest2.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:261
*/
struct RealDivWithSqrtDivisor : public ::mlir::RewritePattern {
  RealDivWithSqrtDivisor(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.RealDiv", 2, context, {"tf.Mul", "tf.Rsqrt"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::RealDivOp src;
    ::mlir::Operation::operand_range arg0(op0->getOperands());
    ::mlir::Operation::operand_range arg1(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::RealDivOp>(op0); (void)castedOp0;
    src = castedOp0;
    arg0 = castedOp0.getODSOperands(0);
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_1(rewriter, op1, tblgen_ops, arg1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::RsqrtOp dest2;
    {
      ::mlir::Value tblgen_value_0 = (*arg1.begin());
      dest2 = rewriter.create<::mlir::TF::RsqrtOp>(odsLoc,
        /*x=*/tblgen_value_0
      );
    }
    ::mlir::TF::MulOp dest1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg0.begin()));
      tblgen_values.push_back((*dest2.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest1 = rewriter.create<::mlir::TF::MulOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest1.getODSResults(0).begin()));
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest2.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:278
*/
struct RedundantReshape : public ::mlir::RewritePattern {
  RedundantReshape(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Reshape", 2, context, {"tf.Reshape"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::ReshapeOp src;
    ::mlir::Operation::operand_range arg(op0->getOperands());
    ::mlir::Operation::operand_range unused(op0->getOperands());
    ::mlir::Operation::operand_range shape(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::ReshapeOp>(op0); (void)castedOp0;
    src = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::ReshapeOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::TF::ReshapeOp type";
        });
      }
      arg = castedOp1.getODSOperands(0);
      unused = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }
    shape = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::ReshapeOp dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg.begin()));
      tblgen_values.push_back((*shape.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::ReshapeOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:394
*/
struct ReluOfMinimum6ToRelu6 : public ::mlir::RewritePattern {
  ReluOfMinimum6ToRelu6(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Relu", 2, context, {"tf.Relu6"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::ReluOp src;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range y(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::ReluOp>(op0); (void)castedOp0;
    src = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::MinimumOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::TF::MinimumOp type";
        });
      }
      x = castedOp1.getODSOperands(0);
      y = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }
    if (!((IsConstantValueOf((*y.begin()),6)))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'y' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::Relu6Op dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*x.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::Relu6Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:282
*/
struct ReshapeToSelfShape : public ::mlir::RewritePattern {
  ReshapeToSelfShape(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Reshape", 2, context, {}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::ReshapeOp op;
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::mlir::Operation::operand_range x0(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::ReshapeOp>(op0); (void)castedOp0;
    op = castedOp0;
    x = castedOp0.getODSOperands(0);
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::ShapeOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::TF::ShapeOp type";
        });
      }
      x0 = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }
    if (!(((*x.begin()).getType() == (*op.getODSResults(0).begin()).getType()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'x, op' failed to satisfy constraint: ''";
      });
    }
    if (!(*x.begin() == *x0.begin())){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "Operands 'x' and 'x0' must be equal";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ x }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:294
*/
struct SquareOfSub : public ::mlir::RewritePattern {
  SquareOfSub(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Square", 2, context, {"tf.SquaredDifference"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::SquareOp src;
    ::mlir::Operation::operand_range arg0(op0->getOperands());
    ::mlir::Operation::operand_range arg1(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::SquareOp>(op0); (void)castedOp0;
    src = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::TF::SubOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::TF::SubOp type";
        });
      }
      if(::mlir::failed(__mlir_ods_local_type_constraint_canonicalize3(rewriter, castedOp1, (*castedOp1.getODSOperands(0).begin()).getType(), "operand 0 of op 'tf.Sub' failed to satisfy constraint: 'tensor of bfloat16 or 128-bit complex or 64-bit complex or 16-bit float or 32-bit float or 32-bit integer or 64-bit integer values'"))) {
        return ::mlir::failure();
      }
      arg0 = castedOp1.getODSOperands(0);
      if(::mlir::failed(__mlir_ods_local_type_constraint_canonicalize3(rewriter, castedOp1, (*castedOp1.getODSOperands(1).begin()).getType(), "operand 1 of op 'tf.Sub' failed to satisfy constraint: 'tensor of bfloat16 or 128-bit complex or 64-bit complex or 16-bit float or 32-bit float or 32-bit integer or 64-bit integer values'"))) {
        return ::mlir::failure();
      }
      arg1 = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::SquaredDifferenceOp dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg0.begin()));
      tblgen_values.push_back((*arg1.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::SquaredDifferenceOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:303
*/
struct SubOfNeg : public ::mlir::RewritePattern {
  SubOfNeg(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Sub", 2, context, {"tf.AddV2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::SubOp src;
    ::mlir::Operation::operand_range arg0(op0->getOperands());
    ::mlir::Operation::operand_range arg1(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::SubOp>(op0); (void)castedOp0;
    src = castedOp0;
    arg0 = castedOp0.getODSOperands(0);
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_0(rewriter, op1, tblgen_ops, arg1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::AddV2Op dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg0.begin()));
      tblgen_values.push_back((*arg1.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::AddV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:311
*/
struct TruncateDivWithSqrtDivisor : public ::mlir::RewritePattern {
  TruncateDivWithSqrtDivisor(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.TruncateDiv", 2, context, {"tf.Mul", "tf.Rsqrt"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::TruncateDivOp src;
    ::mlir::Operation::operand_range arg0(op0->getOperands());
    ::mlir::Operation::operand_range arg1(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::TruncateDivOp>(op0); (void)castedOp0;
    src = castedOp0;
    arg0 = castedOp0.getODSOperands(0);
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_1(rewriter, op1, tblgen_ops, arg1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::RsqrtOp dest2;
    {
      ::mlir::Value tblgen_value_0 = (*arg1.begin());
      dest2 = rewriter.create<::mlir::TF::RsqrtOp>(odsLoc,
        /*x=*/tblgen_value_0
      );
    }
    ::mlir::TF::MulOp dest1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*arg0.begin()));
      tblgen_values.push_back((*dest2.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest1 = rewriter.create<::mlir::TF::MulOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest1.getODSResults(0).begin()));
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest2.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:339
*/
struct VariableToVariableV2 : public ::mlir::RewritePattern {
  VariableToVariableV2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Variable", 1, context, {"tf.VariableV2"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::VariableOp src;
    ::mlir::Attribute shape;
    ::mlir::StringAttr container;
    ::mlir::StringAttr shard_name;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::VariableOp>(op0); (void)castedOp0;
    src = castedOp0;
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("shape");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'tf.Variable' to have attribute 'shape' of type '::mlir::Attribute'";
        });
      }
      shape = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::StringAttr>("container");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getStringAttr("");
      container = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::StringAttr>("shared_name");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getStringAttr("");
      shard_name = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::VariableV2Op dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      if (auto tmpAttr = shape) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("shape"), tmpAttr);
      }
      if (auto tmpAttr = container) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("container"), tmpAttr);
      }
      if (auto tmpAttr = shard_name) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("shared_name"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = rewriter.create<::mlir::TF::VariableV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/canonicalize.td:320
*/
struct XdivyWithSqrtDivisor : public ::mlir::RewritePattern {
  XdivyWithSqrtDivisor(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Xdivy", 2, context, {"tf.MulNoNan", "tf.Rsqrt"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::TF::XdivyOp src;
    ::mlir::Operation::operand_range arg0(op0->getOperands());
    ::mlir::Operation::operand_range arg1(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::TF::XdivyOp>(op0); (void)castedOp0;
    src = castedOp0;
    arg0 = castedOp0.getODSOperands(0);
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 1 of castedOp0";
        });
      }
      if(::mlir::failed(static_dag_matcher_1(rewriter, op1, tblgen_ops, arg1))) {
        return ::mlir::failure();
      }
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::RsqrtOp dest2;
    {
      ::mlir::Value tblgen_value_0 = (*arg1.begin());
      dest2 = rewriter.create<::mlir::TF::RsqrtOp>(odsLoc,
        /*x=*/tblgen_value_0
      );
    }
    ::mlir::TF::MulNoNanOp dest1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*dest2.getODSResults(0).begin()));
      tblgen_values.push_back((*arg0.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest1 = rewriter.create<::mlir::TF::MulNoNanOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest1.getODSResults(0).begin()));
    CopyDeviceAndUnderscoredAttributesAdaptor((*src.getODSResults(0).begin()), (*dest2.getODSResults(0).begin()));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<AddToAddV2>(patterns.getContext());
  patterns.add<AddV2OfNegLeft>(patterns.getContext());
  patterns.add<AddV2OfNegRight>(patterns.getContext());
  patterns.add<BatchMatMulToMatMul>(patterns.getContext());
  patterns.add<BatchMatMulToV2>(patterns.getContext());
  patterns.add<BatchMatMulV2ToMatMul>(patterns.getContext());
  patterns.add<BatchToSpaceToBatchToSpaceND>(patterns.getContext());
  patterns.add<BiasAddV1ToBiasAdd>(patterns.getContext());
  patterns.add<BitcastNested>(patterns.getContext());
  patterns.add<BitcastSameType>(patterns.getContext());
  patterns.add<ConvertToConcatV2>(patterns.getContext());
  patterns.add<DivWithSqrtDivisor>(patterns.getContext());
  patterns.add<GatherToV2>(patterns.getContext());
  patterns.add<HashTableAndInitializeTableToV2>(patterns.getContext());
  patterns.add<HashTableAndLookupTableFindToV2>(patterns.getContext());
  patterns.add<HashTableAndLookupTableSizeToV2>(patterns.getContext());
  patterns.add<LogOfSoftmax>(patterns.getContext());
  patterns.add<LogToLog1p>(patterns.getContext());
  patterns.add<LogicalNotOfEqual>(patterns.getContext());
  patterns.add<LogicalNotOfGreater>(patterns.getContext());
  patterns.add<LogicalNotOfGreaterEqual>(patterns.getContext());
  patterns.add<LogicalNotOfLess>(patterns.getContext());
  patterns.add<LogicalNotOfLessEqual>(patterns.getContext());
  patterns.add<LogicalNotOfNotEqual>(patterns.getContext());
  patterns.add<MatrixDiagToV3>(patterns.getContext());
  patterns.add<MatrixSetDiagToV3>(patterns.getContext());
  patterns.add<MatrixSetDiagV2ToV3>(patterns.getContext());
  patterns.add<MaximumOfZeroToRelu>(patterns.getContext());
  patterns.add<QuantizeAndDequantizeV2ToQuantizeAndDequantizeV4>(patterns.getContext());
  patterns.add<ReadVariableOfCast>(patterns.getContext());
  patterns.add<RealDivWithConstDivisor>(patterns.getContext());
  patterns.add<RealDivWithSqrtDivisor>(patterns.getContext());
  patterns.add<RedundantReshape>(patterns.getContext());
  patterns.add<ReluOfMinimum6ToRelu6>(patterns.getContext());
  patterns.add<ReshapeToSelfShape>(patterns.getContext());
  patterns.add<SquareOfSub>(patterns.getContext());
  patterns.add<SubOfNeg>(patterns.getContext());
  patterns.add<TruncateDivWithSqrtDivisor>(patterns.getContext());
  patterns.add<VariableToVariableV2>(patterns.getContext());
  patterns.add<XdivyWithSqrtDivisor>(patterns.getContext());
}
