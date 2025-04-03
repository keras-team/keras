/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: legalize_to_standard_patterns.td                                     *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


static ::llvm::LogicalResult __mlir_ods_local_type_constraint_legalize_to_standard_patterns1(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Type type,
    ::llvm::StringRef failureStr) {
  if (!(((::llvm::isa<::mlir::RankedTensorType>(type))) && ([](::mlir::Type elementType) { return ((elementType.isFloat8E3M4())) || ((elementType.isFloat8E4M3())) || ((elementType.isFloat8E4M3FN())) || ((elementType.isFloat8E4M3FNUZ())) || ((elementType.isFloat8E4M3B11FNUZ())) || ((elementType.isFloat8E5M2())) || ((elementType.isFloat8E5M2FNUZ())) || ((elementType.isF16())) || ((elementType.isF32())) || ((elementType.isF64())) || ((elementType.isBF16())); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": ranked tensor of f8E3M4 type or f8E4M3 type or f8E4M3FN type or f8E4M3FNUZ type or f8E4M3B11FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_legalize_to_standard_patterns2(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Type type,
    ::llvm::StringRef failureStr) {
  if (!(((::llvm::isa<::mlir::RankedTensorType>(type))) && ([](::mlir::Type elementType) { return (((elementType.isSignlessInteger(2))) || ((elementType.isSignlessInteger(4))) || ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64)))) || (((elementType.isUnsignedInteger(2))) || ((elementType.isUnsignedInteger(4))) || ((elementType.isUnsignedInteger(8))) || ((elementType.isUnsignedInteger(16))) || ((elementType.isUnsignedInteger(32))) || ((elementType.isUnsignedInteger(64)))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": ranked tensor of 2/4/8/16/32/64-bit signless integer or 2/4/8/16/32/64-bit unsigned integer values";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_legalize_to_standard_patterns1(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((::llvm::isa<::mlir::ElementsAttr>(attr)))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": constant vector/tensor attribute";
    });
  }
  return ::mlir::success();
}
/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/legalize_to_standard/legalize_to_standard_patterns.td:28
*/
struct GeneratedConvert0 : public ::mlir::RewritePattern {
  GeneratedConvert0(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.constant", 1, context, {"arith.constant"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::ElementsAttr value;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::ConstantOp>(op0); (void)castedOp0;
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::ElementsAttr>("value");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'mhlo.constant' to have attribute 'value' of type '::mlir::ElementsAttr'";
        });
      }
      if(::mlir::failed(__mlir_ods_local_attr_constraint_legalize_to_standard_patterns1(rewriter, op0, tblgen_attr, "op 'mhlo.constant' attribute 'value' failed to satisfy constraint: 'constant vector/tensor attribute'"))) {
        return ::mlir::failure();
      }
      value = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::ConstantOp tblgen_ConstantOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      if (auto tmpAttr = value) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("value"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ConstantOp_0 = rewriter.create<::mlir::arith::ConstantOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ConstantOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/legalize_to_standard/legalize_to_standard_patterns.td:50
*/
struct GeneratedConvert1 : public ::mlir::RewritePattern {
  GeneratedConvert1(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.ceil", 1, context, {"math.ceil"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range i(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::CeilOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.ceil' failed to satisfy constraint: 'ranked tensor of f8E3M4 type or f8E4M3 type or f8E4M3FN type or f8E4M3FNUZ type or f8E4M3B11FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values'"))) {
      return ::mlir::failure();
    }
    i = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = ::mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), ::mlir::arith::FastMathFlags::none); (void)nativeVar_0;
    ::mlir::math::CeilOp tblgen_CeilOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*i.begin()));
      if (auto tmpAttr = nativeVar_0) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("fastmath"), tmpAttr);
      }
      tblgen_CeilOp_1 = rewriter.create<::mlir::math::CeilOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_CeilOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/legalize_to_standard/legalize_to_standard_patterns.td:53
*/
struct GeneratedConvert2 : public ::mlir::RewritePattern {
  GeneratedConvert2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.and", 1, context, {"arith.andi"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range l(op0->getOperands());
    ::mlir::Operation::operand_range r(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::AndOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.and' failed to satisfy constraint: 'ranked tensor of 2/4/8/16/32/64-bit signless integer or 2/4/8/16/32/64-bit unsigned integer values'"))) {
      return ::mlir::failure();
    }
    l = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'mhlo.and' failed to satisfy constraint: 'ranked tensor of 2/4/8/16/32/64-bit signless integer or 2/4/8/16/32/64-bit unsigned integer values'"))) {
      return ::mlir::failure();
    }
    r = castedOp0.getODSOperands(1);
    if (!((cast<ShapedType>((*l.begin()).getType()).getShape() == cast<ShapedType>((*r.begin()).getType()).getShape()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'l, r' failed to satisfy constraint: 'inputs are same size'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::AndIOp tblgen_AndIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*l.begin()));
      tblgen_values.push_back((*r.begin()));
      tblgen_AndIOp_0 = rewriter.create<::mlir::arith::AndIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AndIOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/legalize_to_standard/legalize_to_standard_patterns.td:56
*/
struct GeneratedConvert3 : public ::mlir::RewritePattern {
  GeneratedConvert3(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.or", 1, context, {"arith.ori"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range l(op0->getOperands());
    ::mlir::Operation::operand_range r(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::OrOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.or' failed to satisfy constraint: 'ranked tensor of 2/4/8/16/32/64-bit signless integer or 2/4/8/16/32/64-bit unsigned integer values'"))) {
      return ::mlir::failure();
    }
    l = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'mhlo.or' failed to satisfy constraint: 'ranked tensor of 2/4/8/16/32/64-bit signless integer or 2/4/8/16/32/64-bit unsigned integer values'"))) {
      return ::mlir::failure();
    }
    r = castedOp0.getODSOperands(1);
    if (!((cast<ShapedType>((*l.begin()).getType()).getShape() == cast<ShapedType>((*r.begin()).getType()).getShape()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'l, r' failed to satisfy constraint: 'inputs are same size'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::OrIOp tblgen_OrIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*l.begin()));
      tblgen_values.push_back((*r.begin()));
      tblgen_OrIOp_0 = rewriter.create<::mlir::arith::OrIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_OrIOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/legalize_to_standard/legalize_to_standard_patterns.td:59
*/
struct GeneratedConvert4 : public ::mlir::RewritePattern {
  GeneratedConvert4(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.add", 1, context, {"arith.addf"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range l(op0->getOperands());
    ::mlir::Operation::operand_range r(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::AddOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.add' failed to satisfy constraint: 'ranked tensor of f8E3M4 type or f8E4M3 type or f8E4M3FN type or f8E4M3FNUZ type or f8E4M3B11FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values'"))) {
      return ::mlir::failure();
    }
    l = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'mhlo.add' failed to satisfy constraint: 'ranked tensor of f8E3M4 type or f8E4M3 type or f8E4M3FN type or f8E4M3FNUZ type or f8E4M3B11FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values'"))) {
      return ::mlir::failure();
    }
    r = castedOp0.getODSOperands(1);
    if (!((cast<ShapedType>((*l.begin()).getType()).getShape() == cast<ShapedType>((*r.begin()).getType()).getShape()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'l, r' failed to satisfy constraint: 'inputs are same size'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = ::mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), ::mlir::arith::FastMathFlags::none); (void)nativeVar_0;
    ::mlir::arith::AddFOp tblgen_AddFOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*l.begin()));
      tblgen_values.push_back((*r.begin()));
      if (auto tmpAttr = nativeVar_0) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("fastmath"), tmpAttr);
      }
      tblgen_AddFOp_1 = rewriter.create<::mlir::arith::AddFOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AddFOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/legalize_to_standard/legalize_to_standard_patterns.td:62
*/
struct GeneratedConvert5 : public ::mlir::RewritePattern {
  GeneratedConvert5(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.subtract", 1, context, {"arith.subf"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range l(op0->getOperands());
    ::mlir::Operation::operand_range r(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::SubtractOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.subtract' failed to satisfy constraint: 'ranked tensor of f8E3M4 type or f8E4M3 type or f8E4M3FN type or f8E4M3FNUZ type or f8E4M3B11FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values'"))) {
      return ::mlir::failure();
    }
    l = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'mhlo.subtract' failed to satisfy constraint: 'ranked tensor of f8E3M4 type or f8E4M3 type or f8E4M3FN type or f8E4M3FNUZ type or f8E4M3B11FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values'"))) {
      return ::mlir::failure();
    }
    r = castedOp0.getODSOperands(1);
    if (!((cast<ShapedType>((*l.begin()).getType()).getShape() == cast<ShapedType>((*r.begin()).getType()).getShape()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'l, r' failed to satisfy constraint: 'inputs are same size'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = ::mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), ::mlir::arith::FastMathFlags::none); (void)nativeVar_0;
    ::mlir::arith::SubFOp tblgen_SubFOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*l.begin()));
      tblgen_values.push_back((*r.begin()));
      if (auto tmpAttr = nativeVar_0) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("fastmath"), tmpAttr);
      }
      tblgen_SubFOp_1 = rewriter.create<::mlir::arith::SubFOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SubFOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/legalize_to_standard/legalize_to_standard_patterns.td:65
*/
struct GeneratedConvert6 : public ::mlir::RewritePattern {
  GeneratedConvert6(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.multiply", 1, context, {"arith.mulf"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range l(op0->getOperands());
    ::mlir::Operation::operand_range r(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::MulOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.multiply' failed to satisfy constraint: 'ranked tensor of f8E3M4 type or f8E4M3 type or f8E4M3FN type or f8E4M3FNUZ type or f8E4M3B11FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values'"))) {
      return ::mlir::failure();
    }
    l = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'mhlo.multiply' failed to satisfy constraint: 'ranked tensor of f8E3M4 type or f8E4M3 type or f8E4M3FN type or f8E4M3FNUZ type or f8E4M3B11FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values'"))) {
      return ::mlir::failure();
    }
    r = castedOp0.getODSOperands(1);
    if (!((cast<ShapedType>((*l.begin()).getType()).getShape() == cast<ShapedType>((*r.begin()).getType()).getShape()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'l, r' failed to satisfy constraint: 'inputs are same size'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = ::mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), ::mlir::arith::FastMathFlags::none); (void)nativeVar_0;
    ::mlir::arith::MulFOp tblgen_MulFOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*l.begin()));
      tblgen_values.push_back((*r.begin()));
      if (auto tmpAttr = nativeVar_0) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("fastmath"), tmpAttr);
      }
      tblgen_MulFOp_1 = rewriter.create<::mlir::arith::MulFOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_MulFOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/legalize_to_standard/legalize_to_standard_patterns.td:68
*/
struct GeneratedConvert7 : public ::mlir::RewritePattern {
  GeneratedConvert7(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.divide", 1, context, {"arith.divf"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range l(op0->getOperands());
    ::mlir::Operation::operand_range r(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::DivOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.divide' failed to satisfy constraint: 'ranked tensor of f8E3M4 type or f8E4M3 type or f8E4M3FN type or f8E4M3FNUZ type or f8E4M3B11FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values'"))) {
      return ::mlir::failure();
    }
    l = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'mhlo.divide' failed to satisfy constraint: 'ranked tensor of f8E3M4 type or f8E4M3 type or f8E4M3FN type or f8E4M3FNUZ type or f8E4M3B11FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values'"))) {
      return ::mlir::failure();
    }
    r = castedOp0.getODSOperands(1);
    if (!((cast<ShapedType>((*l.begin()).getType()).getShape() == cast<ShapedType>((*r.begin()).getType()).getShape()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'l, r' failed to satisfy constraint: 'inputs are same size'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = ::mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), ::mlir::arith::FastMathFlags::none); (void)nativeVar_0;
    ::mlir::arith::DivFOp tblgen_DivFOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*l.begin()));
      tblgen_values.push_back((*r.begin()));
      if (auto tmpAttr = nativeVar_0) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("fastmath"), tmpAttr);
      }
      tblgen_DivFOp_1 = rewriter.create<::mlir::arith::DivFOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_DivFOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/legalize_to_standard/legalize_to_standard_patterns.td:71
*/
struct GeneratedConvert8 : public ::mlir::RewritePattern {
  GeneratedConvert8(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.remainder", 1, context, {"arith.remf"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range l(op0->getOperands());
    ::mlir::Operation::operand_range r(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::RemOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.remainder' failed to satisfy constraint: 'ranked tensor of f8E3M4 type or f8E4M3 type or f8E4M3FN type or f8E4M3FNUZ type or f8E4M3B11FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values'"))) {
      return ::mlir::failure();
    }
    l = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'mhlo.remainder' failed to satisfy constraint: 'ranked tensor of f8E3M4 type or f8E4M3 type or f8E4M3FN type or f8E4M3FNUZ type or f8E4M3B11FNUZ type or f8E5M2 type or f8E5M2FNUZ type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values'"))) {
      return ::mlir::failure();
    }
    r = castedOp0.getODSOperands(1);
    if (!((cast<ShapedType>((*l.begin()).getType()).getShape() == cast<ShapedType>((*r.begin()).getType()).getShape()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'l, r' failed to satisfy constraint: 'inputs are same size'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = ::mlir::arith::FastMathFlagsAttr::get(rewriter.getContext(), ::mlir::arith::FastMathFlags::none); (void)nativeVar_0;
    ::mlir::arith::RemFOp tblgen_RemFOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*l.begin()));
      tblgen_values.push_back((*r.begin()));
      if (auto tmpAttr = nativeVar_0) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("fastmath"), tmpAttr);
      }
      tblgen_RemFOp_1 = rewriter.create<::mlir::arith::RemFOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_RemFOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/legalize_to_standard/legalize_to_standard_patterns.td:74
*/
struct GeneratedConvert9 : public ::mlir::RewritePattern {
  GeneratedConvert9(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.add", 1, context, {"arith.addi"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range l(op0->getOperands());
    ::mlir::Operation::operand_range r(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::AddOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.add' failed to satisfy constraint: 'ranked tensor of 2/4/8/16/32/64-bit signless integer or 2/4/8/16/32/64-bit unsigned integer values'"))) {
      return ::mlir::failure();
    }
    l = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'mhlo.add' failed to satisfy constraint: 'ranked tensor of 2/4/8/16/32/64-bit signless integer or 2/4/8/16/32/64-bit unsigned integer values'"))) {
      return ::mlir::failure();
    }
    r = castedOp0.getODSOperands(1);
    if (!((cast<ShapedType>((*l.begin()).getType()).getShape() == cast<ShapedType>((*r.begin()).getType()).getShape()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'l, r' failed to satisfy constraint: 'inputs are same size'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none); (void)nativeVar_0;
    ::mlir::arith::AddIOp tblgen_AddIOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*l.begin()));
      tblgen_values.push_back((*r.begin()));
      if (auto tmpAttr = nativeVar_0) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("overflowFlags"), tmpAttr);
      }
      tblgen_AddIOp_1 = rewriter.create<::mlir::arith::AddIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AddIOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/legalize_to_standard/legalize_to_standard_patterns.td:77
*/
struct GeneratedConvert10 : public ::mlir::RewritePattern {
  GeneratedConvert10(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.subtract", 1, context, {"arith.subi"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range l(op0->getOperands());
    ::mlir::Operation::operand_range r(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::SubtractOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.subtract' failed to satisfy constraint: 'ranked tensor of 2/4/8/16/32/64-bit signless integer or 2/4/8/16/32/64-bit unsigned integer values'"))) {
      return ::mlir::failure();
    }
    l = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'mhlo.subtract' failed to satisfy constraint: 'ranked tensor of 2/4/8/16/32/64-bit signless integer or 2/4/8/16/32/64-bit unsigned integer values'"))) {
      return ::mlir::failure();
    }
    r = castedOp0.getODSOperands(1);
    if (!((cast<ShapedType>((*l.begin()).getType()).getShape() == cast<ShapedType>((*r.begin()).getType()).getShape()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'l, r' failed to satisfy constraint: 'inputs are same size'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none); (void)nativeVar_0;
    ::mlir::arith::SubIOp tblgen_SubIOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*l.begin()));
      tblgen_values.push_back((*r.begin()));
      if (auto tmpAttr = nativeVar_0) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("overflowFlags"), tmpAttr);
      }
      tblgen_SubIOp_1 = rewriter.create<::mlir::arith::SubIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SubIOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/legalize_to_standard/legalize_to_standard_patterns.td:80
*/
struct GeneratedConvert11 : public ::mlir::RewritePattern {
  GeneratedConvert11(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.multiply", 1, context, {"arith.muli"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range l(op0->getOperands());
    ::mlir::Operation::operand_range r(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::MulOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.multiply' failed to satisfy constraint: 'ranked tensor of 2/4/8/16/32/64-bit signless integer or 2/4/8/16/32/64-bit unsigned integer values'"))) {
      return ::mlir::failure();
    }
    l = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'mhlo.multiply' failed to satisfy constraint: 'ranked tensor of 2/4/8/16/32/64-bit signless integer or 2/4/8/16/32/64-bit unsigned integer values'"))) {
      return ::mlir::failure();
    }
    r = castedOp0.getODSOperands(1);
    if (!((cast<ShapedType>((*l.begin()).getType()).getShape() == cast<ShapedType>((*r.begin()).getType()).getShape()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'l, r' failed to satisfy constraint: 'inputs are same size'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    auto nativeVar_0 = ::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none); (void)nativeVar_0;
    ::mlir::arith::MulIOp tblgen_MulIOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*l.begin()));
      tblgen_values.push_back((*r.begin()));
      if (auto tmpAttr = nativeVar_0) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("overflowFlags"), tmpAttr);
      }
      tblgen_MulIOp_1 = rewriter.create<::mlir::arith::MulIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_MulIOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/legalize_to_standard/legalize_to_standard_patterns.td:83
*/
struct GeneratedConvert12 : public ::mlir::RewritePattern {
  GeneratedConvert12(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.divide", 1, context, {"arith.divsi"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range l(op0->getOperands());
    ::mlir::Operation::operand_range r(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::DivOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.divide' failed to satisfy constraint: 'ranked tensor of 2/4/8/16/32/64-bit signless integer or 2/4/8/16/32/64-bit unsigned integer values'"))) {
      return ::mlir::failure();
    }
    l = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'mhlo.divide' failed to satisfy constraint: 'ranked tensor of 2/4/8/16/32/64-bit signless integer or 2/4/8/16/32/64-bit unsigned integer values'"))) {
      return ::mlir::failure();
    }
    r = castedOp0.getODSOperands(1);
    if (!((cast<ShapedType>((*l.begin()).getType()).getShape() == cast<ShapedType>((*r.begin()).getType()).getShape()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'l, r' failed to satisfy constraint: 'inputs are same size'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::DivSIOp tblgen_DivSIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*l.begin()));
      tblgen_values.push_back((*r.begin()));
      tblgen_DivSIOp_0 = rewriter.create<::mlir::arith::DivSIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_DivSIOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/legalize_to_standard/legalize_to_standard_patterns.td:86
*/
struct GeneratedConvert13 : public ::mlir::RewritePattern {
  GeneratedConvert13(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.remainder", 1, context, {"arith.remsi"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range l(op0->getOperands());
    ::mlir::Operation::operand_range r(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::RemOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.remainder' failed to satisfy constraint: 'ranked tensor of 2/4/8/16/32/64-bit signless integer or 2/4/8/16/32/64-bit unsigned integer values'"))) {
      return ::mlir::failure();
    }
    l = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_legalize_to_standard_patterns2(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'mhlo.remainder' failed to satisfy constraint: 'ranked tensor of 2/4/8/16/32/64-bit signless integer or 2/4/8/16/32/64-bit unsigned integer values'"))) {
      return ::mlir::failure();
    }
    r = castedOp0.getODSOperands(1);
    if (!((cast<ShapedType>((*l.begin()).getType()).getShape() == cast<ShapedType>((*r.begin()).getType()).getShape()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'l, r' failed to satisfy constraint: 'inputs are same size'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::RemSIOp tblgen_RemSIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*l.begin()));
      tblgen_values.push_back((*r.begin()));
      tblgen_RemSIOp_0 = rewriter.create<::mlir::arith::RemSIOp>(odsLoc, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_RemSIOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/legalize_to_standard/legalize_to_standard_patterns.td:90
*/
struct GeneratedConvert14 : public ::mlir::RewritePattern {
  GeneratedConvert14(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.select", 1, context, {"arith.select"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range pred(op0->getOperands());
    ::mlir::Operation::operand_range tv(op0->getOperands());
    ::mlir::Operation::operand_range fv(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::SelectOp>(op0); (void)castedOp0;
    pred = castedOp0.getODSOperands(0);
    tv = castedOp0.getODSOperands(1);
    fv = castedOp0.getODSOperands(2);
    if (!((cast<ShapedType>((*pred.begin()).getType()).getShape() == cast<ShapedType>((*tv.begin()).getType()).getShape()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'pred, tv' failed to satisfy constraint: 'inputs are same size'";
      });
    }
    if (!((cast<ShapedType>((*tv.begin()).getType()).getShape() == cast<ShapedType>((*fv.begin()).getType()).getShape()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'tv, fv' failed to satisfy constraint: 'inputs are same size'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::SelectOp tblgen_SelectOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*pred.begin()));
      tblgen_values.push_back((*tv.begin()));
      tblgen_values.push_back((*fv.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SelectOp_0 = rewriter.create<::mlir::arith::SelectOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SelectOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<GeneratedConvert0>(patterns.getContext());
  patterns.add<GeneratedConvert1>(patterns.getContext());
  patterns.add<GeneratedConvert2>(patterns.getContext());
  patterns.add<GeneratedConvert3>(patterns.getContext());
  patterns.add<GeneratedConvert4>(patterns.getContext());
  patterns.add<GeneratedConvert5>(patterns.getContext());
  patterns.add<GeneratedConvert6>(patterns.getContext());
  patterns.add<GeneratedConvert7>(patterns.getContext());
  patterns.add<GeneratedConvert8>(patterns.getContext());
  patterns.add<GeneratedConvert9>(patterns.getContext());
  patterns.add<GeneratedConvert10>(patterns.getContext());
  patterns.add<GeneratedConvert11>(patterns.getContext());
  patterns.add<GeneratedConvert12>(patterns.getContext());
  patterns.add<GeneratedConvert13>(patterns.getContext());
  patterns.add<GeneratedConvert14>(patterns.getContext());
}
