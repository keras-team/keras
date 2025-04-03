/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: lower_complex_patterns.td                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


static ::llvm::LogicalResult __mlir_ods_local_type_constraint_lower_complex_patterns1(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Type type,
    ::llvm::StringRef failureStr) {
  if (!(((::llvm::isa<::mlir::RankedTensorType>(type))) && ([](::mlir::Type elementType) { return ((::llvm::isa<::mlir::ComplexType>(elementType))) && (((::llvm::cast<::mlir::ComplexType>(elementType).getElementType().isF32())) || ((::llvm::cast<::mlir::ComplexType>(elementType).getElementType().isF64()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": ranked tensor of complex type with 32-bit float or 64-bit float elements values";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_lower_complex_patterns1(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((attr == ::mlir::mhlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::mhlo::ComparisonDirection::NE)))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": constant attribute ::mlir::mhlo::ComparisonDirection::NE";
    });
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_lower_complex_patterns2(
    ::mlir::PatternRewriter &rewriter, ::mlir::Operation *op, ::mlir::Attribute attr,
    ::llvm::StringRef failureStr) {
  if (!((attr == ::mlir::mhlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::mhlo::ComparisonDirection::EQ)))) {
    return rewriter.notifyMatchFailure(op, [&](::mlir::Diagnostic &diag) {
      diag << failureStr << ": constant attribute ::mlir::mhlo::ComparisonDirection::EQ";
    });
  }
  return ::mlir::success();
}
/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/lower_complex/lower_complex_patterns.td:30
*/
struct GeneratedConvert0 : public ::mlir::RewritePattern {
  GeneratedConvert0(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.add", 1, context, {"mhlo.add", "mhlo.complex", "mhlo.imag", "mhlo.real"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::AddOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_complex_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.add' failed to satisfy constraint: 'ranked tensor of complex type with 32-bit float or 64-bit float elements values'"))) {
      return ::mlir::failure();
    }
    lhs = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_complex_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'mhlo.add' failed to satisfy constraint: 'ranked tensor of complex type with 32-bit float or 64-bit float elements values'"))) {
      return ::mlir::failure();
    }
    rhs = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::mhlo::RealOp tblgen_RealOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*lhs.begin());
      tblgen_RealOp_0 = rewriter.create<::mlir::mhlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::RealOp tblgen_RealOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*rhs.begin());
      tblgen_RealOp_1 = rewriter.create<::mlir::mhlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::AddOp tblgen_AddOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_RealOp_0.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_RealOp_1.getODSResults(0).begin());
      tblgen_AddOp_2 = rewriter.create<::mlir::mhlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::ImagOp tblgen_ImagOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*lhs.begin());
      tblgen_ImagOp_3 = rewriter.create<::mlir::mhlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::ImagOp tblgen_ImagOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*rhs.begin());
      tblgen_ImagOp_4 = rewriter.create<::mlir::mhlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::AddOp tblgen_AddOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ImagOp_3.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ImagOp_4.getODSResults(0).begin());
      tblgen_AddOp_5 = rewriter.create<::mlir::mhlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::ComplexOp tblgen_ComplexOp_6;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_AddOp_2.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_AddOp_5.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ComplexOp_6 = rewriter.create<::mlir::mhlo::ComplexOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ComplexOp_6.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/lower_complex/lower_complex_patterns.td:30
*/
struct GeneratedConvert1 : public ::mlir::RewritePattern {
  GeneratedConvert1(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.subtract", 1, context, {"mhlo.complex", "mhlo.imag", "mhlo.real", "mhlo.subtract"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::SubtractOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_complex_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.subtract' failed to satisfy constraint: 'ranked tensor of complex type with 32-bit float or 64-bit float elements values'"))) {
      return ::mlir::failure();
    }
    lhs = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_complex_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'mhlo.subtract' failed to satisfy constraint: 'ranked tensor of complex type with 32-bit float or 64-bit float elements values'"))) {
      return ::mlir::failure();
    }
    rhs = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::mhlo::RealOp tblgen_RealOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*lhs.begin());
      tblgen_RealOp_0 = rewriter.create<::mlir::mhlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::RealOp tblgen_RealOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*rhs.begin());
      tblgen_RealOp_1 = rewriter.create<::mlir::mhlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::SubtractOp tblgen_SubtractOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_RealOp_0.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_RealOp_1.getODSResults(0).begin());
      tblgen_SubtractOp_2 = rewriter.create<::mlir::mhlo::SubtractOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::ImagOp tblgen_ImagOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*lhs.begin());
      tblgen_ImagOp_3 = rewriter.create<::mlir::mhlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::ImagOp tblgen_ImagOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*rhs.begin());
      tblgen_ImagOp_4 = rewriter.create<::mlir::mhlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::SubtractOp tblgen_SubtractOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ImagOp_3.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ImagOp_4.getODSResults(0).begin());
      tblgen_SubtractOp_5 = rewriter.create<::mlir::mhlo::SubtractOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::ComplexOp tblgen_ComplexOp_6;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_SubtractOp_2.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_SubtractOp_5.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ComplexOp_6 = rewriter.create<::mlir::mhlo::ComplexOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ComplexOp_6.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/lower_complex/lower_complex_patterns.td:40
*/
struct GeneratedConvert2 : public ::mlir::RewritePattern {
  GeneratedConvert2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.multiply", 1, context, {"mhlo.add", "mhlo.complex", "mhlo.imag", "mhlo.multiply", "mhlo.real", "mhlo.subtract"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::MulOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_complex_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.multiply' failed to satisfy constraint: 'ranked tensor of complex type with 32-bit float or 64-bit float elements values'"))) {
      return ::mlir::failure();
    }
    lhs = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_complex_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'mhlo.multiply' failed to satisfy constraint: 'ranked tensor of complex type with 32-bit float or 64-bit float elements values'"))) {
      return ::mlir::failure();
    }
    rhs = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::mhlo::RealOp lhs_real;
    {
      ::mlir::Value tblgen_value_0 = (*lhs.begin());
      lhs_real = rewriter.create<::mlir::mhlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::RealOp rhs_real;
    {
      ::mlir::Value tblgen_value_0 = (*rhs.begin());
      rhs_real = rewriter.create<::mlir::mhlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::MulOp tblgen_MulOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*lhs_real.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*rhs_real.getODSResults(0).begin());
      tblgen_MulOp_0 = rewriter.create<::mlir::mhlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::ImagOp lhs_imag;
    {
      ::mlir::Value tblgen_value_0 = (*lhs.begin());
      lhs_imag = rewriter.create<::mlir::mhlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::ImagOp rhs_imag;
    {
      ::mlir::Value tblgen_value_0 = (*rhs.begin());
      rhs_imag = rewriter.create<::mlir::mhlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::MulOp tblgen_MulOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*lhs_imag.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*rhs_imag.getODSResults(0).begin());
      tblgen_MulOp_1 = rewriter.create<::mlir::mhlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::SubtractOp tblgen_SubtractOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_0.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_MulOp_1.getODSResults(0).begin());
      tblgen_SubtractOp_2 = rewriter.create<::mlir::mhlo::SubtractOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::MulOp tblgen_MulOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*lhs_real.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*rhs_imag.getODSResults(0).begin());
      tblgen_MulOp_3 = rewriter.create<::mlir::mhlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::MulOp tblgen_MulOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*lhs_imag.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*rhs_real.getODSResults(0).begin());
      tblgen_MulOp_4 = rewriter.create<::mlir::mhlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::AddOp tblgen_AddOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_3.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_MulOp_4.getODSResults(0).begin());
      tblgen_AddOp_5 = rewriter.create<::mlir::mhlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::ComplexOp tblgen_ComplexOp_6;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_SubtractOp_2.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_AddOp_5.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ComplexOp_6 = rewriter.create<::mlir::mhlo::ComplexOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ComplexOp_6.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/lower_complex/lower_complex_patterns.td:59
*/
struct GeneratedConvert3 : public ::mlir::RewritePattern {
  GeneratedConvert3(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.divide", 1, context, {"mhlo.add", "mhlo.complex", "mhlo.divide", "mhlo.imag", "mhlo.multiply", "mhlo.negate", "mhlo.real"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::DivOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_complex_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.divide' failed to satisfy constraint: 'ranked tensor of complex type with 32-bit float or 64-bit float elements values'"))) {
      return ::mlir::failure();
    }
    lhs = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_complex_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'mhlo.divide' failed to satisfy constraint: 'ranked tensor of complex type with 32-bit float or 64-bit float elements values'"))) {
      return ::mlir::failure();
    }
    rhs = castedOp0.getODSOperands(1);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::mhlo::RealOp tblgen_RealOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*rhs.begin());
      tblgen_RealOp_0 = rewriter.create<::mlir::mhlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::ImagOp tblgen_ImagOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*rhs.begin());
      tblgen_ImagOp_1 = rewriter.create<::mlir::mhlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::NegOp tblgen_NegOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ImagOp_1.getODSResults(0).begin());
      tblgen_NegOp_2 = rewriter.create<::mlir::mhlo::NegOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::ComplexOp conj;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_RealOp_0.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_NegOp_2.getODSResults(0).begin());
      conj = rewriter.create<::mlir::mhlo::ComplexOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::MulOp num;
    {
      ::mlir::Value tblgen_value_0 = (*lhs.begin());
      ::mlir::Value tblgen_value_1 = (*conj.getODSResults(0).begin());
      num = rewriter.create<::mlir::mhlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::RealOp tblgen_RealOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*num.getODSResults(0).begin());
      tblgen_RealOp_3 = rewriter.create<::mlir::mhlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::RealOp tblgen_RealOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*rhs.begin());
      tblgen_RealOp_4 = rewriter.create<::mlir::mhlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::RealOp tblgen_RealOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*rhs.begin());
      tblgen_RealOp_5 = rewriter.create<::mlir::mhlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::MulOp tblgen_MulOp_6;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_RealOp_4.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_RealOp_5.getODSResults(0).begin());
      tblgen_MulOp_6 = rewriter.create<::mlir::mhlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::ImagOp tblgen_ImagOp_7;
    {
      ::mlir::Value tblgen_value_0 = (*rhs.begin());
      tblgen_ImagOp_7 = rewriter.create<::mlir::mhlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::ImagOp tblgen_ImagOp_8;
    {
      ::mlir::Value tblgen_value_0 = (*rhs.begin());
      tblgen_ImagOp_8 = rewriter.create<::mlir::mhlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::MulOp tblgen_MulOp_9;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ImagOp_7.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ImagOp_8.getODSResults(0).begin());
      tblgen_MulOp_9 = rewriter.create<::mlir::mhlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::AddOp den;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_6.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_MulOp_9.getODSResults(0).begin());
      den = rewriter.create<::mlir::mhlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::DivOp tblgen_DivOp_10;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_RealOp_3.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*den.getODSResults(0).begin());
      tblgen_DivOp_10 = rewriter.create<::mlir::mhlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::ImagOp tblgen_ImagOp_11;
    {
      ::mlir::Value tblgen_value_0 = (*num.getODSResults(0).begin());
      tblgen_ImagOp_11 = rewriter.create<::mlir::mhlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::DivOp tblgen_DivOp_12;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ImagOp_11.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*den.getODSResults(0).begin());
      tblgen_DivOp_12 = rewriter.create<::mlir::mhlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::ComplexOp tblgen_ComplexOp_13;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_DivOp_10.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_DivOp_12.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ComplexOp_13 = rewriter.create<::mlir::mhlo::ComplexOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ComplexOp_13.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/lower_complex/lower_complex_patterns.td:73
*/
struct GeneratedConvert4 : public ::mlir::RewritePattern {
  GeneratedConvert4(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.abs", 1, context, {"mhlo.add", "mhlo.imag", "mhlo.multiply", "mhlo.real", "mhlo.sqrt"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range val(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::AbsOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_complex_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.abs' failed to satisfy constraint: 'ranked tensor of complex type with 32-bit float or 64-bit float elements values'"))) {
      return ::mlir::failure();
    }
    val = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::mhlo::RealOp real;
    {
      ::mlir::Value tblgen_value_0 = (*val.begin());
      real = rewriter.create<::mlir::mhlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::MulOp tblgen_MulOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*real.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*real.getODSResults(0).begin());
      tblgen_MulOp_0 = rewriter.create<::mlir::mhlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::ImagOp imag;
    {
      ::mlir::Value tblgen_value_0 = (*val.begin());
      imag = rewriter.create<::mlir::mhlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::MulOp tblgen_MulOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*imag.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*imag.getODSResults(0).begin());
      tblgen_MulOp_1 = rewriter.create<::mlir::mhlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::AddOp tblgen_AddOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_0.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_MulOp_1.getODSResults(0).begin());
      tblgen_AddOp_2 = rewriter.create<::mlir::mhlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::SqrtOp tblgen_SqrtOp_3;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_AddOp_2.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SqrtOp_3 = rewriter.create<::mlir::mhlo::SqrtOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SqrtOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/lower_complex/lower_complex_patterns.td:83
*/
struct GeneratedConvert5 : public ::mlir::RewritePattern {
  GeneratedConvert5(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.sine", 1, context, {"mhlo.add", "mhlo.complex", "mhlo.constant", "mhlo.cosine", "mhlo.divide", "mhlo.exponential", "mhlo.imag", "mhlo.multiply", "mhlo.negate", "mhlo.real", "mhlo.sine", "mhlo.subtract"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range val(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::SineOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_complex_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.sine' failed to satisfy constraint: 'ranked tensor of complex type with 32-bit float or 64-bit float elements values'"))) {
      return ::mlir::failure();
    }
    val = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::mhlo::RealOp real;
    {
      ::mlir::Value tblgen_value_0 = (*val.begin());
      real = rewriter.create<::mlir::mhlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::SineOp tblgen_SineOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*real.getODSResults(0).begin());
      tblgen_SineOp_0 = rewriter.create<::mlir::mhlo::SineOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::ImagOp imag;
    {
      ::mlir::Value tblgen_value_0 = (*val.begin());
      imag = rewriter.create<::mlir::mhlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::ExpOp exp;
    {
      ::mlir::Value tblgen_value_0 = (*imag.getODSResults(0).begin());
      exp = rewriter.create<::mlir::mhlo::ExpOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::NegOp tblgen_NegOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*imag.getODSResults(0).begin());
      tblgen_NegOp_1 = rewriter.create<::mlir::mhlo::NegOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::ExpOp nexp;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_NegOp_1.getODSResults(0).begin());
      nexp = rewriter.create<::mlir::mhlo::ExpOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::AddOp tblgen_AddOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*exp.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*nexp.getODSResults(0).begin());
      tblgen_AddOp_2 = rewriter.create<::mlir::mhlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::MulOp tblgen_MulOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_SineOp_0.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_AddOp_2.getODSResults(0).begin());
      tblgen_MulOp_3 = rewriter.create<::mlir::mhlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto nativeVar_4 = hlo::getSplat(&rewriter, (*real.getODSResults(0).begin()), 2.0); (void)nativeVar_4;
    ::mlir::mhlo::ConstantOp two;
    {
      two = rewriter.create<::mlir::mhlo::ConstantOp>(odsLoc,
        /*value=*/nativeVar_4
      );
    }
    ::mlir::mhlo::DivOp tblgen_DivOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_3.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*two.getODSResults(0).begin());
      tblgen_DivOp_5 = rewriter.create<::mlir::mhlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::CosineOp tblgen_CosineOp_6;
    {
      ::mlir::Value tblgen_value_0 = (*real.getODSResults(0).begin());
      tblgen_CosineOp_6 = rewriter.create<::mlir::mhlo::CosineOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::SubtractOp tblgen_SubtractOp_7;
    {
      ::mlir::Value tblgen_value_0 = (*exp.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*nexp.getODSResults(0).begin());
      tblgen_SubtractOp_7 = rewriter.create<::mlir::mhlo::SubtractOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::MulOp tblgen_MulOp_8;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CosineOp_6.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_SubtractOp_7.getODSResults(0).begin());
      tblgen_MulOp_8 = rewriter.create<::mlir::mhlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::DivOp tblgen_DivOp_9;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_8.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*two.getODSResults(0).begin());
      tblgen_DivOp_9 = rewriter.create<::mlir::mhlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::ComplexOp tblgen_ComplexOp_10;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_DivOp_5.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_DivOp_9.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ComplexOp_10 = rewriter.create<::mlir::mhlo::ComplexOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ComplexOp_10.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/lower_complex/lower_complex_patterns.td:101
*/
struct GeneratedConvert6 : public ::mlir::RewritePattern {
  GeneratedConvert6(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.cosine", 1, context, {"mhlo.add", "mhlo.complex", "mhlo.constant", "mhlo.cosine", "mhlo.divide", "mhlo.exponential", "mhlo.imag", "mhlo.multiply", "mhlo.negate", "mhlo.real", "mhlo.sine", "mhlo.subtract"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range val(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::CosineOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_complex_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.cosine' failed to satisfy constraint: 'ranked tensor of complex type with 32-bit float or 64-bit float elements values'"))) {
      return ::mlir::failure();
    }
    val = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::mhlo::RealOp real;
    {
      ::mlir::Value tblgen_value_0 = (*val.begin());
      real = rewriter.create<::mlir::mhlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::CosineOp tblgen_CosineOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*real.getODSResults(0).begin());
      tblgen_CosineOp_0 = rewriter.create<::mlir::mhlo::CosineOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::ImagOp imag;
    {
      ::mlir::Value tblgen_value_0 = (*val.begin());
      imag = rewriter.create<::mlir::mhlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::ExpOp exp;
    {
      ::mlir::Value tblgen_value_0 = (*imag.getODSResults(0).begin());
      exp = rewriter.create<::mlir::mhlo::ExpOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::NegOp tblgen_NegOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*imag.getODSResults(0).begin());
      tblgen_NegOp_1 = rewriter.create<::mlir::mhlo::NegOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::ExpOp nexp;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_NegOp_1.getODSResults(0).begin());
      nexp = rewriter.create<::mlir::mhlo::ExpOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::AddOp tblgen_AddOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*exp.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*nexp.getODSResults(0).begin());
      tblgen_AddOp_2 = rewriter.create<::mlir::mhlo::AddOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::MulOp tblgen_MulOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CosineOp_0.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_AddOp_2.getODSResults(0).begin());
      tblgen_MulOp_3 = rewriter.create<::mlir::mhlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    auto nativeVar_4 = hlo::getSplat(&rewriter, (*real.getODSResults(0).begin()), 2.0); (void)nativeVar_4;
    ::mlir::mhlo::ConstantOp two;
    {
      two = rewriter.create<::mlir::mhlo::ConstantOp>(odsLoc,
        /*value=*/nativeVar_4
      );
    }
    ::mlir::mhlo::DivOp tblgen_DivOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_3.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*two.getODSResults(0).begin());
      tblgen_DivOp_5 = rewriter.create<::mlir::mhlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::SineOp tblgen_SineOp_6;
    {
      ::mlir::Value tblgen_value_0 = (*real.getODSResults(0).begin());
      tblgen_SineOp_6 = rewriter.create<::mlir::mhlo::SineOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::SubtractOp tblgen_SubtractOp_7;
    {
      ::mlir::Value tblgen_value_0 = (*nexp.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*exp.getODSResults(0).begin());
      tblgen_SubtractOp_7 = rewriter.create<::mlir::mhlo::SubtractOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::MulOp tblgen_MulOp_8;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_SineOp_6.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_SubtractOp_7.getODSResults(0).begin());
      tblgen_MulOp_8 = rewriter.create<::mlir::mhlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::DivOp tblgen_DivOp_9;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_MulOp_8.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*two.getODSResults(0).begin());
      tblgen_DivOp_9 = rewriter.create<::mlir::mhlo::DivOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::ComplexOp tblgen_ComplexOp_10;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_DivOp_5.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_DivOp_9.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ComplexOp_10 = rewriter.create<::mlir::mhlo::ComplexOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ComplexOp_10.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/lower_complex/lower_complex_patterns.td:123
*/
struct GeneratedConvert7 : public ::mlir::RewritePattern {
  GeneratedConvert7(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.exponential", 1, context, {"mhlo.complex", "mhlo.cosine", "mhlo.exponential", "mhlo.imag", "mhlo.multiply", "mhlo.real", "mhlo.sine"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range val(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::ExpOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_complex_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.exponential' failed to satisfy constraint: 'ranked tensor of complex type with 32-bit float or 64-bit float elements values'"))) {
      return ::mlir::failure();
    }
    val = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::mhlo::ImagOp imag;
    {
      ::mlir::Value tblgen_value_0 = (*val.begin());
      imag = rewriter.create<::mlir::mhlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::CosineOp tblgen_CosineOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*imag.getODSResults(0).begin());
      tblgen_CosineOp_0 = rewriter.create<::mlir::mhlo::CosineOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::RealOp real;
    {
      ::mlir::Value tblgen_value_0 = (*val.begin());
      real = rewriter.create<::mlir::mhlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::ExpOp exp;
    {
      ::mlir::Value tblgen_value_0 = (*real.getODSResults(0).begin());
      exp = rewriter.create<::mlir::mhlo::ExpOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::MulOp tblgen_MulOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_CosineOp_0.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*exp.getODSResults(0).begin());
      tblgen_MulOp_1 = rewriter.create<::mlir::mhlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::SineOp tblgen_SineOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*imag.getODSResults(0).begin());
      tblgen_SineOp_2 = rewriter.create<::mlir::mhlo::SineOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::MulOp tblgen_MulOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_SineOp_2.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*exp.getODSResults(0).begin());
      tblgen_MulOp_3 = rewriter.create<::mlir::mhlo::MulOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1
      );
    }
    ::mlir::mhlo::ComplexOp tblgen_ComplexOp_4;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_MulOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_MulOp_3.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_ComplexOp_4 = rewriter.create<::mlir::mhlo::ComplexOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_ComplexOp_4.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/lower_complex/lower_complex_patterns.td:132
*/
struct GeneratedConvert8 : public ::mlir::RewritePattern {
  GeneratedConvert8(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.compare", 1, context, {"mhlo.compare", "mhlo.imag", "mhlo.or", "mhlo.real"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::mlir::mhlo::ComparisonTypeAttr compare_type;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::CompareOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_complex_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.compare' failed to satisfy constraint: 'ranked tensor of complex type with 32-bit float or 64-bit float elements values'"))) {
      return ::mlir::failure();
    }
    lhs = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_complex_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'mhlo.compare' failed to satisfy constraint: 'ranked tensor of complex type with 32-bit float or 64-bit float elements values'"))) {
      return ::mlir::failure();
    }
    rhs = castedOp0.getODSOperands(1);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::mhlo::ComparisonDirectionAttr>("comparison_direction");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'mhlo.compare' to have attribute 'comparison_direction' of type '::mlir::mhlo::ComparisonDirectionAttr'";
        });
      }
      if(::mlir::failed(__mlir_ods_local_attr_constraint_lower_complex_patterns1(rewriter, op0, tblgen_attr, "op 'mhlo.compare' attribute 'comparison_direction' failed to satisfy constraint: 'constant attribute ::mlir::mhlo::ComparisonDirection::NE'"))) {
        return ::mlir::failure();
      }
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::mhlo::ComparisonTypeAttr>("compare_type");(void)tblgen_attr;
      compare_type = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::mhlo::RealOp tblgen_RealOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*lhs.begin());
      tblgen_RealOp_0 = rewriter.create<::mlir::mhlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::RealOp tblgen_RealOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*rhs.begin());
      tblgen_RealOp_1 = rewriter.create<::mlir::mhlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::CompareOp tblgen_CompareOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_RealOp_0.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_RealOp_1.getODSResults(0).begin());
      tblgen_CompareOp_2 = rewriter.create<::mlir::mhlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::mhlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::mhlo::ComparisonDirection::NE),
        /*compare_type=*/compare_type
      );
    }
    ::mlir::mhlo::ImagOp tblgen_ImagOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*lhs.begin());
      tblgen_ImagOp_3 = rewriter.create<::mlir::mhlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::ImagOp tblgen_ImagOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*rhs.begin());
      tblgen_ImagOp_4 = rewriter.create<::mlir::mhlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::CompareOp tblgen_CompareOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ImagOp_3.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ImagOp_4.getODSResults(0).begin());
      tblgen_CompareOp_5 = rewriter.create<::mlir::mhlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::mhlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::mhlo::ComparisonDirection::NE),
        /*compare_type=*/compare_type
      );
    }
    ::mlir::mhlo::OrOp tblgen_OrOp_6;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_CompareOp_2.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_CompareOp_5.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_OrOp_6 = rewriter.create<::mlir::mhlo::OrOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_OrOp_6.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/local_xla/xla/mlir_hlo/mhlo/transforms/lower_complex/lower_complex_patterns.td:132
*/
struct GeneratedConvert9 : public ::mlir::RewritePattern {
  GeneratedConvert9(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("mhlo.compare", 1, context, {"mhlo.and", "mhlo.compare", "mhlo.imag", "mhlo.real"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range lhs(op0->getOperands());
    ::mlir::Operation::operand_range rhs(op0->getOperands());
    ::mlir::mhlo::ComparisonTypeAttr compare_type;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::mhlo::CompareOp>(op0); (void)castedOp0;
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_complex_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(0).begin()).getType(), "operand 0 of op 'mhlo.compare' failed to satisfy constraint: 'ranked tensor of complex type with 32-bit float or 64-bit float elements values'"))) {
      return ::mlir::failure();
    }
    lhs = castedOp0.getODSOperands(0);
    if(::mlir::failed(__mlir_ods_local_type_constraint_lower_complex_patterns1(rewriter, castedOp0, (*castedOp0.getODSOperands(1).begin()).getType(), "operand 1 of op 'mhlo.compare' failed to satisfy constraint: 'ranked tensor of complex type with 32-bit float or 64-bit float elements values'"))) {
      return ::mlir::failure();
    }
    rhs = castedOp0.getODSOperands(1);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::mhlo::ComparisonDirectionAttr>("comparison_direction");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'mhlo.compare' to have attribute 'comparison_direction' of type '::mlir::mhlo::ComparisonDirectionAttr'";
        });
      }
      if(::mlir::failed(__mlir_ods_local_attr_constraint_lower_complex_patterns2(rewriter, op0, tblgen_attr, "op 'mhlo.compare' attribute 'comparison_direction' failed to satisfy constraint: 'constant attribute ::mlir::mhlo::ComparisonDirection::EQ'"))) {
        return ::mlir::failure();
      }
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::mhlo::ComparisonTypeAttr>("compare_type");(void)tblgen_attr;
      compare_type = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::mhlo::RealOp tblgen_RealOp_0;
    {
      ::mlir::Value tblgen_value_0 = (*lhs.begin());
      tblgen_RealOp_0 = rewriter.create<::mlir::mhlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::RealOp tblgen_RealOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*rhs.begin());
      tblgen_RealOp_1 = rewriter.create<::mlir::mhlo::RealOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::CompareOp tblgen_CompareOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_RealOp_0.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_RealOp_1.getODSResults(0).begin());
      tblgen_CompareOp_2 = rewriter.create<::mlir::mhlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::mhlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::mhlo::ComparisonDirection::EQ),
        /*compare_type=*/compare_type
      );
    }
    ::mlir::mhlo::ImagOp tblgen_ImagOp_3;
    {
      ::mlir::Value tblgen_value_0 = (*lhs.begin());
      tblgen_ImagOp_3 = rewriter.create<::mlir::mhlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::ImagOp tblgen_ImagOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*rhs.begin());
      tblgen_ImagOp_4 = rewriter.create<::mlir::mhlo::ImagOp>(odsLoc,
        /*operand=*/tblgen_value_0
      );
    }
    ::mlir::mhlo::CompareOp tblgen_CompareOp_5;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ImagOp_3.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ImagOp_4.getODSResults(0).begin());
      tblgen_CompareOp_5 = rewriter.create<::mlir::mhlo::CompareOp>(odsLoc,
        /*lhs=*/tblgen_value_0,
        /*rhs=*/tblgen_value_1,
        ::mlir::mhlo::ComparisonDirectionAttr::get(rewriter.getContext(), ::mlir::mhlo::ComparisonDirection::EQ),
        /*compare_type=*/compare_type
      );
    }
    ::mlir::mhlo::AndOp tblgen_AndOp_6;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_CompareOp_2.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_CompareOp_5.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_AndOp_6 = rewriter.create<::mlir::mhlo::AndOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AndOp_6.getODSResults(0) }) {
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
}
